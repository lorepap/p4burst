/* P4-16 */

/*

TODO (data collection):
- remove the flow header
- add different deflection thresholds
- deflection strategy (1: random, 2: round robin, 3: least loaded ?)

*/


#include <core.p4>
#include <v1model.p4>

#include "includes/sd_consts.p4"
#include "includes/sd_headers.p4"
#include "includes/sd_parser.p4"
#include "includes/sd_checksums.p4"
#include "sd_controls.p4"

/*
    Switch ingress pipeline
*/
control SimpleDeflectionIngress(inout header_t hdr,
                  inout metadata_t meta,
                  inout standard_metadata_t standard_metadata) {

    Forward() forward;

    // Other registers
    register<bit<19>>(8) queue_depth_info;
    register<bit<1>>(8) neighbor_switch_indicator;
    register<bit<1>>(1) is_fw_port_full_register;
    register<bit<3>>(1)max_free_space_queue_id;

    counter(1, CounterType.packets) normal_ctr;
    counter(1, CounterType.packets) deflected_ctr;
    counter(1, CounterType.packets) dropped_ctr;
    counter(1, CounterType.packets) flow_header_counter;

    action set_physical_deflect_port_from_id(bit<9> physical_port) {
        meta.deflect_port = physical_port;
    }

    table set_physical_deflect_port_from_id_table {
        key = {
            meta.deflect_port: exact;
        }
        actions = {
            set_physical_deflect_port_from_id;
        }
        size = 8;
    }


    action drop() {
        log_msg("Packet dropped: ingress_port={}, reason=explicit_drop", {standard_metadata.ingress_port});
        mark_to_drop(standard_metadata);
    }

    /*
    action set_deflect_egress_port_action(bit<9> idx) {
        standard_metadata.egress_spec = idx;
    }

    table set_deflect_egress_port_table {
        key = {
            meta.output_port_idx : exact;
        }
        actions = {
            set_deflect_egress_port_action;
            drop;
        }
        size = TABLE_SIZE;
        default_action = drop();
    }
    */

    table debug_enq_qdepth_table {
        key = {
            standard_metadata.enq_qdepth: exact;
        }
        actions = {
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
     

    apply {
        if (hdr.bee.isValid()) {
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator, (bit<32>)hdr.bee.port_idx_in_reg);
            queue_depth_info.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_depth);
            if(meta.neighbor_switch_indicator == 1){
                queue_depth_info.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_depth);
                bit<3> max_free_space_queue_id_tmp;
                max_free_space_queue_id.read(max_free_space_queue_id_tmp, (bit<32>)hdr.bee.port_idx_in_reg);
                bit<19> max_free_space_queue_occupancy;
                queue_depth_info.read(max_free_space_queue_occupancy, (bit<32>)max_free_space_queue_id_tmp);
                bit<19> max_free_space_queue = QUEUE_CAPACITY - max_free_space_queue_occupancy;
                bit<19> free_space = QUEUE_CAPACITY - hdr.bee.queue_depth;
                if (free_space > max_free_space_queue) {
                    max_free_space_queue_id.write((bit<32>)0, (bit<3>)hdr.bee.port_idx_in_reg);
                }
            }
            // log_msg("BeeIngress: port={}, queue_occ={}, queue_depth={}", 
            //        {hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info, hdr.bee.queue_depth});
        } else {
            //ingress_ctr.count(ingress_ctr_index);
            forward.apply(hdr, meta, standard_metadata);
            debug_enq_qdepth_table.apply();
            
            if (hdr.ipv4.isValid()) {
                
                max_free_space_queue_id.read(meta.deflect_port_id, (bit<32>)0);
                set_physical_deflect_port_from_id_table.apply();
                queue_depth_info.read(meta.queue_length, (bit<32>)meta.fw_port_idx);
                queue_depth_info.read(meta.deflect_queue_length, (bit<32>)meta.deflect_port_id);
                
                if (meta.queue_length < QUEUE_CAPACITY) {
                    // normal forwarding
                    log_msg("Normal: port={}", {meta.fw_port_idx+1});
                    normal_ctr.count(0);
                    
                } else if(meta.deflect_queue_length < QUEUE_CAPACITY) {
                    // deflection
                    standard_metadata.egress_spec = meta.deflect_port;
                    deflected_ctr.count(0);
                    //log_msg("Deflection: original_port={}, deflected_to={}, random_number={}, fw_port_depth={}",
                    //        {meta.fw_port_idx+1, meta.output_port_idx+1, meta.random_number, fw_port_depth});
                } else {
                    // drop
                    dropped_ctr.count(0);
                    //log_msg("Drop: original_port={}, deflected_to={}, random_number={}, fw_port_depth={}",
                    //        {meta.fw_port_idx+1, meta.output_port_idx+1, meta.random_number, fw_port_depth});
                    drop();  
                }
            }
        }
    }
}


/*
    Switch Egress pipeline
*/
control SimpleDeflectionEgress(inout header_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {

    register<bit<19>>(8) queue_depth_info;  // Change to 8 entries to store depth for each port
    register<bit<32>>(8) arrival_counter;

    // TODO: Following action and table can be avoided if we unify output_port_idx and fw_port_idx.
    //       This would avoid a table lookup, but like that we can count the number of deflected packets.
    action get_eg_port_idx_in_reg_action(bit<16> index) {
        meta.port_idx_in_reg = index;
    }
    
    table get_eg_port_idx_in_reg_table {
        key = {
            standard_metadata.egress_port: exact;
        }
        actions = {
            get_eg_port_idx_in_reg_action;
        }
        size = TABLE_SIZE;
        // const default_action = set_eg_queue_length_action;
    }

    apply {
        if (hdr.bee.isValid()) {
            // Bee packets read both queue occupancy and queue depth
            queue_depth_info.read(hdr.bee.queue_depth, (bit<32>)hdr.bee.port_idx_in_reg);
            
            // log_msg("BeeEgress: port={}, queue_occ={}, queue_depth={}", 
            //        {hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info, hdr.bee.queue_depth});
            
            recirculate_preserving_field_list(0);
            
        } else {
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                // Normal packet processing
                
                get_eg_port_idx_in_reg_table.apply();
     
                queue_depth_info.write((bit<32>)meta.port_idx_in_reg, standard_metadata.enq_qdepth);
            }
        }
    }
}


// Switch architecture

V1Switch(
    SimpleDeflectionParser(),
    SimpleDeflectionVerifyChecksum(),
    SimpleDeflectionIngress(),
    SimpleDeflectionEgress(), 
    SimpleDeflectionComputeChecksum(),
    SimpleDeflectionDeparser()
) main;