#include <core.p4>
#include <v1model.p4>

#include "includes/distpd_consts.p4"
#include "includes/distpd_headers.p4"
#include "includes/distpd_parser.p4"
#include "includes/distpd_checksums.p4"
#include "distpd_controls.p4"

/*
    Switch ingress pipeline
*/
control SwitchIngress(
    inout header_t            hdr,
    inout metadata_t          meta,
    inout standard_metadata_t standard_metadata)
{
    
    Routing()               routing;
    DeflectRouting()        deflection_routing;
    GetMinRelPrioQueueLen() get_min_rel_prio_queue_len;
    DeflectGetMinRelPrioQueueLen() deflect_get_min_rel_prio_queue_len;

    register<bit<32>>(NUM_LOGICAL_PORTS) ig_queue_length_reg;
    register<bit<32>>(NUM_LOGICAL_PORTS) ig_m_reg;
    
    action drop() {
        mark_to_drop(standard_metadata);
    }
    
    action get_flow_priority_action(bit<32> rank) {
        meta.rank = rank;
    }
    table get_flow_priority_table {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: exact;
        }
        actions = { get_flow_priority_action; }
        size = TABLE_SIZE;
    }
    
    action shift_queue_length() {
        meta.queue_length = meta.queue_length << EQUATION_MULT_SHIFT;
    }
    
    action shift_deflect_queue_length() {
        meta.deflect_queue_length = meta.deflect_queue_length << EQUATION_MULT_SHIFT;
    }
    
    action get_rel_prio_action(bit<32> rel_prio) {
        meta.rel_prio = rel_prio;
    }
    action get_deflect_rel_prio_action(bit<32> rel_prio) {
        meta.deflect_rel_prio = rel_prio;
    }
    table get_rel_prio_table {
        key = {
            meta.rank[19:0]: range;
            meta.m[19:0] : range;
        }
        actions = { get_rel_prio_action; }
        size = TABLE_SIZE;
    }
    table get_deflect_rel_prio_table {
        key = {
            meta.rank[19:0]: range;
            meta.deflect_m[19:0] : range;
        }
        actions = { get_deflect_rel_prio_action; }
        size = TABLE_SIZE;
    }
    
    apply {
        if (hdr.bee.isValid()) {
            ig_queue_length_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_length);
            ig_m_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.M);
        }
        else if (hdr.ipv4.isValid() &&
                (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                 hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
            
            get_flow_priority_table.apply();
            routing.apply(hdr, meta, standard_metadata);
            deflection_routing.apply(hdr, meta, standard_metadata);
            ig_queue_length_reg.read(meta.queue_length, (bit<32>)meta.fw_port_idx);
            ig_queue_length_reg.read(meta.deflect_queue_length, (bit<32>)meta.fw_port_idx);
            ig_m_reg.read(meta.m, (bit<32>)meta.fw_port_idx);
            ig_m_reg.read(meta.deflect_m, (bit<32>)meta.fw_port_idx);
            
            get_rel_prio_table.apply();
            get_deflect_rel_prio_table.apply();
            
            if (meta.queue_length < QUEUE_SIZE) {
                meta.queue_length = QUEUE_SIZE - meta.queue_length;
            } else {
                meta.queue_length = 0;
            }
            if (meta.deflect_queue_length < QUEUE_SIZE) {
                meta.deflect_queue_length = QUEUE_SIZE - meta.deflect_queue_length;
            } else {
                meta.deflect_queue_length = 0;
            }
            shift_queue_length();
            shift_deflect_queue_length();
            
            get_min_rel_prio_queue_len.apply(meta);
            deflect_get_min_rel_prio_queue_len.apply(meta);
            
            if (meta.min_value_rel_prio_queue_len == meta.queue_length) {
                meta.m = (bit<32>) meta.deflect_m;
                if (meta.deflect_min_value_rel_prio_queue_len != meta.deflect_queue_length) {
                    standard_metadata.egress_spec = meta.deflect_egress_spec;
                } else {
                    meta.m = 0;
                    drop();
                } 
                //TODO: looks like in TNA packets are dropped immediately, while in v1model often packets marked to be dropped in the imgress are dropped after the egress.
                //TODO: check if this is the case here, otherwise packets that in the original implementation does not count for stats (because are immediatly dropped) will count for stats.
            }
        }
    }
}


/*
    Switch Egress pipeline
*/
control SwitchEgress(
    inout header_t            hdr,
    inout metadata_t          meta,
    inout standard_metadata_t standard_metadata)
{


    register<bit<32>>(NUM_LOGICAL_PORTS) eg_queue_length_reg;
    register<bit<32>>(NUM_LOGICAL_PORTS) eg_m_reg;

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
    }
            
    action get_newm_action(bit<32> new_m) {
        meta.new_m = new_m;
    }

    table get_newm_table {
        key = {
            meta.rank[19:0]: range;
            meta.m[19:0]: range;
        }
        actions = {
            get_newm_action;
        }
        size = TABLE_SIZE;
    }

    apply {

        if (hdr.bee.isValid()) {
            eg_queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_idx_in_reg);
            eg_m_reg.read(hdr.bee.M, (bit<32>)hdr.bee.port_idx_in_reg);
            recirculate_preserving_field_list(0);
        } else if (hdr.ipv4.isValid() && 
                   meta.m != 0 &&
                   (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                    hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {

            get_eg_port_idx_in_reg_table.apply();
            eg_queue_length_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)(standard_metadata.deq_qdepth)); //Dopo che il pacchetto lascia la coda, pensaci!!
            get_newm_table.apply();
            eg_m_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)meta.new_m);
        }
                
    }
    
}



//switch architecture
V1Switch(SwitchParser(),
         SwitchVerifyChecksum(),
         SwitchIngress(),
         SwitchEgress(),
         SwitchComputeChecksum(),
         SwitchDeparser()
) main;