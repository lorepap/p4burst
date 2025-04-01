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
    register<bit<1>>(8) queue_occupancy_info;
    register<bit<19>>(8) queue_depth_info;
    register<bit<1>>(8) neighbor_switch_indicator;
    register<bit<1>>(1) is_fw_port_full_register;

    counter(1, CounterType.packets) normal_ctr;
    counter(1, CounterType.packets) deflected_ctr;
    counter(1, CounterType.packets) flow_header_counter;

    action drop() {
        log_msg("Packet dropped: ingress_port={}, reason=explicit_drop", {standard_metadata.ingress_port});
        mark_to_drop(standard_metadata);
    }

    action generate_random() {
        random(meta.random_number, 0, 7);
    }
    
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
            queue_occupancy_info.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info);
            queue_depth_info.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_depth);
            
            // log_msg("BeeIngress: port={}, queue_occ={}, queue_depth={}", 
            //        {hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info, hdr.bee.queue_depth});
        } else {
            //ingress_ctr.count(ingress_ctr_index);
            forward.apply(hdr, meta, standard_metadata);
            debug_enq_qdepth_table.apply();
            
            if (hdr.ipv4.isValid()) {

                
                if (hdr.tcp.isValid()) {
                    log_msg("Ingress: port={}, size={}, timestamp={}", {standard_metadata.ingress_port, hdr.ipv4.totalLen, standard_metadata.ingress_global_timestamp});
                    log_msg("TCP: src_ip={}, dst_ip={}, src_port={}, dst_port={}, seq={}", 
                        {hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.srcPort, hdr.tcp.dstPort, hdr.tcp.seqNo});
                }
                
                // if (hdr.flow.isValid()) {
                //     flow_header_counter.count(0);
                // }
                
                queue_occupancy_info.read(meta.is_queue_full_0, (bit<32>)0);
                queue_occupancy_info.read(meta.is_queue_full_1, (bit<32>)1);
                queue_occupancy_info.read(meta.is_queue_full_2, (bit<32>)2);
                queue_occupancy_info.read(meta.is_queue_full_3, (bit<32>)3);
                queue_occupancy_info.read(meta.is_queue_full_4, (bit<32>)4);
                queue_occupancy_info.read(meta.is_queue_full_5, (bit<32>)5);
                queue_occupancy_info.read(meta.is_queue_full_6, (bit<32>)6);
                queue_occupancy_info.read(meta.is_queue_full_7, (bit<32>)7);

                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_0, (bit<32>)0);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_1, (bit<32>)1);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_2, (bit<32>)2);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_3, (bit<32>)3);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_4, (bit<32>)4);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_5, (bit<32>)5);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_6, (bit<32>)6);
                neighbor_switch_indicator.read(meta.neighbor_switch_indicator_7, (bit<32>)7);

                queue_occupancy_info.read(meta.is_fw_port_full, (bit<32>)meta.fw_port_idx);
                
                // Read queue depth of forward port
                bit<19> fw_port_depth;
                queue_depth_info.read(fw_port_depth, (bit<32>)meta.fw_port_idx);
                
                // debug
                is_fw_port_full_register.write(0, meta.is_fw_port_full);

                generate_random();
                
                // make sure not to consider ports toward neighboring switches
                meta.is_queue_full_0 = meta.is_queue_full_0 | meta.neighbor_switch_indicator_0;
                meta.is_queue_full_1 = meta.is_queue_full_1 | meta.neighbor_switch_indicator_1;
                meta.is_queue_full_2 = meta.is_queue_full_2 | meta.neighbor_switch_indicator_2;
                meta.is_queue_full_3 = meta.is_queue_full_3 | meta.neighbor_switch_indicator_3;
                meta.is_queue_full_4 = meta.is_queue_full_4 | meta.neighbor_switch_indicator_4;
                meta.is_queue_full_5 = meta.is_queue_full_5 | meta.neighbor_switch_indicator_5;
                meta.is_queue_full_6 = meta.is_queue_full_6 | meta.neighbor_switch_indicator_6;
                meta.is_queue_full_7 = meta.is_queue_full_7 | meta.neighbor_switch_indicator_7;

                if (meta.is_fw_port_full == 1) {
                    // queue is full
                    if (meta.random_number <= 0 && meta.is_queue_full_0 == 0) {
                        meta.output_port_idx = 0;
                    } else if (meta.random_number <= 1 && meta.is_queue_full_1 == 0) {
                        meta.output_port_idx = 1;
                    } else if (meta.random_number <= 2 && meta.is_queue_full_2 == 0) {
                        meta.output_port_idx = 2;
                    } else if (meta.random_number <= 3 && meta.is_queue_full_3 == 0) {
                        meta.output_port_idx = 3;
                    } else if (meta.random_number <= 4 && meta.is_queue_full_4 == 0) {
                        meta.output_port_idx = 4;
                    } else if (meta.random_number <= 5 && meta.is_queue_full_5 == 0) {
                        meta.output_port_idx = 5;
                    } else if (meta.random_number <= 6 && meta.is_queue_full_6 == 0) {
                        meta.output_port_idx = 6;
                    } else if (meta.random_number <= 7 && meta.is_queue_full_7 == 0) {
                        meta.output_port_idx = 7;
                    } else {
                        // it's a loop check
                        if (meta.is_queue_full_0 == 0) {
                            meta.output_port_idx = 0;
                        } else if (meta.is_queue_full_1 == 0) {
                            meta.output_port_idx = 1;
                        } else if (meta.is_queue_full_2 == 0) {
                            meta.output_port_idx = 2;
                        } else if (meta.is_queue_full_3 == 0) {
                            meta.output_port_idx = 3;
                        } else if (meta.is_queue_full_4 == 0) {
                            meta.output_port_idx = 4;
                        } else if (meta.is_queue_full_5 == 0) {
                            meta.output_port_idx = 5;
                        } else if (meta.is_queue_full_6 == 0) {
                            meta.output_port_idx = 6;
                        } else if (meta.is_queue_full_7 == 0) {
                            meta.output_port_idx = 7;
                        }
                    }
                    
                    set_deflect_egress_port_table.apply();
                    deflected_ctr.count(0);
                    
                    log_msg("Deflection: original_port={}, deflected_to={}, random_number={}, fw_port_depth={}",
                            {meta.fw_port_idx+1, meta.output_port_idx+1, meta.random_number, fw_port_depth});
                }
                else {
                    // normal forwarding
                    log_msg("Normal: port={}", {meta.fw_port_idx+1});
                    normal_ctr.count(0);
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

    register<bit<1>>(8) queue_occupancy_info;
    register<bit<19>>(8) queue_depth_info;  // Change to 8 entries to store depth for each port
    register<bit<19>>(1) debug_qdepth;
    register<bit<9>>(1) debug_eg_port;
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

    table debug_deq_qdepth_table {
        key = {
            standard_metadata.deq_qdepth: exact;
        }
        actions = {
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    apply {
        if (hdr.bee.isValid()) {
            // Bee packets read both queue occupancy and queue depth
            queue_occupancy_info.read(hdr.bee.queue_occ_info, (bit<32>)hdr.bee.port_idx_in_reg);
            queue_depth_info.read(hdr.bee.queue_depth, (bit<32>)hdr.bee.port_idx_in_reg);
            
            // log_msg("BeeEgress: port={}, queue_occ={}, queue_depth={}", 
            //        {hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info, hdr.bee.queue_depth});
            
            recirculate_preserving_field_list(0);
            
        } else {
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                // Normal packet processing
                
                // Store both queue occupancy and actual queue depth
                if (standard_metadata.deq_qdepth < QUEUE_CAPACITY) {
                    meta.is_fw_port_full = 0;
                } else {
                    meta.is_fw_port_full = 1;
                }

                get_eg_port_idx_in_reg_table.apply();
                debug_eg_port.write((bit<32>)0, standard_metadata.egress_port);
                debug_qdepth.write((bit<32>)0, standard_metadata.deq_qdepth);

                // Update arrival counter as before
                bit<32> count;
                arrival_counter.read(count, (bit<32>)standard_metadata.egress_port);
                count = count + 1;
                arrival_counter.write((bit<32>)standard_metadata.egress_port, count);
                // log_msg("Arrival: port={}, count={}", {standard_metadata.egress_port, count});

                // Write both queue occupancy and queue depth to registers
                queue_occupancy_info.write((bit<32>)meta.port_idx_in_reg, meta.is_fw_port_full);
                queue_depth_info.write((bit<32>)meta.port_idx_in_reg, standard_metadata.deq_qdepth);

                // Queue depths
                bit<19> q0;
                bit<19> q1;
                bit<19> q2;
                bit<19> q3;
                bit<19> q4;
                bit<19> q5;
                bit<19> q6;
                bit<19> q7;
                queue_depth_info.read(q0, 0);
                queue_depth_info.read(q1, 1);
                queue_depth_info.read(q2, 2);
                queue_depth_info.read(q3, 3);
                queue_depth_info.read(q4, 4);
                queue_depth_info.read(q5, 5);
                queue_depth_info.read(q6, 6);
                queue_depth_info.read(q7, 7);
                
                log_msg("Queue depths: q0={} q1={} q2={} q3={} q4={} q5={} q6={} q7={}",
                        {q0, q1, q2, q3, q4, q5, q6, q7});

                bit<1> occ0; bit<1> occ1; bit<1> occ2; bit<1> occ3;
                bit<1> occ4; bit<1> occ5; bit<1> occ6; bit<1> occ7;
                queue_occupancy_info.read(occ0, (bit<32>)0);
                queue_occupancy_info.read(occ1, (bit<32>)1);
                queue_occupancy_info.read(occ2, (bit<32>)2);
                queue_occupancy_info.read(occ3, (bit<32>)3);
                queue_occupancy_info.read(occ4, (bit<32>)4);
                queue_occupancy_info.read(occ5, (bit<32>)5);
                queue_occupancy_info.read(occ6, (bit<32>)6);
                queue_occupancy_info.read(occ7, (bit<32>)7);

                // log_msg("Queue occupancy: q0={} q1={} q2={} q3={} q4={} q5={} q6={} q7={}",
                //         {occ0, occ1, occ2, occ3, occ4, occ5, occ6, occ7});
                
                // Log the forward port depth in egress to match the queue depths
                bit<19> fw_port_depth;
                queue_depth_info.read(fw_port_depth, (bit<32>)meta.fw_port_idx);
                log_msg("Forward port depth: port={}, depth={}", {meta.fw_port_idx+1, fw_port_depth});

                // Dequeue queue depth
                debug_deq_qdepth_table.apply();
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