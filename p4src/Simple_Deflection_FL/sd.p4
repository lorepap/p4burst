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
control SwitchIngress(inout header_t hdr,
                  inout metadata_t meta,
                  inout standard_metadata_t standard_metadata) {

    Routing() routing;

    register<bit<1>>(8) queue_occupancy_info;

    // Better to use constants, since they are not changing. For the moment a regiter is more usefull since avoids to create different costants for each switch.
    register<bit<1>>(8) neighbor_switch_indicator;

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action generate_random() {
        hash(meta.random_number,
        HashAlgorithm.crc16,  // Using CRC16 for deterministic output
        (bit<16>)0,           // Base value
        { hdr.ipv4.srcAddr,   // Source IP
          hdr.ipv4.dstAddr,   // Destination IP
          hdr.ipv4.protocol,  // Protocol
          hdr.tcp.srcPort,    // Source port (if TCP)
          hdr.tcp.dstPort },  // Destination port (if TCP)
        (bit<16>)8);
    }
    
    action set_deflect_eggress_port_action(bit<9> idx) {
        standard_metadata.egress_spec = idx;
    }

    table set_deflect_eggress_port_table {
        key = {
            meta.output_port_idx : exact;
        }
        actions = {
            set_deflect_eggress_port_action;
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
            //meta.is_recirculated = 1;
            //resubmit_preserving_field_list((bit<8>)1);
        } else {
            //ingress_ctr.count(ingress_ctr_index);
            routing.apply(hdr, meta, standard_metadata);
            debug_enq_qdepth_table.apply();
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                
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
                    
                    set_deflect_eggress_port_table.apply();
                    //deflect_ctr.count(deflect_ctr_index);
                    //ucast_port_debug_alu.execute(0);    // what is the value for idx
                }
            } 
        }
    }
}


/*
    Switch Egress pipeline
*/
control SwitchEgress(inout header_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {

    register<bit<1>>(8) queue_occupancy_info;

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
            // At the egress, worker packets should only read from the queue occupancy register array
            // TODO: In BMv2 metadata can be preserved during recirculation. No need for bee header.
            //       Also, bee packets don't need to reach UDP, can we send L2 packets with specific metadata?
            queue_occupancy_info.read(hdr.bee.queue_occ_info, (bit<32>)hdr.bee.port_idx_in_reg);
            recirculate_preserving_field_list(0);
            
        } else {
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                // At the egress, data packets should write into the queue occupancy register array
                debug_deq_qdepth_table.apply();
                if (standard_metadata.deq_qdepth < QUEUE_SIZE) {
                    meta.is_fw_port_full = 0; // Possible to write the register directly, but this is more readable
                } else {
                    meta.is_fw_port_full = 1;
                }

                get_eg_port_idx_in_reg_table.apply();

                queue_occupancy_info.write((bit<32>)meta.port_idx_in_reg, meta.is_fw_port_full);
            }
        }
    }
}


// Switch architecture

V1Switch(
    SwitchParser(),
    SwitchVerifyChecksum(),
    SwitchIngress(),
    SwitchEgress(), 
    SwitchComputeChecksum(),
    SwitchDeparser()
) main;