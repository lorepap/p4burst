#include <core.p4>
#include <v1model.p4>

#include "/home/ubuntu/extern_lib/declaration.p4"
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
    Time() timer;

    register<bit<1>>(32) queue_occupancy_info;

    // Better to use constants, since they are not changing. For the moment a regiter is more usefull since avoids to create different costants for each switch.
    register<bit<1>>(32) neighbor_switch_indicator;

    // Contatori per la misurazione delle prestazioni
    counter(1, CounterType.packets) drop_counter;
    counter(1, CounterType.packets) deflect_counter;
    counter(1, CounterType.packets) packet_counter;
    counter(1, CounterType.packets) implicitely_dropped;
    // Counter per i pacchetti processati completamente dall'ingress
    counter(1, CounterType.packets) ingress_packet_counter;

    // Registro per tempistica ingress
    register<bit<64>>(1) reg_ing_sum;
    register<bit<64>>(1) reg_ing_max_time;

    action drop() {
        implicitely_dropped.count(0);
        mark_to_drop(standard_metadata);
    }

    action generate_random() {
        random(meta.random_number, 0, 31);
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
     

    apply {

        timer.get_time_ns(meta.t1);
        
        if (hdr.bee.isValid()) {
            queue_occupancy_info.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info);
            //meta.is_recirculated = 1;
            //resubmit_preserving_field_list((bit<8>)1);
        } else if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
            packet_counter.count(0);
            //ingress_ctr.count(ingress_ctr_index);
            routing.apply(hdr, meta, standard_metadata);
                
            queue_occupancy_info.read(meta.is_queue_full_0, (bit<32>)0);
            queue_occupancy_info.read(meta.is_queue_full_1, (bit<32>)1);
            queue_occupancy_info.read(meta.is_queue_full_2, (bit<32>)2);
            queue_occupancy_info.read(meta.is_queue_full_3, (bit<32>)3);
            queue_occupancy_info.read(meta.is_queue_full_4, (bit<32>)4);
            queue_occupancy_info.read(meta.is_queue_full_5, (bit<32>)5);
            queue_occupancy_info.read(meta.is_queue_full_6, (bit<32>)6);
            queue_occupancy_info.read(meta.is_queue_full_7, (bit<32>)7);
            queue_occupancy_info.read(meta.is_queue_full_8, (bit<32>)8);
            queue_occupancy_info.read(meta.is_queue_full_9, (bit<32>)9);
            queue_occupancy_info.read(meta.is_queue_full_10, (bit<32>)10);
            queue_occupancy_info.read(meta.is_queue_full_11, (bit<32>)11);
            queue_occupancy_info.read(meta.is_queue_full_12, (bit<32>)12);
            queue_occupancy_info.read(meta.is_queue_full_13, (bit<32>)13);
            queue_occupancy_info.read(meta.is_queue_full_14, (bit<32>)14);
            queue_occupancy_info.read(meta.is_queue_full_15, (bit<32>)15);
            queue_occupancy_info.read(meta.is_queue_full_16, (bit<32>)16);
            queue_occupancy_info.read(meta.is_queue_full_17, (bit<32>)17);
            queue_occupancy_info.read(meta.is_queue_full_18, (bit<32>)18);
            queue_occupancy_info.read(meta.is_queue_full_19, (bit<32>)19);
            queue_occupancy_info.read(meta.is_queue_full_20, (bit<32>)20);
            queue_occupancy_info.read(meta.is_queue_full_21, (bit<32>)21);
            queue_occupancy_info.read(meta.is_queue_full_22, (bit<32>)22);
            queue_occupancy_info.read(meta.is_queue_full_23, (bit<32>)23);
            queue_occupancy_info.read(meta.is_queue_full_24, (bit<32>)24);
            queue_occupancy_info.read(meta.is_queue_full_25, (bit<32>)25);
            queue_occupancy_info.read(meta.is_queue_full_26, (bit<32>)26);
            queue_occupancy_info.read(meta.is_queue_full_27, (bit<32>)27);
            queue_occupancy_info.read(meta.is_queue_full_28, (bit<32>)28);
            queue_occupancy_info.read(meta.is_queue_full_29, (bit<32>)29);
            queue_occupancy_info.read(meta.is_queue_full_30, (bit<32>)30);
            queue_occupancy_info.read(meta.is_queue_full_31, (bit<32>)31);

            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_0, (bit<32>)0);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_1, (bit<32>)1);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_2, (bit<32>)2);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_3, (bit<32>)3);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_4, (bit<32>)4);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_5, (bit<32>)5);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_6, (bit<32>)6);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_7, (bit<32>)7);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_8, (bit<32>)8);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_9, (bit<32>)9);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_10, (bit<32>)10);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_11, (bit<32>)11);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_12, (bit<32>)12);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_13, (bit<32>)13);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_14, (bit<32>)14);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_15, (bit<32>)15);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_16, (bit<32>)16);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_17, (bit<32>)17);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_18, (bit<32>)18);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_19, (bit<32>)19);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_20, (bit<32>)20);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_21, (bit<32>)21);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_22, (bit<32>)22);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_23, (bit<32>)23);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_24, (bit<32>)24);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_25, (bit<32>)25);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_26, (bit<32>)26);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_27, (bit<32>)27);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_28, (bit<32>)28);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_29, (bit<32>)29);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_30, (bit<32>)30);
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator_31, (bit<32>)31);

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
            meta.is_queue_full_8 = meta.is_queue_full_8 | meta.neighbor_switch_indicator_8;
            meta.is_queue_full_9 = meta.is_queue_full_9 | meta.neighbor_switch_indicator_9;
            meta.is_queue_full_10 = meta.is_queue_full_10 | meta.neighbor_switch_indicator_10;
            meta.is_queue_full_11 = meta.is_queue_full_11 | meta.neighbor_switch_indicator_11;
            meta.is_queue_full_12 = meta.is_queue_full_12 | meta.neighbor_switch_indicator_12;
            meta.is_queue_full_13 = meta.is_queue_full_13 | meta.neighbor_switch_indicator_13;
            meta.is_queue_full_14 = meta.is_queue_full_14 | meta.neighbor_switch_indicator_14;
            meta.is_queue_full_15 = meta.is_queue_full_15 | meta.neighbor_switch_indicator_15;
            meta.is_queue_full_16 = meta.is_queue_full_16 | meta.neighbor_switch_indicator_16;
            meta.is_queue_full_17 = meta.is_queue_full_17 | meta.neighbor_switch_indicator_17;
            meta.is_queue_full_18 = meta.is_queue_full_18 | meta.neighbor_switch_indicator_18;
            meta.is_queue_full_19 = meta.is_queue_full_19 | meta.neighbor_switch_indicator_19;
            meta.is_queue_full_20 = meta.is_queue_full_20 | meta.neighbor_switch_indicator_20;
            meta.is_queue_full_21 = meta.is_queue_full_21 | meta.neighbor_switch_indicator_21;
            meta.is_queue_full_22 = meta.is_queue_full_22 | meta.neighbor_switch_indicator_22;
            meta.is_queue_full_23 = meta.is_queue_full_23 | meta.neighbor_switch_indicator_23;
            meta.is_queue_full_24 = meta.is_queue_full_24 | meta.neighbor_switch_indicator_24;
            meta.is_queue_full_25 = meta.is_queue_full_25 | meta.neighbor_switch_indicator_25;
            meta.is_queue_full_26 = meta.is_queue_full_26 | meta.neighbor_switch_indicator_26;
            meta.is_queue_full_27 = meta.is_queue_full_27 | meta.neighbor_switch_indicator_27;
            meta.is_queue_full_28 = meta.is_queue_full_28 | meta.neighbor_switch_indicator_28;
            meta.is_queue_full_29 = meta.is_queue_full_29 | meta.neighbor_switch_indicator_29;
            meta.is_queue_full_30 = meta.is_queue_full_30 | meta.neighbor_switch_indicator_30;
            meta.is_queue_full_31 = meta.is_queue_full_31 | meta.neighbor_switch_indicator_31;

            if (meta.is_fw_port_full == 1) {
                // queue is full
                if (meta.random_number == 0 && meta.is_queue_full_0 == 0) {
                    meta.output_port_idx = 0;
                } else if (meta.random_number == 1 && meta.is_queue_full_1 == 0) {
                    meta.output_port_idx = 1;
                } else if (meta.random_number == 2 && meta.is_queue_full_2 == 0) {
                    meta.output_port_idx = 2;
                } else if (meta.random_number == 3 && meta.is_queue_full_3 == 0) {
                    meta.output_port_idx = 3;
                } else if (meta.random_number == 4 && meta.is_queue_full_4 == 0) {
                    meta.output_port_idx = 4;
                } else if (meta.random_number == 5 && meta.is_queue_full_5 == 0) {
                    meta.output_port_idx = 5;
                } else if (meta.random_number == 6 && meta.is_queue_full_6 == 0) {
                    meta.output_port_idx = 6;
                } else if (meta.random_number == 7 && meta.is_queue_full_7 == 0) {
                    meta.output_port_idx = 7;
                } else if (meta.random_number == 8 && meta.is_queue_full_8 == 0) {
                    meta.output_port_idx = 8;
                } else if (meta.random_number == 9 && meta.is_queue_full_9 == 0) {
                    meta.output_port_idx = 9;
                } else if (meta.random_number == 10 && meta.is_queue_full_10 == 0) {
                    meta.output_port_idx = 10;
                } else if (meta.random_number == 11 && meta.is_queue_full_11 == 0) {
                    meta.output_port_idx = 11;
                } else if (meta.random_number == 12 && meta.is_queue_full_12 == 0) {
                    meta.output_port_idx = 12;
                } else if (meta.random_number == 13 && meta.is_queue_full_13 == 0) {
                    meta.output_port_idx = 13;
                } else if (meta.random_number == 14 && meta.is_queue_full_14 == 0) {
                    meta.output_port_idx = 14;
                } else if (meta.random_number == 15 && meta.is_queue_full_15 == 0) {
                    meta.output_port_idx = 15;
                } else if (meta.random_number == 16 && meta.is_queue_full_16 == 0) {
                    meta.output_port_idx = 16;
                } else if (meta.random_number == 17 && meta.is_queue_full_17 == 0) {
                    meta.output_port_idx = 17;
                } else if (meta.random_number == 18 && meta.is_queue_full_18 == 0) {
                    meta.output_port_idx = 18;
                } else if (meta.random_number == 19 && meta.is_queue_full_19 == 0) {
                    meta.output_port_idx = 19;
                } else if (meta.random_number == 20 && meta.is_queue_full_20 == 0) {
                    meta.output_port_idx = 20;
                } else if (meta.random_number == 21 && meta.is_queue_full_21 == 0) {
                    meta.output_port_idx = 21;
                } else if (meta.random_number == 22 && meta.is_queue_full_22 == 0) {
                    meta.output_port_idx = 22;
                } else if (meta.random_number == 23 && meta.is_queue_full_23 == 0) {
                    meta.output_port_idx = 23;
                } else if (meta.random_number == 24 && meta.is_queue_full_24 == 0) {
                    meta.output_port_idx = 24;
                } else if (meta.random_number == 25 && meta.is_queue_full_25 == 0) {
                    meta.output_port_idx = 25;
                } else if (meta.random_number == 26 && meta.is_queue_full_26 == 0) {
                    meta.output_port_idx = 26;
                } else if (meta.random_number == 27 && meta.is_queue_full_27 == 0) {
                    meta.output_port_idx = 27;
                } else if (meta.random_number == 28 && meta.is_queue_full_28 == 0) {
                    meta.output_port_idx = 28;
                } else if (meta.random_number == 29 && meta.is_queue_full_29 == 0) {
                    meta.output_port_idx = 29;
                } else if (meta.random_number == 30 && meta.is_queue_full_30 == 0) {
                    meta.output_port_idx = 30;
                } else if (meta.random_number == 31 && meta.is_queue_full_31 == 0) {
                    meta.output_port_idx = 31;
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
                    } else if (meta.is_queue_full_8 == 0) {
                        meta.output_port_idx = 8;
                    } else if (meta.is_queue_full_9 == 0) {
                        meta.output_port_idx = 9;
                    } else if (meta.is_queue_full_10 == 0) {
                        meta.output_port_idx = 10;
                    } else if (meta.is_queue_full_11 == 0) {
                        meta.output_port_idx = 11;
                    } else if (meta.is_queue_full_12 == 0) {
                        meta.output_port_idx = 12;
                    } else if (meta.is_queue_full_13 == 0) {
                        meta.output_port_idx = 13;
                    } else if (meta.is_queue_full_14 == 0) {
                        meta.output_port_idx = 14;
                    } else if (meta.is_queue_full_15 == 0) {
                        meta.output_port_idx = 15;
                    } else if (meta.is_queue_full_16 == 0) {
                        meta.output_port_idx = 16;
                    } else if (meta.is_queue_full_17 == 0) {
                        meta.output_port_idx = 17;
                    } else if (meta.is_queue_full_18 == 0) {
                        meta.output_port_idx = 18;
                    } else if (meta.is_queue_full_19 == 0) {
                        meta.output_port_idx = 19;
                    } else if (meta.is_queue_full_20 == 0) {
                        meta.output_port_idx = 20;
                    } else if (meta.is_queue_full_21 == 0) {
                        meta.output_port_idx = 21;
                    } else if (meta.is_queue_full_22 == 0) {
                        meta.output_port_idx = 22;
                    } else if (meta.is_queue_full_23 == 0) {
                        meta.output_port_idx = 23;
                    } else if (meta.is_queue_full_24 == 0) {
                        meta.output_port_idx = 24;
                    } else if (meta.is_queue_full_25 == 0) {
                        meta.output_port_idx = 25;
                    } else if (meta.is_queue_full_26 == 0) {
                        meta.output_port_idx = 26;
                    } else if (meta.is_queue_full_27 == 0) {
                        meta.output_port_idx = 27;
                    } else if (meta.is_queue_full_28 == 0) {
                        meta.output_port_idx = 28;
                    } else if (meta.is_queue_full_29 == 0) {
                        meta.output_port_idx = 29;
                    } else if (meta.is_queue_full_30 == 0) {
                        meta.output_port_idx = 30;
                    } else if (meta.is_queue_full_31 == 0) {
                        meta.output_port_idx = 31;
                    } else {
                        // drop
                        drop_counter.count(0);
                        drop();
                        meta.dropped = 1;
                    }
                }
                    
                if(meta.dropped != 1){
                    set_deflect_eggress_port_table.apply();
                    deflect_counter.count(0);
                }
            }
            // Alla fine dell'ingress, registriamo il timestamp finale
            timer.get_time_ns(meta.t_ing_end);
            // Contiamo i pacchetti che completano l'ingress
            ingress_packet_counter.count(0);
            // Salviamo il tempo di elaborazione ingress 
            bit<64> ing_process_time = meta.t_ing_end - meta.t1;
            bit<64> ing_sum;
            reg_ing_sum.read(ing_sum, 0);
            ing_sum = ing_sum + ing_process_time;
            reg_ing_sum.write(0, ing_sum);

            // Aggiorniamo il tempo massimo di ingress
            bit<64> ing_max_time;
            reg_ing_max_time.read(ing_max_time, 0);
            if (ing_process_time > ing_max_time) {
                reg_ing_max_time.write(0, ing_process_time);
            }
        }
        // Alla fine dell'ingress, registriamo il timestamp finale
        timer.get_time_ns(meta.t_ing_end);
        // Contiamo i pacchetti che completano l'ingress
        ingress_packet_counter.count(0);
        // Salviamo il tempo di elaborazione ingress 
        bit<64> ing_process_time = meta.t_ing_end - meta.t1;
        bit<64> ing_sum;
        reg_ing_sum.read(ing_sum, 0);
        ing_sum = ing_sum + ing_process_time;
        reg_ing_sum.write(0, ing_sum);
            
            // Aggiorniamo il tempo massimo di ingress
        bit<64> ing_max_time;
        reg_ing_max_time.read(ing_max_time, 0);
        if (ing_process_time > ing_max_time) {
            reg_ing_max_time.write(0, ing_process_time);
        } 
    }  
}


/*
    Switch Egress pipeline
*/
control SwitchEgress(inout header_t hdr,
                 inout metadata_t meta,
                 inout standard_metadata_t standard_metadata) {

    register<bit<1>>(32) queue_occupancy_info;
    Time() timer;
    counter(1, CounterType.packets) egress_packet_counter;
    register<bit<64>>(1) reg_sum;
    // Registro per memorizzare il tempo di processamento massimo
    register<bit<64>>(1) reg_max_time;
    // Registro per tempistica egress
    register<bit<64>>(1) reg_egr_sum;
    register<bit<64>>(1) reg_egr_max_time;

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

        timer.get_time_ns(meta.t_egr_start);

        if (hdr.bee.isValid()) {
            // At the egress, worker packets should only read from the queue occupancy register array
            // TODO: In BMv2 metadata can be preserved during recirculation. No need for bee header.
            //       Also, bee packets don't need to reach UDP, can we send L2 packets with specific metadata?
            queue_occupancy_info.read(hdr.bee.queue_occ_info, (bit<32>)hdr.bee.port_idx_in_reg);
            recirculate_preserving_field_list(0);
            
        } else if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
        
            if (standard_metadata.deq_qdepth < QUEUE_CAPACITY) {
                meta.is_fw_port_full = 0; // Possible to write the register directly, but this is more readable
            } else {
                meta.is_fw_port_full = 1;
            }

            get_eg_port_idx_in_reg_table.apply();

            queue_occupancy_info.write((bit<32>)meta.port_idx_in_reg, meta.is_fw_port_full);
                
            timer.get_time_ns(meta.t2);
            // Tempo totale di elaborazione
            bit<64> process_time;
            bit<64> sum;
            reg_sum.read(sum, 0);
            process_time = meta.t2 - meta.t1;
            sum = sum + process_time;
            reg_sum.write(0, sum);
                
                // Tempo di elaborazione egress
            bit<64> egr_process_time = meta.t2 - meta.t_egr_start;
            bit<64> egr_sum;
            reg_egr_sum.read(egr_sum, 0);
            egr_sum = egr_sum + egr_process_time;
            reg_egr_sum.write(0, egr_sum);
                
                // Aggiorniamo i tempi massimi
            bit<64> max_time;
            reg_max_time.read(max_time, 0);
            if (process_time > max_time) {
                reg_max_time.write(0, process_time);
            }
                
            bit<64> egr_max_time;
            reg_egr_max_time.read(egr_max_time, 0);
            if (egr_process_time > egr_max_time) {
                reg_egr_max_time.write(0, egr_process_time);
            }
                
            egress_packet_counter.count(0);
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