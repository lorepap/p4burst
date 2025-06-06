#include <core.p4>
#include <v1model.p4>

#include "../extern_lib/declaration.p4"
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
    Time() timer;

    register<bit<32>>(NUM_LOGICAL_PORTS) ig_queue_length_reg;
    register<bit<32>>(NUM_LOGICAL_PORTS) ig_m_reg;

    counter(1, CounterType.packets) drop_counter;
    counter(1, CounterType.packets) deflect_counter;
    counter(1, CounterType.packets) packet_counter;
    counter(1, CounterType.packets) implicitely_dropped;
    // Counter per i pacchetti processati completamente dall'ingress
    //counter(1, CounterType.packets) ingress_packet_counter;
    
    // Registro per tempistica ingress
    //register<bit<64>>(1) reg_ing_sum;
    //register<bit<64>>(1) reg_ing_max_time;

    action drop() {
        implicitely_dropped.count(0);
        mark_to_drop(standard_metadata);
    }
    
    action get_flow_priority_action(bit<32> rank) {
        meta.rank = rank;
    }
    table get_flow_priority_table {
        key = {
            hdr.tcp.dstPort: exact;
        }
        actions = { get_flow_priority_action; }
        size = 2048;
    }
    
    
    action shift_queue_length() {
        meta.queue_length = meta.queue_length << 2;
    }
    
    action shift_deflect_queue_length() {
        meta.deflect_queue_length = meta.deflect_queue_length << 2;
    }
    
    
    action get_rel_prio_action(bit<32> rel_prio) {
        meta.rel_prio = rel_prio; // << PRIO_SHIFT;
    }
    action get_deflect_rel_prio_action(bit<32> rel_prio) {
        meta.deflect_rel_prio = rel_prio; // << PRIO_SHIFT;
    }
    table get_rel_prio_table {
        key = {
            meta.rank: range;
            meta.m : range;
        }
        actions = { get_rel_prio_action; }
        size = TABLE_SIZE;
    }
    table get_deflect_rel_prio_table {
        key = {
            meta.rank: range;
            meta.deflect_m : range;
        }
        actions = { get_deflect_rel_prio_action; }
        size = TABLE_SIZE;
    }
    
    apply {
        timer.get_time_ns(meta.t1);
        if (hdr.bee.isValid()) {
            ig_queue_length_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_length);
            ig_m_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.M);
        }
        else if (hdr.ipv4.isValid() &&
                (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                 hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                    
            packet_counter.count(0);
            
            // Prima di applicare i control, imposta needs_drop a 0
            meta.needs_drop = 0;
            
            get_flow_priority_table.apply();
            routing.apply(hdr, meta, standard_metadata);
            deflection_routing.apply(hdr, meta, standard_metadata);
            
            // Dopo i control, controlla se needs_drop è stato impostato a 1
            if (meta.needs_drop == 1) {
                drop();  // Chiama l'azione drop locale che incrementa il contatore e fa mark_to_drop
            }
            
            ig_queue_length_reg.read(meta.queue_length, (bit<32>)meta.fw_port_idx);
            ig_queue_length_reg.read(meta.deflect_queue_length, (bit<32>)meta.deflect_fw_port_idx);
            ig_m_reg.read(meta.m, (bit<32>)meta.fw_port_idx);
            ig_m_reg.read(meta.deflect_m, (bit<32>)meta.deflect_fw_port_idx);
            
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
            
            // Diminusce la prbabilità di deflessione commentare per testare la deflessione.
            shift_queue_length();
            shift_deflect_queue_length();
            
            get_min_rel_prio_queue_len.apply(meta);
            deflect_get_min_rel_prio_queue_len.apply(meta);
            
            if (meta.min_value_rel_prio_queue_len == meta.queue_length) {
                meta.m = (bit<32>) meta.deflect_m;
                if (meta.deflect_min_value_rel_prio_queue_len != meta.deflect_queue_length) {
                    deflect_counter.count(0);
                    standard_metadata.egress_spec = meta.deflect_egress_spec;
                } else {
                    drop_counter.count(0);
                    drop();
                } 
                //TODO: looks like in TNA packets are dropped immediately, while in v1model often packets marked to be dropped in the imgress are dropped after the egress.
                //TODO: check if this is the case here, otherwise packets that in the original implementation does not count for stats (because are immediatly dropped) will count for stats.
            }
            // Alla fine dell'ingress, registriamo il timestamp finale
            //timer.get_time_ns(meta.t_ing_end);
            // Contiamo i pacchetti che completano l'ingress
            //ingress_packet_counter.count(0);
            // Salviamo il tempo di elaborazione ingress 
            /*bit<64> ing_process_time = meta.t_ing_end - meta.t1;
            bit<64> ing_sum;
            reg_ing_sum.read(ing_sum, 0);
            ing_sum = ing_sum + ing_process_time;
            reg_ing_sum.write(0, ing_sum);

            // Aggiorniamo il tempo massimo di ingress
            bit<64> ing_max_time;
            reg_ing_max_time.read(ing_max_time, 0);
            if (ing_process_time > ing_max_time) {
                reg_ing_max_time.write(0, ing_process_time);
            }*/
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

    Time() timer;
    register<bit<32>>(NUM_LOGICAL_PORTS) eg_queue_length_reg;
    register<bit<32>>(NUM_LOGICAL_PORTS) eg_m_reg;
    counter(1, CounterType.packets) egress_packet_counter;
    register<bit<64>>(1) reg_sum;
    // Registro per memorizzare il tempo di processamento massimo
    register<bit<64>>(1) reg_max_time;
    // Registro per tempistica egress
    //register<bit<64>>(1) reg_egr_sum;
    //register<bit<64>>(1) reg_egr_max_time;

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

        //timer.get_time_ns(meta.t_egr_start);
        
        if (hdr.bee.isValid()) {
            eg_queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_idx_in_reg);
            eg_m_reg.read(hdr.bee.M, (bit<32>)hdr.bee.port_idx_in_reg);
            recirculate_preserving_field_list(0);
        } else if (hdr.ipv4.isValid() && 
                   (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                    hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {

            egress_packet_counter.count(0);

            get_eg_port_idx_in_reg_table.apply();
            eg_queue_length_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)(standard_metadata.deq_qdepth)); //Dopo che il pacchetto lascia la coda, pensaci!!
            get_newm_table.apply();
            eg_m_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)meta.new_m);
            timer.get_time_ns(meta.t2);
            bit<64> process_time;
            bit<64> sum;
            reg_sum.read(sum, 0);
            bit<64> max_time;
            reg_max_time.read(max_time, 0);
            process_time = meta.t2 - meta.t1;
            bit<64> sleep_time = 50000 - process_time;
            if (sleep_time > 0) {
                timer.sleep(sleep_time);
            }
            timer.get_time_ns(meta.t2);
            process_time = meta.t2 - meta.t1;
            sum = sum + process_time;
            reg_sum.write(0, sum);
            
            // Tempo di elaborazione egress
            /*bit<64> egr_process_time = meta.t2 - meta.t_egr_start;
            bit<64> egr_sum;
            reg_egr_sum.read(egr_sum, 0);
            egr_sum = egr_sum + egr_process_time;
            reg_egr_sum.write(0, egr_sum);*/
            
            // Aggiornamento del tempo massimo
            if (process_time > max_time) {
                reg_max_time.write(0, process_time);
            }
            
            /*bit<64> egr_max_time;
            reg_egr_max_time.read(egr_max_time, 0);
            if (egr_process_time > egr_max_time) {
                reg_egr_max_time.write(0, egr_process_time);
            }*/

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