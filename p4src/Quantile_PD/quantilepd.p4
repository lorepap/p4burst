#include <core.p4>
#include <v1model.p4>

#include "/home/ubuntu/extern_lib/declaration.p4"
#include "includes/quantilepd_consts.p4"
#include "includes/quantilepd_headers.p4"
#include "includes/quantilepd_parser.p4"
#include "includes/quantilepd_checksums.p4"
#include "quantilepd_controls.p4"



/*
    Switch ingress pipeline
*/
control SwitchIngress(
    inout header_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_metadata)
{

    register<bit<32>>(NUM_LOGICAL_PORTS) queue_length_reg;
    register<bit<16>>(1) tail_low_reg;
    register<bit<16>>(1) tail_high_reg;
    
    Time() timer;

    Routing()           routing;
    DeflectRouting()    deflection_routing;
    GetQuantile()       get_quantile;
    GetMin()            get_min;
    DeflectGetMin()     deflect_get_min;

    // Aggiunti contatori per la misurazione delle prestazioni
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

    
    action get_flow_priority_action(bit<32> rank) {
        meta.rank = rank;
    }

    table get_flow_priority_table {
        key = {
            hdr.tcp.dstPort: exact;
        }
        actions = { get_flow_priority_action; }
        size = TABLE_SIZE;
    }

    action get_tail_action() {
        bit<16> t_low;
        bit<16> t_high;
        tail_low_reg.read(t_low, 0);
        tail_high_reg.read(t_high, 0);
        if (t_low < (16 * SAMPLE_COUNT - 1)) {
            t_low = t_low + 1;
        } else {
            t_low = 0;
        }
        tail_low_reg.write(0, t_low);
        tail_high_reg.write(0, t_high);
        meta.tail = t_low;
    }

    apply {
        
        timer.get_time_ns(meta.t1);
        
        //log_msg("Ingress pipeline -- prova");
        if (hdr.bee.isValid()) {
            queue_length_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_length);
        } else if (hdr.ipv4.isValid() &&
                  (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                   hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
            
            packet_counter.count(0);
            get_flow_priority_table.apply();
            routing.apply(hdr, meta, standard_metadata);
            deflection_routing.apply(hdr, meta, standard_metadata);
            queue_length_reg.read(meta.queue_length, (bit<32>)meta.fw_port_idx);
            queue_length_reg.read(meta.deflect_queue_length, (bit<32>)meta.deflect_fw_port_idx);

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

            get_tail_action();
            get_quantile.apply(meta);
            get_min.apply(meta);
            deflect_get_min.apply(meta);

            if (meta.min_value == meta.queue_length) {
                if (meta.deflect_min_value != meta.deflect_queue_length) {
                    standard_metadata.egress_spec = meta.deflect_egress_spec;
                    deflect_counter.count(0);
                    //meta.fw_port_idx = meta.deflect_fw_port_idx;
                } else {
                    drop_counter.count(0);
                    drop();
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
}



control SwitchEgress(
    inout header_t            hdr,
    inout metadata_t          meta,
    inout standard_metadata_t standard_metadata)
{
    register<bit<32>>(NUM_LOGICAL_PORTS) eg_queue_length_reg;
    Time() timer;
    counter(1, CounterType.packets) egress_packet_counter;
    register<bit<64>>(1) reg_sum;
    // Registro per memorizzare il tempo di processamento massimo
    register<bit<64>>(1) reg_max_time;
    // Registro per tempistica egress
    register<bit<64>>(1) reg_egr_sum;
    register<bit<64>>(1) reg_egr_max_time;

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

    apply {

        timer.get_time_ns(meta.t_egr_start);

        if (hdr.bee.isValid()) {
            eg_queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_idx_in_reg);
            recirculate_preserving_field_list(0);
        } else if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {

            get_eg_port_idx_in_reg_table.apply();

            eg_queue_length_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)(standard_metadata.deq_qdepth));
                
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


//switch architecture
V1Switch(SwitchParser(),
         SwitchVerifyChecksum(),
         SwitchIngress(),
         SwitchEgress(),
         SwitchComputeChecksum(),
         SwitchDeparser()
) main;