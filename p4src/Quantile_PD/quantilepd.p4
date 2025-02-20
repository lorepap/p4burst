#include <core.p4>
#include <v1model.p4>

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

    Routing()           routing;
    DeflectRouting()    deflection_routing;
    GetQuantile()       get_quantile;
    GetMin()            get_min;
    DeflectGetMin()     deflect_get_min;

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

    table debug_deflected {
        key = {
            standard_metadata.egress_spec: exact;
            meta.deflect_egress_spec: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

    table debug_queue_length {
        key = {
            //meta.fw_port_idx: exact;
            meta.queue_length: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

    table debug_queue_length_deflect {
        key = {
            //meta.deflect_fw_port_idx: exact;
            meta.deflect_queue_length: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

    table debug_deflection {
        key = {
            meta.fw_port_idx: exact;
            meta.deflect_fw_port_idx: exact;
            meta.queue_length: exact;
            meta.min_value: exact;
            meta.deflect_min_value: exact;
            meta.deflect_queue_length: exact;
            standard_metadata.egress_spec: exact;
            meta.deflect_egress_spec: exact;
            meta.count_all: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
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

    table debug_bee_ingress {
        key = {
            hdr.bee.port_idx_in_reg: exact;
            hdr.bee.queue_length: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

    apply {
        //log_msg("Ingress pipeline -- prova");
        if (hdr.bee.isValid()) {
            queue_length_reg.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_length);
            debug_bee_ingress.apply();
        } else if (hdr.ipv4.isValid() &&
                  (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                   hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
            
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

            //debug_queue_length.apply();
            //debug_queue_length_deflect.apply();

            get_tail_action();
            get_quantile.apply(meta);
            get_min.apply(meta);
            deflect_get_min.apply(meta);


            debug_deflection.apply();
            if (meta.min_value == meta.queue_length) {
                if (meta.deflect_min_value != meta.deflect_queue_length) {
                    debug_deflected.apply();
                    standard_metadata.egress_spec = meta.deflect_egress_spec;
                    //meta.fw_port_idx = meta.deflect_fw_port_idx;
                } else {
                    drop();
                }
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

    table debug_bee_egress {
        key = {
            hdr.bee.port_idx_in_reg: exact;
            hdr.bee.queue_length: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

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

    table debug_deq_qdepth {
        key = {
            standard_metadata.deq_qdepth: exact;
            meta.port_idx_in_reg: exact;
            standard_metadata.egress_port: exact;
        }
        actions = { NoAction; }
        default_action = NoAction();
        size = TABLE_SIZE;
    }

    apply {
        if (hdr.bee.isValid()) {
            eg_queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_idx_in_reg);
            debug_bee_egress.apply();
            recirculate_preserving_field_list(0);
        } else {
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {

                get_eg_port_idx_in_reg_table.apply();

                debug_deq_qdepth.apply();

                eg_queue_length_reg.write((bit<32>)meta.port_idx_in_reg, (bit<32>)(standard_metadata.deq_qdepth)); //Dopo che il pacchetto lascia la coda, pensaci!!
            }
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