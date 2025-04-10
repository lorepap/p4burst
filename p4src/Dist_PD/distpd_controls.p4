#ifndef _CONTROLS_
#define _CONTROLS_

#include "includes/distpd_consts.p4"
#include "includes/distpd_headers.p4"

control Routing(inout header_t hdr,
               inout metadata_t meta,
               inout standard_metadata_t standard_metadata) {
    
    action drop() {
        mark_to_drop(standard_metadata);
    }

    action get_fw_port_idx_action(bit<9> port, bit<16> fw_port_idx, bit<48> dst_mac) {
        standard_metadata.egress_spec = port;
        meta.fw_port_idx = fw_port_idx;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr; // l2 simplified forwarding to enable comunications between hosts
        hdr.ethernet.dstAddr = dst_mac;
    }

    table get_fw_port_idx_table {
        key = {
            hdr.ipv4.dstAddr : lpm;
        }
        actions = {
            get_fw_port_idx_action;
            drop;
        }
        size = TABLE_SIZE;
        default_action = drop();
    }

    action fw_l2_action(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action broadcast() {
        standard_metadata.mcast_grp = 1;
    }

    table fw_l2_table {
        key = {
            hdr.ethernet.dstAddr : exact;
        }
        actions = {
            fw_l2_action;
            broadcast;
            drop;
        }
        size = TABLE_SIZE;
        default_action = broadcast();
    }

    apply {
        if (!hdr.ipv4.isValid()) {
            fw_l2_table.apply();
        } else {
            hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
            if (hdr.ipv4.ttl == 0) {
                drop();
            }
            get_fw_port_idx_table.apply();
        }
    }
}

control DeflectRouting(inout header_t hdr,
               inout metadata_t meta,
               inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action deflect_get_fw_port_idx_action(bit<9> port, bit<16> fw_port_idx) {
        meta.deflect_egress_spec = port;
        meta.deflect_fw_port_idx = fw_port_idx;
    }

    action no_deflection() {
        meta.deflect_egress_spec = standard_metadata.egress_spec;
        meta.deflect_fw_port_idx = meta.fw_port_idx;
    }

    table deflect_get_fw_port_idx_table {
        key = {
            hdr.ipv4.dstAddr : lpm;
        }
        actions = {
            deflect_get_fw_port_idx_action;
            no_deflection;
        }
        size = TABLE_SIZE;
        default_action = no_deflection();
    }
    
    apply {
        deflect_get_fw_port_idx_table.apply();
    }

}



control GetMinRelPrioQueueLen(inout metadata_t ig_md) {

    action get_min_rel_prio_queue_len() {
        if (ig_md.rel_prio < ig_md.queue_length) {
            ig_md.min_value_rel_prio_queue_len = ig_md.rel_prio;
        } else {
            ig_md.min_value_rel_prio_queue_len = ig_md.queue_length;
        }
    }

    apply {
        get_min_rel_prio_queue_len();
    }
}

control DeflectGetMinRelPrioQueueLen(inout metadata_t ig_md) {

    action deflect_get_min_rel_prio_queue_len() {
        if (ig_md.deflect_rel_prio < ig_md.deflect_queue_length) {
            ig_md.deflect_min_value_rel_prio_queue_len = ig_md.deflect_rel_prio;
        } else {
            ig_md.deflect_min_value_rel_prio_queue_len = ig_md.deflect_queue_length;
        }
    }

    apply {
        deflect_get_min_rel_prio_queue_len();
    }
}

#endif
