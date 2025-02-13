#ifndef _CONTROLS_
#define _CONTROLS_

#include "includes/quantilepd_consts.p4"
#include "includes/quantilepd_headers.p4"

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

control GetQuantile(inout metadata_t meta) {
    register<bit<32>>(20) window_register;

    #define CHECK_WINDOW_ACTION(INDEX)                                      \
        action check_window_action_##INDEX() {                              \
            bit<32> reg_val;                                                \
            window_register.read(reg_val, INDEX);                           \
            if (meta.rank < reg_val) {                                      \
                meta.check_results_##INDEX = 1;                             \
            } else {                                                        \
                meta.check_results_##INDEX = 0;                             \
            }                                                               \
            if (meta.tail == (bit<16>)((INDEX) * SAMPLE_COUNT)) {           \
                reg_val = meta.rank;                                        \
            }                                                               \
            window_register.write(INDEX, meta.rank);                        \
        }


    action sum_columns_and_compute_count_all() {
        bit<32> col0;
        bit<32> col1;
        bit<32> col2;
        bit<32> col3;
        
        // Somma per ciascuna colonna:
        // Colonna 0: indici 0, 4, 8, 12, 16
        col0 = (bit<32>)meta.check_results_0 + (bit<32>)meta.check_results_4 +
               (bit<32>)meta.check_results_8 + (bit<32>)meta.check_results_12 +
               (bit<32>)meta.check_results_16;
        // Colonna 1: indici 1, 5, 9, 13, 17
        col1 = (bit<32>)meta.check_results_1 + (bit<32>)meta.check_results_5 +
               (bit<32>)meta.check_results_9 + (bit<32>)meta.check_results_13 +
               (bit<32>)meta.check_results_17;
        // Colonna 2: indici 2, 6, 10, 14, 18
        col2 = (bit<32>)meta.check_results_2 + (bit<32>)meta.check_results_6 +
               (bit<32>)meta.check_results_10 + (bit<32>)meta.check_results_14 +
               (bit<32>)meta.check_results_18;
        // Colonna 3: indici 3, 7, 11, 15, 19
        col3 = (bit<32>)meta.check_results_3 + (bit<32>)meta.check_results_7 +
               (bit<32>)meta.check_results_11 + (bit<32>)meta.check_results_15 +
               (bit<32>)meta.check_results_19;

        // Logica per impostare il count finale (adattabile alla logica originale)
        if (col0 == 0 && col2 == 0) {
            meta.count_all = 0;
        } else if ((col0 == 1 && col2 == 0) || (col0 == 0 && col2 == 1)) {
            meta.count_all = 1;
        } else {
            meta.count_all = 2;
        }

        // Moltiplichiamo count_all (shift left di 10 bit equivale a moltiplicare per 2^10)
        meta.count_all = meta.count_all << 10;
    }

    CHECK_WINDOW_ACTION(0)
    CHECK_WINDOW_ACTION(1)
    CHECK_WINDOW_ACTION(2)
    CHECK_WINDOW_ACTION(3)
    CHECK_WINDOW_ACTION(4)
    CHECK_WINDOW_ACTION(5)
    CHECK_WINDOW_ACTION(6)
    CHECK_WINDOW_ACTION(7)
    CHECK_WINDOW_ACTION(8)
    CHECK_WINDOW_ACTION(9)
    CHECK_WINDOW_ACTION(10)
    CHECK_WINDOW_ACTION(11)
    CHECK_WINDOW_ACTION(12)
    CHECK_WINDOW_ACTION(13)
    CHECK_WINDOW_ACTION(14)
    CHECK_WINDOW_ACTION(15)
    CHECK_WINDOW_ACTION(16)
    CHECK_WINDOW_ACTION(17)
    CHECK_WINDOW_ACTION(18)
    CHECK_WINDOW_ACTION(19)

    apply {
        // In questo esempio, "unroliamo" manualmente la chiamata per ciascuna finestra (0..19)
        check_window_action_0();
        check_window_action_1();
        check_window_action_2();
        check_window_action_3();
        check_window_action_4();
        check_window_action_5();
        check_window_action_6();
        check_window_action_7();
        check_window_action_8();
        check_window_action_9();
        check_window_action_10();
        check_window_action_11();
        check_window_action_12();
        check_window_action_13();
        check_window_action_14();
        check_window_action_15();
        check_window_action_16();
        check_window_action_17();
        check_window_action_18();
        check_window_action_19();

        // Dopo aver eseguito tutti i check, sommiamo i risultati e calcoliamo il valore finale
        sum_columns_and_compute_count_all();
    }

}

control GetMin(inout metadata_t meta) {

    action get_min_action() {
        // Esegue il minimo tra count_all e queue_length
        if (meta.count_all < meta.queue_length) {
            meta.min_value = meta.count_all;
        } else {
            meta.min_value = meta.queue_length;
        }
    }

    apply {
        get_min_action();
    }
}



control DeflectGetMin(inout metadata_t meta) {

    action deflect_get_min_action() {
        // Esegue il minimo tra count_all e deflect_queue_length
        if (meta.count_all < meta.deflect_queue_length) {
            meta.deflect_min_value = meta.count_all;
        } else {
            meta.deflect_min_value = meta.deflect_queue_length;
        }
    }

    apply {
        deflect_get_min_action();
    }
}

#endif
