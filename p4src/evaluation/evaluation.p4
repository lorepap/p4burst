
#include <core.p4>
#include <v1model.p4>

#include "includes/evaluation_consts.p4"
#include "includes/evaluation_headers.p4"
#include "includes/evaluation_parser.p4"
#include "includes/evaluation_checksums.p4"
#include "evaluation_controls.p4"

/*
    Switch ingress pipeline
*/
control SwitchIngress(
    inout header_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_metadata)
{

    register<bit<16>>(8) queue_length_reg;
    register<bit<1>>(8) neighbor_switch_indicator;
    register<bit<3>>(1) max_free_space_queue_id; 

    Routing() routing;

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

    action set_switch_id(bit<8> switch_id) {
        meta.switch_id = switch_id;
    }

    table set_switch_id_table {
        actions = {
            set_switch_id;
        }
        size = 1;
    }

    action decision_meta_switch_id_action(bit<16> val){
        meta.DT_field = (bit<16>)meta.switch_id;
        meta.DT_val = val;
    }
    action decision_meta_queue_lenght_4_action(bit<16> val){
        meta.DT_field = (bit<16>)meta.queue_lenght_4;
        meta.DT_val = val;
    }
    action decision_meta_queue_lenght_5_action(bit<16> val){
        meta.DT_field = (bit<16>)meta.queue_lenght_5;
        meta.DT_val = val;
    }
    action decision_meta_queue_lenght_6_action(bit<16> val){
        meta.DT_field = (bit<16>)meta.queue_lenght_6;
        meta.DT_val = val;
    }
    action decision_meta_queue_lenght_7_action(bit<16> val){
        meta.DT_field = (bit<16>)meta.queue_lenght_7;
        meta.DT_val = val;
    }

    action deflect(){
        meta.DT_leaf_reached = 1;
        standard_metadata.egress_spec = meta.deflect_port;
    }

    action forward(){
        meta.DT_leaf_reached = 1;
    }

    action exit_BDT(){
        meta.DT_leaf_reached = 1;
    }
    table BDT_table_lev0_cond0{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev1_cond0{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev1_cond1{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev2_cond0{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev2_cond1{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev2_cond2{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev2_cond3{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond0{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond1{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond2{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond3{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond4{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond5{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond6{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev3_cond7{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond0{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond1{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond2{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond3{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond4{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond5{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond6{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond7{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond8{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond9{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond10{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond11{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond12{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond13{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond14{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    
    table BDT_table_lev4_cond15{
        actions = {
            decision_meta_queue_lenght_7_action;
            decision_meta_queue_lenght_6_action;
            exit_BDT;
            decision_meta_queue_lenght_4_action;
            decision_meta_switch_id_action;
            deflect;
            decision_meta_queue_lenght_5_action;
            forward;
        }
        default_action=exit_BDT();
    }
    

    apply {
        if (hdr.bee.isValid()) {
            log_msg("Ingress Bee packet port={} length={}", {hdr.bee.port_id, hdr.bee.queue_length});
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator, (bit<32>)hdr.bee.port_id);
            if(meta.neighbor_switch_indicator == 1){
                queue_length_reg.write((bit<32>)hdr.bee.port_id, (bit<16>)hdr.bee.queue_length);
                bit<3> max_free_space_queue_id_tmp;
                max_free_space_queue_id.read(max_free_space_queue_id_tmp, (bit<32>)hdr.bee.port_id);
                bit<16> max_free_space_queue_occupancy;
                queue_length_reg.read(max_free_space_queue_occupancy, (bit<32>)max_free_space_queue_id_tmp);
                bit<16> max_free_space_queue = QUEUE_CAPACITY - max_free_space_queue_occupancy;
                bit<16> free_space = QUEUE_CAPACITY - hdr.bee.queue_length;
                if (free_space > max_free_space_queue) {
                    max_free_space_queue_id.write((bit<32>)0, (bit<3>)hdr.bee.port_id);
                }
            }
        } else if (hdr.ipv4.isValid() &&
                  (hdr.ipv4.protocol == IP_PROTOCOLS_TCP ||
                   hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {

            log_msg("Non bee packet");
        
            routing.apply(hdr, meta, standard_metadata);
            
            max_free_space_queue_id.read(meta.deflect_port_id, (bit<32>)0);
            set_physical_deflect_port_from_id_table.apply();

            //TODO: metto solo quelle che mi interessano
            //queue_length_reg.read(meta.queue_lenght_0, (bit<32>)0);
            //queue_length_reg.read(meta.queue_lenght_1, (bit<32>)1);
            //queue_length_reg.read(meta.queue_lenght_2, (bit<32>)2);
            //queue_length_reg.read(meta.queue_lenght_3, (bit<32>)3);
            queue_length_reg.read(meta.queue_lenght_4, (bit<32>)4);
            queue_length_reg.read(meta.queue_lenght_5, (bit<32>)5);
            queue_length_reg.read(meta.queue_lenght_6, (bit<32>)6);
            queue_length_reg.read(meta.queue_lenght_7, (bit<32>)7);

            set_switch_id_table.apply();
            
            meta.DT_leaf_reached = 0;
            BDT_table_lev0_cond0.apply();
            if(meta.DT_leaf_reached == 0){
                if(meta.DT_field <= meta.DT_val){
                    BDT_table_lev1_cond0.apply();
                    if(meta.DT_leaf_reached == 0){
                        if(meta.DT_field <= meta.DT_val){
                            BDT_table_lev2_cond0.apply();
                            if(meta.DT_leaf_reached == 0){
                                if(meta.DT_field <= meta.DT_val){
                                    BDT_table_lev3_cond0.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond0.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond1.apply();
                                        }
                                    }
                                }
                                else{
                                    BDT_table_lev3_cond1.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond2.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond3.apply();
                                        }
                                    }
                                }
                            }
                        }
                        else{
                            BDT_table_lev2_cond1.apply();
                            if(meta.DT_leaf_reached == 0){
                                if(meta.DT_field <= meta.DT_val){
                                    BDT_table_lev3_cond2.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond4.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond5.apply();
                                        }
                                    }
                                }
                                else{
                                    BDT_table_lev3_cond3.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond6.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond7.apply();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else{
                    BDT_table_lev1_cond1.apply();
                    if(meta.DT_leaf_reached == 0){
                        if(meta.DT_field <= meta.DT_val){
                            BDT_table_lev2_cond2.apply();
                            if(meta.DT_leaf_reached == 0){
                                if(meta.DT_field <= meta.DT_val){
                                    BDT_table_lev3_cond4.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond8.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond9.apply();
                                        }
                                    }
                                }
                                else{
                                    BDT_table_lev3_cond5.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond10.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond11.apply();
                                        }
                                    }
                                }
                            }
                        }
                        else{
                            BDT_table_lev2_cond3.apply();
                            if(meta.DT_leaf_reached == 0){
                                if(meta.DT_field <= meta.DT_val){
                                    BDT_table_lev3_cond6.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond12.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond13.apply();
                                        }
                                    }
                                }
                                else{
                                    BDT_table_lev3_cond7.apply();
                                    if(meta.DT_leaf_reached == 0){
                                        if(meta.DT_field <= meta.DT_val){
                                            BDT_table_lev4_cond14.apply();
                                        }
                                        else{
                                            BDT_table_lev4_cond15.apply();
                                        }
                                    }
                                }
                            }
                        }
                    }
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
    register<bit<16>>(8) queue_length_reg;

    action get_eg_port_id_action(bit<3> index) {
        meta.port_id = index;
    }
    
    table get_eg_port_id_table {
        key = {
            standard_metadata.egress_port: exact;
        }
        actions = {
            get_eg_port_id_action;
        }
        size = TABLE_SIZE;
    }

    apply {
        if (hdr.bee.isValid()) {
            log_msg("egress bee packet port={} length={}", {hdr.bee.port_id, hdr.bee.queue_length});
            queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_id);
            recirculate_preserving_field_list(0);
        } else {
            log_msg("egress not bee");
            if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
                log_msg("qdepth={}", {standard_metadata.deq_qdepth});
                get_eg_port_id_table.apply();
                queue_length_reg.write((bit<32>)meta.port_id, (bit<16>)(standard_metadata.deq_qdepth)); //Dopo che il pacchetto lascia la coda, pensaci!!
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
