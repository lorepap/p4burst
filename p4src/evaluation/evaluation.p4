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

    register<bit<16>>(32) queue_length_reg;     // Changed from 8 to 32
    register<bit<1>>(32) neighbor_switch_indicator;  // Changed from 8 to 32
    register<bit<5>>(1) max_free_space_queue_id;    // Changed bit<3> to bit<5>

    Routing() routing;
    
    action drop() {
        mark_to_drop(standard_metadata);
    }

    action set_max_port_occupancies(bit<16> forward_occupancy, bit<16> deflect_occupancy) {
        meta.max_forward_queue_occupancy = forward_occupancy;
        meta.max_deflect_queue_occupancy = deflect_occupancy;
    }
    
    table set_max_port_occupancies_table {
        actions = {
            set_max_port_occupancies;
        }
        default_action = set_max_port_occupancies(QUEUE_CAPACITY, QUEUE_CAPACITY);
    }

    apply {
        if (hdr.bee.isValid()) {
            //log_msg("Ingress Bee packet port={} length={}", {hdr.bee.port_id, hdr.bee.queue_length});
            neighbor_switch_indicator.read(meta.neighbor_switch_indicator, (bit<32>)hdr.bee.port_id);
            queue_length_reg.write((bit<32>)hdr.bee.port_id, (bit<16>)hdr.bee.queue_length);
            if(meta.neighbor_switch_indicator == 1){
                bit<5> max_free_space_queue_id_tmp;
                bit<16> max_free_space_queue_occupancy;
                max_free_space_queue_id.read(max_free_space_queue_id_tmp, (bit<32>)0);
                queue_length_reg.read(max_free_space_queue_occupancy, (bit<32>)max_free_space_queue_id_tmp);
                bit<16> max_free_space_queue = QUEUE_CAPACITY - max_free_space_queue_occupancy;
                bit<16> free_space = QUEUE_CAPACITY - hdr.bee.queue_length;
                if (free_space > max_free_space_queue) {
                    max_free_space_queue_id.write((bit<32>)0, (bit<5>)hdr.bee.port_id);
                }
            }
        } else if (hdr.ipv4.isValid()) {

        
            routing.apply(hdr, meta, standard_metadata);
            
            // With our topology physical_ports = logial_ports + 1
            // if we want to generalize this we need to use conversion tables
            bit<5> forward_port_id;
            bit<5> deflect_port_id;
            bit<16> forward_queue_occupancy;
            bit<16> deflect_queue_occupancy;
            forward_port_id = (bit<5>)standard_metadata.egress_spec - 1;
            max_free_space_queue_id.read(deflect_port_id, (bit<32>)0);
            queue_length_reg.read(forward_queue_occupancy, (bit<32>)forward_port_id);
            queue_length_reg.read(deflect_queue_occupancy, (bit<32>)deflect_port_id);
            set_max_port_occupancies_table.apply();
            if(forward_queue_occupancy > meta.max_forward_queue_occupancy){
                if (deflect_queue_occupancy > meta.max_deflect_queue_occupancy) {
                    standard_metadata.egress_spec = (bit<9>)deflect_port_id + 1; // Convert back to physical port and deflect
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
    register<bit<16>>(32) queue_length_reg;  // Changed from 8 to 32

    apply {
        if (hdr.bee.isValid()) {
            log_msg("egress bee packet port={} length={}", {hdr.bee.port_id, hdr.bee.queue_length});
            queue_length_reg.read(hdr.bee.queue_length, (bit<32>)hdr.bee.port_id);
            recirculate_preserving_field_list(0);
        } else if (hdr.ipv4.isValid() && (hdr.ipv4.protocol == IP_PROTOCOLS_TCP || hdr.ipv4.protocol == IP_PROTOCOLS_UDP)) {
            log_msg("qdepth={}", {standard_metadata.deq_qdepth});
            bit<32> port_id = (bit<32>)(standard_metadata.egress_spec - 1); // Convert physical port to logical port
            queue_length_reg.write(port_id, (bit<16>)(standard_metadata.deq_qdepth)); //Dopo che il pacchetto lascia la coda, pensaci!!
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
