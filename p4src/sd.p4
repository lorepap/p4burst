#include <core.p4>
#include <v1model.p4>

// Header definitions
header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<4>  res;
    bit<8>  flags;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

header bee_t {
    bit<8> port_idx_in_reg;
    bit<8> queue_occ_info;
}

struct headers {
    ethernet_t ethernet;
    ipv4_t     ipv4;
    tcp_t      tcp;
    udp_t      udp;
    bee_t      bee;
}

struct metadata {
    bit<16> fw_port_idx;
    bit<1>  is_fw_port_full;
    bit<5>  random_number;
    bit<16> output_port_idx;
    bit<1> need_deflection;
    bit<9> current_try_port;
    bit<8>  is_queue_full_0;
    bit<8>  is_queue_full_1;
    bit<8>  is_queue_full_2;
    bit<8>  is_queue_full_3;
    bit<8>  is_queue_full_4;
    bit<8>  is_queue_full_5;
    bit<8>  is_queue_full_6;
    bit<8>  is_queue_full_7;
    @field_list(1)  // Add annotation for field to preserve
    bit<1>  is_recirculated;
}

// Parser
parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x0800: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6: parse_tcp;   // TCP
            17: parse_udp;  // UDP
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition select(hdr.udp.dstPort) {
            9999: parse_bee;  // BEE_PORT
            default: accept;
        }
    }

    state parse_bee {
        packet.extract(hdr.bee);
        transition accept;
    }
}

// Checksum verification control
control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {
        verify_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

// Ingress control
control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    register<bit<8>>(8) queue_status;
    
    action drop() {
        mark_to_drop(standard_metadata);
    }

    // action forward(bit<9> port) {
    //     standard_metadata.egress_spec = port;
    // }

    // L3 forwarding
    action get_fw_port_idx_action(bit<9> port, bit<16> fw_port_idx) {
        standard_metadata.egress_spec = port;
        meta.fw_port_idx = fw_port_idx;
    }

    // L2 forwarding
    action fw_l2_action(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action broadcast() {
        // BMv2 multicast implementation
        standard_metadata.mcast_grp = 1;
    }


    action generate_random() {
        random(meta.random_number, 0, 7);
    }

    action try_port(bit<9> port_num) {
        bit<8> try_status;
        queue_status.read(try_status, (bit<32>)port_num);
        if (try_status == 0) {
            standard_metadata.egress_spec = port_num;
            meta.need_deflection = 0;
        }
    }
    
    action deflect() {
        bit<8> port_status;
        bit<3> port = (bit<3>)meta.random_number;
        
        // Read queue status for randomly selected port
        queue_status.read(port_status, (bit<32>)port);
        
        if (port_status == 0) {
            // Port not congested, use it
            standard_metadata.egress_spec = (bit<9>)port;
        } else {
            meta.need_deflection = 1;
            meta.current_try_port = (bit<9>)port;
        }
    }

    table get_fw_port_idx_table {
        key = {
            hdr.ipv4.dstAddr : lpm;
        }
        actions = {
            get_fw_port_idx_action;
            drop;
        }
        default_action = drop();
        size = 256;
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
        size = 256;
        default_action = broadcast();
    }

      table try_next_port {
        key = {
            meta.current_try_port: exact;
        }
        actions = {
            try_port;
            drop;
        }
        const entries = {
            0 : try_port(1);
            1 : try_port(2);
            2 : try_port(3);
            3 : try_port(4);
            4 : try_port(5);
            5 : try_port(6);
            6 : try_port(7);
            7 : try_port(0);
        }
        default_action = drop();
    }

     apply {
        if (hdr.bee.isValid()) {
            queue_status.write((bit<32>)hdr.bee.port_idx_in_reg, hdr.bee.queue_occ_info);
            meta.is_recirculated = 1;
            resubmit_preserving_field_list((bit<8>)1);
        } else {
            if (hdr.ipv4.isValid()) {
                hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
                if (hdr.ipv4.ttl == 0) {
                    drop();
                } else {
                    get_fw_port_idx_table.apply();
                    bit<8> dst_port_status;
                    queue_status.read(dst_port_status, (bit<32>)standard_metadata.egress_spec);
                    if (dst_port_status == 1) {
                        generate_random();
                        deflect();
                        if (meta.need_deflection == 1) {
                            try_next_port.apply();
                        }
                    }
                }
            } else {
                fw_l2_table.apply();
            }
        }
    }
}

// Egress control
control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    
    register<bit<8>>(8) queue_status;
    
    apply {
        // Update queue status based on queue depth
        if (standard_metadata.enq_qdepth >= 1024) { // Queue capacity threshold
            queue_status.write((bit<32>)standard_metadata.egress_port, 1);
        } else {
            queue_status.write((bit<32>)standard_metadata.egress_port, 0);
        }
    }
}

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4 .totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.bee);
    }
}

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;