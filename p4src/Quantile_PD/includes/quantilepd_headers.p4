#ifndef _HEADERS_
#define _HEADERS_

header ethernet_h {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header tcp_h {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4> dataOffset;
    bit<3> res;
    bit<3> ecn;
    bit<6> ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgent_ptr;
}

header udp_h {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> pkt_length;
    bit<16> checksum;
}

header bee_h {
    bit<16> port_idx_in_reg;
    bit<32> queue_length;
}

struct metadata_t {
    
    @field_list(0) // to avoid warning "no user metadata fields tagged with @field_list(0)"
    bit<1> dummy;

    bit<16> order;
    bit<16> tail;

    bit<32> queue_length;
    bit<32> deflect_queue_length;
    
    bit<32> min_value;

    bit<16> deflect_fw_port_idx;
    bit<16> fw_port_idx;

    bit<32> count_all;

    bit<32> deflect_min_value;

    bit<32> rank;

    bit<1> check_results_0;
    bit<1> check_results_1;
    bit<1> check_results_2;
    bit<1> check_results_3;
    bit<1> check_results_4;
    bit<1> check_results_5;
    bit<1> check_results_6;
    bit<1> check_results_7;
    bit<1> check_results_8;
    bit<1> check_results_9;
    bit<1> check_results_10;
    bit<1> check_results_11;
    bit<1> check_results_12;
    bit<1> check_results_13;
    bit<1> check_results_14;
    bit<1> check_results_15;
    bit<1> check_results_16;
    bit<1> check_results_17;
    bit<1> check_results_18;
    bit<1> check_results_19;

    bit<9> deflect_egress_spec;
    bit<16> port_idx_in_reg;
}


struct header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
    bee_h bee;
}


#endif /* _HEADERS_ */