#ifndef _HEADERS_
#define _HEADERS_

typedef bit<8> ip_protocol_t;

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
    bit<31> port_idx_in_reg;
    bit<1> queue_occ_info;
}

struct metadata_t {

    @field_list(0) // to avoid warning "no user metadata fields tagged with @field_list(0)"
    bit<1> dummy;

    bit<5> random_number;

    bit<32> queue_length;
    bit<32> queue_length2;
    bit<16> fw_port_idx; 

    bit<32> output_port_idx;

    bit<1> is_queue_full_0;
    bit<1> is_queue_full_1;
    bit<1> is_queue_full_2;
    bit<1> is_queue_full_3;
    bit<1> is_queue_full_4;
    bit<1> is_queue_full_5;
    bit<1> is_queue_full_6;
    bit<1> is_queue_full_7;
    bit<1> is_queue_full_8;
    bit<1> is_queue_full_9;
    bit<1> is_queue_full_10;
    bit<1> is_queue_full_11;
    bit<1> is_queue_full_12;
    bit<1> is_queue_full_13;
    bit<1> is_queue_full_14;
    bit<1> is_queue_full_15;
    bit<1> is_queue_full_16;
    bit<1> is_queue_full_17;
    bit<1> is_queue_full_18;
    bit<1> is_queue_full_19;
    bit<1> is_queue_full_20;
    bit<1> is_queue_full_21;
    bit<1> is_queue_full_22;
    bit<1> is_queue_full_23;
    bit<1> is_queue_full_24;
    bit<1> is_queue_full_25;
    bit<1> is_queue_full_26;
    bit<1> is_queue_full_27;
    bit<1> is_queue_full_28;
    bit<1> is_queue_full_29;
    bit<1> is_queue_full_30;
    bit<1> is_queue_full_31;

    bit<1> neighbor_switch_indicator_0;
    bit<1> neighbor_switch_indicator_1;
    bit<1> neighbor_switch_indicator_2;
    bit<1> neighbor_switch_indicator_3;
    bit<1> neighbor_switch_indicator_4;
    bit<1> neighbor_switch_indicator_5;
    bit<1> neighbor_switch_indicator_6;
    bit<1> neighbor_switch_indicator_7;
    bit<1> neighbor_switch_indicator_8;
    bit<1> neighbor_switch_indicator_9;
    bit<1> neighbor_switch_indicator_10;
    bit<1> neighbor_switch_indicator_11;
    bit<1> neighbor_switch_indicator_12;
    bit<1> neighbor_switch_indicator_13;
    bit<1> neighbor_switch_indicator_14;
    bit<1> neighbor_switch_indicator_15;
    bit<1> neighbor_switch_indicator_16;
    bit<1> neighbor_switch_indicator_17;
    bit<1> neighbor_switch_indicator_18;
    bit<1> neighbor_switch_indicator_19;
    bit<1> neighbor_switch_indicator_20;
    bit<1> neighbor_switch_indicator_21;
    bit<1> neighbor_switch_indicator_22;
    bit<1> neighbor_switch_indicator_23;
    bit<1> neighbor_switch_indicator_24;
    bit<1> neighbor_switch_indicator_25;
    bit<1> neighbor_switch_indicator_26;
    bit<1> neighbor_switch_indicator_27;
    bit<1> neighbor_switch_indicator_28;
    bit<1> neighbor_switch_indicator_29;
    bit<1> neighbor_switch_indicator_30;
    bit<1> neighbor_switch_indicator_31;

    bit<16> port_idx_in_reg;
    bit<1> is_fw_port_full;
    
    // Campi per il calcolo dei tempi
    bit<64> t1;          // Ingresso pacchetto ingress
    bit<64> t_ing_end;   // Uscita pacchetto ingress
    bit<64> t_egr_start; // Ingresso pacchetto egress
    bit<64> t2;          // Uscita pacchetto egress
    bit<1> dropped;
}

struct header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
    bee_h bee;
}

struct pair16 {
    bit<16>     low;
    bit<16>     high;
}


#endif /* _HEADERS_ */