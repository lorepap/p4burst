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
    bit<16> port_id;
    bit<16> queue_length;
}

struct metadata_t {

    @field_list(0) // to avoid warning "no user metadata fields tagged with @field_list(0)"
    bit<1> dummy;
    //bit<16> queue_lenght_0;
    //bit<16> queue_lenght_1;
    //bit<16> queue_lenght_2;
    //bit<16> queue_lenght_3;
    bit<16> queue_lenght_4;
    bit<16> queue_lenght_5;
    bit<16> queue_lenght_6;
    bit<16> queue_lenght_7;

    bit<1> neighbor_switch_indicator;
    bit<9> deflect_port;
    bit<3> deflect_port_id;
    bit<3> port_logical_id;
    bit<8> switch_id;
    bit<3> max_free_space_queue_id_tmp;
    bit<16> max_free_space_queue;
    bit<3> port_id;
    bit<16> DT_field;
    bit<16> DT_val;
    bit<1> DT_leaf_reached;

}


struct header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    tcp_h tcp;
    udp_h udp;
    bee_h bee;
}



#endif /* _HEADERS_ */
