#ifndef _HEADERS_
#define _HEADERS_

struct port_metadata_t {
    bit<3> port_pcp;
    bit<12> port_vid;
    bit<9> l2_xid;
}

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
    bit<32> M;
}

struct metadata_t {
    
    @field_list(0) // to avoid warning "no user metadata fields tagged with @field_list(0)"
    bit<1> dummy;


    bit<16> deflect_fw_port_idx;
    bit<16> fw_port_idx;

    bit<32> rel_prio;
    bit<32> deflect_rel_prio;

    bit<9> deflect_egress_spec;
    
    bit<32> queue_length;
    bit<32> m;
    bit<32> new_m;
    bit<32> rank;
    bit<32> min_value_rel_prio_queue_len;

    bit<32> deflect_queue_length;
    bit<32> deflect_m;
    bit<32> deflect_min_value_rel_prio_queue_len;

    bit<9> deflect_ucast_egress_port;
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