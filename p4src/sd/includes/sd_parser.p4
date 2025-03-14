#ifndef _PARSER_
#define _PARSER_

#include "sd_headers.p4"
#include "sd_consts.p4"

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
parser SimpleDeflectionParser(packet_in packet,
                out header_t hdr,
                inout metadata_t meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTOCOLS_TCP : parse_tcp;
            IP_PROTOCOLS_UDP : parse_udp;
            default : accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition select(hdr.tcp.dstPort) {
            BEE_PORT: parse_bee;
            default: parse_flow;
        }
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition select(hdr.udp.dstPort) {
            BEE_PORT: parse_bee;
            default: parse_flow;
        }
    }

    state parse_bee {
        packet.extract(hdr.bee);
        transition accept;
    }


    state parse_flow {
        packet.extract(hdr.flow);
        log_msg("Parsing flow header: flow_id={}, seq={}", {hdr.flow.flow_id, hdr.flow.seq});
        transition accept;
    }
}


// ---------------------------------------------------------------------------
// Deparser
// ---------------------------------------------------------------------------
control SimpleDeflectionDeparser(packet_out packet, in header_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.flow);
        packet.emit(hdr.bee);
    }
}

#endif