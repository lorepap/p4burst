#ifndef _PARSER_
#define _PARSER_

#include "quantilepd_headers.p4"
#include "quantilepd_consts.p4"

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
parser SwitchParser(packet_in packet,
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
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition select(hdr.udp.dstPort) {
            BEE_PORT: parse_bee;
            default: accept;
        }
    }

    state parse_bee {
        packet.extract(hdr.bee);
        transition accept;
    }
}


// ---------------------------------------------------------------------------
// Deparser
// ---------------------------------------------------------------------------
control SwitchDeparser(packet_out packet, in header_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
        packet.emit(hdr.bee);
    }
}

#endif