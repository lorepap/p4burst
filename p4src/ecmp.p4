#include <core.p4>
#include <v1model.p4>
#include "/home/ubuntu/extern_lib/declaration.p4"

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_TCP  = 6;
const bit<8>  TYPE_UDP  = 17;

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
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

struct metadata {
    bit<14> ecmp_hash;
    bit<14> ecmp_group_id;
    
    bit<64> t1;          // Ingresso pacchetto ingress
    bit<64> t_ing_end;   // Uscita pacchetto ingress
    bit<64> t_egr_start; // Ingresso pacchetto egress
    bit<64> t2;          // Uscita pacchetto egress
}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
    udp_t        udp;
}

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
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition accept;
    }
}

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply { }
}

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    
    // Aggiunta Time extern e counter
    Time() timer;
    counter(1, CounterType.packets) packet_counter;
    counter(1, CounterType.packets) implicitely_dropped;

    // Counter per i pacchetti processati completamente dall'ingress
    counter(1, CounterType.packets) ingress_packet_counter;

    // Registro per tempistica ingress
    register<bit<64>>(1) reg_ing_sum;
    register<bit<64>>(1) reg_ing_max_time;
    
    action drop() {
        /*log_msg("dropped --- dst={}.{}.{}.{}",
            {(bit<32>)(hdr.ipv4.dstAddr >> 24), 
             (bit<32>)(hdr.ipv4.dstAddr >> 16 & 0xFF),
             (bit<32>)(hdr.ipv4.dstAddr >> 8 & 0xFF),
             (bit<32>)(hdr.ipv4.dstAddr & 0xFF)});*/
        implicitely_dropped.count(0);
        mark_to_drop(standard_metadata);
    }

    action set_ecmp_select(bit<14> ecmp_group_id, bit<16> num_nhops) {
        //log_msg("set_ecmp_select --- ecmp_group_id={}", {ecmp_group_id});
        hash(meta.ecmp_hash,
            HashAlgorithm.crc16,
            (bit<1>)0,
            { hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr,
              hdr.ipv4.protocol,
              hdr.tcp.srcPort,
              hdr.tcp.dstPort },
            num_nhops);

        meta.ecmp_group_id = ecmp_group_id;
    }

    action set_nhop(macAddr_t dstAddr, egressSpec_t port) {
        //log_msg("set_nhop --- dstAddr={}", {dstAddr});
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
    }

    table ecmp_nhop {
        key = {
            meta.ecmp_group_id: exact;
            meta.ecmp_hash: exact;
        }
        actions = {
            drop;
            set_nhop;
        }
        size = 2048;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            set_ecmp_select;
            set_nhop;
            drop;
        }
        size = 1024;
        default_action = drop();
    }

    apply {
        
        timer.get_time_ns(meta.t1);
        if (hdr.ipv4.isValid() &&
                  (hdr.ipv4.protocol == TYPE_TCP ||
                   hdr.ipv4.protocol == TYPE_UDP)) {
            packet_counter.count(0);
            // Conteggio dei pacchetti in ingresso
            hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
            if (hdr.ipv4.ttl == 0) {
                drop();
            }
            switch (ipv4_lpm.apply().action_run){
                set_ecmp_select: {
                    ecmp_nhop.apply();
                }
            }
            // Alla fine dell'ingress, registriamo il timestamp finale
            timer.get_time_ns(meta.t_ing_end);
            // Contiamo i pacchetti che completano l'ingress
            ingress_packet_counter.count(0);
            // Salviamo il tempo di elaborazione ingress 
            bit<64> ing_process_time = meta.t_ing_end - meta.t1;
            bit<64> ing_sum;
            reg_ing_sum.read(ing_sum, 0);
            ing_sum = ing_sum + ing_process_time;
            reg_ing_sum.write(0, ing_sum);

            // Aggiorniamo il tempo massimo di ingress
            bit<64> ing_max_time;
            reg_ing_max_time.read(ing_max_time, 0);
            if (ing_process_time > ing_max_time) {
                reg_ing_max_time.write(0, ing_process_time);
            }
        }
    }
}

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    
    // Aggiunta Time extern, counter e registri per tempi
    Time() timer;
    counter(1, CounterType.packets) egress_packet_counter;
    register<bit<64>>(1) reg_sum;
    // Registro per memorizzare il tempo di processamento massimo
    register<bit<64>>(1) reg_max_time;

    register<bit<64>>(1) reg_egr_sum;
    register<bit<64>>(1) reg_egr_max_time;
    
    apply {
        // Acquisizione timestamp in ingresso nell'egress
        timer.get_time_ns(meta.t_egr_start);
        
        if (hdr.ipv4.isValid() &&
                  (hdr.ipv4.protocol == TYPE_TCP ||
                   hdr.ipv4.protocol == TYPE_UDP)) {
            // Conteggio dei pacchetti in uscita
            egress_packet_counter.count(0);
            
            // Calcolo del tempo di attraversamento del pacchetto
            timer.get_time_ns(meta.t2);
            bit<64> process_time;
            bit<64> sum;
            reg_sum.read(sum, 0);
            process_time = meta.t2 - meta.t1;
            sum = sum + process_time;
            reg_sum.write(0, sum);

            // Tempo di elaborazione egress
            bit<64> egr_process_time = meta.t2 - meta.t_egr_start;
            bit<64> egr_sum;
            reg_egr_sum.read(egr_sum, 0);
            egr_sum = egr_sum + egr_process_time;
            reg_egr_sum.write(0, egr_sum);
            
            // Aggiornamento del tempo massimo
            bit<64> max_time;
            reg_max_time.read(max_time, 0);
            if (process_time > max_time) {
                reg_max_time.write(0, process_time);
            }

            bit<64> egr_max_time;
            reg_egr_max_time.read(egr_max_time, 0);
            if (egr_process_time > egr_max_time) {
                reg_egr_max_time.write(0, egr_process_time);
            }
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

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
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