#ifndef _CONSTS_
#define _CONSTS_


const bit<16> ETHERTYPE_IPV4  = 0x0800;
const bit<8> IP_PROTOCOLS_TCP = 6;
const bit<8> IP_PROTOCOLS_UDP = 17;
const bit<16> BEE_PORT        = 9999;
const bit<32> TABLE_SIZE      = 256;
const bit<19> QUEUE_SIZE  = 1;    // Value overriden from config.py

#endif