#ifndef _CONSTS_
#define _CONSTS_

const bit<16> ETHERTYPE_IPV4  = 0x0800;
const bit<8> IP_PROTOCOLS_TCP = 6;
const bit<8> IP_PROTOCOLS_UDP = 17;
const bit<16> BEE_PORT        = 9999;
const bit<32> TABLE_SIZE      = 256;
const bit<16> QUANTILEPD_PORT = 9000;
const bit<16> NUM_FLOWS       = 500;
const bit<16> NUM_PREFIXES    = 500;
const bit<32> QUEUE_SIZE      = 1000;    // Value overriden from config.py
const bit<16> SAMPLE_COUNT    = 1;
const bit<32> NUM_LOGICAL_PORTS = 8;
const bit<8> EQUATION_MULT_SHIFT = 3;

#endif
