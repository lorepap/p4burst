#!/usr/bin/env python3

import os
import sys
import argparse
import csv
import logging
import subprocess
import tempfile
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TCP flag definitions
TCP_FLAGS = {
    0x01: 'FIN',
    0x02: 'SYN',
    0x04: 'RST',
    0x08: 'PSH',
    0x10: 'ACK',
    0x20: 'URG',
    0x40: 'ECE',
    0x80: 'CWR'
}

def flags_to_string(flags_value):
    """Convert numeric TCP flags to human-readable string format."""
    if not flags_value:
        return "-"
    flag_names = []
    for bit, name in TCP_FLAGS.items():
        if flags_value & bit:
            flag_names.append(name)
    return "+".join(flag_names) if flag_names else "?"

class PacketRTT:
    """Class to track RTT for a packet."""
    def __init__(self, flow_id, seq_num, send_time, data_len, flags):
        self.flow_id = flow_id
        self.seq_num = seq_num 
        self.send_time = send_time
        self.ack_time = None
        self.rtt = None
        self.data_len = data_len
        self.flags = flags
        self.flags_str = flags_to_string(flags)

    def set_ack(self, ack_time):
        """Set the ACK time and calculate RTT."""
        self.ack_time = ack_time
        self.rtt = ack_time - self.send_time
        return self.rtt

    def __str__(self):
        if self.rtt is not None:
            return f"Seq: {self.seq_num}, RTT: {self.rtt*1000:.2f}ms, Flags: {self.flags_str}"
        else:
            return f"Seq: {self.seq_num}, Not ACKed, Flags: {self.flags_str}"

class FlowStats:
    def __init__(self, flow_id, src_ip, dst_ip, src_port, dst_port):
        self.flow_id = flow_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.syn_time = None
        self.last_ack_time = None
        self.bytes_sent = 0
        self.fct = None
        self.packets_sent = 0
        self.packets_received = 0
        self.last_seq = None
        self.highest_ack = 0
        self.complete = False
        self.first_flag = None
        self.last_flag = None
        # RTT tracking
        self.rtts = []
        self.packet_rtts = {}  # seq_num -> PacketRTT
        self.retransmissions = 0

    def add_packet(self, seq_num, send_time, data_len, flags):
        """Add an outgoing packet for RTT tracking."""
        key = seq_num
        # Check if this is a retransmission
        if key in self.packet_rtts:
            self.retransmissions += 1
            # Update if this is newer
            if send_time > self.packet_rtts[key].send_time:
                self.packet_rtts[key].send_time = send_time
        else:
            self.packet_rtts[key] = PacketRTT(self.flow_id, seq_num, send_time, data_len, flags)

    def process_ack(self, ack_num, ack_time):
        """Process an incoming ACK, matching it to sent packets and calculating RTT."""
        # ACK n acknowledges bytes up to but not including n
        # Find all packets that this ACK acknowledges
        newly_acked = []
        
        for seq_num, packet in list(self.packet_rtts.items()):
            # If this packet is being ACKed and hasn't been ACKed before
            if seq_num + packet.data_len <= ack_num and packet.ack_time is None:
                rtt = packet.set_ack(ack_time)
                newly_acked.append(packet)
                self.rtts.append(rtt)
        
        return newly_acked

    def compute_fct(self):
        if self.syn_time and self.last_ack_time:
            self.fct = self.last_ack_time - self.syn_time
            return self.fct
        return None

    def get_rtt_stats(self):
        """Return statistics about RTTs in this flow."""
        if not self.rtts:
            return {
                'min_rtt': None,
                'max_rtt': None,
                'avg_rtt': None,
                'median_rtt': None,
                'retransmissions': self.retransmissions,
                'total_packets': len(self.packet_rtts)
            }
        
        rtts = sorted(self.rtts)
        return {
            'min_rtt': min(rtts),
            'max_rtt': max(rtts),
            'avg_rtt': sum(rtts) / len(rtts),
            'median_rtt': rtts[len(rtts) // 2],
            'retransmissions': self.retransmissions,
            'total_packets': len(self.packet_rtts)
        }

    def __str__(self):
        flag_info = ""
        if self.first_flag:
            flag_info = f", First: {flags_to_string(self.first_flag)}, Last: {flags_to_string(self.last_flag)}"
        
        rtt_stats = self.get_rtt_stats()
        rtt_info = ""
        if rtt_stats['avg_rtt'] is not None:
            rtt_info = f", Avg RTT: {rtt_stats['avg_rtt']*1000:.2f}ms"
        
        return (f"Flow {self.flow_id}: {self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port}, "
                f"FCT: {self.fct:.6f}s, Bytes: {self.bytes_sent}, Packets: {self.packets_sent}{flag_info}{rtt_info}")

def parse_pcap_with_tshark(pcap_file):
    """
    Parse pcap file using tshark to extract flow information.
    This approach is faster than using Python libraries like scapy for large pcap files.
    """
    logging.info(f"Parsing pcap file: {pcap_file}")
    
    # Create a temporary file to store tshark output
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
    
    # Run tshark to extract TCP flow information
    cmd = [
        "tshark", "-r", pcap_file,
        "-Y", "tcp",
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "tcp.seq",
        "-e", "tcp.ack",
        "-e", "tcp.len",
        "-e", "tcp.flags",
        "-E", "separator=,"
    ]
    
    try:
        subprocess.run(cmd, stdout=open(temp_filename, 'w'), check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running tshark: {e}")
        os.unlink(temp_filename)
        return None
    
    flows = defaultdict(dict)
    flow_stats = {}
    
    # Define TCP flags
    SYN = 0x02
    ACK = 0x10
    FIN = 0x01
    
    # Parse the tshark output
    with open(temp_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            time_epoch, src_ip, dst_ip, src_port, dst_port, seq, ack, tcp_len, flags = parts
            
            time_epoch = float(time_epoch)
            src_port = int(src_port)
            dst_port = int(dst_port)
            seq = int(seq) if seq else 0
            ack = int(ack) if ack else 0
            tcp_len = int(tcp_len) if tcp_len else 0
            flags = int(flags, 16) if flags else 0
            
            # Create a flow ID
            flow_tuple = (src_ip, dst_ip, src_port, dst_port)
            reverse_flow_tuple = (dst_ip, src_ip, dst_port, src_port)
            
            # Check if this is part of an existing flow
            if flow_tuple in flow_stats:
                flow = flow_stats[flow_tuple]
                # Update last flag seen
                flow.last_flag = flags
            elif reverse_flow_tuple in flow_stats:
                # This is a packet in the reverse direction
                flow = flow_stats[reverse_flow_tuple]
                # Update last flag seen
                flow.last_flag = flags
            else:
                # New flow
                flow_id = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}"
                flow = FlowStats(flow_id, src_ip, dst_ip, src_port, dst_port)
                flow_stats[flow_tuple] = flow
                flow.first_flag = flags  # Record the first flag for the flow
            
            # Check for SYN packets from client to server
            if (flags & SYN) and not (flags & ACK) and src_ip == flow.src_ip:
                flow.syn_time = time_epoch
                # Add SYN packet for RTT tracking
                flow.add_packet(seq, time_epoch, 1, flags)  # SYN consumes 1 sequence number
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"SYN packet detected at {time_epoch}: {src_ip}:{src_port} → {dst_ip}:{dst_port}")
            
            # Track data packets sent from client to server
            if src_ip == flow.src_ip:
                # Add all packets from client to server for RTT tracking
                data_len = tcp_len if tcp_len > 0 else 1  # Ensure at least 1 for SYN/FIN
                flow.add_packet(seq, time_epoch, data_len, flags)
                
                if tcp_len > 0:
                    flow.bytes_sent += tcp_len
                    flow.packets_sent += 1
                    flow.last_seq = seq + tcp_len
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Data packet sent: {tcp_len} bytes, flags={flags_to_string(flags)}")
            
            # Track ACKs from server to client
            if src_ip == flow.dst_ip and (flags & ACK):
                flow.packets_received += 1
                
                # Process this ACK for RTT calculation
                newly_acked = flow.process_ack(ack, time_epoch)
                if logging.getLogger().isEnabledFor(logging.DEBUG) and newly_acked:
                    for packet in newly_acked:
                        logging.debug(f"RTT calculated: {packet.rtt*1000:.2f}ms for seq {packet.seq_num}")
                
                # Check if this ACK acknowledges all data sent
                if ack > flow.highest_ack:
                    flow.highest_ack = ack
                    
                    # If this ACK acknowledges all data sent by the client
                    if flow.last_seq and ack >= flow.last_seq and flow.syn_time:
                        flow.last_ack_time = time_epoch
                        flow.complete = True
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(f"Complete ACK received at {time_epoch}, flags={flags_to_string(flags)}")
    
    # Cleanup temp file
    os.unlink(temp_filename)
    
    # Compute FCT for each flow
    result_flows = []
    for flow in flow_stats.values():
        flow.compute_fct()
        if flow.complete:
            result_flows.append(flow)
    
    return result_flows

def write_stats_to_csv(flows, output_file):
    """Write flow statistics to a CSV file."""
    logging.info(f"Writing {len(flows)} flows to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
            "fct", "bytes_sent", "packets_sent", "packets_received",
            "first_flag", "last_flag", "min_rtt", "max_rtt", "avg_rtt", "median_rtt", 
            "retransmissions"
        ])
        
        for flow in flows:
            rtt_stats = flow.get_rtt_stats()
            writer.writerow([
                flow.flow_id,
                flow.src_ip,
                flow.dst_ip,
                flow.src_port,
                flow.dst_port,
                flow.fct,
                flow.bytes_sent,
                flow.packets_sent,
                flow.packets_received,
                flags_to_string(flow.first_flag),
                flags_to_string(flow.last_flag),
                rtt_stats['min_rtt']*1000 if rtt_stats['min_rtt'] is not None else None,  # Convert to ms
                rtt_stats['max_rtt']*1000 if rtt_stats['max_rtt'] is not None else None,  # Convert to ms
                rtt_stats['avg_rtt']*1000 if rtt_stats['avg_rtt'] is not None else None,  # Convert to ms
                rtt_stats['median_rtt']*1000 if rtt_stats['median_rtt'] is not None else None,  # Convert to ms
                rtt_stats['retransmissions']
            ])

def write_packet_rtts_to_csv(flows, output_file):
    """Write per-packet RTT details to a separate CSV file."""
    logging.info(f"Writing packet RTT data to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
            "seq_num", "data_len", "send_time", "ack_time", "rtt_ms", "flags"
        ])
        
        for flow in flows:
            for seq_num, packet in sorted(flow.packet_rtts.items()):
                writer.writerow([
                    flow.flow_id,
                    flow.src_ip,
                    flow.dst_ip,
                    flow.src_port,
                    flow.dst_port,
                    packet.seq_num,
                    packet.data_len,
                    packet.send_time,
                    packet.ack_time,
                    packet.rtt*1000 if packet.rtt is not None else None,  # Convert to ms
                    packet.flags_str
                ])

def analyze_flows(flows):
    """Analyze the FCT distribution of the flows."""
    if not flows:
        logging.warning("No flows to analyze")
        return
    
    # Calculate FCT statistics
    fcts = [flow.fct for flow in flows if flow.fct is not None]
    if not fcts:
        logging.warning("No valid FCTs found")
        return
    
    fcts.sort()
    
    # Calculate basic statistics
    avg_fct = sum(fcts) / len(fcts)
    median_fct = fcts[len(fcts) // 2]
    min_fct = min(fcts)
    max_fct = max(fcts)
    p99_fct = fcts[int(len(fcts) * 0.99)] if len(fcts) >= 100 else max_fct
    
    logging.info(f"FCT Analysis for {len(flows)} flows:")
    logging.info(f"  Average FCT: {avg_fct*1000:.2f} ms")
    logging.info(f"  Median FCT: {median_fct*1000:.2f} ms")
    logging.info(f"  Min FCT: {min_fct*1000:.2f} ms")
    logging.info(f"  Max FCT: {max_fct*1000:.2f} ms")
    logging.info(f"  99th percentile FCT: {p99_fct*1000:.2f} ms")
    
    # RTT analysis across all flows
    all_rtts = []
    for flow in flows:
        all_rtts.extend(flow.rtts)
    
    if all_rtts:
        all_rtts.sort()
        avg_rtt = sum(all_rtts) / len(all_rtts)
        median_rtt = all_rtts[len(all_rtts) // 2]
        min_rtt = min(all_rtts)
        max_rtt = max(all_rtts)
        p99_rtt = all_rtts[int(len(all_rtts) * 0.99)] if len(all_rtts) >= 100 else max_rtt
        
        logging.info(f"\nRTT Analysis for {len(all_rtts)} packets:")
        logging.info(f"  Average RTT: {avg_rtt*1000:.2f} ms")
        logging.info(f"  Median RTT: {median_rtt*1000:.2f} ms")
        logging.info(f"  Min RTT: {min_rtt*1000:.2f} ms")
        logging.info(f"  Max RTT: {max_rtt*1000:.2f} ms")
        logging.info(f"  99th percentile RTT: {p99_rtt*1000:.2f} ms")
    else:
        logging.warning("No RTT data available")
    
    # Count flows by byte size
    size_counts = defaultdict(int)
    for flow in flows:
        size_category = f"{flow.bytes_sent // 1000}k" if flow.bytes_sent >= 1000 else f"{flow.bytes_sent}B"
        size_counts[size_category] += 1
    
    logging.info("\nFlow size distribution:")
    for size, count in sorted(size_counts.items(), key=lambda x: (len(x[0]), x[0])):
        logging.info(f"  {size}: {count} flows")
    
    # Display a sample of flows with their flags and RTT stats
    logging.info("\nSample flows with flags and RTT:")
    for i, flow in enumerate(flows[:5]):  # Show first 5 flows
        rtt_stats = flow.get_rtt_stats()
        rtt_info = "No RTT data"
        if rtt_stats['avg_rtt'] is not None:
            rtt_info = f"RTT: {rtt_stats['avg_rtt']*1000:.2f}ms (min={rtt_stats['min_rtt']*1000:.2f}ms, max={rtt_stats['max_rtt']*1000:.2f}ms)"
        
        logging.info(f"  {i+1}. {flow.src_ip}:{flow.src_port} → {flow.dst_ip}:{flow.dst_port}, "
                   f"First: {flags_to_string(flow.first_flag)}, Last: {flags_to_string(flow.last_flag)}, "
                   f"FCT: {flow.fct*1000:.2f}ms, {rtt_info}")

def main():
    parser = argparse.ArgumentParser(description='Compute TCP flow completion times from pcap files')
    parser.add_argument('pcap_file', help='Path to the pcap file')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('-r', '--rtt-output', help='Output CSV file for packet RTT data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.pcap_file):
        logging.error(f"PCAP file not found: {args.pcap_file}")
        return 1
    
    flows = parse_pcap_with_tshark(args.pcap_file)
    
    if not flows:
        logging.error("No flows found in the pcap file")
        return 1
    
    analyze_flows(flows)
    
    if args.output:
        write_stats_to_csv(flows, args.output)
        logging.info(f"Flow statistics written to {args.output}")
    
    # Generate RTT output filename if not specified
    rtt_output = args.rtt_output
    if not rtt_output and args.output:
        # Derive name from main output
        base_name = os.path.splitext(args.output)[0]
        rtt_output = f"{base_name}_rtt.csv"
    
    if rtt_output:
        write_packet_rtts_to_csv(flows, rtt_output)
        logging.info(f"Packet RTT data written to {rtt_output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 