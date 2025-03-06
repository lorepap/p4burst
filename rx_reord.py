#!/usr/bin/env python3
from scapy.all import sniff, Ether, IP, UDP, Packet, BitField, bind_layers
from argparse import ArgumentParser
from server import BaseServer
import time, csv
import threading

class FlowHeader(Packet):
    name = "Flow"
    fields_desc = [
        BitField("flow_id", 0, 32),
        BitField("seq", 0, 32)
    ]

# Parse arguments first so we can get the log file path
parser = ArgumentParser()
parser.add_argument("--intf", help="Interface to sniff packets from")
parser.add_argument("--port", help="Port to start the server on")
parser.add_argument("--log", help="Log file to write the received packets to", default="receiver_log.csv")
args = parser.parse_args()

# This dictionary will store, per flow (or port), the highest sequence number seen so far.
highest_seq = {}

# Log file for the receiver
log_file = args.log
with open(log_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "src_ip", "dst_ip", "udp_dport", "flow_id", "seq", "reordering_flag"])

def process_packet(pkt):
    print("Processing packet: ", pkt.summary())
    if UDP in pkt:
        if FlowHeader not in pkt:
            print("Received a UDP packet without FlowHeader")
            return        
        raw_payload = bytes(pkt[UDP].payload)
        # Extract the sequence number and other info.
        flow_id = pkt[FlowHeader].flow_id
        seq = pkt[FlowHeader].seq
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        dport = pkt[UDP].dport
        arrival_time = time.time()  # current timestamp in seconds
        key = (src_ip, dst_ip, dport)  # define a flow key

        # Check if this packet is out-of-order:
        reorder_flag = 0
        if key in highest_seq:
            if seq < highest_seq[key]:
                reorder_flag = 1  # packet arrives with a lower seq than previously seen
            else:
                highest_seq[key] = seq
        else:
            highest_seq[key] = seq

        print(f"Received packet with seq {seq} from {src_ip} to {dst_ip}:{dport} at {arrival_time}. Reorder flag: {reorder_flag}")

        # Log the packet information
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([arrival_time, src_ip, dst_ip, dport, flow_id, seq, reorder_flag])
    else:
        print("Received a non-UDP packet")

def start_server(port):
    print("Starting server")
    # Start the server
    server = BaseServer(port=port, ip="0.0.0.0")
    server.start()

# Start the server in a separate thread
server_thread = threading.Thread(target=start_server, args=(int(args.port),))
server_thread.daemon = True
server_thread.start()
bind_layers(UDP, FlowHeader, dport=int(args.port))
print("Starting sniffing")
sniff(iface=args.intf, filter=f"udp and dst port {args.port}", prn=process_packet, store=0)


