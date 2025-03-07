#!/usr/bin/env python3
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp, bind_layers, sendpfast, wrpcap
import time
from argparse import ArgumentParser
import traceback
import random

class FlowHeader(Packet):
    name = "Flow"
    fields_desc = [
        BitField("flow_id", 0, 32),
        BitField("seq", 0, 32)
    ]

def send_out_of_order_pkts(src_ip, iface, dst_ip, dst_port, num_packets=1000, interval=0.001, 
                        out_of_order_probability=0.2, num_flows=1):
    print(f"Sending {num_packets} packets from {src_ip} to {dst_ip}:{dst_port}")
    print(f"Using {num_flows} flows with {out_of_order_probability*100}% out-of-order probability")
    
    # Create a list to hold all packets from all flows
    all_packets = []
    
    # Generate packets for each flow
    for flow_num in range(num_flows):
        flow_id = random.randint(0, 2**32-1)
        print(f"Flow {flow_num+1}: ID = {flow_id}")
        
        # Generate packets in sequence for this flow
        flow_packets = []
        for seq in range(num_packets):
            pkt = (Ether(dst="00:00:0a:00:02:02") /
                  IP(src=src_ip, dst=dst_ip) /
                  UDP(sport=12345, dport=dst_port) /
                  FlowHeader(flow_id=flow_id, seq=seq))
            flow_packets.append((seq, pkt))
        
        # Introduce out-of-order packets within this flow
        if out_of_order_probability > 0:
            # Identify packets that will be sent out of order
            num_out_of_order = int(num_packets * out_of_order_probability)
            
            # Select pairs of packets to swap
            for _ in range(num_out_of_order // 2):
                idx1 = random.randint(0, len(flow_packets) - 2)
                idx2 = idx1 + 1
                flow_packets[idx1], flow_packets[idx2] = flow_packets[idx2], flow_packets[idx1]
        
        # Add flow packets to the main packet list with their original sequence number
        all_packets.extend(flow_packets)
    
    # Optionally interleave packets from different flows to simulate concurrent traffic
    # if num_flows > 1:
    #     random.shuffle(all_packets)
    
    # Send packets in the (potentially modified) order
    print(f"Sending {len(all_packets)} total packets...")
    
    for seq, pkt in all_packets:
        sendp(pkt, iface=iface, verbose=False)
        time.sleep(interval)
        
    print("Done sending packets.")


def send_custom_traffic(src_ip, iface, dst_ip, dst_port, num_packets=1000, interval=0.001):
    print(f"Sending {num_packets} packets from {src_ip} to {dst_ip}:{dst_port} with {interval} s inter-arrival.")
    packet_list = []
    # generate random flow_id
    flow_id = random.randint(0, 2**32-1)
    print(f"Sending Flow ID: {flow_id}")
    for seq in range(num_packets):
        pkt = (Ether(dst="00:00:0a:00:02:02") /
               IP(src=src_ip, dst=dst_ip) /
               UDP(sport=12345, dport=dst_port) /
               FlowHeader(flow_id=flow_id, seq=seq))
        packet_list.append(pkt)

    # sendp with a list of packets will wait 'interval' seconds between each transmission
    sendp(packet_list, iface=iface, inter=interval, verbose=False)
    print("Done sending packets.")

if __name__ == '__main__':
    # Example usage: send 1000 packets from interface intf to server at server_ip port port
    parser = ArgumentParser()
    parser.add_argument("--intf", help="Interface to send packets from")
    parser.add_argument("--src_ip", help="Source IP address")
    parser.add_argument("--dst_ip", help="Destination IP address")
    parser.add_argument("--port", type=int, help="Destination port")
    parser.add_argument("--num_packets", type=int, help="Number of packets per flow", default=100)
    parser.add_argument("--interval", type=float, help="Interval between packets", default=0.001)
    parser.add_argument("--reorder_prob", type=float, help="Probability of out-of-order packets", default=0)
    parser.add_argument("--num_flows", type=int, help="Number of flows to generate", default=1)

    args = parser.parse_args()
    # send 1 flow of #num_packets packets
    # loopppa on the number of flows to debug different flow ids
    bind_layers(UDP, FlowHeader, dport=int(args.port))
    #send_custom_traffic(args.src_ip, args.intf, args.dst_ip, int(args.port), num_packets=args.num_packets, interval=args.interval)
    send_out_of_order_pkts(args.src_ip, args.intf, args.dst_ip, int(args.port), num_packets=args.num_packets, interval=args.interval, out_of_order_probability=args.reorder_prob, num_flows=args.num_flows)
