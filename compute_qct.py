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
import traceback

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

class QueryStats:
    """Class to track statistics for a query (burst)."""
    def __init__(self, query_id):
        self.query_id = query_id
        self.first_request_time = None
        self.last_response_time = None
        self.servers = set()
        self.bytes_received = 0
        self.packets_received = 0
        self.qct = None
        self.flows = {}  # Map of flow_id -> FlowStats
    
    def compute_qct(self):
        """Compute the Query Completion Time."""
        if self.first_request_time and self.last_response_time:
            self.qct = self.last_response_time - self.first_request_time
            return self.qct
        return None
    
    def add_server(self, server_ip):
        """Add a server to this query."""
        self.servers.add(server_ip)
    
    def update_request_time(self, time_epoch):
        """Update the first request time if this is earlier."""
        if self.first_request_time is None or time_epoch < self.first_request_time:
            self.first_request_time = time_epoch
    
    def update_response_time(self, time_epoch):
        """Update the last response time if this is later."""
        if self.last_response_time is None or time_epoch > self.last_response_time:
            self.last_response_time = time_epoch
    
    def finalize(self):
        """Compute the QCT (query completion time) for this query."""
        # Only calculate QCT if there is actual data transfer (bytes received)
        if self.bytes_received > 0 and self.first_request_time and self.last_response_time:
            if self.first_request_time < self.last_response_time:
                self.qct = self.last_response_time - self.first_request_time
        
        # Finalize all flows
        for flow in self.flows.values():
            flow.finalize()
            
        return self
    
    def __str__(self):
        return (f"Query {self.query_id}: QCT: {self.qct:.6f}s, "
                f"Servers: {len(self.servers)}, Bytes: {self.bytes_received}, "
                f"Start: {self.first_request_time}, End: {self.last_response_time}")

class FlowStats:
    """Class to track statistics for a flow within a query."""
    def __init__(self, flow_id, src_ip, dst_ip, src_port, dst_port):
        self.flow_id = flow_id
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.request_time = None
        self.last_response_time = None
        self.bytes_received = 0
        self.packets_received = 0
        self.packets_sent = 0
        self.fct = None
        self.complete = False
    
    def finalize(self):
        """Compute the FCT (flow completion time) for this flow."""
        # Only calculate FCT if there is actual data transfer (bytes received)
        if self.bytes_received > 0 and self.request_time and self.last_response_time:
            if self.request_time < self.last_response_time:
                self.fct = self.last_response_time - self.request_time
        return self
    
    def __str__(self):
        fct_str = f"{self.fct:.6f}s" if self.fct else "N/A"
        return (f"Flow {self.flow_id}: {self.src_ip}:{self.src_port} → {self.dst_ip}:{self.dst_port}, "
                f"FCT: {fct_str}, Bytes: {self.bytes_received}")

def parse_pcap_with_tshark(pcap_file):
    """
    Parse pcap file using tshark to extract query and flow information.
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
    
    # Define TCP flags
    SYN = 0x02
    ACK = 0x10
    FIN = 0x01
    PSH = 0x08
    
    # Containers for our data
    query_stats = {}  # Map of source_port_base -> QueryStats
    flow_stats = {}   # Map of (src_ip, dst_ip, src_port, dst_port) -> FlowStats
    
    # Identify client IP (source of the pcap)
    client_ip = None
    port_to_query = {}  # Map from source port to query ID
    flow_tuples = set()  # Set of unique flow tuples for deduplication
    
    # First scan to identify client IP and group ports by query
    with open(temp_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            time_epoch, src_ip, dst_ip, src_port, dst_port, seq, ack, tcp_len, flags = parts
            
            src_port = int(src_port) if src_port else 0
            dst_port = int(dst_port) if dst_port else 0
            
            # Identify client IP based on port 12346 (burst port)
            if dst_port == 12346 and client_ip is None:
                client_ip = src_ip
                logging.info(f"Identified client IP: {client_ip}")
            
            # Skip if client IP not identified yet
            if client_ip is None:
                continue
            
            # Consider outgoing connections to bursty server port
            if src_ip == client_ip and dst_port == 12346:
                # Use individual source port as the query identifier instead of grouping
                port_base = src_port  # Each source port is a separate query
                
                # Create new query if not seen
                if port_base not in port_to_query:
                    query_id = f"query_{port_base}"
                    port_to_query[port_base] = query_id
                    query_stats[query_id] = QueryStats(query_id)
                
                # Add flow tuple
                flow_tuple = (src_ip, dst_ip, src_port, dst_port)
                flow_tuples.add(flow_tuple)
    
    # If no client IP or queries found, exit
    if client_ip is None:
        logging.error("Could not identify client IP in the pcap file")
        os.unlink(temp_filename)
        return None
    
    if not query_stats:
        logging.error("No queries/bursts found in the pcap file")
        os.unlink(temp_filename)
        return None
    
    logging.info(f"Found {len(query_stats)} potential queries and {len(flow_tuples)} flows")
    
    # Initialize flow stats objects
    for src_ip, dst_ip, src_port, dst_port in flow_tuples:
        # Each source port is its own query
        port_base = src_port
        if port_base in port_to_query:
            query_id = port_to_query[port_base]
            flow_id = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}"
            flow = FlowStats(flow_id, src_ip, dst_ip, src_port, dst_port)
            flow_stats[(src_ip, dst_ip, src_port, dst_port)] = flow
            
            # Link the flow to its query
            query_stats[query_id].flows[flow_id] = flow
            query_stats[query_id].add_server(dst_ip)
    
    # Second scan to process packet details
    with open(temp_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            time_epoch, src_ip, dst_ip, src_port, dst_port, seq, ack, tcp_len, flags = parts
            
            time_epoch = float(time_epoch)
            src_port = int(src_port) if src_port else 0
            dst_port = int(dst_port) if dst_port else 0
            flags = int(flags, 16) if flags else 0
            tcp_len = int(tcp_len) if tcp_len else 0
            
            # Process outgoing requests (client to server)
            if src_ip == client_ip and dst_port == 12346:
                flow_tuple = (src_ip, dst_ip, src_port, dst_port)
                if flow_tuple in flow_stats:
                    flow = flow_stats[flow_tuple]
                    
                    # Update request time
                    if flow.request_time is None or time_epoch < flow.request_time:
                        flow.request_time = time_epoch
                    
                    # Update the corresponding query's request time
                    port_base = src_port
                    if port_base in port_to_query:
                        query_id = port_to_query[port_base]
                        query = query_stats[query_id]
                        query.update_request_time(time_epoch)
            
            # Process incoming responses (server to client)
            elif dst_ip == client_ip and src_port == 12346:
                flow_tuple = (dst_ip, src_ip, dst_port, src_port)  # Reverse the tuple
                if flow_tuple in flow_stats:
                    flow = flow_stats[flow_tuple]
                    
                    # Update stats
                    flow.bytes_received += tcp_len
                    flow.packets_received += 1
                    
                    # Update last response time
                    if flow.last_response_time is None or time_epoch > flow.last_response_time:
                        flow.last_response_time = time_epoch
                    
                    # Mark as complete if FIN flag is set
                    if flags & FIN:
                        flow.complete = True
                    
                    # Calculate FCT if we have enough data
                    if flow.request_time and flow.last_response_time:
                        flow.fct = flow.last_response_time - flow.request_time
                    
                    # Update the corresponding query's response time and stats
                    port_base = dst_port
                    if port_base in port_to_query:
                        query_id = port_to_query[port_base]
                        query = query_stats[query_id]
                        query.update_response_time(time_epoch)
                        query.bytes_received += tcp_len
                        query.packets_received += 1
    
    # Compute QCT for each query
    for query in query_stats.values():
        query.finalize()
    
    # Cleanup temp file
    os.unlink(temp_filename)
    
    # Only return queries with valid QCT
    valid_queries = [q for q in query_stats.values() if q.qct is not None]
    logging.info(f"Found {len(valid_queries)} valid queries with QCT")
    
    return valid_queries

def write_stats(query_stats, output_file, flow_details_file=None):
    """Write query stats to a CSV file."""
    # Write query stats summary
    logging.info(f"Writing query stats to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'qct', 'total_bytes', 'total_packets', 'num_flows', 'num_servers', 
                         'first_request_time', 'last_response_time'])
        
        for query_id, query in query_stats.items():
            # Convert None to empty string for CSV
            qct_value = f"{query.qct:.6f}" if query.qct is not None else ""
            
            # Only include times if QCT was properly calculated
            first_request = f"{query.first_request_time:.6f}" if query.qct is not None else ""
            last_response = f"{query.last_response_time:.6f}" if query.qct is not None else ""
            
            total_bytes = query.bytes_received
            total_packets = query.packets_received
            
            writer.writerow([
                query_id, 
                qct_value,
                total_bytes,
                total_packets,
                len(query.flows),
                len(query.servers),
                first_request,
                last_response
            ])
    
    # Write flow details if requested
    if flow_details_file:
        logging.info(f"Writing flow details to {flow_details_file}")
        with open(flow_details_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'query_id', 'flow_id', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
                'fct', 'bytes_received', 'packets_received', 'request_time', 'last_response_time'
            ])
            
            for query_id, query in query_stats.items():
                for flow_id, flow in query.flows.items():
                    # Convert None to empty string for CSV
                    fct_value = f"{flow.fct:.6f}" if flow.fct is not None else ""
                    
                    # Only include times if FCT was properly calculated 
                    request_time = f"{flow.request_time:.6f}" if flow.fct is not None else ""
                    last_response_time = f"{flow.last_response_time:.6f}" if flow.fct is not None else ""
                    
                    writer.writerow([
                        query_id,
                        flow_id,
                        flow.src_ip,
                        flow.dst_ip,
                        flow.src_port,
                        flow.dst_port,
                        fct_value,
                        flow.bytes_received,
                        flow.packets_received,
                        request_time,
                        last_response_time
                    ])

def analyze_queries(queries):
    """Analyze the QCT distribution of the queries."""
    if not queries:
        logging.warning("No queries to analyze")
        return
    
    # Calculate QCT statistics
    qcts = [q.qct for q in queries if q.qct is not None]
    if not qcts:
        logging.warning("No valid QCTs found")
        return
    
    qcts.sort()
    
    # Calculate basic statistics
    avg_qct = sum(qcts) / len(qcts)
    median_qct = qcts[len(qcts) // 2]
    min_qct = min(qcts)
    max_qct = max(qcts)
    p99_qct = qcts[int(len(qcts) * 0.99)] if len(qcts) >= 100 else max_qct
    
    logging.info(f"QCT Analysis for {len(queries)} queries:")
    logging.info(f"  Average QCT: {avg_qct*1000:.2f} ms")
    logging.info(f"  Median QCT: {median_qct*1000:.2f} ms")
    logging.info(f"  Min QCT: {min_qct*1000:.2f} ms")
    logging.info(f"  Max QCT: {max_qct*1000:.2f} ms")
    logging.info(f"  99th percentile QCT: {p99_qct*1000:.2f} ms")
    
    # Server analysis
    server_counts = [len(q.servers) for q in queries]
    avg_servers = sum(server_counts) / len(server_counts)
    logging.info(f"  Average servers per query: {avg_servers:.2f}")
    
    # Size analysis
    sizes = [q.bytes_received for q in queries]
    avg_size = sum(sizes) / len(sizes)
    logging.info(f"  Average response size: {avg_size:.2f} bytes")
    
    # Display a sample of queries with details
    logging.info("\nSample queries:")
    for i, query in enumerate(queries[:5]):  # Show first 5 queries
        logging.info(f"  {i+1}. {query.query_id}: QCT={query.qct*1000:.2f}ms, "
                   f"Servers={len(query.servers)}, Bytes={query.bytes_received}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute Query Completion Times (QCT) from pcap files')
    parser.add_argument('pcap_file', help='Path to the pcap file')
    parser.add_argument('-o', '--output', help='Output CSV file path for QCT data')
    parser.add_argument('-f', '--flow-output', help='Output CSV file for per-flow data')
    parser.add_argument('-c', '--client-ip', help='Client IP address (will be detected if not specified)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate pcap file exists
    if not os.path.exists(args.pcap_file):
        logging.error(f"PCAP file not found: {args.pcap_file}")
        sys.exit(1)
        
    return args

def determine_client_ip(pcap_file, provided_ip=None):
    """
    Determine the client IP address from a pcap file or use the provided IP.
    Assumes the client is the source of connections to port 12346 (bursty server port).
    """
    if provided_ip:
        return provided_ip
        
    # Create a temporary file for tshark output
    with tempfile.NamedTemporaryFile() as temp:
        # Run tshark to extract source IPs for connections to port 12346
        cmd = [
            'tshark', '-r', pcap_file, 
            '-Y', 'tcp.dstport == 12346',
            '-T', 'fields', '-e', 'ip.src',
            '-E', 'header=y', '-E', 'separator=,'
        ]
        
        try:
            subprocess.run(cmd, stdout=temp, check=True)
            temp.flush()
            
            # Read and count source IPs
            ip_counts = {}
            with open(temp.name, 'r') as f:
                # Skip header
                next(f, None)
                
                for line in f:
                    ip = line.strip()
                    if ip:
                        ip_counts[ip] = ip_counts.get(ip, 0) + 1
            
            # The client IP is likely the one with the most connections to port 12346
            if ip_counts:
                client_ip = max(ip_counts.items(), key=lambda x: x[1])[0]
                logging.info(f"Determined client IP: {client_ip}")
                return client_ip
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running tshark: {e}")
        except Exception as e:
            logging.error(f"Error determining client IP: {e}")
            
    return None

def process_pcap(pcap_file, client_ip=None):
    """
    Process a pcap file to extract query and flow information.
    This approach uses tshark to efficiently parse large pcap files.
    
    Args:
        pcap_file: Path to the pcap file
        client_ip: Client IP address (if None, will attempt to determine from pcap)
        
    Returns:
        Tuple of (flow_stats, query_stats) dictionaries
    """
    logging.info(f"Processing pcap file: {pcap_file}")
    
    # Determine client IP if not provided
    if not client_ip:
        client_ip = determine_client_ip(pcap_file)
        if not client_ip:
            logging.error("Could not identify client IP in the pcap file")
            return {}, {}
    
    logging.info(f"Using client IP: {client_ip}")
    
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
        return {}, {}
    
    # Define TCP flags
    SYN = 0x02
    ACK = 0x10
    FIN = 0x01
    PSH = 0x08
    
    # Containers for our data
    query_stats = {}  # Map of source_port_base -> QueryStats
    flow_stats = {}   # Map of (src_ip, dst_ip, src_port, dst_port) -> FlowStats
    
    # Map from source port to query ID
    port_to_query = {}  # Map from source port to query ID
    flow_tuples = set()  # Set of unique flow tuples for deduplication
    
    # First scan to identify client IP and group ports by query
    with open(temp_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            time_epoch, src_ip, dst_ip, src_port, dst_port, seq, ack, tcp_len, flags = parts
            
            src_port = int(src_port) if src_port else 0
            dst_port = int(dst_port) if dst_port else 0
            
            # Consider outgoing connections to bursty server port
            if src_ip == client_ip and dst_port == 12346:
                # Use individual source port as the query identifier
                port_base = src_port  # Each source port is a separate query
                
                # Create new query if not seen
                if port_base not in port_to_query:
                    query_id = f"query_{port_base}"
                    port_to_query[port_base] = query_id
                    query_stats[query_id] = QueryStats(query_id)
                
                # Add flow tuple
                flow_tuple = (src_ip, dst_ip, src_port, dst_port)
                flow_tuples.add(flow_tuple)
    
    # If no queries found, exit
    if not query_stats:
        logging.error("No queries/bursts found in the pcap file")
        os.unlink(temp_filename)
        return {}, {}
    
    logging.info(f"Found {len(query_stats)} potential queries and {len(flow_tuples)} flows")
    
    # Initialize flow stats objects
    for src_ip, dst_ip, src_port, dst_port in flow_tuples:
        # Each source port is its own query
        port_base = src_port
        if port_base in port_to_query:
            query_id = port_to_query[port_base]
            flow_id = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}"
            flow = FlowStats(flow_id, src_ip, dst_ip, src_port, dst_port)
            flow_stats[(src_ip, dst_ip, src_port, dst_port)] = flow
            
            # Link the flow to its query
            query_stats[query_id].flows[flow_id] = flow
            query_stats[query_id].add_server(dst_ip)
    
    # Second scan to process packet details
    with open(temp_filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            time_epoch, src_ip, dst_ip, src_port, dst_port, seq, ack, tcp_len, flags = parts
            
            time_epoch = float(time_epoch)
            src_port = int(src_port) if src_port else 0
            dst_port = int(dst_port) if dst_port else 0
            flags = int(flags, 16) if flags else 0
            tcp_len = int(tcp_len) if tcp_len else 0
            
            # Process outgoing requests (client to server)
            if src_ip == client_ip and dst_port == 12346:
                flow_tuple = (src_ip, dst_ip, src_port, dst_port)
                if flow_tuple in flow_stats:
                    flow = flow_stats[flow_tuple]
                    
                    # Update request time only if it's a SYN packet or contains data
                    if (flags & SYN) or tcp_len > 0:
                        # Count outgoing packets too
                        flow.packets_sent += 1
                        
                        # Update request time
                        if flow.request_time is None or time_epoch < flow.request_time:
                            flow.request_time = time_epoch
                        
                        # Update the corresponding query's request time
                        port_base = src_port
                        if port_base in port_to_query:
                            query_id = port_to_query[port_base]
                            query = query_stats[query_id]
                            query.update_request_time(time_epoch)
            
            # Process incoming responses (server to client)
            elif dst_ip == client_ip and src_port == 12346:
                flow_tuple = (dst_ip, src_ip, dst_port, src_port)  # Reverse the tuple
                if flow_tuple in flow_stats:
                    flow = flow_stats[flow_tuple]
                    
                    # Update stats
                    if tcp_len > 0:  # Only count data packets
                        flow.bytes_received += tcp_len
                    
                    # Always count the packet
                    flow.packets_received += 1
                    
                    # Update last response time if this packet has data or is a FIN
                    if tcp_len > 0 or (flags & FIN):
                        if flow.last_response_time is None or time_epoch > flow.last_response_time:
                            flow.last_response_time = time_epoch
                        
                        # Update the corresponding query's response time and stats
                        port_base = dst_port
                        if port_base in port_to_query:
                            query_id = port_to_query[port_base]
                            query = query_stats[query_id]
                            query.update_response_time(time_epoch)
                            query.bytes_received += tcp_len
                            query.packets_received += 1
    
    # Cleanup temp file
    os.unlink(temp_filename)
    
    # Filter out incomplete flows with no meaningful data transfer
    valid_flow_stats = {}
    for flow_tuple, flow in flow_stats.items():
        # Must have at least 2 packets received and some bytes transferred
        # to be considered a valid flow
        if flow.packets_received >= 2 and flow.bytes_received > 0:
            if flow.request_time is not None and flow.last_response_time is not None:
                if flow.request_time < flow.last_response_time:
                    valid_flow_stats[flow_tuple] = flow
    
    # Update query flows to include only valid flows
    for query_id, query in query_stats.items():
        valid_flows = {}
        for flow_id, flow in query.flows.items():
            flow_tuple = (flow.src_ip, flow.dst_ip, flow.src_port, flow.dst_port)
            if flow_tuple in valid_flow_stats:
                valid_flows[flow_id] = flow
        
        query.flows = valid_flows
    
    # Filter out queries with no valid flows
    valid_query_stats = {}
    for query_id, query in query_stats.items():
        if query.flows and len(query.flows) > 0:
            valid_query_stats[query_id] = query
    
    return valid_flow_stats, valid_query_stats

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Get client IP if not provided
        client_ip = args.client_ip
        if not client_ip:
            client_ip = determine_client_ip(args.pcap_file, None)
            if not client_ip:
                logging.error("Could not determine client IP. Please specify using --client_ip.")
                return 1
            logging.info(f"Determined client IP: {client_ip}")
        
        # Process PCAP file to get flow and query statistics
        flow_stats, query_stats = process_pcap(args.pcap_file, client_ip)
        
        # Finalize all queries to calculate QCT
        for query in query_stats.values():
            query.finalize()
        
        # Log overall stats
        total_flows = len(flow_stats)
        total_queries = len(query_stats)
        logging.info(f"Processed {total_flows} flows in {total_queries} queries")
        
        # Generate flow output filename if not specified
        flow_output = args.flow_output
        if not flow_output and args.output:
            # Derive name from main output
            base_name = os.path.splitext(args.output)[0]
            flow_output = f"{base_name}_flows.csv"
        
        # Write statistics to output files
        if args.output:
            write_stats(query_stats, args.output, flow_output)
            logging.info(f"Query statistics written to {args.output}")
            
            if flow_output:
                logging.info(f"Flow details written to {flow_output}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error processing PCAP file: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 