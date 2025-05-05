#!/bin/bash

# Script to run TCP client and analyze FCT from pcap files
set -e

# Default values
DURATION=60
FLOW_SIZE=10000
FLOW_IAT=0.001
QUEUE_RATE=1000
QUEUE_DEPTH=64
N_CLIENTS=5
N_SERVERS=5
N_HOSTS=14
N_LEAF=2
N_SPINE=2
BW=10
BURST_SERVERS=2
BURST_CLIENTS=2
BURST_INTERVAL=0.2
BURST_REPLY_SIZE=4000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --flow_size)
      FLOW_SIZE="$2"
      shift 2
      ;;
    --flow_iat)
      FLOW_IAT="$2"
      shift 2
      ;;
    --queue_rate)
      QUEUE_RATE="$2"
      shift 2
      ;;
    --queue_depth)
      QUEUE_DEPTH="$2"
      shift 2
      ;;
    --n_clients)
      N_CLIENTS="$2"
      shift 2
      ;;
    --n_servers)
      N_SERVERS="$2"
      shift 2
      ;;
    --n_hosts)
      N_HOSTS="$2"
      shift 2
      ;;
    --n_leaf)
      N_LEAF="$2"
      shift 2
      ;;
    --n_spine)
      N_SPINE="$2"
      shift 2
      ;;
    --bw)
      BW="$2"
      shift 2
      ;;
    --burst_servers)
      BURST_SERVERS="$2"
      shift 2
      ;;
    --burst_clients)
      BURST_CLIENTS="$2"
      shift 2
      ;;
    --burst_interval)
      BURST_INTERVAL="$2"
      shift 2
      ;;
    --burst_reply_size)
      BURST_REPLY_SIZE="$2"
      shift 2
      ;;
    --exp_id)
      EXP_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Generate a unique experiment ID based on timestamp if not provided
if [ -z "$EXP_ID" ]; then
  EXP_ID=$(date +%Y%m%d_%H%M%S)
fi

OUTPUT_DIR="tmp/${EXP_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/fct_stats"
mkdir -p "$OUTPUT_DIR/fct_stats/plots"
mkdir -p "$OUTPUT_DIR/utilization"
mkdir -p "$OUTPUT_DIR/qct_stats"
mkdir -p "$OUTPUT_DIR/qct_stats/plots"

echo "Starting experiment $EXP_ID with settings:"
echo "  Duration: $DURATION seconds"
echo "  Flow size: $FLOW_SIZE bytes"
echo "  Flow IAT: $FLOW_IAT seconds"
echo "  Queue rate: $QUEUE_RATE pps"
echo "  Queue depth: $QUEUE_DEPTH packets"
echo "  Clients: $N_CLIENTS"
echo "  Servers: $N_SERVERS"
echo "  Hosts: $N_HOSTS"
echo "  Leaf switches: $N_LEAF"
echo "  Spine switches: $N_SPINE"
echo "  Bandwidth: $BW Mbps"
echo "  Burst interval: $BURST_INTERVAL seconds"
echo "  Burst reply size: $BURST_REPLY_SIZE bytes"
echo "  Output directory: $OUTPUT_DIR"

# Run the collection_runner.py script
echo "Starting network simulation..."
sudo -E python3 collection_runner.py \
  --duration "$DURATION" \
  --n_hosts "$N_HOSTS" \
  --n_leaf "$N_LEAF" \
  --n_spine "$N_SPINE" \
  --bw "$BW" \
  --n_clients "$N_CLIENTS" \
  --n_servers "$N_SERVERS" \
  --flow_iat "$FLOW_IAT" \
  --flow_size "$FLOW_SIZE" \
  --burst_interval "$BURST_INTERVAL" \
  --bursty_reply_size "$BURST_REPLY_SIZE" \
  --burst_servers "$BURST_SERVERS" \
  --burst_clients "$BURST_CLIENTS" \
  --queue_rate "$QUEUE_RATE" \
  --queue_depth "$QUEUE_DEPTH" \
  --exp_id "$EXP_ID" \
  --switch_pcap

echo "Experiment completed. Processing pcap files..."

# Create directories for client and server analysis
CLIENT_DIR="$OUTPUT_DIR/bg_clients"
SERVER_DIR="$OUTPUT_DIR/bg_servers"
BURSTY_DIR="$OUTPUT_DIR/bursty_clients"

mkdir -p "$CLIENT_DIR"
mkdir -p "$SERVER_DIR"
mkdir -p "$BURSTY_DIR"

# Move client and server pcap files to their respective directories
echo "Organizing pcap files..."
find "$OUTPUT_DIR" -name "bg_client_*.pcap" -exec cp {} "$CLIENT_DIR/" \;
find "$OUTPUT_DIR" -name "bg_server_*.pcap" -exec cp {} "$SERVER_DIR/" \;
find "$OUTPUT_DIR" -name "bursty_client_*.pcap" -exec cp {} "$BURSTY_DIR/" \;

# Function to check if pcap contains valid TCP flows
check_pcap_has_flows() {
  local pcap_file="$1"
  # Count TCP packets in the pcap file
  local flow_count=$(tshark -r "$pcap_file" -Y "tcp" -c 10 2>/dev/null | wc -l)
  
  # Return 0 (success) if flows found, 1 (failure) if no flows
  if [ "$flow_count" -gt 0 ]; then
    return 0  # Has flows
  else
    return 1  # No flows
  fi
}

# Function to calculate link utilization for a client
calculate_link_utilization() {
  local pcap_file="$1"
  local output_dir="$2"
  local client_name="$3"
  local bw_mbps="$4"  # Link bandwidth in Mbps
  
  echo "Calculating link utilization for $client_name..."
  
  # Call the Python script to calculate utilization
  ./calculate_link_utilization.py "$pcap_file" "$output_dir" "$client_name" --bw "$bw_mbps" || \
    echo "  Error calculating utilization for $client_name"
}

# Process all client pcap files
echo "Analyzing client pcap files for FCT and RTT..."
CLIENT_COUNT=0
for CLIENT_PCAP in "$CLIENT_DIR"/*.pcap; do
  if [ -f "$CLIENT_PCAP" ]; then
    CLIENT_NAME=$(basename "$CLIENT_PCAP" .pcap)
    
    # Check if this client has any flows
    if check_pcap_has_flows "$CLIENT_PCAP"; then
      CLIENT_COUNT=$((CLIENT_COUNT + 1))
      echo "Processing client: $CLIENT_NAME"
      
      # Run the FCT analysis script with RTT output
      python3 compute_fct.py "$CLIENT_PCAP" -o "$OUTPUT_DIR/fct_stats/${CLIENT_NAME}_stats.csv" -r "$OUTPUT_DIR/fct_stats/${CLIENT_NAME}_stats_rtt.csv" 2>&1 | grep -v "No flows found"
      
      echo "FCT and RTT statistics for $CLIENT_NAME written to $OUTPUT_DIR/fct_stats/"
      
      # Calculate link utilization for this client
      calculate_link_utilization "$CLIENT_PCAP" "$OUTPUT_DIR" "$CLIENT_NAME" "$BW"
    else
      echo "Skipping client $CLIENT_NAME (no TCP flows found)"
    fi
  fi
done

# If no client pcap files were found, display an error
if [ "$CLIENT_COUNT" -eq 0 ]; then
  echo "Error: No client pcap files with valid flows found in $CLIENT_DIR"
  exit 1
fi

# Also process the server pcap files to ensure we capture both sides of the connection
echo "Analyzing server pcap files for RTT..."
SERVER_COUNT=0
ACTIVE_SERVER_COUNT=0
for SERVER_PCAP in "$SERVER_DIR"/*.pcap; do
  if [ -f "$SERVER_PCAP" ]; then
    SERVER_COUNT=$((SERVER_COUNT + 1))
    SERVER_NAME=$(basename "$SERVER_PCAP" .pcap)
    
    # Check if this server has any flows before processing
    if check_pcap_has_flows "$SERVER_PCAP"; then
      ACTIVE_SERVER_COUNT=$((ACTIVE_SERVER_COUNT + 1))
      echo "Processing server: $SERVER_NAME"
      
      # Run the FCT analysis script for the server side
      python3 compute_fct.py "$SERVER_PCAP" -o "$OUTPUT_DIR/fct_stats/${SERVER_NAME}_stats.csv" -r "$OUTPUT_DIR/fct_stats/${SERVER_NAME}_stats_rtt.csv" 2>&1 | grep -v "No flows found"
      
      echo "FCT and RTT statistics for $SERVER_NAME written to $OUTPUT_DIR/fct_stats/"
    else
      echo "Skipping server $SERVER_NAME (idle server with no TCP flows)"
    fi
  fi
done

# Process bursty client pcap files for QCT analysis
echo "Analyzing bursty client pcap files for QCT..."
BURSTY_CLIENT_COUNT=0
for BURSTY_PCAP in "$BURSTY_DIR"/*.pcap; do
  if [ -f "$BURSTY_PCAP" ]; then
    BURSTY_NAME=$(basename "$BURSTY_PCAP" .pcap)
    CLIENT_IP=$(echo "$BURSTY_NAME" | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
    
    # Check if this client has any flows
    if check_pcap_has_flows "$BURSTY_PCAP"; then
      BURSTY_CLIENT_COUNT=$((BURSTY_CLIENT_COUNT + 1))
      echo "Processing bursty client: $BURSTY_NAME with IP $CLIENT_IP"
      
      # Run the QCT analysis script with client IP explicitly specified
      python3 compute_qct.py "$BURSTY_PCAP" -o "$OUTPUT_DIR/qct_stats/${BURSTY_NAME}_stats.csv" -f "$OUTPUT_DIR/qct_stats/${BURSTY_NAME}_flows.csv" -c "$CLIENT_IP" 2>&1 | grep -v "No flows found"
      
      echo "QCT statistics for $BURSTY_NAME written to $OUTPUT_DIR/qct_stats/"
    else
      echo "Skipping bursty client $BURSTY_NAME (no TCP flows found)"
    fi
  fi
done

# If no bursty client pcap files were found, display a message
if [ "$BURSTY_CLIENT_COUNT" -eq 0 ]; then
  echo "No bursty client pcap files with valid flows found in $BURSTY_DIR"
else
  echo "Successfully analyzed $BURSTY_CLIENT_COUNT bursty clients for QCT"
  
  # Combine QCT statistics from all bursty clients
  if [ -d "$OUTPUT_DIR/qct_stats" ] && [ "$(ls -A "$OUTPUT_DIR/qct_stats" 2>/dev/null)" ]; then
    echo "Combining QCT statistics and generating visualizations..."
    ./combine_qct_stats.py "$OUTPUT_DIR/qct_stats" -o "$OUTPUT_DIR/qct_stats"
    if [ $? -eq 0 ]; then
      echo "QCT Visualizations generated successfully:"
      echo "  - Histogram: $OUTPUT_DIR/qct_stats/plots/qct_histogram.png"
      echo "  - CDF: $OUTPUT_DIR/qct_stats/plots/qct_cdf.png"
      echo "  - Boxplot: $OUTPUT_DIR/qct_stats/plots/qct_boxplot.png"
      echo "  - Combined stats: $OUTPUT_DIR/qct_stats/combined_qct_stats.csv"
      echo "  - Flow stats: $OUTPUT_DIR/qct_stats/combined_flow_stats.csv"
    else
      echo "Warning: Could not generate combined QCT visualizations"
    fi
  fi
fi

# Also extract the switch pcap files if they exist
echo "Analyzing switch pcap files..."
for PCAP in pcap/*.pcap; do
  if [ -f "$PCAP" ]; then
    SWITCH_NAME=$(basename "$PCAP" .pcap)
    echo "Processing switch pcap: $SWITCH_NAME"
    
    # Extract TCP flows from switch pcap
    tshark -r "$PCAP" -Y tcp \
      -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport \
      -e tcp.seq -e tcp.len -e tcp.flags -e tcp.flags.str \
      -E header=y -E separator=, > "$OUTPUT_DIR/switch_${SWITCH_NAME}.csv"
      
    echo "  Extracted to $OUTPUT_DIR/switch_${SWITCH_NAME}.csv"
  fi
done

# Generate an overall summary
echo "Generating summary statistics..."
echo "Experiment: $EXP_ID" > "$OUTPUT_DIR/summary.txt"
echo "Clients analyzed: $CLIENT_COUNT" >> "$OUTPUT_DIR/summary.txt"
echo "Servers found: $SERVER_COUNT (active: $ACTIVE_SERVER_COUNT)" >> "$OUTPUT_DIR/summary.txt"
echo "Configuration:" >> "$OUTPUT_DIR/summary.txt"
echo "  Queue rate: $QUEUE_RATE pps" >> "$OUTPUT_DIR/summary.txt"
echo "  Queue depth: $QUEUE_DEPTH packets" >> "$OUTPUT_DIR/summary.txt"
echo "  Flow size: $FLOW_SIZE bytes" >> "$OUTPUT_DIR/summary.txt"
echo "  Flow IAT: $FLOW_IAT seconds" >> "$OUTPUT_DIR/summary.txt"
echo "  Link bandwidth: $BW Mbps" >> "$OUTPUT_DIR/summary.txt"

# Calculate average FCT and RTT if we have client data
if [ "$CLIENT_COUNT" -gt 0 ]; then
  echo "Calculating average FCT and RTT statistics across client flows..."
  ./calculate_average_statistics.py "$OUTPUT_DIR/fct_stats" "$OUTPUT_DIR/summary.txt" || \
    echo "Error generating summary statistics"
fi

# Generate aggregated link utilization report
echo "Generating aggregated link utilization report..."
./generate_aggregated_utilization_report.py "$OUTPUT_DIR" || \
  echo "Error generating aggregated link utilization report"

# Add link utilization statistics to the summary
if [ -f "$OUTPUT_DIR/link_utilization_report.txt" ]; then
  echo "Adding link utilization statistics to summary..."
  echo -e "\nLink Utilization Statistics:" >> "$OUTPUT_DIR/summary.txt"
  grep -A 100 "Per-Client Statistics" "$OUTPUT_DIR/link_utilization_report.txt" >> "$OUTPUT_DIR/summary.txt"
fi

# Add QCT information to summary if available
if [ "$BURSTY_CLIENT_COUNT" -gt 0 ]; then
  echo -e "\nBursty Traffic Statistics:" >> "$OUTPUT_DIR/summary.txt"
  echo "  Bursty clients analyzed: $BURSTY_CLIENT_COUNT" >> "$OUTPUT_DIR/summary.txt"
  
  # Add average QCT if available
  if [ -f "$OUTPUT_DIR/qct_stats/combined_qct_stats.csv" ]; then
    # Extract average QCT from the combined stats using Python
    AVG_QCT=$(python3 -c "import pandas as pd; df = pd.read_csv('$OUTPUT_DIR/qct_stats/combined_qct_stats.csv'); print(f'{df[\"qct\"].mean()*1000:.2f}')")
    echo "  Average QCT: ${AVG_QCT} ms" >> "$OUTPUT_DIR/summary.txt"
    
    # Extract number of servers from the combined stats
    AVG_SERVERS=$(python3 -c "import pandas as pd; df = pd.read_csv('$OUTPUT_DIR/qct_stats/combined_qct_stats.csv'); print(f'{df[\"num_servers\"].mean():.2f}') if 'num_servers' in df.columns else print('N/A')")
    echo "  Average servers per query: ${AVG_SERVERS}" >> "$OUTPUT_DIR/summary.txt"
  fi
fi

# Combine FCT and RTT statistics from all clients and generate visualizations
echo "Combining FCT and RTT statistics and generating visualizations..."
./combine_fct_stats.py "$OUTPUT_DIR/fct_stats" -o "$OUTPUT_DIR/fct_stats"
if [ $? -eq 0 ]; then
  echo "Visualizations generated successfully:"
  echo "  FCT Visualizations:"
  echo "  - Histogram: $OUTPUT_DIR/fct_stats/plots/fct_histogram.png"
  echo "  - Boxplot: $OUTPUT_DIR/fct_stats/plots/fct_boxplot.png"
  echo "  - CDF: $OUTPUT_DIR/fct_stats/plots/fct_cdf.png"
  echo "  - Combined stats: $OUTPUT_DIR/fct_stats/combined_fct_stats.csv"
  echo "  RTT Visualizations:"
  echo "  - Histogram: $OUTPUT_DIR/fct_stats/plots/rtt_histogram.png"
  echo "  - Boxplot: $OUTPUT_DIR/fct_stats/plots/rtt_boxplot.png"
  echo "  - CDF: $OUTPUT_DIR/fct_stats/plots/rtt_cdf.png"
  echo "  - RTT over time: $OUTPUT_DIR/fct_stats/plots/rtt_over_time.png"
  echo "  - Client RTT over time: $OUTPUT_DIR/fct_stats/plots/client_rtt_over_time.png"
  echo "  - Combined stats: $OUTPUT_DIR/fct_stats/combined_rtt_stats.csv"
  echo "  Link Utilization and Throughput Visualizations:"
  echo "  - Per client utilization: $OUTPUT_DIR/plots/[client]_utilization.png"
  echo "  - Per client throughput: $OUTPUT_DIR/plots/[client]_throughput.png"
  echo "  - Per client active throughput: $OUTPUT_DIR/plots/[client]_active_throughput.png"
  echo "  - Average utilization by client: $OUTPUT_DIR/plots/avg_utilization_by_client.png"
  echo "  - Average active utilization by client: $OUTPUT_DIR/plots/avg_active_utilization_by_client.png"
  echo "  - Average throughput by client: $OUTPUT_DIR/plots/avg_throughput_by_client.png"
  echo "  - Throughput comparison (overall vs active): $OUTPUT_DIR/plots/throughput_comparison.png"
  echo "  - Utilization time series: $OUTPUT_DIR/plots/all_clients_utilization_time_series.png"
  echo "  - Throughput time series: $OUTPUT_DIR/plots/all_clients_throughput_time_series.png"
else
  echo "Warning: Could not generate combined visualizations"
fi

echo "Analysis complete!"
echo "To view the results:"
echo "  - Summary statistics: $OUTPUT_DIR/summary.txt"
echo "  - Individual client/server FCT statistics: $OUTPUT_DIR/fct_stats/"
echo "  - Combined FCT statistics: $OUTPUT_DIR/fct_stats/combined_fct_stats.csv"
echo "  - Combined RTT statistics: $OUTPUT_DIR/fct_stats/combined_rtt_stats.csv"
echo "  - FCT and RTT visualizations: $OUTPUT_DIR/fct_stats/plots/"
echo "  - Link utilization: $OUTPUT_DIR/utilization/"
echo "  - Link utilization report: $OUTPUT_DIR/link_utilization_report.txt"
echo "  - Link utilization plots: $OUTPUT_DIR/plots/"

# Add QCT info to final message if available
if [ "$BURSTY_CLIENT_COUNT" -gt 0 ]; then
  echo "  - Combined QCT statistics: $OUTPUT_DIR/qct_stats/combined_qct_stats.csv"
  echo "  - QCT visualizations: $OUTPUT_DIR/qct_stats/plots/"
fi

echo "  - Switch data: $OUTPUT_DIR/switch_*.csv"
echo "  - Application logs: $OUTPUT_DIR/app.log"
echo "  - All data saved in: $OUTPUT_DIR/" 