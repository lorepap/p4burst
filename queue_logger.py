import subprocess
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Queue logger for P4 switch")
    parser.add_argument("--log", "-l", default="tmp/0000-deflection/queue_log.txt", 
                        help="Path to the output log file (default: tmp/0000-deflection/queue_log.txt)")
    return parser.parse_args()

def extract_values(output, register_name):
    """ Extracts numeric values from a register output, handling errors and unexpected text. """
    if f"{register_name}=" in output:
        try:
            values_str = output.split("=")[1].strip()
            values_list = [int(x) for x in values_str.split(",") if x.strip().isdigit()]
            return values_list
        except ValueError:
            print(f"[!] Warning: Could not parse {register_name} values correctly.")
            return []
    return []

def extract_single_value(output, register_name):
    """ Extracts a single numeric value from a register output, handling errors and unexpected text. """
    if f"{register_name}=" in output:
        try:
            value_str = output.split("=")[1].strip()
            value = value_str.split('\n')[0].strip()
            return int(value)
        except ValueError:
            print(f"[!] Warning: Could not parse {register_name} value correctly.")
            return None
    return None

def main(args):
    log_file = args.log

    # BMv2 CLI commands
    cli_cmd_queue = "echo 'register_read SimpleDeflectionIngress.queue_occupancy_info' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_fw_full = "echo 'register_read SimpleDeflectionIngress.is_fw_port_full_register' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_deq_depth_eg = "echo 'register_read SimpleDeflectionEgress.debug_qdepth' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_eg_port = "echo 'register_read SimpleDeflectionEgress.debug_eg_port' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_normal_counter = "echo 'counter_read SimpleDeflectionIngress.normal_ctr 0' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_deflected_counter = "echo 'counter_read SimpleDeflectionIngress.deflected_ctr 0' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_queue_depths = "echo 'register_read SimpleDeflectionEgress.queue_depth_info' | simple_switch_CLI --thrift-port 9090"
    cli_cmd_flow_pkt_counter = "echo 'counter_read SimpleDeflectionIngress.flow_header_counter 0' | simple_switch_CLI --thrift-port 9090"


    with open(log_file, "w") as f:
        # f.write("Timestamp | Port | All Queues Occupancy (0/1) | Full? | Crt Enq Depth\n")
        f.write("----------------------------------------------------------------------------------------------\n")

    try:
        while True:
            # Run BMv2 CLI commands
            result_queue = subprocess.run(cli_cmd_queue, shell=True, capture_output=True, text=True)
            result_fw_full = subprocess.run(cli_cmd_fw_full, shell=True, capture_output=True, text=True)
            result_deq_depth_eg = subprocess.run(cli_cmd_deq_depth_eg, shell=True, capture_output=True, text=True)
            result_debug_eg_port = subprocess.run(cli_cmd_eg_port, shell=True, capture_output=True, text=True)
            result_counter_normal = subprocess.run(cli_cmd_normal_counter, shell=True, capture_output=True, text=True)
            result_counter_deflected = subprocess.run(cli_cmd_deflected_counter, shell=True, capture_output=True, text=True)
            result_queue_depths = subprocess.run(cli_cmd_queue_depths, shell=True, capture_output=True, text=True)
            result_flow_pkt_counter = subprocess.run(cli_cmd_flow_pkt_counter, shell=True, capture_output=True, text=True)

            output_queue = result_queue.stdout.strip()
            output_fw_full = result_fw_full.stdout.strip()
            output_deq_depth_eg = result_deq_depth_eg.stdout.strip()
            output_debug_eg_port = result_debug_eg_port.stdout.strip()
            output_counter_normal = result_counter_normal.stdout.strip()
            output_counter_deflected = result_counter_deflected.stdout.strip()
            output_queue_depths = result_queue_depths.stdout.strip()
            output_flow_pkt_counter = result_flow_pkt_counter.stdout.strip()

            # Debugging: Print raw output
            # print("Raw Queue Output:", output_queue)
            # print("Raw FW Full Output:", output_fw_full)
            # print("Raw Deq Depth Output:", output_deq_depth_eg)
            # print("Raw Eg Port Output:", output_debug_eg_port)

            with open(log_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                #f.write(f"{output_queue}\n")
                #f.write(f"{output_fw_full}\n")
                #f.write(f"{output_deq_depth_eg}\n")
                #f.write(f"{output_debug_eg_port}\n")
                f.write(f"{output_counter_normal}\n")
                f.write(f"{output_counter_deflected}\n")
                f.write(f"{output_queue_depths}\n")
                f.write(f"{output_flow_pkt_counter}\n")
                f.write("\n")


            # # Extract values from output
            # queue_values = extract_values(output_queue, "SimpleDeflectionIngress.queue_occupancy_info")
            # is_fw_values = extract_single_value(output_fw_full, "SimpleDeflectionIngress.is_fw_port_full_register")
            # deq_depth_values_eg = extract_single_value(output_deq_depth_eg, "SimpleDeflectionEgress.debug_qdepth")
            # eg_port_values = extract_single_value(output_debug_eg_port, "SimpleDeflectionEgress.debug_eg_port")

            # # Log data
            # with open(log_file, "a") as f:
            #     f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Crt Port: {eg_port_values} | Queue Occ.: {queue_values} | is_fw_port_full: {is_fw_values} | Enq Depth: {deq_depth_values_eg}\n")

            # # Iterate through ports (0-7) and log only the full ones
            # for port_idx in range(len(is_fw_values)):
            #     # if is_fw_values[port_idx] == 1:
            #         timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            #         output_capacity = queue_values[port_idx] if port_idx < len(queue_values) else "N/A"
            #         deq_depth_ig = deq_depth_values_ig[port_idx] if port_idx < len(deq_depth_values_ig) else "N/A"
            #         deq_depth_eg = deq_depth_values_eg[port_idx] if port_idx < len(deq_depth_values_eg) else "N/A"

            #         # Log data
            #         with open(log_file, "a") as f:
            #             f.write(f"{timestamp} | Port {port_idx} | Queue: {queue_values} | is_fw_port_full: {is_fw_values[port_idx]} | Ig Deq Depth: {deq_depth_ig} | Eg Deq Depth: {deq_depth_eg} | Capacity: {output_capacity}\n")

            #         # Print for live monitoring
            #         print(f"{timestamp} | Port {port_idx} | Queue: {queue_values} | is_fw_port_full: {is_fw_values[port_idx]} | Ig Deq Depth: {deq_depth_ig} | Eg Deq Depth: {deq_depth_eg} | Capacity: {output_capacity}")

            # Sleep before next reading (adjust interval if needed)
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[*] Stopping queue logging. Log saved in queue_log.txt")



if __name__ == "__main__":
    args = parse_args()
    main(args)