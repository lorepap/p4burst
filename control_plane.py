'''
from abc import ABC, abstractmethod
from p4utils.mininetlib.network_API import NetworkAPI
import os 
import ipaddress
from topology import LeafSpineTopology, DumbbellTopology  # Add this line to import topology classes
import utils
import utils.bee_packets

class BaseControlPlane(ABC):
    def __init__(self, topology, cmd_path='p4cli'):
        self.topology = topology
        self.net_api = topology.net
        self.path = cmd_path

    @abstractmethod
    def generate_control_plane(self):
        pass

    def get_host_ip(self, host):
        return self.net_api.getNode(host).get('ip')

    def get_host_mac(self, host):
        return self.net_api.getNode(host).get('mac')
    
    def get_switch_mac(self, sw1, sw2):
        """
        Returns the MAC address of the interface on sw2.
        """
        for node1, node2, info in self.net_api.links(withInfo=True):
            if sw1 in [node1, node2] and sw2 in [node1, node2]:
                return info['addr1'] if sw2 == node1 else info['addr2']

    def get_node_interfaces(self, node):
        return self.net_api.node_intfs()[node]

    def save_commands(self, switch, commands):
        with open(f'{self.path}/s{switch}-commands.txt', 'w') as f:
            f.write('\n'.join(commands))

class ECMPControlPlane(BaseControlPlane):
    def __init__(self, topology, leaf_switches, spine_switches):
        super().__init__(topology)
        self.num_leaf = leaf_switches
        self.num_spine = spine_switches

    def generate_control_plane(self):
        if not isinstance(self.topology, LeafSpineTopology):
            raise ValueError("ECMPControlPlane can only be used with LeafSpineTopology")
        for switch in self.net_api.switches():
            commands = []
            switch_num = int(switch[1:])
            is_spine = switch_num > self.num_leaf

            if is_spine:
                self._generate_spine_commands(switch, commands)
            else:
                self._generate_leaf_commands(switch, commands)

            self.save_commands(switch_num, commands)

    def _generate_spine_commands(self, switch, commands):
        commands.append("table_set_default ipv4_lpm drop")
        for leaf in range(1, self.num_leaf + 1):
            leaf_switch = f's{leaf}'
            for port, nodes in self.net_api.node_ports()[switch].items():
                if leaf_switch in nodes:
                    leaf_mac = self.get_switch_mac(switch, leaf_switch)
                    for host in self.net_api.hosts():
                        if self.is_host_connected_to_leaf(host, leaf_switch):
                            host_ip = f'10.0.{leaf}.{host[1:]}/32'
                            commands.append(f"table_add ipv4_lpm ipv4_forward {host_ip} => {leaf_mac} {port}")
    
    def _generate_leaf_commands(self, switch, commands):
        commands.append("table_set_default ipv4_lpm drop")
        commands.append("table_set_default ecmp_group drop")
        commands.append("table_set_default ecmp_nhop drop")

        # Handle local hosts
        for host in self.net_api.hosts():
            if self.is_host_connected_to_leaf(host, switch):
                host_ip = f'10.0.{switch[1:]}.{host[1:]}/32'
                for port, nodes in self.net_api.node_ports()[switch].items():
                    if host in nodes:
                        host_mac = self.get_host_mac(host)
                        commands.append(f"table_add ipv4_lpm set_nhop {host_ip} => {host_mac} {port}")

        # Handle remote hosts (ECMP to spine switches)
        commands.append(f"table_add ecmp_group set_ecmp_select 0.0.0.0/0 => 1 {self.num_spine}")
        for i, spine in enumerate(range(self.num_leaf + 1, self.num_leaf + self.num_spine + 1)):
            spine_switch = f's{spine}'
            for port, nodes in self.net_api.node_ports()[switch].items():
                if spine_switch in nodes:
                    spine_mac = self.get_switch_mac(switch, spine_switch)
                    commands.append(f"table_add ecmp_nhop set_nhop 1 {i} => {spine_mac} {port}")

    def is_host_connected_to_leaf(self, host, leaf_switch):
        for port, nodes in self.net_api.node_ports()[leaf_switch].items():
            if host in nodes:
                return True
        return False

class L3ForwardingControlPlane(BaseControlPlane):
    """
    L3 forwarding control plane for dumbbell topology (2 switches)
    @TODO - add support for leaf-spine topology
    """
    def __init__(self, topology, cmd_path='p4cli'):
        super().__init__(topology, cmd_path)

    def generate_control_plane(self):
        if isinstance(self.topology, DumbbellTopology):
            for switch in self.net_api.switches():
                commands = []
                commands.append("table_set_default MyIngress.ipv4_lpm drop")
                host_entries = {}  # Track individual host entries for each switch

                for host in self.net_api.hosts():
                    is_remote = True
                    host_ip = self.get_host_ip(host).split('/')[0]
                    other_switch = 's2' if switch == 's1' else 's1'

                    # Find local hosts and add entry for each specific host IP
                    for port, nodes in self.net_api.node_ports()[switch].items():
                        if host in nodes:
                            is_remote = False
                            # Only add a new entry if this host IP hasn't been added yet
                            if host_ip not in host_entries:
                                host_mac = self.get_host_mac(host)
                                commands.append(
                                    f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {port}"
                                )
                                host_entries[host_ip] = (host_mac, port)
                            break  # Stop after adding entry for this host IP

                    # If the host is remote, add forwarding rule for the other switch
                    if is_remote:
                        for port, nodes in self.net_api.node_ports()[switch].items():
                            if other_switch in nodes:
                                # Only add remote subnet entry if it hasn't been added yet
                                subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                                if subnet not in host_entries:
                                    commands.append(
                                        f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => 00:00:00:00:00:00 {port}"
                                    )
                                    host_entries[subnet] = ("00:00:00:00:00:00", port)
                                break  # Stop after adding one entry per subnet
                
                self.save_commands(switch[1:], commands)   

        elif isinstance(self.topology, LeafSpineTopology):
            # Iterate over all switches in the network
            for switch in self.net_api.switches():
                commands = []
                commands.append("table_set_default MyIngress.ipv4_lpm drop")
                host_entries = {}  # to avoid duplicate entries on the same switch

                # Determine if this switch is a leaf switch.
                # (Assume leaf if any port has a node whose name starts with 'h')
                is_leaf = False
                for port, nodes in self.net_api.node_ports()[switch].items():
                    if any(node.startswith("h") for node in nodes[:2]):
                        is_leaf = True
                        break

                if is_leaf:
                    # === Leaf Switch Logic ===
                    for host in self.net_api.hosts():
                        host_ip = self.get_host_ip(host).split('/')[0]

                        # Check if host is directly attached (local) to this leaf switch.
                        local = False
                        for port, nodes in self.net_api.node_ports()[switch].items():
                            if host in nodes:
                                local = True
                                if host_ip not in host_entries:
                                    host_mac = self.get_host_mac(host)
                                    commands.append(
                                        f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {port}"
                                    )
                                    host_entries[host_ip] = (host_mac, port)
                                break  # Entry added for this host, move to next host

                        if not local:
                            # Remote host: forward traffic via one of this leaf's spine ports.
                            # Find a port that connects to a spine switch (assuming spine names start with 's').
                            for port, nodes in self.net_api.node_ports()[switch].items():
                                if any(node.startswith("s") for node in nodes[:2]):
                                    # Use an aggregated subnet entry for remote hosts.
                                    subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                                    if subnet not in host_entries:
                                        # Using a placeholder MAC (e.g. 00:00:00:00:00:00) to indicate a next-hop lookup.
                                        commands.append(
                                            f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => 00:00:00:00:00:00 {port}"
                                        )
                                        host_entries[subnet] = ("00:00:00:00:00:00", port)
                                    break  # One remote rule per subnet is sufficient

                else:
                    # === Spine Switch Logic ===
                    # Spine switches do not have hosts attached directly.
                    # They must forward traffic to the appropriate leaf, so install one entry per leaf subnet.
                    # First, find all leaf switches in the network.
                    leaf_switches = [
                        sw for sw in self.net_api.switches()
                        if any(
                            node.startswith("h")
                            for port in self.net_api.node_ports()[sw]
                            for node in self.net_api.node_ports()[sw][port][:2]
                        )
                    ]

                    for leaf in leaf_switches:
                        # Pick one host attached to the leaf to infer its subnet.
                        leaf_host = None
                        for port, nodes in self.net_api.node_ports()[leaf].items():
                            for node in nodes[:2]:
                                if node.startswith("h"):
                                    leaf_host = node
                                    break
                            if leaf_host:
                                break

                        if leaf_host:
                            leaf_host_ip = self.get_host_ip(leaf_host).split('/')[0]
                            subnet = '.'.join(leaf_host_ip.split('.')[:3]) + ".0/24"
                            # On the spine switch, find a port that connects to this leaf.
                            for port, nodes in self.net_api.node_ports()[switch].items():
                                if leaf in nodes[:2]:
                                    if subnet not in host_entries:
                                        commands.append(
                                            f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => 00:00:00:00:00:00 {port}"
                                        )
                                        host_entries[subnet] = ("00:00:00:00:00:00", port)
                                    break
                
                self.save_commands(switch[1:], commands)   
        else:
            raise ValueError("Unsupported topology type")


class SimpleDeflectionControlPlane(BaseControlPlane):
    def __init__(self, topology, cmd_path='p4cli', queue_rate=100, queue_depth=100):
        super().__init__(topology, cmd_path)
        self.queue_rate = queue_rate
        self.queue_depth = queue_depth

    @staticmethod
    def send_bee_packets(switch):
        utils.bee_packets.send_bee_packets(switch)

    def generate_control_plane(self):

        if isinstance(self.topology, LeafSpineTopology):
            host_connections = self.topology.get_host_connections()
            port_mappings = self.topology.get_port_mappings()

            switch_commands = dict()
            for switch in self.net_api.switches():
                switch_commands[switch] = set()

            leaf_switches = self.topology.get_leaf_switches()
            spine_switches = self.topology.get_spine_switches()

            # Process each host and add forwarding rules
            for host in self.net_api.hosts():
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                connected_sw, port = host_connections[host]
                subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                switch_mac = "00:00:00:00:00:00" # TODO: uncorrect to use this mac, however p4 program does not use it
                host_mac = self.topology.get_host_mac(host)

                # Add direct connection rule to leaf switch
                logical_port = port_mappings[connected_sw][port]
                switch_commands[connected_sw].add(
                    f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {port} {logical_port} {host_mac}"
                )

                # Add subnet rules to spine switches
                for spine in spine_switches:
                    port = self.topology.get_connecting_port(spine, connected_sw)
                    switch_commands[spine].add(
                        f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => {switch_mac} {port}"
                    )

                # Add routing to spine for other leaf switches
                for leaf in (l for l in leaf_switches if l != connected_sw):
                    ports = self.topology.get_spine_ports(leaf)
                    selected_port_idx = hash(subnet) % len(ports) # same subnet always goes to the same spine, but different subnet can go to different spine
                    port = ports[selected_port_idx]
                    logical_port = port_mappings[leaf][port]
                    switch_commands[leaf].add(
                        f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table get_fw_port_idx_action {subnet} => {port} {logical_port} {switch_mac}"
                    )

            spine_defaults = [
                "table_set_default MyIngress.ipv4_lpm drop",
            ]

            leaf_defaults = [
                "table_set_default SimpleDeflectionIngress.forward.get_fw_port_idx_table SimpleDeflectionIngress.forward.drop",
                "table_set_default SimpleDeflectionIngress.forward.fw_l2_table broadcast",
                #"table_set_default SimpleDeflectionIngress.set_deflect_egress_port_table drop"
            ]

            # Save all commands
            for i, leaf in enumerate(leaf_switches):

                spine_logical_ports = {port_mappings[leaf][port] for port in self.topology.get_spine_ports(leaf)}

                register_commands = [
                    f"register_write SimpleDeflectionIngress.neighbor_switch_indicator {logical_port} 1" 
                    for logical_port in range(8) if logical_port not in spine_logical_ports
                ]
                    
                deflection_table_commands = [
                    f"table_add SimpleDeflectionIngress.set_deflect_egress_port_table set_deflect_egress_port_action {logical_port} => {physical_port}" 
                    for physical_port, logical_port in port_mappings[leaf].items()
                ]
                    
                port_index_commands = [
                    f"table_add SimpleDeflectionEgress.get_eg_port_idx_in_reg_table get_eg_port_idx_in_reg_action {physical_port} => {logical_port}"
                    for physical_port, logical_port in port_mappings[leaf].items()
                ]

                queue_commands = [f"set_queue_rate {self.queue_rate}", f"set_queue_depth {self.queue_depth}"]

                # Combine all commands
                commands = (
                    leaf_defaults + 
                    list(switch_commands[leaf]) +
                    register_commands +
                    deflection_table_commands +
                    port_index_commands +
                    queue_commands
                )
                
                self.save_commands(leaf[1:], commands)
                
            for spine in spine_switches:
                commands = spine_defaults + list(switch_commands[spine])
                self.save_commands(spine[1:], commands)

        elif isinstance(self.topology, DumbbellTopology):
            """
            Configure the control plane for a dumbbell topology:
            - Switch "s1" runs the Simple Deflection program (sd.p4)
            - Switch "s2" runs standard L3 forwarding (l3_forwarding.p4)
            
            Assumptions:
            * There are exactly two switches: "s1" and "s2".
            * The inter-switch link uses physical port 1 on both switches.
            * Hosts can be connected to any port except port 1.
            """

            # Define the two switches.
            s1 = "s1"  # deflection switch
            s2 = "s2"  # L3 forwarding switch

            s1_inter = self.topology.get_connecting_port(s1, s2)
            s2_inter = self.topology.get_connecting_port(s2, s1)

            # Obtain port mappings and host connections from the topology.
            # port_mappings { switch: { physical_port: logical_port, ... }, ... }
            port_mappings = self.topology.get_port_mappings()
            # host_connections { host: (switch, physical_port), ... }
            host_connections = self.topology.get_host_connections()

            # Initialize a dictionary to hold all control plane commands for each switch.
            switch_commands = { s1: set(), s2: set() }

            # Process each host.
            for host in self.net_api.hosts():
                # Get host IP (removing any prefix) and MAC.
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                host_mac = self.topology.get_host_mac(host)
                # Determine which switch the host is connected to and on which physical port.
                connected_sw, host_port = host_connections[host]

                if connected_sw == s1:
                    # --- For a host connected to s1 ---
                    # (a) Install a local forwarding rule on s1.
                    logical_port = port_mappings[s1][host_port]
                    switch_commands[s1].add(
                        f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table "
                        f"get_fw_port_idx_action {host_ip}/32 => {host_port} {logical_port} {host_mac}"
                    )
                    # (b) On s2, add an inter-switch rule so that packets destined to this host are forwarded via s2's inter-switch port.
                    switch_commands[s2].add(
                        f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {s2_inter}"
                    )
                elif connected_sw == s2:
                    # --- For a host connected to s2 ---
                    # (a) Install a local L3 forwarding rule on s2.
                    switch_commands[s2].add(
                        f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {host_port}"
                    )
                    # (b) On s1, install an inter-switch rule so that packets destined to this host use the inter-switch link.
                    # A dummy MAC is used here since the sd.p4 program does not use it.
                    dummy_mac = "00:00:00:00:00:00"
                    logical_inter = port_mappings[s1][s1_inter]
                    switch_commands[s1].add(
                        f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table "
                        f"get_fw_port_idx_action {host_ip}/32 => {s1_inter} {logical_inter} {dummy_mac}"
                    )

            # Set default actions.
            # For s1 (Simple Deflection switch), we set a default for the L2 table.
            s1_defaults = [
                # Uncomment the next line if you need a default drop for the get_fw_port_idx_table:
                "table_set_default SimpleDeflectionIngress.forward.get_fw_port_idx_table SimpleDeflectionIngress.forward.drop",
                "table_set_default SimpleDeflectionIngress.forward.fw_l2_table broadcast",
                "table_set_default SimpleDeflectionIngress.set_deflect_egress_port_table SimpleDeflectionIngress.drop"
            ]
            # For s2 (L3 forwarding switch), drop if no rule matches.
            s2_defaults = [
                "table_set_default MyIngress.ipv4_lpm drop",
            ]

            # Additional configuration for s1 (the deflection switch):
            # 1. Configure the neighbor_switch_indicator register.
            #    For s1, only the inter-switch port should be available for deflection.
            s1_register_commands = [
                f"register_write SimpleDeflectionIngress.neighbor_switch_indicator {logical_port} 1"
                for physical_port, logical_port in port_mappings[s1].items()
                if physical_port != s1_inter
            ]
            # 2. Populate the deflection mapping table.
            s1_deflection_table_commands = [
                f"table_add SimpleDeflectionIngress.set_deflect_egress_port_table set_deflect_egress_port_action {logical_port} => {physical_port}"
                for physical_port, logical_port in port_mappings[s1].items()
            ]
            # 3. Populate the port-index table used in the egress pipeline.
            s1_port_index_commands = [
                f"table_add SimpleDeflectionEgress.get_eg_port_idx_in_reg_table get_eg_port_idx_in_reg_action {physical_port} => {logical_port}"
                for physical_port, logical_port in port_mappings[s1].items()
            ]
            # 4. Set queue configuration commands.
            s1_queue_commands = [ f"set_queue_rate {self.queue_rate}", f"set_queue_depth {self.queue_depth}" ]
            #s1_queue_commands = [""]

            # Combine all commands for s1.
            s1_commands = (
                s1_defaults +
                list(switch_commands[s1]) +
                s1_register_commands +
                s1_deflection_table_commands +
                s1_port_index_commands +
                s1_queue_commands
            )

            # Combine all commands for s2.
            s2_commands = s2_defaults + list(switch_commands[s2])

            # Save the commands for each switch.
            self.save_commands(s1[1:], s1_commands)
            self.save_commands(s2[1:], s2_commands)
'''

from abc import ABC, abstractmethod
from p4utils.mininetlib.network_API import NetworkAPI
import itertools
from topology import LeafSpineTopology, DumbbellTopology  # Add this line to import topology classes
from collections import defaultdict
import math
import utils.bee_packets

class BaseControlPlane(ABC):
    def __init__(self, topology, cmd_path='p4cli', queue_rate=100, queue_depth=100):
        self.topology = topology
        self.net_api = topology.net
        self.path = cmd_path
        self.queue_rate = queue_rate
        self.queue_depth = queue_depth

    @abstractmethod
    def generate_control_plane(self):
        pass

    def save_commands(self, switch, commands, mode = 'w'):
        with open(f'{self.path}/s{switch}-commands.txt', mode) as f:
            if mode == 'a':
                f.write('\n')
            f.write('\n'.join(commands))

class ECMPControlPlane(BaseControlPlane):
    def __init__(self, topology, leaf_switches, spine_switches):
        super().__init__(topology)
        self.num_leaf = leaf_switches
        self.num_spine = spine_switches

    def generate_control_plane(self):
        if not isinstance(self.topology, LeafSpineTopology):
            raise ValueError("ECMPControlPlane can only be used with LeafSpineTopology")
        for switch in self.net_api.switches():
            commands = []
            switch_num = int(switch[1:])
            is_spine = switch_num > self.num_leaf

            if is_spine:
                self._generate_spine_commands(switch, commands)
            else:
                self._generate_leaf_commands(switch, commands)

            self.save_commands(switch_num, commands)

    def _generate_spine_commands(self, switch, commands):
        commands.append("table_set_default ipv4_lpm drop")
        for leaf in range(1, self.num_leaf + 1):
            leaf_switch = f's{leaf}'
            for port, nodes in self.net_api.node_ports()[switch].items():
                if leaf_switch in nodes:
                    leaf_mac = self.topology.get_switch_mac(switch, leaf_switch)
                    for host in self.net_api.hosts():
                        if self.is_host_connected_to_leaf(host, leaf_switch):
                            host_ip = f'10.0.{leaf}.{host[1:]}/32'
                            commands.append(f"table_add ipv4_lpm ipv4_forward {host_ip} => {leaf_mac} {port}")
    
    def _generate_leaf_commands(self, switch, commands):
        commands.append("table_set_default ipv4_lpm drop")
        commands.append("table_set_default ecmp_group drop")
        commands.append("table_set_default ecmp_nhop drop")

        # Handle local hosts
        for host in self.net_api.hosts():
            if self.is_host_connected_to_leaf(host, switch):
                host_ip = f'10.0.{switch[1:]}.{host[1:]}/32'
                for port, nodes in self.net_api.node_ports()[switch].items():
                    if host in nodes:
                        host_mac = self.topology.get_host_mac(host)
                        commands.append(f"table_add ipv4_lpm set_nhop {host_ip} => {host_mac} {port}")

        # Handle remote hosts (ECMP to spine switches)
        commands.append(f"table_add ecmp_group set_ecmp_select 0.0.0.0/0 => 1 {self.num_spine}")
        for i, spine in enumerate(range(self.num_leaf + 1, self.num_leaf + self.num_spine + 1)):
            spine_switch = f's{spine}'
            for port, nodes in self.net_api.node_ports()[switch].items():
                if spine_switch in nodes:
                    spine_mac = self.topology.get_switch_mac(switch, spine_switch)
                    commands.append(f"table_add ecmp_nhop set_nhop 1 {i} => {spine_mac} {port}")

    def is_host_connected_to_leaf(self, host, leaf_switch):
        for port, nodes in self.net_api.node_ports()[leaf_switch].items():
            if host in nodes:
                return True
        return False

class L3ForwardingControlPlane(BaseControlPlane):
    def generate_control_plane(self):
        for switch in self.net_api.switches():
            commands = []
            commands.append("table_set_default MyIngress.ipv4_lpm drop")
            host_entries = {}  # Track individual host entries for each switch

            for host in self.net_api.hosts():
                is_remote = True
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                other_switch = 's2' if switch == 's1' else 's1'

                # Find local hosts and add entry for each specific host IP
                for port, nodes in self.net_api.node_ports()[switch].items():
                    if host in nodes:
                        is_remote = False
                        # Only add a new entry if this host IP hasn't been added yet
                        if host_ip not in host_entries:
                            host_mac = self.topology.get_host_mac(host)
                            commands.append(
                                f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {port}"
                            )
                            host_entries[host_ip] = (host_mac, port)
                        break  # Stop after adding entry for this host IP

                # If the host is remote, add forwarding rule for the other switch
                if is_remote:
                    for port, nodes in self.net_api.node_ports()[switch].items():
                        if other_switch in nodes:
                            # Only add remote subnet entry if it hasn't been added yet
                            subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                            if subnet not in host_entries:
                                commands.append(
                                    f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => 00:00:00:00:00:00 {port}"
                                )
                                host_entries[subnet] = ("00:00:00:00:00:00", port)
                            break  # Stop after adding one entry per subnet

            # Save generated commands for each switch
            self.save_commands(switch[1:], commands)

class BaseDeflectionControlPlane(BaseControlPlane):
    def generate_control_plane(self):
        if isinstance(self.topology, LeafSpineTopology):
            self.generate_leaf_spine_control_plane()
        # TODO - we need to refactor the topology - control plane interaction
        elif isinstance(self.topology, DumbbellTopology):
            self.generate_dumbbell_control_plane()
    
    @abstractmethod
    def generate_leaf_spine_control_plane(self):
        self.host_connections = self.topology.get_host_connections()
        self.port_mappings = self.topology.get_port_mappings()

        switch_commands = defaultdict(set)

        self.leaf_switches = self.topology.get_leaf_switches()
        self.spine_switches = self.topology.get_spine_switches()

        # Process each host and add forwarding rules
        for host in self.net_api.hosts():

            host_ip = self.topology.get_host_ip(host).split('/')[0]
            connected_sw, port = self.host_connections[host]
            subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
            switch_mac = "00:00:00:00:00:00" # TODO: uncorrect to use this mac, however p4 program does not use it
            host_mac = self.topology.get_host_mac(host)

            # Add direct connection rule to leaf switch
            logical_port = self.port_mappings[connected_sw][port]
            switch_commands[connected_sw].add(
                f"table_add SwitchIngress.routing.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {port} {logical_port} {host_mac}"
            )

                # Add subnet rules to spine switches
            for spine in self.spine_switches:
                port = self.topology.get_connecting_port(spine, connected_sw)
                switch_commands[spine].add(
                    f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => {switch_mac} {port}"
                )

            # Add routing to spine for other leaf switches
            for leaf in (l for l in self.leaf_switches if l != connected_sw):
                ports = self.topology.get_spine_ports(leaf)
                selected_port_idx = hash(subnet) % len(ports) # same subnet always goes to the same spine, but different subnet can go to different spine
                port = ports[selected_port_idx]
                logical_port = self.port_mappings[leaf][port]
                switch_commands[leaf].add(
                    f"table_add SwitchIngress.routing.get_fw_port_idx_table get_fw_port_idx_action {subnet} => {port} {logical_port} {switch_mac}"
                )

        spine_defaults = [
            "table_set_default MyIngress.ipv4_lpm drop",
        ]

        leaf_defaults = [
            #"table_set_default SwitchIngress.routing.get_fw_port_idx_table drop",
            "table_set_default SwitchIngress.routing.fw_l2_table broadcast",
            #"table_set_default SwitchIngress.set_deflect_eggress_port_table drop"
        ]

        # Save all commands
        for leaf in self.leaf_switches:
                
            port_index_commands = [
                f"table_add SwitchEgress.get_eg_port_idx_in_reg_table get_eg_port_idx_in_reg_action {physical_port} => {logical_port}"
                for physical_port, logical_port in self.port_mappings[leaf].items()
            ]

            #queue_commands = [f"set_queue_rate {self.queue_rate}", f"set_queue_depth {self.queue_depth}", f"set_queue_rate 1000 9999"]
            queue_commands = [f"set_queue_rate {self.queue_rate}", f"set_queue_depth {self.queue_depth}"]

            # Combine all commands
            commands = (
                leaf_defaults + 
                list(switch_commands[leaf]) +
                port_index_commands +
                queue_commands
            )
            
            self.save_commands(leaf[1:], commands)
            
        for spine in self.spine_switches:
            commands = spine_defaults + list(switch_commands[spine])
            self.save_commands(spine[1:], commands)
    

    @abstractmethod
    def generate_dumbbell_control_plane(self):
        pass
            

class SimpleDeflectionControlPlane(BaseDeflectionControlPlane):


    @staticmethod
    def send_bee_packets(switch):
        utils.bee_packets.send_bee_packets_s(switch)

    def generate_leaf_spine_control_plane(self):
        super().generate_leaf_spine_control_plane()
        for leaf in self.leaf_switches:

            spine_logical_ports = {self.port_mappings[leaf][port] for port in self.topology.get_spine_ports(leaf)}

            register_commands = [
                f"register_write SwitchIngress.neighbor_switch_indicator {logical_port} 1" 
                for logical_port in range(8) if logical_port not in spine_logical_ports
            ]
                
            deflection_table_commands = [
                f"table_add SwitchIngress.set_deflect_eggress_port_table set_deflect_eggress_port_action {logical_port} => {physical_port}" 
                for physical_port, logical_port in self.port_mappings[leaf].items()
            ]

            # Combine all commands
            commands = (
                register_commands +
                deflection_table_commands
            )
            
            self.save_commands(leaf[1:], commands, mode='a')
    

    def generate_dumbbell_control_plane(self):
        for switch in self.net_api.switches():
            commands = []
                
            # Default actions
            commands.append("table_set_default MyIngress.get_fw_port_idx_table drop")
            commands.append("table_set_default MyIngress.fw_l2_table broadcast")
                
                # Track added entries
            host_entries = {}
                
                # Get the other switch in dumbbell
            other_switch = None
            for sw in self.net_api.switches():
                if sw != switch: 
                    other_switch = sw
                    break
                
            # Find inter-switch port
            interswitch_port = None
            for port, nodes in self.net_api.node_ports()[switch].items():
                if other_switch in nodes:
                    interswitch_port = port
                    break
                
                # Process hosts
            for host in self.net_api.hosts():
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                host_mac = self.topology.get_host_mac(host)
                    
                    # Find the switch connected to the host
                connected_sw = None
                host_port = None
                for sw in self.net_api.switches():
                    for port, nodes in self.net_api.node_ports()[sw].items():
                        if host in nodes:
                            connected_sw = sw
                            host_port = port
                            break
                    if connected_sw:
                        break
                    
                is_remote = connected_sw != switch
                    
                if not is_remote:
                    # Directly connected host
                    port = host_port  # Use the port we found earlier instead of node_to_node_port
                    commands.append(
                        f"table_add MyIngress.get_fw_port_idx_table get_fw_port_idx_action {host_ip} => {port} {port}"
                    )
                    commands.append(
                        f"table_add MyIngress.fw_l2_table fw_l2_action {host_mac} => {port}"
                    )
                else:
                    # Remote host - route through other switch
                    subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                    if subnet not in host_entries:
                        commands.append(
                            f"table_add MyIngress.get_fw_port_idx_table get_fw_port_idx_action {subnet} => {interswitch_port} {interswitch_port}"
                        )
                        host_entries[subnet] = interswitch_port
                
            # Save commands for this switch (fix indentation - move inside the switch loop)
            self.save_commands(switch[1:], commands)

class BasePreemptiveDeflectionControlPlane(BaseDeflectionControlPlane):
    def generate_leaf_spine_control_plane(self):
        super().generate_leaf_spine_control_plane()
        hosts_pairs = itertools.combinations(self.net_api.hosts(), 2)
        switch_commands = defaultdict(set)
        for host1, host2 in hosts_pairs:
            h1_ip = self.topology.get_host_ip_no_mask(host1)
            h2_ip = self.topology.get_host_ip_no_mask(host2)
            rank = self.calculate_rank(h1_ip, h2_ip)
            h1_connected_sw, _ = self.host_connections[host1]
            h2_connected_sw, _ = self.host_connections[host2]
            rank_commands = [
                f"table_add SwitchIngress.get_flow_priority_table get_flow_priority_action {h1_ip} {h2_ip} => {rank}",
                f"table_add SwitchIngress.get_flow_priority_table get_flow_priority_action {h2_ip} {h1_ip} => {rank}"
            ]
            switch_commands[h1_connected_sw].update(rank_commands)
            switch_commands[h2_connected_sw].update(rank_commands)

        for host in self.net_api.hosts():

            host_ip = self.topology.get_host_ip_no_mask(host)
            connected_sw, _ = self.host_connections[host]
            subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"

            # Add routing to spine for other leaf switches
            for leaf in (l for l in self.leaf_switches if l != connected_sw):
                ports = self.topology.get_spine_ports(leaf)
                deflection_port_idx = (hash(subnet) + 1) % len(ports) # same subnet always goes to the same spine, but different subnet can go to different spine
                deflection_port = ports[deflection_port_idx]
                logical_deflection_port = self.port_mappings[leaf][deflection_port]
                switch_commands[leaf].add(
                    f"table_add SwitchIngress.deflection_routing.deflect_get_fw_port_idx_table deflect_get_fw_port_idx_action {subnet} => {deflection_port} {logical_deflection_port}"
                )
        
        for leaf in self.leaf_switches:
            commands = list(switch_commands[leaf])
            self.save_commands(leaf[1:], commands, mode='a')
        



    @abstractmethod
    def calculate_rank(ip_address_1, ip_address_2):
        pass


class QuantilePreemptiveDeflectionControlPlane(BasePreemptiveDeflectionControlPlane):
    def calculate_rank(self, ip_address_1, ip_address_2): # TODO: we have to think about priorities of packets
        # Sort IPs to ensure consistent ordering
        ips = sorted([ip_address_1, ip_address_2])
    
        # Create a hash from the two IPs
        combined = f"{ips[0]},{ips[1]}".encode()
        hash_value = hash(combined) & 0xFFFFFFFF  # Get positive 32-bit value
    
        # Map to 1 or 2 with 75%-25% distribution
        #return 1 if hash_value % 4 < 3 else 2
        return 1 if hash_value % 2==0 else 2
    
    def generate_leaf_spine_control_plane(self):
        super().generate_leaf_spine_control_plane()

    def generate_dumbbell_control_plane(self):
        raise NotImplementedError("QuantilePreemptiveDeflectionControlPlane is not implemented for DumbbellTopology")
    
class DistPreemptiveDeflectionControlPlane(BasePreemptiveDeflectionControlPlane):
    def calculate_rank(self, ip_address_1, ip_address_2): # TODO: we have to think about priorities of packets
        # Sort IPs to ensure consistent ordering
        ips = sorted([ip_address_1, ip_address_2])
    
        # Create a hash from the two IPs
        combined = f"{ips[0]},{ips[1]}".encode()
        hash_value = hash(combined) & 0xFFFFFFFF  # Get positive 32-bit value
    
        # Map to 45 or 2 with 80%-20% distribution
        return 45 if hash_value % 5 < 4 else 2
    
    def generate_leaf_spine_control_plane(self):
        super().generate_leaf_spine_control_plane()
        commands = []
        C = self.queue_depth - 1
        alpha = config.ALPHA
        m_prio_num_entries = config.M_PRIO_NUM_ENTRIES
        m_prio_rank_entries = config.M_PRIO_RANK_ENTRIES
        m_newm_num_entries = config.M_NEWM_NUM_ENTRIES
        m_newm_rank_entries = config.M_NEWM_RANK_ENTRIES

        for i in range(m_prio_num_entries):
            m_start, m_end, mid_m = DistPreemptiveDeflectionControlPlane.compute_interval_and_midpoint(i)
            for j in range(m_prio_rank_entries):
                rank_start, rank_end, mid_rank = DistPreemptiveDeflectionControlPlane.compute_interval_and_midpoint(j)
                rel_prio = math.floor(C * alpha * (1 - math.exp(- (mid_rank / mid_m))))
                commands.append(
                    f"table_add SwitchIngress.get_rel_prio_table get_rel_prio_action {rank_start}->{rank_end} {m_start}->{m_end} => {rel_prio} 1"
                )
                commands.append(
                    f"table_add SwitchIngress.get_deflect_rel_prio_table get_deflect_rel_prio_action {rank_start}->{rank_end} {m_start}->{m_end} => {rel_prio} 1"
                )

        for i in range(m_newm_num_entries):
            m_start, m_end, mid_m = DistPreemptiveDeflectionControlPlane.compute_interval_and_midpoint(i)
            for j in range(m_newm_rank_entries):
                rank_start, rank_end, mid_rank = DistPreemptiveDeflectionControlPlane.compute_interval_and_midpoint(j)
                new_m = self.compute_new_m(mid_m, mid_rank)
                commands.append(
                    f"table_add SwitchEgress.get_newm_table get_newm_action {rank_start}->{rank_end} {m_start}->{m_end} => {new_m} 1"
                )
        
        for leaf in self.leaf_switches:
            self.save_commands(leaf[1:], commands, mode='a')
    
    def generate_dumbbell_control_plane(self):
        raise NotImplementedError("DistPreemptiveDeflectionControlPlane is not implemented for DumbbellTopology")

    '''
    @staticmethod
    def _compute_interval_and_midpoint(index):
        start = (2 << index) + 1
        end = (2 << (index + 1))
        return start, end, (start + end) / 2.0
        '''
    @staticmethod
    def compute_interval_and_midpoint(index): # diverso da practical deflection (guarda sopra), ma in questo modo gli intervalli partono da 0
        start = (2 << index) - 2
        end = (2 << (index + 1)) - 3
        return start, end, (start + end) / 2.0
    
    @staticmethod
    def compute_new_m(mid_m, mid_rank):
        return math.floor((49 * mid_m + mid_rank) / 50)
    
    @staticmethod
    def compute_rel_prio(mid_rank, mid_m, C, alpha):
        return math.floor(C * alpha * (1 - math.exp(- (mid_rank / mid_m))))
    

        

class TestControlPlane(BaseControlPlane):
    """
    Works only for dumbbell topology
    For testing purposes
    updates:
    - 
    """
    def __init__(self, topology, deflection_switch='s1'):
        super().__init__(topology)
        self.deflection_switch = deflection_switch

    def generate_control_plane(self):
        if not isinstance(self.topology, DumbbellTopology):
            raise ValueError("TestControlPlane can only be used with DumbbellTopology")

        for switch in self.net_api.switches():
            commands = []
            if switch == self.deflection_switch:
                # Use deflection routing on deflection_switch
                commands.append("table_set_default MyIngress.get_fw_port_idx_table drop")
                commands.append("table_set_default MyIngress.fw_l2_table broadcast")
            else:
                # Use simple L3 forwarding on other switch
                commands.append("table_set_default MyIngress.ipv4_lpm drop")

            # Get the other switch and inter-switch port
            other_switch = [sw for sw in self.net_api.switches() if sw != switch][0]
            interswitch_port = None
            for port, nodes in self.net_api.node_ports()[switch].items():
                if other_switch in nodes:
                    interswitch_port = port
                    break

            # Process all hosts
            for host in self.net_api.hosts():
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                host_mac = self.topology.get_host_mac(host)
                is_local = False
                local_port = None

                # Find if host is local to this switch
                for port, nodes in self.net_api.node_ports()[switch].items():
                    if host in nodes:
                        is_local = True
                        local_port = port
                        break

                if is_local:
                    if switch == self.deflection_switch:
                        # Local host on deflection switch - use deflection tables
                        commands.append(
                            f"table_add MyIngress.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {local_port} {local_port}"
                        )
                        commands.append(
                            f"table_add MyIngress.fw_l2_table fw_l2_action {host_mac} => {local_port}"
                        )
                    else:
                        # Local host on L3 switch - use L3 forwarding
                        commands.append(
                            f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {local_port}"
                        )
                else:
                    # Remote hosts
                    subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                    if switch == self.deflection_switch:
                        # Remote on deflection switch - use deflection with interswitch port
                        commands.append(
                            f"table_add MyIngress.get_fw_port_idx_table get_fw_port_idx_action {subnet} => {interswitch_port} {interswitch_port}"
                        )
                    else:
                        # Remote on L3 switch - use simple forwarding through interswitch port
                        commands.append(
                            f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => 00:00:00:00:00:00 {interswitch_port}"
                        )

            self.save_commands(switch[1:], commands)

class RLDeflectionControlPlane(BaseControlPlane):
    def __init__(self, topology, cmd_path='p4cli', queue_rate=100, queue_depth=100):
        super().__init__(topology, cmd_path)
        self.queue_rate = queue_rate
        self.queue_depth = queue_depth

    @staticmethod
    def send_bee_packets(switch):
        utils.bee_packets.send_bee_packets_rl(switch)

    def generate_control_plane(self):

        if isinstance(self.topology, LeafSpineTopology):
            host_connections = self.topology.get_host_connections()
            port_mappings = self.topology.get_port_mappings()

            switch_commands = dict()
            for switch in self.net_api.switches():
                switch_commands[switch] = set()

            leaf_switches = self.topology.get_leaf_switches()
            spine_switches = self.topology.get_spine_switches()

            # Process each host and add forwarding rules
            for host in self.net_api.hosts():
                host_ip = self.topology.get_host_ip(host).split('/')[0]
                connected_sw, port = host_connections[host]
                subnet = '.'.join(host_ip.split('.')[:3]) + ".0/24"
                switch_mac = "00:00:00:00:00:00" # TODO: uncorrect to use this mac, however p4 program does not use it
                host_mac = self.topology.get_host_mac(host)

                # Add direct connection rule to leaf switch
                logical_port = port_mappings[connected_sw][port]
                switch_commands[connected_sw].add(
                    f"table_add SwitchIngress.routing.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {port} {host_mac}"
                )

                # Add subnet rules to spine switches
                for spine in spine_switches:
                    port = self.topology.get_connecting_port(spine, connected_sw)
                    switch_commands[spine].add(
                        f"table_add MyIngress.ipv4_lpm ipv4_forward {subnet} => {switch_mac} {port}"
                    )

                # Add routing to spine for other leaf switches
                for leaf in (l for l in leaf_switches if l != connected_sw):
                    ports = self.topology.get_spine_ports(leaf)
                    selected_port_idx = hash(subnet) % len(ports) # same subnet always goes to the same spine, but different subnet can go to different spine
                    port = ports[selected_port_idx]
                    logical_port = port_mappings[leaf][port]
                    switch_commands[leaf].add(
                        f"table_add SwitchIngress.routing.get_fw_port_idx_table get_fw_port_idx_action {subnet} => {port} {switch_mac}"
                    )

            spine_defaults = [
                "table_set_default MyIngress.ipv4_lpm drop",
            ]

            leaf_defaults = [
                "table_set_default SwitchIngress.routing.get_fw_port_idx_table drop",
                "table_set_default SwitchIngress.routing.fw_l2_table broadcast",
            ]

            tree_commands = [
                    f"table_set_default SwitchIngress.BDT_table_lev0_cond0 decision_meta_switch_id_action 1",
                    f"table_set_default SwitchIngress.BDT_table_lev1_cond0 forward",
                    f"table_set_default SwitchIngress.BDT_table_lev1_cond1 decision_meta_queue_lenght_7_action 0",
                    f"table_set_default SwitchIngress.BDT_table_lev2_cond2 decision_meta_queue_lenght_6_action 0",
                    f"table_set_default SwitchIngress.BDT_table_lev2_cond3 forward",
                    f"table_set_default SwitchIngress.BDT_table_lev3_cond4 decision_meta_queue_lenght_4_action 0",
                    f"table_set_default SwitchIngress.BDT_table_lev3_cond5 forward",
                    f"table_set_default SwitchIngress.BDT_table_lev4_cond8 deflect",
                    f"table_set_default SwitchIngress.BDT_table_lev4_cond9 forward",
                    #f"table_set_default SwitchIngress.BDT_table_lev4_cond8 decision_meta_queue_lenght_5_action => 0",
                    #f"table_set_default SwitchIngress.BDT_table_lev4_cond9 forward",
                    #f"table_set_default SwitchIngress.BDT_table_lev5_cond16 deflect",
                    #f"table_set_default SwitchIngress.BDT_table_lev5_cond17 forward"
            ]

            for i, leaf in enumerate(leaf_switches):

                switch_id_command = [f"table_set_default SwitchIngress.set_switch_id_table set_switch_id {i}"]

                spine_logical_ports = {port_mappings[leaf][port] for port in self.topology.get_spine_ports(leaf)}

                register_commands = [
                    f"register_write SwitchIngress.neighbor_switch_indicator {logical_port} 1" 
                    for logical_port in range(8) if logical_port not in spine_logical_ports
                ]
                    
                deflection_table_commands = [
                    f"table_add SwitchIngress.set_physical_deflect_port_from_id_table set_physical_deflect_port_from_id {logical_port} => {physical_port}" 
                    for physical_port, logical_port in port_mappings[leaf].items()
                ]
                    
                port_index_commands = [
                    f"table_add SwitchEgress.get_eg_port_id_table get_eg_port_id_action {physical_port} => {logical_port}"
                    for physical_port, logical_port in port_mappings[leaf].items()
                ]

                queue_commands = [f"set_queue_rate {self.queue_rate}", f"set_queue_depth {self.queue_depth}"]

                commands = (
                    switch_id_command +
                    leaf_defaults + 
                    list(switch_commands[leaf]) +
                    register_commands +
                    deflection_table_commands +
                    port_index_commands +
                    queue_commands +
                    tree_commands
                )
                
                self.save_commands(leaf[1:], commands)
                
            for spine in spine_switches:
                commands = spine_defaults + list(switch_commands[spine])
                self.save_commands(spine[1:], commands)

        else:
            raise ValueError("Unsupported topology type")


