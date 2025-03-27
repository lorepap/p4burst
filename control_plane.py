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
        """
        Configure the control plane:
        - `s1` runs the Simple Deflection P4 program (`sd.p4`)
        - `s2` runs a normal L3 forwarding (`l3_forwarding.p4`)
        """

        for switch in self.net_api.switches():
            commands = []
            print(f"[INFO] Configuring switch {switch}")

            # Find the other switch (to get the inter-switch link)
            other_switch = None
            for sw in self.net_api.switches():
                if sw != switch:
                    other_switch = sw
                    break

            # Find the inter-switch port
            interswitch_port = None
            for port, nodes in self.net_api.node_ports()[switch].items():
                if other_switch in nodes:
                    interswitch_port = port
                    break

            for host in self.net_api.hosts():
                host_ip = self.get_host_ip(host).split('/')[0]
                host_mac = self.get_host_mac(host)

                # Identify the switch the host is connected to
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

                ## 🎯 **Switch S1: Implements Simple Deflection (`sd.p4`)**
                if switch == "s1":
                    commands.append("table_set_default SimpleDeflectionIngress.forward.get_fw_port_idx_table drop")
                    commands.append("table_set_default SimpleDeflectionIngress.forward.fw_l2_table broadcast")

                    # Initialize queue occupancy registers (assuming 8 ports)
                    for port in range(8):
                        commands.append(f"register_write SimpleDeflectionIngress.queue_occupancy_info {port} 0")

                    if connected_sw == "s1":
                        # Local forwarding (e.g., h1 → h2)
                        print(f"[INFO] (S1) Adding L3 forwarding: {host_ip} -> port {host_port}")
                        commands.append(
                            f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {host_port} {host_port}"
                        )
                    else:
                        # Inter-switch forwarding (s1 -> s2)
                        print(f"[INFO] (S1) Adding deflection rule for {host_ip}/32 -> inter-switch port {interswitch_port}")
                        commands.append(
                            f"table_add SimpleDeflectionIngress.forward.get_fw_port_idx_table get_fw_port_idx_action {host_ip}/32 => {interswitch_port} {interswitch_port}"
                        )
                        commands.append(
                            f"register_write SimpleDeflectionIngress.queue_occupancy_info {interswitch_port} 1"
                        )

                ## 🎯 **Switch S2: Implements Standard L3 Forwarding (`l3_forwarding.p4`)**
                elif switch == "s2":
                    commands.append("table_set_default MyIngress.ipv4_lpm drop")

                    if connected_sw == "s2":
                        # Local forwarding (e.g., h3 → h4)
                        print(f"[INFO] (S2) Adding L3 forwarding: {host_ip} -> port {host_port}")
                        commands.append(
                            f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {host_port}"
                        )
                    else:
                        # Inter-switch forwarding (s2 -> s1)
                        print(f"[INFO] (S2) Adding inter-switch forwarding: {host_ip}/32 -> port {interswitch_port}")
                        commands.append(
                            f"table_add MyIngress.ipv4_lpm ipv4_forward {host_ip}/32 => {host_mac} {interswitch_port}"
                        )

            # Save commands for debugging
            self.save_commands(switch[1:], commands)