from abc import ABC, abstractmethod
import config
from p4utils.mininetlib.network_API import NetworkAPI
import itertools
from topology import LeafSpineTopology, DumbbellTopology  # Add this line to import topology classes
from collections import defaultdict
import math

class BaseControlPlane(ABC):
    def __init__(self, topology, cmd_path='p4cli'):
        self.topology = topology
        self.net_api = topology.net
        self.path = cmd_path

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

            #queue_commands = [f"set_queue_rate {config.QUEUE_RATE}", f"set_queue_depth {config.QUEUE_SIZE}", f"set_queue_rate 1000 9999"]
            queue_commands = [f"set_queue_rate {config.QUEUE_RATE}", f"set_queue_depth {config.QUEUE_SIZE}"]

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
        C = config.QUEUE_SIZE - 1
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