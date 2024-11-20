from abc import ABC, abstractmethod
from p4utils.mininetlib.network_API import NetworkAPI
import os

P4_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'p4src')


class BaseTopology(ABC):
    def __init__(self, num_hosts, bw, latency):
        self.num_hosts = num_hosts
        self.bw = bw
        self.latency = latency
        self.net = NetworkAPI()
        self.path = 'p4cli' # path to output dir for the p4cli commands
        self.p4_program = None
        # Create log, src and p4cli directories
        os.makedirs('p4src', exist_ok=True)
        os.makedirs('p4cli', exist_ok=True)

        # Network general options
        self.net.setLogLevel('info')
        self.net.disableCli()

    @abstractmethod
    def generate_topology(self):
        pass

    def create_switch_commands(self, switch_num):
        # Create txt files for each switch
        for i in range(1, switch_num + 1):
            with open(f'{self.path}/s{i}-commands.txt', 'w') as f:
                f.write('')        

    def start_network(self):
        # Nodes general options
        self.net.enablePcapDumpAll()
        self.net.enableLogAll()

        # Assignment strategy
        self.net.mixed()

        # Start the network
        self.net.startNetwork()


class LeafSpineTopology(BaseTopology):
    def __init__(self, num_hosts, num_leaf, num_spine, bw, latency):
        super().__init__(num_hosts, bw, latency)
        self.num_leaf = num_leaf
        self.num_spine = num_spine
        self.p4_program = 'ecmp.p4'
        self.create_switch_commands(num_leaf + num_spine)

    def generate_topology(self):
        hosts_per_leaf = self.num_hosts // self.num_leaf

        # Generate switches
        for i in range(1, self.num_leaf + self.num_spine + 1):
            self.net.addP4Switch(f's{i}', cli_input=os.path.join(self.path, f's{i}-commands.txt'))
        self.net.setP4SourceAll(os.path.join(P4_PATH, self.p4_program))

        # Generate hosts
        for i in range(1, self.num_hosts + 1):
            self.net.addHost(f'h{i}')

        # Connect hosts to leaf switches
        for i in range(1, self.num_hosts + 1):
            # leaf_num = ((i - 1) // hosts_per_leaf) + 1
            leaf_id = ((i - 1) % self.num_leaf) + 1
            print(f'Connecting h{i} to s{leaf_id}')
            self.net.addLink(f'h{i}', f's{leaf_id}', bw=self.bw, delay=f'{self.latency}ms')

        # Connect leaf switches to spine switches
        for leaf in range(1, self.num_leaf + 1):
            for spine in range(self.num_leaf + 1, self.num_leaf + self.num_spine + 1):
                self.net.addLink(f's{leaf}', f's{spine}', bw=self.bw, delay=f'{self.latency}ms')


class DumbbellTopology(BaseTopology):
    def __init__(self, num_hosts, bw, latency):
        super().__init__(num_hosts, bw, latency)
        self.p4_program = 'l3_forwarding.p4' # TODO: parametrize this
        self.create_switch_commands(2)
        if self.num_hosts < 2:
            raise ValueError("DumbbellTopology requires at least 2 hosts")

    def generate_topology(self):
        # Generate switches
        self.net.addP4Switch('s1', cli_input=os.path.join(self.path, 's1-commands.txt'))
        self.net.addP4Switch('s2', cli_input=os.path.join(self.path, 's2-commands.txt'))
        self.net.setP4SourceAll(os.path.join(P4_PATH, self.p4_program))

        # Generate hosts
        hosts_per_switch = self.num_hosts // 2
        for i in range(1, self.num_hosts + 1):
            self.net.addHost(f'h{i}')

        # Connect hosts to switches
        for i in range(1, self.num_hosts + 1):
            switch_num = 1 if i <= hosts_per_switch else 2
            host_port = i if switch_num == 1 else i - hosts_per_switch
            switch_port = i if switch_num == 1 else i - hosts_per_switch
            self.net.addLink(f'h{i}', f's{switch_num}', 
                             bw=self.bw, delay=f'{self.latency}ms')

        # Connect switches
        self.net.addLink('s1', 's2', 
                         bw=self.bw, delay=f'{self.latency}ms')
