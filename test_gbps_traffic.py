import os
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.cli import CLI
from mininet.log import setLogLevel
from p4utils.mininetlib.node import P4Host, P4Switch
from p4utils.mininetlib.network_API import NetworkAPI
from p4utils.mininetlib.net import P4Mininet
from p4utils.mininetlib.log import setLogLevel, debug, info, output, warning, error
from control_plane import L3ForwardingControlPlane
from topology import BaseTopology
import subprocess

P4_PATH = 'p4src'

# Define a custom topology with a single switch, client, and server
class SimpleTopo(BaseTopology):
    def __init__(self, *args, **params):
        # Init superclass
        super().__init__(*args, **params)
        self.bw = 1000
        self.delay = 0.0001
        self.p4_program = 'l3_forwarding.p4'

    def generate_topology(self):
        # Create a BMv2 switch
        # switch = self.addSwitch('s1', cls=OVSSwitch) # TODO: replace with bmv2
        self.net.addP4Switch('s1', cli_input=os.path.join('p4cli/s1-commands.txt'), max_queue_size=1000)
        self.net.addP4Switch('s2', cli_input=os.path.join('p4cli/s2-commands.txt'), max_queue_size=1000)
        self.net.setP4SourceAll(os.path.join(P4_PATH, 'l3_forwarding.p4'))
        
        # Create hosts
        client = self.net.addHost('h1')
        server = self.net.addHost('h2')

        # Add links with Gbps bandwidth
        self.net.addLink('h1', 's1', bw=self.bw, delay=self.delay, cls=TCLink)
        self.net.addLink('h2', 's1', bw=self.bw, delay=self.delay, cls=TCLink)
        self.net.addLink('s1', 's2', bw=self.bw, delay=self.delay, cls=TCLink)

    def run_network(self):
        print("Retrieving hosts...")
        # Manually set IPs and MAC addresses
        client = self.net.net.get('h1')
        server = self.net.net.get('h2')

        # Configure server to listen for iperf3 traffic
        print("Starting iperf3 server on h2...")
        s_log_file = "tmp/iperf_server_log.txt"
        server.cmd('iperf3 -s > {} &'.format(s_log_file))  # Run iperf3 server as a background process

        # Specify the log file for the client
        log_file = "tmp/iperf_client_log.txt"
        
        # Generate Gbps traffic from client to server and log the output
        print("Sending Gbps traffic from h1 to h2 and logging to {}".format(log_file))
        print("h1 IP: {}".format(client.IP()))
        print("h2 IP: {}".format(server.IP()))
        client.cmd('iperf3 -c {} -t 30  > {} 2>&1'.format(server.IP(), log_file))

        # Display the iperf3 log content
        with open(log_file, 'r') as f:
            print("\nIperf3 Client Log:")
            print(f.read())

    def setup_control_plane(self):
       cp = L3ForwardingControlPlane(self)
       cp.generate_control_plane()

    def setup_network(self):
        # topo = SimpleTopo()
        # self.auto_assignment()
        # net = P4Mininet(topo=topo, link=HighBWLink)
        self.net.disableCli()
        self.net.disableDebuggerAll()
        # self.cleanup()
        # self.compile()
        # net = Mininet(topo = topo, controller = None, link = TCLink, intf=HighBWIntf, host=P4Host, switch=P4Switch)
        # net.start()
        self.net.l3()
        self.net.startNetwork()

if __name__ == '__main__':
    setLogLevel('info')
    topo = SimpleTopo()
    topo.generate_topology()
    topo.setup_network() # cannot program switches without control plane
    topo.setup_control_plane() # cannot setup control plane without assigning ips, macs etc.
    info('Programming switches...\n')
    topo.net.program_switches()
    output('Switches programmed correctly!\n')
    topo.run_network()
