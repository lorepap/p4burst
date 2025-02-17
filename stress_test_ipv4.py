#!/usr/bin/env python3
# Copyright 2013-present Barefoot Networks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import socket
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.log import setLogLevel, info
# from p4utils.mininetlib.node import P4Host, P4SwitchBmv2
# from p4_mininet import P4Switch, P4Host
from test_p4_mininet import P4Switch, P4Host

import argparse
from time import sleep
import subprocess
import os
import re
import sys

parser = argparse.ArgumentParser(description='Mininet demo')
parser.add_argument('--thrift-port',
                    help='Thrift server port for table updates',
                    type=int, action="store", default=9090)
parser.add_argument('--repeat', '-r',
                    help='Number of times to run test',
                    type=int, action="store", default=5)
args = parser.parse_args()

class SingleSwitchTopo(Topo):
    "Single switch connected to 2 hosts."
    def __init__(self, sw_path, json_path, thrift_port, **opts):
        # Initialize topology and default options
        Topo.__init__(self, **opts)

        switch = self.addSwitch('s1',
                                sw_path = sw_path,
                                json_path = json_path,
                                thrift_port = thrift_port,
                                pcap_dump = False,
                                device_id=0)

        for h in range(2):
            host = self.addHost('h%d' % (h + 1),
                                ip = "10.0.%d.10/24" % h,
                                mac = '00:04:00:00:00:%02x' %h)
            self.addLink(host, switch)

def start_iperf(net, client_name, server_name):
    h2 = net.getNodeByName(server_name)
    # print("Starting iperf server...")
    server = h2.popen("iperf -s")
    h1 = net.getNodeByName(client_name)
    # -f m: report in Mbits
    return h1.popen("iperf -f m -c %s -t 30" % (h2.IP()), stdout=subprocess.PIPE)

# is there a need to make this function more general?
def configure_dp(commands_path, thrift_port):
    cmd = ["/home/ubuntu/p4-tools/bmv2/tools/runtime_CLI.py",
           "--thrift-port", str(thrift_port)]
    with open(commands_path, "r") as f:
        print(" ".join(cmd))
        sub_env = os.environ.copy()
        pythonpath = ""
        if "PYTHONPATH" in sub_env:
            pythonpath = sub_env["PYTHONPATH"] + ":"
        sub_env["PYTHONPATH"] = pythonpath + \
                                "/home/ubuntu/p4-tools/bmv2/thrift_src/gen-py/"
        subprocess.Popen(cmd, stdin = f, env = sub_env).wait()

def run_measurement(net, client_name, server_name):
    iperf_proc = start_iperf(net, client_name, server_name)
    out, _ = iperf_proc.communicate()
    res = re.findall(r"(\d+) Mbits/sec", out.decode('utf-8'))
    return res[-1]

def check_deps():
    try:
        _ = subprocess.check_output(["/sbin/ethtool", "--version"],
                                    stderr=subprocess.STDOUT)
    except:
        print("ethtool not available")
        print("On Debian systems you can install it with:")
        print("sudo apt install ethtool")
        sys.exit(1)

def main():
    check_deps()

    thrift_port = args.thrift_port
    num_hosts = 2

    sw_path = "/home/ubuntu/p4-tools/bmv2/targets/simple_switch/simple_switch"
    json_path = "/home/ubuntu/p4-tools/bmv2/mininet/simple_router.json"
    topo = SingleSwitchTopo(sw_path, json_path, thrift_port)
    net = Mininet(topo = topo, host = P4Host, switch = P4Switch,
                  controller = None)
    net.start()

    sw_mac = ["00:aa:bb:00:00:%02x" % n for n in range(num_hosts)]

    sw_addr = ["10.0.%d.1" % n for n in range(num_hosts)]

    for n in range(num_hosts):
        h = net.get('h%d' % (n + 1))
        h.setARP(sw_addr[n], sw_mac[n])
        h.setDefaultRoute("dev eth0 via %s" % sw_addr[n])

    for n in range(num_hosts):
        h = net.get('h%d' % (n + 1))
        h.describe()

    sleep(1)

    configure_dp("/home/ubuntu/p4-tools/bmv2/mininet/stress_test_commands.txt", thrift_port)

    sleep(1)

    print("Ready !")

    net.pingAll()

    throughputs = []
    for i in range(args.repeat):
        sleep(1)
        print("Running iperf measurement {} of {}".format(i + 1, args.repeat))
        t = run_measurement(net, "h1", "h2")
        print(t, "Mbps")
        throughputs.append(t)
    throughputs.sort()
    print("Median throughput is", throughputs[int(args.repeat / 2)], "Mbps")

    net.stop()
    # just in case...
    subprocess.Popen("pgrep -f iperf | xargs kill -9", shell=True).wait()

if __name__ == '__main__':
    setLogLevel( 'info' )
    main()
