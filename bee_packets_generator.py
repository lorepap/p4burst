import time
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
from abc import abstractmethod

from topology import DumbbellTopology, LeafSpineTopology

class BeePackets:

    def __init__(self, topology):
        self.topology = topology

    @abstractmethod
    def send_bee_packets(self):
        pass

class SimpleDeflectionBeePackets(BeePackets):

    class BeeHeader(Packet):
        name = "BeeHeader"
        fields_desc = [
            BitField("port_idx_in_reg", 0, 31),
            BitField("queue_occ_info", 0, 1)
        ]
    
    def send_bee_packets(self):
        if isinstance(self.topology, LeafSpineTopology):
            self.send_bee_packets_leaf_spine()
        elif isinstance(self.topology, DumbbellTopology):
            self.send_bee_packets_fat_tree()
    
    def send_bee_packets_leaf_spine(self):
        leafs = self.topology.get_leaf_switches()
        for leaf_name in leafs:
            iface = leaf_name + '-eth1'
            for port in range(8):
                pkt = (Ether() / 
                      IP(src="0.0.0.0", dst="0.0.0.0") /  # Minimal IP header
                      UDP(dport=9999) /
                      SimpleDeflectionBeePackets.BeeHeader(port_idx_in_reg=port))
                sendp(pkt, iface=iface, verbose=False)
                time.sleep(0.1) 

    def send_bee_packets_dumbbell(self):
        raise NotImplementedError("Dumbbell topology not implemented yet")
    