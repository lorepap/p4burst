import time
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
from abc import ABC, abstractmethod

from topology import DumbbellTopology, LeafSpineTopology

class BeePackets(ABC):
    def __init__(self, topology):
        self.topology = topology

    def send_bee_packets(self):
        if isinstance(self.topology, LeafSpineTopology):
            self.send_bee_packets_leaf_spine()
        elif isinstance(self.topology, DumbbellTopology):
            self.send_bee_packets_dumbbell()

    def send_bee_packets_leaf_spine(self):
        leafs = self.topology.get_leaf_switches()
        for leaf_name in leafs:
            iface = leaf_name + '-eth1'
            for port in range(8):
                # Creiamo l'header usando il metodo astratto
                bee_header = self.create_bee_header(port)
                pkt = (Ether() /
                       IP(src="0.0.0.0", dst="0.0.0.0") / 
                       UDP(dport=9999) /
                       bee_header)
                sendp(pkt, iface=iface, verbose=False)
                time.sleep(0.1)

    @abstractmethod
    def create_bee_header(self, port):
        """Create a BeeHeader with the given port"""
        pass


class SimpleDeflectionBeePackets(BeePackets):
    class BeeHeader(Packet):
        name = "BeeHeader"
        fields_desc = [
            BitField("port_idx_in_reg", 0, 31),
            BitField("queue_occ_info", 0, 1)
        ]
    def create_bee_header(self, port):
        return self.BeeHeader(port_idx_in_reg=port)
    
class QuantilePreemptiveDeflectionBeePackets(BeePackets):
    class BeeHeader(Packet):
        name = "BeeHeader"
        fields_desc = [
            BitField("port_idx_in_reg", 0, 16),  # TODO: ridurre a 16 o meno
            BitField("queue_length", 0, 32),
            BitField("M", 0, 32)
        ]
    def create_bee_header(self, port):
        return self.BeeHeader(port_idx_in_reg=port)

class DistPreemptiveDeflectionBeePackets(BeePackets):
    class BeeHeader(Packet):
        name = "BeeHeader"
        fields_desc = [
            BitField("port_idx_in_reg", 0, 16),
            BitField("queue_length", 0, 32),
            BitField("M", 0, 32),
        ]
    def create_bee_header(self, port):
        return self.BeeHeader(port_idx_in_reg=port)
