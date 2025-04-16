from scapy.all import Ether, IP, UDP, Packet, BitField, sendp, bind_layers

class BeeHeaderRL(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 28),
        BitField("queue_occ_info", 0, 1),
        BitField("queue_depth", 0, 19) 
    ]

class BeeHeaderS(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 31),
        BitField("queue_occ_info", 0, 1)
    ]

class BeeHeaderQPD(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 16),  # TODO: ridurre a 16 o meno
        BitField("queue_length", 0, 32),
        BitField("M", 0, 32)
    ]

class BeeHeaderDPD(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 16),
        BitField("queue_length", 0, 32),
        BitField("M", 0, 32),
    ]

    
def send_bee_packets_rl(switch: str):
    switch_name = switch
    iface = switch_name + '-eth1'
    for port in range(32):
        pkt = (Ether() /  # Use broadcast MAC
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeaderRL(port_idx_in_reg=port, ))
        sendp(pkt, iface=iface, verbose=False)

        
def send_bee_packets_s(switch: str):
    switch_name = switch
    iface = switch_name + '-eth1'
    for port in range(32):
        pkt = (Ether() /  # Use broadcast MAC
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeaderS(port_idx_in_reg=port, ))
        sendp(pkt, iface=iface, verbose=False)


def send_bee_packets_qpd(switch: str):
    switch_name = switch
    iface = switch_name + '-eth1'
    for port in range(32):
        pkt = (Ether() /  # Use broadcast MAC
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeaderQPD(port_idx_in_reg=port, ))
        sendp(pkt, iface=iface, verbose=False)

def send_bee_packets_dpd(switch: str):
    switch_name = switch
    iface = switch_name + '-eth1'
    for port in range(32):
        pkt = (Ether() /  # Use broadcast MAC
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeaderDPD(port_idx_in_reg=port, ))
        sendp(pkt, iface=iface, verbose=False)