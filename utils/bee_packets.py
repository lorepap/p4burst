from scapy.all import Ether, IP, UDP, Packet, BitField, sendp, bind_layers

class BeeHeader(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 28),
        BitField("queue_occ_info", 0, 1),
        BitField("queue_depth", 0, 19) 
    ]
    
def send_bee_packets(switch: str):
    switch_name = switch
    iface = switch_name + '-eth1'
    for port in range(8):
        pkt = (Ether() /  # Use broadcast MAC
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeader(port_idx_in_reg=port, ))
        sendp(pkt, iface=iface, verbose=False)

