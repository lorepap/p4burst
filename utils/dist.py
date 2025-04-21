import math
def compute_interval_and_midpoint(index): # diverso da practical deflection (guarda sopra), ma in questo modo gli intervalli partono da 0
    start = (2 << index) - 2
    end = (2 << (index + 1)) - 3
    return start, end, (start + end) / 2.0
    
def compute_new_m(mid_m, mid_rank):
    return math.floor((49 * mid_m + mid_rank) / 50)
    
def compute_rel_prio(mid_rank, mid_m, C, alpha):
    return math.floor(C * alpha * (1 - math.exp(- (mid_rank / mid_m))))