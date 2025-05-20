import math
from utils.dist import compute_interval_and_midpoint, compute_rel_prio
def update_p4_consts(p4_file, queue_size, logaritmic_deflecting_margin, alpha, m_prio_num_entries, m_prio_rank_entries):
    with open(p4_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith('const'):
            # Extract const name
            const_name = line.split('=')[0].split()[-1]
            if const_name == 'QUEUE_SIZE' or const_name == 'QUEUE_CAPACITY':
                base = line.split('=')[0]
                lines[i] = f"{base}= {queue_size - 1};    // Value overriden\n"
                
            if const_name == 'COUNT_ALL_SHIFT':
                if logaritmic_deflecting_margin is None:
                    raise ValueError("logaritmic_deflecting_margin cannot be None when COUNT_ALL_SHIFT is set.")
                base = line.split('=')[0]
                deflection_margin = logaritmic_deflecting_margin if logaritmic_deflecting_margin > 0 else 1
                shift = math.floor(math.log2(queue_size - 1)) - deflection_margin
                if shift < 0:
                    shift = 0
                lines[i] = f"{base} = {shift};    // Value overriden\n"

            if const_name == 'PRIO_SHIFT':
                if logaritmic_deflecting_margin is None:
                    raise ValueError("logaritmic_deflecting_margin cannot be None when PRIO_SHIFT is set.")
                if m_prio_num_entries is None:
                    raise ValueError("m_prio_num_entries cannot be None when PRIO_SHIFT is set.")
                if m_prio_rank_entries is None:
                    raise ValueError("m_prio_rank_entries cannot be None when PRIO_SHIFT is set.")
                if alpha is None:
                    raise ValueError("alpha cannot be None when PRIO_SHIFT is set.")
                base = line.split('=')[0]
                deflection_margin = logaritmic_deflecting_margin if logaritmic_deflecting_margin > 0 else 1
                C = queue_size - 1
                max_m_index = m_prio_num_entries - 1
                m_rank_index = m_prio_rank_entries -1
                base = line.split('=')[0]
                _, _, mid_rank = compute_interval_and_midpoint(m_rank_index)
                _, _, mid_m = compute_interval_and_midpoint(max_m_index)
                rel_prio = compute_rel_prio(mid_rank, mid_m, C, alpha)
                shift = math.floor(math.log2(C / rel_prio)) - deflection_margin
                if shift < 0:
                    shift = 0
                lines[i] = f"{base} = {shift};    // Value overriden\n"        
    with open(p4_file, 'w') as f:
        f.writelines(lines)