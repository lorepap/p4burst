
def update_p4_queue_size(p4_file, queue_capacity, threshold):

    print(f"Updating P4 queue size to {queue_capacity*threshold} for {p4_file}")
    with open(p4_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith('const'):
            # Extract const name
            const_name = line.split('=')[0].split()[-1]
            if const_name == 'QUEUE_CAPACITY':
                base = line.split('=')[0]
                lines[i] = f"{base}= {int(queue_capacity*threshold) - 1};    // Value overriden from config.py\n"
    with open(p4_file, 'w') as f:
            f.writelines(lines)
