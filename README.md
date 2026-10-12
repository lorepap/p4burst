# p4burst
A repository to experiment with the use of P4 language to build burst-tolerant in-network algorithms for data center networks.

```
                    +----+      +----+
                 +--| S1 |------| S2 |--+
                 |  +----+      +----+  |
                 |                      |
             +---+--+              +----+--+
        +----| Leaf1|              | Leaf2 |----+
        |    +---+--+              +----+--+    |
        |        |                      |       |
    +---+---+    |                      |    +--+----+
    | Host1 |    |                      |    | Host2 |
    +-------+    |                      |    +-------+
              +--+----+              +--+----+
              | Host3 |              | Host4 |
              +-------+              +-------+


```

## Installation

Before running the code, you need to install all the necessary dependencies. Use the provided `install.sh` file to install the following components:

* PI
* Behavioral Model (BMv2)
* P4C
* Mininet
* FRRouting
* P4-Utils

To install, run:

```bash
chmod +x install.sh
sudo ./install.sh
```

Note: P4-Learning depends on these software components. Please refer to their respective documentation for any troubleshooting during installation.
This repository is tested on the Ubuntu 20.04 Linux distribution.

## Running the Simulation

To run an example simulation, use the following command:

```bash
sudo python runner.py -t leafspine -l 2 -s 2 -n 4 --cli
```

This command creates a leaf-spine topology with 2 leaf switches, 2 spine switches, and 4 hosts, and launches a client-server application to generate incast traffic using ECMP algorithm.

Choose the forwarding scheme with `-c`: `ecmp` (default), `simple_deflection`, `simple_deflection_fl`, `quantile_preemptive_deflection` or `dist_preemptive_deflection`. For example:

```bash
sudo python runner.py -t leafspine -l 2 -s 2 -n 4 -c simple_deflection --cli
```

## Incast Application

The bursty application in p4burst is implemented by reproducing the concepts presented in [1]. The application simulates a scenario where a client performs queries to a set of servers that will reply simultaneously, creating an incast traffic pattern.

1. Each client initiates a query to N servers simultaneously.
2. Upon receiving a query, each server responds immediately with a 40 KB flow (MTU=1500B).
3. This simultaneous response from multiple servers to a single client creates an incast traffic pattern in the DCN.

This implementation allows us to study and optimize network performance under bursty, incast-prone workloads.

## Deflection Algorithms

When the output queue of a switch is full, packet deflection sends a packet through another port instead of dropping it. The repository implements four variants in P4, each in its own folder under `p4src/`:

- `Simple_Deflection`: deflects to a random port when the output queue is full.
- `Simple_Deflection_FL`: picks the alternative port pseudo-randomly per flow, so that packets of the same flow tend to follow the same path and arrive in order.
- `Quantile_PD` and `Dist_PD`: two forms of preemptive deflection [1], which deflect some packets before the queue is full, based on its occupancy.

BMv2 does not expose egress queue lengths at ingress, where the forwarding decision is taken. The switches therefore exchange small "bee" packets that copy the queue state from egress to ingress registers (`bee_packets_generator.py`).

Further experiments (asynchronous traffic generators, timing externs, data collection for a learned deflection policy) are in the `evaluation` and `data-collection` branches.

[1] S. Abdous, E. Sharafzadeh, S. Ghorbani, [Practical Packet Deflection in Datacenters](https://dl.acm.org/doi/pdf/10.1145/3629147), CoNEXT 2023.

## Project Structure

### Topology

The file `topology.py` is responsible for creating the Mininet topology using the P4Mininet APIs. It defines the network structure, including switches, hosts, and their connections.

You can add more topologies to this file to experiment with different network configurations.

### Control Plane

The file `control_plane.py` generates the `.txt` files used by P4-utils to program the switches. It defines the control plane logic for different network topologies.

You can modify this file to implement different control plane behaviors or add support for new topologies.

## Extending the Project

To add a new topology:
1. Define the topology in `topology.py`
2. Implement the corresponding control plane logic in `control_plane.py`
3. Add a new p4 source program (by default, all sources are located in `p4src`)
4. Update `runner.py` to support the new topology option
