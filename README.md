# MultPIM: Fast Stateful Multiplication for Processing-in-Memory
## Overview
This is a logic simulator to verify the theoretical results (latency & area) of MultPIM. This logic simulator models a memristive crossbar row with partitions, and then simulates running the MultPIM algorithm on 
the row. Crucially, the simulator models all stateful operations exactly, including initialization cycles.

## Results
We present the various results for MultPIM and other works below. Note that 
the simulator is deterministic. 

Below are results for full-precision 32-bit fixed-point multiplication. Latency is improved by 4.2x over the previous state-of-the-art (RIME [2]).

| Algorithm      | Latency (Cycles) | Area (Memristors) | Partitions | Gates
| ---- | :----: | :----: | :----: | :----: |
| Haj-Ali *et al.* [1] | 12870 | 635 | 1 | NOT/NOR |
| RIME [2] | **2541** | 468 | 31 | NOT/NOR/NAND/Min3 |
| MultPIM (This work) | **611** | 441 | 31 | NOT/Min3 |
| MultPIM-Area (This work) | 899 | 320 | 31 | NOT/Min3 |

The general expressions for N-bit fixed-point multiplication are below. Asymptotic latency is reduced from the state-of-the-art O(N<sup>2</sup>) to O(NlogN).

| Algorithm      | Latency (Cycles) | Area (Memristors) | Partitions | Gates
| ---- | :----: | :----: | :----: | :----: |
| Haj-Ali *et al.* [1] | 13N<sup>2</sup> - 14N + 6 | 20N - 5 | 1 | NOT/NOR |
| RIME [2] | 2N<sup>2</sup> + 16N - 19 | 15N-12 | N - 1 | NOT/NOR/NAND/Min3 |
| MultPIM (This work) | Nlog(N) + 14N + 3  | 14N - 7 | N - 1 | NOT/Min3 |
| MultPIM-Area (This work) | Nlog(N) + 23N + 3 | 10N | N - 1 | NOT/Min3 |

We also include alternative implementations that assume different gate types (see folders). Their results for 32-bit multiplication:

| Gates | Latency (Cycles) | Area (Memristors) |
| :---- | :----: | :----: |
| NOT/Min3 | 611 | 441 |
| NOT/Min3-Area | 899 | 320 |
| NOT/NOR | TODO | TODO |
| NOT/NOR-Area | TODO | TODO |

## User Information
### Dependencies
In order to use the project, you will need:
1. python3
2. tqdm
### User Manual
Running `python mult.py` will run MultPIM on the simulator for a random sample of numbers. The simulator verifies the correctness
of the simulator output and counts the exact number of cycles used. As MultPIM is deterministic, this cycle count is identical for all samples.

Running `python mult-area.py` will run MultPIM-area on the simulator for a random sample of numbers. The simulator verifies the correctness
of the simulator output and counts the exact number of cycles used. As MultPIM-area is deterministic, this cycle count is identical for all samples.

Running `python mv.py` will simulate MultPIM-optimized matrix-vector multiplication. As seen in the paper, matrix-vector multiplication
involves performing an inner product in each row in parallel. Hence, we only model performing an inner product in the simulator
as the expansion to matrix-vector multiplication follows trivially. 

Running `python mv-area.py` will simulate MultPIM-area-optimized matrix-vector multiplication. As seen in the paper, matrix-vector multiplication
involves performing an inner product in each row in parallel. Hence, we only model performing an inner product in the simulator
as the expansion to matrix-vector multiplication follows trivially. 

## Implementation Details
The implementation is divided into the following files: 
1. `simulator.py`. Provides the interface for a memristive crossbar row.
2. `mult.py`. Simulates the MultPIM algorithm for multiplying two fixed-point numbers.
3. `mult-area.py`. Simulates the MultPIM-area algorithm for multiplying two fixed-point numbers.
3. `mv.py`. Simulates the MultPIM-optimized matrix-vector multiplication algorithm.
3. `mv-area.py`. Simulates the MultPIM-area-optimized matrix-vector multiplication algorithm.

### References

[1] A. Haj-Ali, R. Ben-Hur, N. Wald and S. Kvatinsky, "Efficient Algorithms for In-Memory Fixed Point Multiplication Using MAGIC," 2018 IEEE International Symposium on Circuits and Systems (ISCAS), 2018, pp. 1-5, doi: 10.1109/ISCAS.2018.8351561.

[2] Z. Lu, M. T. Arafin and G. Qu, "RIME: A Scalable and Energy-Efficient Processing-In-Memory Architecture for Floating-Point Operations," 2021 26th Asia and South Pacific Design Automation Conference (ASP-DAC), 2021, pp. 120-125.