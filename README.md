# MultPIM: Fast Stateful Multiplication for Processing-in-Memory
## Overview
This is a logic simulator to verify the theoretical results (latency & area) of MultPIM. This logic simulator models a memristive crossbar row with partitions, and then simulates running the MultPIM algorithm on 
the row. Crucially, the simulator models all stateful operations exactly, including initialization cycles.

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
