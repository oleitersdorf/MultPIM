
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint


def MultPIMArea(crossbar: Crossbar, N: int):
    """
    Performs the MultPIMArea algorithm on the given crossbar
    :param crossbar: The crossbar to perform the algorithm on. Assumes partition 0 contains only the two inputs,
    that there are N partitions (indices 1 to N) with unknown init status, and a partition at the end for the output.
    :param N: The number of bits in each number. Assumes w.l.o.g. N is a power of two: this is not a necessary
    assumption, but it does simplify the implementation of the function here.
    """

    # Legend
    ABIT = 0
    ABBIT = 1
    SBIT = 2  # stores S
    CBIT = 3  # stores C
    TEMP1 = 4
    TEMP2 = 5
    TEMP3 = 6

    # Init entries
    # --- 2 OPs --- #
    crossbar.perform(
        Operation([Gate(GateType.INIT1, [],
                  [(partition, ABIT), (partition, ABBIT), (partition, TEMP1), (partition, TEMP2), (partition, TEMP3)])
                for partition in range(1, N+1)] + [Gate(GateType.INIT1, [], [(N + 1, i) for i in range(2*N)])]))
    crossbar.perform(
        Operation([Gate(GateType.INIT0, [],
                  [(partition, SBIT), (partition, CBIT)])
                for partition in range(1, N+1)]))

    # Store a's bits in the partitions
    # --- N OPs --- #
    for i in range(N):
        crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, i)], [(N-i, ABIT)])]))

    # Iterate over all 2*N stages
    for k in range(2 * N):

        is_notted = [False] * (N + 1)

        if k < N:

            is_notted[1:] = [bool(bin(p).count('1') % 2 == 1) for p in range(N)]

            # Copy b_k to all partitions using log_2(N) ops
            # --- log_2(N) OPs --- #
            log2_N = N.bit_length() - 1
            for i in range(log2_N):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(j, TEMP3 if is_notted[j] else ABBIT) if j != 1 else (0, N+k)],
                         [(j + (N >> (i+1)), TEMP3 if is_notted[j + (N >> (i+1))] else ABBIT)])
                    for j in range(1, N+1, 1 << (log2_N - i))
                ]))

            # Compute partial products
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAJ3, [(j, ABIT), (j, TEMP3), (j, TEMP2)], [(j, TEMP1)]) if is_notted[j] else
                Gate(GateType.MAGIC_NOT, [(j, ABIT)], [(j, ABBIT)])
                for j in range(2, N + 1)
            ] + [Gate(GateType.MAGIC_NOT, [(0, N + k)], [(1, ABBIT)])]))

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, TEMP1)], [(j, ABBIT)])
                for j in range(2, N + 1) if is_notted[j]
            ]))

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, TEMP1), (j, TEMP3)])
                for j in range(2, N + 1)
            ]))

        # TEMP2 <- not(Cin)
        # TEMP1 <- not(TEMP2) = Cin
        # Init(CBIT)
        # CBIT  <- Maj(ABBIT, SBIT, TEMP1) = Maj(A, B, Cin)

        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, CBIT)], [(j, TEMP2)])
            for j in range(2, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, TEMP2)], [(j, TEMP1)])
            for j in range(2, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, CBIT)])
            for j in range(2, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAJ3, [(j, ABBIT), (j, SBIT), (j, TEMP1)], [(j, CBIT)])
            for j in range(2, N + 1)
        ]))

        # TEMP3 <- Maj(ABBIT, SBIT, TEMP2) = Maj(A, B, not Cin)
        # Init(TEMP2)
        # TEMP2 <- not(CBIT) = Min(A, B, Cin)

        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAJ3, [(j, ABBIT), (j, SBIT), (j, TEMP2)], [(j, TEMP3)])
            for j in range(2, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, TEMP2)])
            for j in range(2, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, CBIT)], [(j, TEMP2)])
            for j in range(2, N + 1)
        ]))

        # Init(SBIT)
        # SBIT <- Maj(TEMP3, TEMP1, TEMP2) = Maj(Maj(A, B, not Cin), Cin, Min(A, B, C))

        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, SBIT)])
            for j in range(2, N + 1)
        ] + ([Gate(GateType.MAJ3, [(1, ABIT), (1, ABBIT), (1, TEMP1)], [(1, TEMP2)])] if k < N else [])))

        # Compute S across adjacent partitions
        # --- 2 OPs --- #
        crossbar.perform(Operation([
            Gate(GateType.MAJ3, [(j, TEMP3), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
            for j in range(3, N + 1, 2)
        ] + [Gate(GateType.MAGIC_NOT, [(1, TEMP2)], [(2, SBIT)])
            if k < N else Gate(GateType.INIT0, [], [(2, SBIT)])]))
        crossbar.perform(Operation([
            Gate(GateType.MAJ3, [(j, TEMP3), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
            for j in range(2, N, 2)
        ] + [Gate(GateType.MAJ3, [(N, TEMP3), (N, TEMP1), (N, TEMP2)], [(N + 1, k)])]))

        # Init the temps for next time
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], ([(j, ABBIT)] if k < N else []) + [(j, TEMP1), (j, TEMP2), (j, TEMP3)])
            for j in range(1, N + 1)
        ]))

        if k == N-1:
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT0, [], [(j, ABBIT)])
                for j in range(1, N + 1)
            ]))


# Parameters
N = 32

# Crossbar
partitionSize = 7
crossbar = Crossbar([2 * N] + [partitionSize] * N + [2 * N])

num_samples = 10
for sample in tqdm(range(num_samples)):

    # Sample a and b
    a = randint(0, (1 << N) - 1)
    b = randint(0, (1 << N) - 1)

    # Crossbar initialization
    crossbar.op_counter = 0
    crossbar.partitions[0] = [bool((a & (1 << i)) >> i) for i in range(N)] + [bool((b & (1 << i)) >> i) for i in range(N)]

    # Perform multiplication
    MultPIMArea(crossbar, N)

    # Verify results
    assert(sum([int(crossbar.partitions[N+1][i]) << i for i in range(2*N)]) == a * b)

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per multiplication')

