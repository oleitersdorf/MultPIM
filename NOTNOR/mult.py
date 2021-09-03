
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint


def MultPIM(crossbar: Crossbar, N: int):
    """
    Performs the MultPIM algorithm on the given crossbar
    :param crossbar: The crossbar to perform the algorithm on. Assumes partition 0 contains only the two inputs,
    that there are N partitions (indices 1 to N) with unknown init status, and a partition at the end for the output.
    :param N: The number of bits in each number. Assumes w.l.o.g. N is a power of two: this is not a necessary
    assumption, but it does simplify the implementation of the function here.
    """

    # Legend
    TEMP1 = 0
    ABIT = 1
    BBIT = 2
    ABBIT = 3  # only in partitions that receive b'
    SBIT = 4  # stores S
    CBIT = 5  # stores C
    TEMP2 = 6
    TEMP3 = 7

    # Init all entries in partitions 1 to N+1
    # --- 2 OPs --- #
    crossbar.perform(
        Operation([Gate(GateType.INIT1, [], [(1, TEMP1), (1, ABIT), (1, BBIT)])] + [Gate(GateType.INIT1, [],
                  [(partition, ABIT), (partition, BBIT), (partition, ABBIT),
                   (partition, TEMP1), (partition, TEMP2), (partition, TEMP3)])
                for partition in range(2, N+1)] + [Gate(GateType.INIT1, [], [(N + 1, i) for i in range(2*N)])]))
    crossbar.perform(
        Operation([Gate(GateType.INIT0, [],
                  [(partition, SBIT), (partition, CBIT)])
                for partition in range(2, N+1)]))

    # Store a's bits in the partitions
    # --- N OPs --- #
    for i in range(N):
        crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, i)], [(N-i, ABIT)])]))

    # Iterate over all 2*N stages
    for k in range(2 * N):

        is_notted = [False] * (N + 1)

        if k < N:

            # Copy b_k to all partitions using log_2(N) ops
            # --- log_2(N) OPs --- #
            log2_N = N.bit_length() - 1
            for i in range(log2_N):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(j, BBIT) if j != 1 else (0, N+k)], [(j + (N >> (i+1)), BBIT)])
                    for j in range(1, N+1, 1 << (log2_N - i))
                ]))
            is_notted[1:] = [bool(bin(p).count('1') % 2 == 1) for p in range(N)]

            # Compute partial products
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABIT), (j, BBIT)], [(j, ABBIT)]) if is_notted[j] else
                Gate(GateType.MAGIC_NOT, [(j, ABIT)], [(j, BBIT)])
                for j in range(2, N + 1)
            ] + [Gate(GateType.MAGIC_NOT, [(0, N + k)], [(1, BBIT)])]))

            # TEMP1 = NOR(ABBIT if is_notted[j] else BBIT, SBIT) = NOR(A, B)
            # TEMP2 = NOR(ABBIT if is_notted[j] else BBIT, TEMP1) = NOR(A, NOR(A, B))
            # TEMP3 = NOR(TEMP1, SBIT) = NOR(NOR(A, B), B)
            # INIT(ABBIT, SBIT)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABBIT if is_notted[j] else BBIT), (j, SBIT)], [(j, TEMP1)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABBIT if is_notted[j] else BBIT), (j, TEMP1)], [(j, TEMP2)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP1), (j, SBIT)], [(j, TEMP3)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, ABBIT), (j, SBIT)])
                for j in range(2, N + 1)
            ]))

            # ABBIT = NOR(TEMP2, TEMP3)
            # SBIT = NOR(ABBIT, CBIT)
            # INIT(TEMP2, TEMP3)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP2), (j, TEMP3)], [(j, ABBIT)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABBIT), (j, CBIT)], [(j, SBIT)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, TEMP2), (j, TEMP3)])
                for j in range(2, N + 1)
            ]))

            # TEMP2 = NOR(ABBIT, SBIT)
            # TEMP3 = NOR(SBIT, CBIT)
            # INIT(CBIT)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABBIT), (j, SBIT)], [(j, TEMP2)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, SBIT), (j, CBIT)], [(j, TEMP3)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, CBIT)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP1), (j, SBIT)], [(j, CBIT)])
                for j in range(2, N + 1)
            ]))

            # INIT(SBIT)
            # SBIT = NOR(TEMP2, TEMP3)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, SBIT)])
                for j in range(2, N + 1)
            ]))
            # Compute S across adjacent partitions
            # --- 2 OPs --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP2), (j, TEMP3)], [(j + 1, SBIT)])
                for j in range(3, N + 1, 2)
            ] + [Gate(GateType.MAGIC_NOR, [(1, ABIT), (1, BBIT)], [(2, SBIT)]) if k < N else Gate(GateType.INIT0, [], [(2, SBIT)])]))
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP2), (j, TEMP3)], [(j + 1, SBIT)])
                for j in range(2, N, 2)
            ] + [Gate(GateType.MAGIC_NOR, [(N, TEMP2), (N, TEMP3)], [(N + 1, k)])]))

        else:

            # TEMP1 = NOR(SBIT, CBIT)
            # TEMP2 = NOT(SBIT)
            # TEMP3 = NOT(CBIT)
            # INIT(CBIT)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, SBIT), (j, CBIT)], [(j, TEMP1)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, SBIT)], [(j, TEMP2)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, CBIT)], [(j, TEMP3)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, CBIT)])
                for j in range(2, N + 1)
            ]))

            # CBIT = NOR(TEMP2, TEMP3)
            # INIT(SBIT)
            # SUM = NOR(CBIT, TEMP1)

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, TEMP2), (j, TEMP3)], [(j, CBIT)])
                for j in range(2, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, SBIT)])
                for j in range(2, N + 1)
            ]))
            # Compute S across adjacent partitions
            # --- 2 OPs --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, CBIT), (j, TEMP1)], [(j + 1, SBIT)])
                for j in range(3, N + 1, 2)
            ] + [Gate(GateType.MAGIC_NOR, [(1, ABIT), (1, BBIT)], [(2, SBIT)]) if k < N else Gate(GateType.INIT0, [], [(2, SBIT)])]))
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, CBIT), (j, TEMP1)], [(j + 1, SBIT)])
                for j in range(2, N, 2)
            ] + [Gate(GateType.MAGIC_NOR, [(N, CBIT), (N, TEMP1)], [(N + 1, k)])]))

        # Init the temps for next time
        # --- 1 OP --- #
        crossbar.perform(Operation([Gate(GateType.INIT1, [], [(1, BBIT)])] + [
            Gate(GateType.INIT1, [], ([(j, BBIT)] if k < N else []) + [(j, ABBIT), (j, TEMP1), (j, TEMP2), (j, TEMP3)])
            for j in range(2, N + 1)
        ]))

        if k == N-1:
            # Set AB to zero for last k iterations
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT0, [], [(j, BBIT)])
                for j in range(1, N + 1)
            ]))


# Parameters
N = 32

# Crossbar
partitionSize = 8
crossbar = Crossbar([2 * N] + [3] + [partitionSize] * (N - 1) + [2 * N])

num_samples = 10
for sample in tqdm(range(num_samples)):

    # Sample a and b
    a = randint(0, (1 << N) - 1)
    b = randint(0, (1 << N) - 1)

    # Crossbar initialization
    crossbar.op_counter = 0
    crossbar.partitions[0] = [bool((a & (1 << i)) >> i) for i in range(N)] + [bool((b & (1 << i)) >> i) for i in range(N)]

    # Perform multiplication
    MultPIM(crossbar, N)

    # Verify results
    assert(sum([int(crossbar.partitions[N+1][i]) << i for i in range(2*N)]) == a * b)

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per multiplication')

