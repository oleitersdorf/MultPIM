
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint


def FSM(crossbar: Crossbar, N: int):
    """
    Performs the FSM algorithm on the given crossbar
    :param crossbar: The crossbar to perform the algorithm on. Assumes partition 0 contains only the two inputs,
    that there are N partitions (indices 1 to N) with unknown init status, and a partition at the end for the output.
    :param N: The number of bits in each number. Assumes w.l.o.g. N is a power of two: this is not a necessary
    assumption, but it does simplify the implementation of the function here.
    """

    # Legend
    ABIT = 0
    BBIT = 1
    ABBIT = 2  # only in partitions that receive b'
    SBITEven = 3  # stores S for even iterations
    CBITEven = 4  # stores C for even iterations
    NotCBITEven = 5  # stores not(C) for even iterations
    SBITOdd = 6  # stores S for odd iterations
    CBITOdd = 7  # stores C for odd iterations
    NotCBITOdd = 8  # stores not(C) for odd iterations
    TEMP = 9

    # Init all entries in partitions 1 to N+1
    # --- 2 OPs --- #
    crossbar.perform(
        Operation([Gate(GateType.INIT1, [],
                  [(partition, ABIT), (partition, BBIT), (partition, ABBIT), (partition, NotCBITEven),
                   (partition, SBITOdd), (partition, CBITOdd), (partition, NotCBITOdd), (partition, TEMP), (partition, TEMP)])
                for partition in range(1, N+1)] + [Gate(GateType.INIT1, [], [(N + 1, i) for i in range(2*N)])]))
    crossbar.perform(
        Operation([Gate(GateType.INIT0, [],
                  [(partition, SBITEven), (partition, CBITEven)])
                for partition in range(1, N+1)]))

    # Store a's bits in the partitions
    # --- N OPs --- #
    for i in range(N):
        crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, i)], [(N-i, ABIT)])]))

    # Iterate over first N stages
    for k in range(2 * N):

        # The bit locations relevant to this iteration
        iterSBIT = SBITEven if k % 2 == 0 else SBITOdd
        iterCBIT = CBITEven if k % 2 == 0 else CBITOdd
        iterNotCBIT = NotCBITEven if k % 2 == 0 else NotCBITOdd
        nextSBIT = SBITEven if k % 2 == 1 else SBITOdd
        nextCBIT = CBITEven if k % 2 == 1 else CBITOdd
        nextNotCBIT = NotCBITEven if k % 2 == 1 else NotCBITOdd
        is_notted = [False] * (N + 1)

        if k < N:

            # Used to understand which partitions receive b_k and which receive not(b_k)
            numStepsToReach = [0] * (N + 1)

            # Copy b_k to all partitions using log_2(N) ops
            # --- log_2(N) OPs --- #
            log2_N = N.bit_length() - 1
            for i in range(log2_N):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(j, BBIT) if j != 1 else (0, N+k)], [(j + (N >> (i+1)), BBIT)])
                    for j in range(1, N+1, 1 << (log2_N - i))
                ]))
                for j in range(1, N + 1, 1 << (log2_N - i)):
                    numStepsToReach[j + (N >> (i+1))] = numStepsToReach[j] + 1
            is_notted = [bool(steps % 2) for steps in numStepsToReach]

            # Compute partial products
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABIT), (j, BBIT), (j, TEMP)], [(j, ABBIT)]) if is_notted[j] else
                Gate(GateType.MAGIC_NOT, [(j, ABIT)], [(j, BBIT)])
                for j in range(2, N + 1)
            ] + [Gate(GateType.MAGIC_NOT, [(0, N + k)], [(1, BBIT)])]))

        # Compute new not(carry)
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, ABBIT if is_notted[j] else BBIT), (j, iterSBIT), (j, iterCBIT)], [(j, nextNotCBIT)])
            for j in range(2, N + 1)
        ]))
        # Compute new carry
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, nextNotCBIT)], [(j, nextCBIT)])
            for j in range(2, N + 1)
        ]))

        # Compute Min3(AB, S, not(C))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, ABBIT if is_notted[j] else BBIT), (j, iterSBIT), (j, iterNotCBIT)], [(j, TEMP)])
            for j in range(2, N + 1)
        ]))

        # Compute S across adjacent partitions
        # --- 2 OPs --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)], [(j + 1, nextSBIT)])
            for j in range(3, N + 1, 2)
        ] + [Gate(GateType.FELIX_MIN3, [(1, ABIT), (1, BBIT), (1, TEMP)], [(2, nextSBIT)]) if k < N else Gate(GateType.INIT0, [], [(2, nextSBIT)])]))
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)], [(j + 1, nextSBIT)])
            for j in range(2, N, 2)
        ] + [Gate(GateType.FELIX_MIN3, [(N, nextCBIT), (N, iterNotCBIT), (N, TEMP)], [(N + 1, k)])]))

        # Init the temps for next time
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], ([(j, BBIT)] if k < N else []) + [(j, ABBIT), (j, iterSBIT), (j, iterCBIT), (j, iterNotCBIT), (j, TEMP)])
            for j in range(1, N + 1)
        ]))

        if k == N-1:
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT0, [], [(j, BBIT)])
                for j in range(1, N + 1)
            ]))


# Parameters
N = 32

# Crossbar
partitionSize = 10
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
    FSM(crossbar, N)

    # Verify results
    assert(sum([int(crossbar.partitions[N+1][i]) << i for i in range(2*N)]) == a * b)

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per multiplication')

