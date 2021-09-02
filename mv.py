
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint
from itertools import chain


def MV(crossbar: Crossbar, N: int, n: int):
    """
    Performs the MV algorithm on the given crossbar. As the crossbar is modeled by a single row,
    performs a inner product in the memristive row.
    :param crossbar: The crossbar to perform the algorithm on. Assumes partition 0 contains the two vector inputs,
    that there are N partitions (indices 1 to N) with unknown init status, and a partition at the end for the output.
    :param N: The number of bits in each number. Assumes w.l.o.g. N is a power of two: this is not a necessary
    assumption, but it does simplify the implementation of the function here.
    :param n: The vector dimension.
    """

    # Legend
    ABIT = 5
    BBIT = 6
    ABBIT = 7  # only in partitions that receive b'
    SBITEven = 8  # stores S for even iterations
    CBITEven = 0  # stores C for even iterations
    NotCBITEven = 1  # stores not(C) for even iterations
    SBITOdd = 9  # stores S for odd iterations
    CBITOdd = 2  # stores C for odd iterations
    NotCBITOdd = 3  # stores not(C) for odd iterations
    TEMP = 4

    # Init entries
    # --- 2 OPs --- #
    crossbar.perform(
        Operation([Gate(GateType.INIT1, [], [(partition, ABIT), (partition, BBIT), (partition, ABBIT), (partition, NotCBITEven),
                   (partition, SBITOdd), (partition, CBITOdd), (partition, NotCBITOdd), (partition, TEMP)])
                for partition in range(1, N + 1)] + [Gate(GateType.INIT1, [], [(N + 1, i) for i in range(2*N)])]
                  + [Gate(GateType.INIT1, [], [(0, 2*n*N + 2*N + NotCBITEven),
                    (0, 2*n*N + 2*N + CBITOdd), (0, 2*n*N + 2*N + NotCBITOdd), (0, 2*n*N + 2*N + TEMP)])]))
    crossbar.perform(
        Operation([Gate(GateType.INIT0, [],
                  [(partition, SBITEven), (partition, CBITEven)])
                for partition in range(1, N + 1)] + [Gate(GateType.INIT0, [], [(0, 2*n*N + 2*N + CBITEven)])]))

    # Iterate over multiplication element pair
    for element in range(n):

        # Init temp carry in S-C addition
        # --- 2 OPs --- #
        if element > 0:
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(0, 2 * n * N + 2 * N + NotCBITEven)])]))
            crossbar.perform(Operation([Gate(GateType.INIT0, [], [(0, 2*n*N + 2*N + CBITEven)])]))

        # Store a's bits in the partitions
        # --- N OPs --- #
        for i in range(N):
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, i + element*N)], [(N - i, ABIT)])]))

        # Iterate over stages
        for k in range(N):

            # The bit locations relevant to this iteration
            iterSBIT = SBITEven if k % 2 == 0 else SBITOdd
            iterCBIT = CBITEven if k % 2 == 0 else CBITOdd
            iterNotCBIT = NotCBITEven if k % 2 == 0 else NotCBITOdd
            nextSBIT = SBITEven if k % 2 == 1 else SBITOdd
            nextCBIT = CBITEven if k % 2 == 1 else CBITOdd
            nextNotCBIT = NotCBITEven if k % 2 == 1 else NotCBITOdd

            # Used to understand which partitions receive b_k and which receive not(b_k)
            numStepsToReach = [0] * (N + 1)

            # Copy b_k to p_1
            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, n*N + element*N + k)], [(1, BBIT)])]))
            numStepsToReach[1] = 1

            # Copy b_k to all partitions using log_2(N) ops
            # --- log_2(N) OPs --- #
            log2_N = N.bit_length() - 1
            for i in range(log2_N):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(j, BBIT)], [(j + (N >> (i + 1)), BBIT)])
                    for j in range(1, N + 1, 1 << (log2_N - i))
                ]))
                for j in range(1, N + 1, 1 << (log2_N - i)):
                    numStepsToReach[j + (N >> (i + 1))] = numStepsToReach[j] + 1
            is_notted = [bool(steps % 2) for steps in numStepsToReach]

            # Compute partial products
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOR, [(j, ABIT), (j, BBIT)], [(j, ABBIT)]) if is_notted[j] else
                Gate(GateType.MAGIC_NOT, [(j, ABIT)], [(j, BBIT)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.FELIX_MIN3, [(0, 2*n*N + N-k-1), (0, 2*n*N + N + N-k-1), (0, 2*n*N + 2*N + iterCBIT)], [(0, 2*n*N + 2*N + nextNotCBIT)])]))

            # Compute new not(carry)
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABBIT if is_notted[j] else BBIT), (j, iterSBIT), (j, iterCBIT)],
                     [(j, nextNotCBIT)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.MAGIC_NOT, [(0, 2*n*N + 2*N + nextNotCBIT)], [(0, 2*n*N + 2*N + nextCBIT)])]))
            # Compute new carry
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, nextNotCBIT)], [(j, nextCBIT)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.FELIX_MIN3, [(0, 2*n*N + N-k-1), (0, 2*n*N + N + N-k-1), (0, 2*n*N + 2*N + iterNotCBIT)],
                      [(0, 2*n*N + 2*N + TEMP)])]))

            # Compute Min3(AB, S, not(C))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABBIT if is_notted[j] else BBIT), (j, iterSBIT), (j, iterNotCBIT)],
                     [(j, TEMP)])
                for j in range(1, N + 1)
            ]))

            # Compute S across adjacent partitions
            # --- 2 OPs --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)],
                     [(j + 1, nextSBIT)])
                for j in range(1, N + 1, 2)
            ]))
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)],
                     [(j + 1, nextSBIT)])
                for j in range(2, N, 2)
            ] + [Gate(GateType.FELIX_MIN3, [(N, nextCBIT), (N, iterNotCBIT), (N, TEMP)], [(N + 1, k)]),
            Gate(GateType.FELIX_MIN3, [(0, 2 * n * N + 2 * N + nextCBIT), (0, 2 * n * N + 2 * N + iterNotCBIT),
                (0, 2 * n * N + 2 * N + TEMP)], [(1, nextSBIT)]) if element > 0 else Gate(GateType.INIT0, [], [(1, nextSBIT)])]))

            # Init the temps for next time
            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [],
                     [(j, BBIT), (j, ABBIT), (j, iterSBIT), (j, iterCBIT), (j, iterNotCBIT), (j, TEMP)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.INIT1, [], [(0, 2*n*N + 2*N + iterCBIT), (0, 2*n*N + 2*N + iterNotCBIT), (0, 2*n*N + 2*N + TEMP)])]))

        if element < n-1:

            # Init the sum/carry temp area in p0
            # --- 1 OP --- #
            crossbar.perform(
                Operation([Gate(GateType.INIT1, [], [(0, i) for i in range(2 * n * N, 2 * n * N + 2 * N)])]))

            # Copy work partitions carry back to temp zone
            # --- N OPs --- #
            for i in range(N):
                crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i+1, NotCBITEven)], [(0, 2*n*N + N + i)])]))

            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(partition, CBITEven), (partition, NotCBITEven)]) for partition in range(1, N + 1)]))

            # Copy sum from pN+1 back to the work partitions carry and copy work partitions sum back at same time
            # --- N + 4 OPs --- #
            crossbar.perform(
                Operation([Gate(GateType.MAGIC_NOT, [(i, SBITEven)], [(i, TEMP)]) for i in range(1, N + 1)]))
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(N, TEMP)], [(0, 2*n*N+N-1)])]))
            for i in range(N-1):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(N + 1, i)], [(N-i, NotCBITEven)]),
                    Gate(GateType.MAGIC_NOT, [(N-i-1, TEMP)], [(0, 2*n*N + N-i-1-1)])]))
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(N + 1, N-1)], [(1, NotCBITEven)])]))
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i + 1, NotCBITEven)], [(i + 1, CBITEven)]) for i in range(N)]))

            # Init entries
            # --- 2 OPs --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [],
                 [(partition, ABIT), (partition, BBIT), (partition, ABBIT),
                 (partition, SBITOdd), (partition, CBITOdd), (partition, NotCBITOdd), (partition, TEMP)])
                for partition in range(1, N + 1)] + [Gate(GateType.INIT1, [], [(N + 1, i) for i in range(2 * N)])]
                + [Gate(GateType.INIT1, [], [(0, 2 * n * N + 2 * N + CBITOdd), (0, 2 * n * N + 2 * N + NotCBITOdd),
                   (0, 2 * n * N + 2 * N + TEMP)])]))
            crossbar.perform(Operation([Gate(GateType.INIT0, [],
                [(partition, SBITEven)]) for partition in range(1, N + 1)]
                + [Gate(GateType.INIT0, [], [(0, 2 * n * N + 2 * N + CBITEven)])]))

        else:

            # Init the temps for the final addition
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, ABIT), (j, BBIT), (j, ABBIT), (j, NotCBITEven), (j, NotCBITOdd), (j, SBITOdd), (j, CBITOdd)])
                for j in range(1, N + 1)
            ]))

    # Set AB to zero for last k iterations
    # --- 1 OP --- #
    crossbar.perform(Operation([
        Gate(GateType.INIT0, [], [(j, BBIT)])
        for j in range(1, N + 1)
    ]))
    # --- 1 OP --- #
    crossbar.perform(Operation([
        Gate(GateType.MAGIC_NOT, [(j, CBITEven)], [(j, NotCBITEven)])
        for j in range(1, N + 1)
    ]))

    # Iterate over first N stages
    for k in range(N):

        # The bit locations relevant to this iteration
        iterSBIT = SBITEven if k % 2 == 0 else SBITOdd
        iterCBIT = CBITEven if k % 2 == 0 else CBITOdd
        iterNotCBIT = NotCBITEven if k % 2 == 0 else NotCBITOdd
        nextSBIT = SBITEven if k % 2 == 1 else SBITOdd
        nextCBIT = CBITEven if k % 2 == 1 else CBITOdd
        nextNotCBIT = NotCBITEven if k % 2 == 1 else NotCBITOdd

        # Compute new not(carry)
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, BBIT), (j, iterSBIT), (j, iterCBIT)],
                 [(j, nextNotCBIT)])
            for j in range(1, N + 1)
        ]))
        # Compute new carry
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, nextNotCBIT)], [(j, nextCBIT)])
            for j in range(1, N + 1)
        ]))

        # Compute Min3(AB, S, not(C))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, BBIT), (j, iterSBIT), (j, iterNotCBIT)], [(j, TEMP)])
            for j in range(1, N + 1)
        ]))

        # Compute S across adjacent partitions
        # --- 2 OPs --- #
        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)], [(j + 1, nextSBIT)])
            for j in range(1, N + 1, 2)
        ]))
        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(j, nextCBIT), (j, iterNotCBIT), (j, TEMP)],
                 [(j + 1, nextSBIT)])
            for j in range(2, N, 2)
        ] + [Gate(GateType.INIT0, [], [(1, nextSBIT)])] + [Gate(GateType.FELIX_MIN3, [(N, nextCBIT), (N, iterNotCBIT), (N, TEMP)], [(N + 1, N + k)])]))

        # Init the temps for next time
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [],
                 [(j, ABBIT), (j, iterSBIT), (j, iterCBIT), (j, iterNotCBIT), (j, TEMP)])
            for j in range(1, N + 1)
        ]))


# Parameters
N = 32
n = 8

# Crossbar
partitionSize = 10
crossbar = Crossbar([2 * N * n + 2 * N + 5] + [partitionSize] * N + [2 * N])

num_samples = 10
for sample in tqdm(range(num_samples)):

    # Sample n random pairs
    a_s = [randint(0, (1 << N) - 1) for _ in range(n)]
    b_s = [randint(0, (1 << N) - 1) for _ in range(n)]

    # Crossbar initialization
    crossbar.op_counter = 0
    crossbar.partitions[0][:2*N*n] = \
        list(chain(*([[bool((a_s[j] & (1 << i)) >> i) for i in range(N)] for j in range(n)] +
                     [[bool((b_s[j] & (1 << i)) >> i) for i in range(N)] for j in range(n)])))

    # Perform multiplication
    MV(crossbar, N, n)

    # Verify results
    result = sum([int(crossbar.partitions[N+1][i]) << i for i in range(2*N)])
    assert(result == (sum([a_s[i] * b_s[i] for i in range(n)]) % (1 << 2*N)))

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per inner product')
