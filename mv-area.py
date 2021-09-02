
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint
from itertools import chain


def MVArea(crossbar: Crossbar, N: int, n: int):
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
    ABIT = 0
    ABBIT = 1
    SBIT = 2
    CBIT = 3
    TEMP1 = 4
    TEMP2 = 5

    # Init entries
    # --- 2 OPs --- #
    crossbar.perform(
        Operation([Gate(GateType.INIT1, [], [(partition, ABIT), (partition, ABBIT), (partition, TEMP1), (partition, TEMP2)])
                for partition in range(1, N + 1)]
                  + [Gate(GateType.INIT1, [], [(0, 2*n*N + 2*N + i) for i in range(10)])]))
    crossbar.perform(
        Operation([Gate(GateType.INIT0, [], [(partition, SBIT), (partition, CBIT)])
                for partition in range(1, N + 1)] + [Gate(GateType.INIT0, [], [(0, 2*n*N + 2*N + 0)])]))

    # Iterate over multiplication element pair
    for element in range(n):

        # Store a's bits in the partitions
        # --- N OPs --- #
        for i in range(N):
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, i + element*N)], [(N - i, ABIT)])]))

        # Iterate over stages
        for k in range(N):

            # The bit locations relevant to this iteration -- relevant to p0 only
            p0iterCBIT = 0 if k % 2 == 0 else 2
            p0iterNotCBIT = 1 if k % 2 == 0 else 3
            p0nextCBIT = 0 if k % 2 == 1 else 2
            p0nextNotCBIT = 1 if k % 2 == 1 else 3
            p0TEMP = 4

            is_notted = [False] + [bool(bin(p).count('1') % 2 == 0) for p in range(N)]

            # Copy b_k to p_1
            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, n*N + element*N + k)],
                [(1, TEMP1 if is_notted[1] else ABBIT)])]))

            # Copy b_k to all partitions using log_2(N) ops
            # --- log_2(N) OPs --- #
            log2_N = N.bit_length() - 1
            for i in range(log2_N):
                crossbar.perform(Operation([
                    Gate(GateType.MAGIC_NOT, [(j, TEMP1 if is_notted[j] else ABBIT)],
                         [(j + (N >> (i + 1)), TEMP1 if is_notted[j + (N >> (i+1))] else ABBIT)])
                    for j in range(1, N + 1, 1 << (log2_N - i))
                ]))

            # Compute partial products
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABIT), (j, TEMP1), (j, TEMP2)], [(j, ABBIT)]) if is_notted[j] else
                Gate(GateType.MAGIC_NOT, [(j, ABIT)], [(j, ABBIT)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.FELIX_MIN3, [(0, 2*n*N + k), (0, 2*n*N + N + k), (0, 2*n*N + 2*N + p0iterCBIT)],
                [(0, 2*n*N + 2*N + p0nextNotCBIT)])]))

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, TEMP1)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.MAGIC_NOT, [(0, 2*n*N + 2*N + p0nextNotCBIT)], [(0, 2*n*N + 2*N + p0nextCBIT)])]))

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, CBIT)], [(j, TEMP2)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.FELIX_MIN3, [(0, 2*n*N + k), (0, 2*n*N + N + k), (0, 2*n*N + 2*N + p0iterNotCBIT)],
                      [(0, 2*n*N + 2*N + p0TEMP)])]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABBIT), (j, SBIT), (j, CBIT)], [(j, TEMP1)])
                for j in range(1, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, CBIT)])
                for j in range(1, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.MAGIC_NOT, [(j, TEMP1)], [(j, CBIT)])
                for j in range(1, N + 1)
            ]))

            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, TEMP1)])
                for j in range(1, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, ABBIT), (j, SBIT), (j, TEMP2)], [(j, TEMP1)])
                for j in range(1, N + 1)
            ]))
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, SBIT)])
                for j in range(1, N + 1)
            ]))

            # Compute S across adjacent partitions
            # --- 2 OPs --- #
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, CBIT), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
                for j in range(1, N + 1, 2)
            ] + [Gate(GateType.INIT1, [], [(0, 2 * n * N + k)])]))
            crossbar.perform(Operation([
                Gate(GateType.FELIX_MIN3, [(j, CBIT), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
                for j in range(2, N, 2)
            ] + [Gate(GateType.FELIX_MIN3, [(0, 2 * n * N + 2 * N + p0nextCBIT), (0, 2 * n * N + 2 * N + p0iterNotCBIT),
                                          (0, 2 * n * N + 2 * N + p0TEMP)], [(1, SBIT)]) if element > 0 else Gate(GateType.INIT0, [], [(1, SBIT)])]))

            crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(N, CBIT), (N, TEMP1), (N, TEMP2)], [(0, 2 * n * N + k)])]))

            # Init the temps for next time
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, ABBIT), (j, TEMP1), (j, TEMP2)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.INIT1, [], [(0, 2*n*N + 2*N + p0iterCBIT), (0, 2*n*N + 2*N + p0iterNotCBIT), (0, 2*n*N + 2*N + p0TEMP)])]))

        if element < n-1:

            # Init the sum/carry temp area in p0
            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(0, i) for i in range(2 * n * N + N, 2 * n * N + 2*N)])]))

            # Copy work partitions carry back to temp zone
            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i, CBIT)], [(i, TEMP1)]) for i in range(1, N + 1)]))
            # --- N OPs --- #
            for i in range(N):
                crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i + 1, TEMP1)], [(0, 2*n*N + N + (N-i-1))])]))

            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(partition, TEMP1), (partition, CBIT)]) for partition in range(1, N + 1)]))

            # Copy the computed sum into the new carry positions
            for i in range(N):
                crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, 2 * n * N + i)], [(N-i, TEMP2)])]))
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i + 1, TEMP2)], [(i + 1, CBIT)]) for i in range(N)]))

            # --- 1 OP --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(0, i) for i in range(2 * n * N, 2 * n * N + N)])]))

            # Copy old sum back into the temp position
            # --- N + 4 OPs --- #
            crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(i, SBIT)], [(i, TEMP1)]) for i in range(1, N + 1)]))
            for i in range(N):
                crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(N-i, TEMP1)], [(0, 2*n*N + i)])]))

            # Init entries
            # --- 2 OPs --- #
            crossbar.perform(Operation([Gate(GateType.INIT1, [],
                 [(partition, ABIT), (partition, ABBIT), (partition, TEMP1), (partition, TEMP2)])
                for partition in range(1, N + 1)]
                + [Gate(GateType.INIT1, [], [(0, 2 * n * N + 2 * N + i) for i in range(1, 5)])]))
            crossbar.perform(Operation([Gate(GateType.INIT0, [],
                [(partition, SBIT)]) for partition in range(1, N + 1)]
                + [Gate(GateType.INIT0, [], [(0, 2 * n * N + 2 * N + 0)])]))

        else:

            # Init the temps for the final addition
            # --- 1 OP --- #
            crossbar.perform(Operation([
                Gate(GateType.INIT1, [], [(j, ABIT), (j, ABBIT), (j, TEMP1), (j, TEMP2)])
                for j in range(1, N + 1)
            ] + [Gate(GateType.INIT1, [], [(0, 2*n*N + N + i) for i in range(N)])]))

    # Set AB to zero for last k iterations
    # --- 1 OP --- #
    crossbar.perform(Operation([
        Gate(GateType.INIT0, [], [(j, ABBIT)])
        for j in range(1, N + 1)
    ]))

    # Iterate over first N stages
    for k in range(N):

        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, CBIT)], [(j, TEMP2)])
            for j in range(1, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, ABBIT), (j, SBIT), (j, CBIT)], [(j, TEMP1)])
            for j in range(1, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, CBIT)])
            for j in range(1, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.MAGIC_NOT, [(j, TEMP1)], [(j, CBIT)])
            for j in range(1, N + 1)
        ]))

        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, TEMP1)])
            for j in range(1, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, ABBIT), (j, SBIT), (j, TEMP2)], [(j, TEMP1)])
            for j in range(1, N + 1)
        ]))
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, SBIT)])
            for j in range(1, N + 1)
        ]))

        # Compute S across adjacent partitions
        # --- 2 OPs --- #
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, CBIT), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
            for j in range(1, N + 1, 2)
        ]))
        crossbar.perform(Operation([
            Gate(GateType.FELIX_MIN3, [(j, CBIT), (j, TEMP1), (j, TEMP2)], [(j + 1, SBIT)])
            for j in range(2, N, 2)
        ]))

        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(N, CBIT), (N, TEMP1), (N, TEMP2)], [(0, 2 * n * N + N + k)])]))

        # Init the temps for next time
        # --- 1 OP --- #
        crossbar.perform(Operation([
            Gate(GateType.INIT1, [], [(j, TEMP1), (j, TEMP2)])
            for j in range(1, N + 1)
        ]))


# Parameters
N = 32
n = 8

# Crossbar
partitionSize = 6
crossbar = Crossbar([2 * N * n + 2 * N + 10] + [partitionSize] * N)

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
    MVArea(crossbar, N, n)

    # Verify results
    result = sum([int(crossbar.partitions[0][2*n*N+i]) << i for i in range(2*N)])
    assert(result == (sum([a_s[i] * b_s[i] for i in range(n)]) % (1 << 2*N)))

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per inner product')
