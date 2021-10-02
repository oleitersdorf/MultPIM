
from simulator import Crossbar, Gate, GateType, Operation
from tqdm import tqdm
from random import randint


def RippleAdder(crossbar: Crossbar, N: int):
    """
    Performs the ripple-carry algorithm on the given crossbar (with the novel MultPIM full-adder - footnote 6 in the paper)
    :param crossbar: The crossbar to perform the algorithm on.
    :param N: The number of bits in each number.
    """

    # Init carry to zero
    crossbar.perform(Operation([Gate(GateType.INIT0, [], [(0, 3 * N)])]))
    crossbar.perform(Operation([Gate(GateType.INIT1, [], [(0, 3 * N + 2), (0, 3*N + 4), (0, 3*N + 1), (0, 3*N + 3)] + [(0, k) for k in range(2*N, 3*N)])]))

    for k in range(N):

        # Legend
        carry_loc = 3*N + (0 if k % 2 == 0 else 1)
        new_carry_loc = 3 * N + (1 if k % 2 == 0 else 0)
        not_carry_loc = 3*N + (2 if k % 2 == 0 else 3)
        new_not_carry_loc = 3 * N + (3 if k % 2 == 0 else 2)
        temp_loc = 3 * N + 4

        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(0, k), (0, N + k), (0, carry_loc)], [(0, new_not_carry_loc)])]))
        crossbar.perform(Operation([Gate(GateType.MAGIC_NOT, [(0, new_not_carry_loc)], [(0, new_carry_loc)])]))
        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(0, k), (0, N + k), (0, not_carry_loc)], [(0, temp_loc)])]))
        crossbar.perform(Operation([Gate(GateType.FELIX_MIN3, [(0, new_carry_loc), (0, not_carry_loc), (0, temp_loc)], [(0, 2 * N + k)])]))

        if k < N-1:
            crossbar.perform(Operation([Gate(GateType.INIT1, [], [(0, temp_loc), (0, carry_loc), (0, not_carry_loc)])]))


# Parameters
N = 32

# Crossbar
crossbar = Crossbar([3 * N + 5])

num_samples = 1000
for sample in tqdm(range(num_samples)):

    # Sample a and b
    a = randint(0, (1 << N) - 1)
    b = randint(0, (1 << N) - 1)

    # Crossbar initialization
    crossbar.op_counter = 0
    crossbar.partitions[0][:2*N] = [bool((a & (1 << i)) >> i) for i in range(N)] + [bool((b & (1 << i)) >> i) for i in range(N)]

    # Perform multiplication
    RippleAdder(crossbar, N)

    res = sum([int(crossbar.partitions[0][2*N + i]) << i for i in range(N)])

    # Verify results
    assert(res == ((a + b) % (1 << N)))

print(f'Success with {crossbar.op_counter} cycles and {sum([len(partition) for partition in crossbar.partitions])} memristors per addition')

