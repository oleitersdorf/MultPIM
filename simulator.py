from typing import List
from enum import Enum


class GateType(Enum):
    """
    Represents a type of gate.
    """

    MAGIC_NOT = 0
    MAGIC_NOR = 1
    FELIX_NAND = 2
    FELIX_OR = 3
    FELIX_MIN3 = 4
    MAJ3 = 5
    INIT0 = 6
    INIT1 = 7


class Gate:
    """
    Represents a single gate.
    """

    def __init__(self, type: GateType, inputs: List[tuple], outputs: List[tuple]):
        """
        Initializes the gate.
        :param type: The type of the gate
        :param inputs: An array of tuples representing the inputs. Each tuple contains the partition index, and the
            index of the input within the partition
        :param outputs: An array of tuples representing the outputs. Each tuple contains the partition index, and the
            index of the output within the partition
        """
        self.type = type
        self.inputs = inputs
        self.outputs = outputs

    def perform(self, partitions: List[List[bool]]):
        """
        Performs the gate on the given partitions list.
        :param partitions: the given partitions array.
        """
        if self.type == GateType.MAGIC_NOT:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                not (partitions[self.inputs[0][0]][self.inputs[0][1]]) and partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.MAGIC_NOR:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                not (partitions[self.inputs[0][0]][self.inputs[0][1]] or partitions[self.inputs[1][0]][self.inputs[1][1]]) and partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.FELIX_NAND:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                not (partitions[self.inputs[0][0]][self.inputs[0][1]] and partitions[self.inputs[1][0]][self.inputs[1][1]]) and partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.FELIX_OR:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                (partitions[self.inputs[0][0]][self.inputs[0][1]] or partitions[self.inputs[1][0]][self.inputs[1][1]]) and partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.FELIX_MIN3:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                not ((partitions[self.inputs[0][0]][self.inputs[0][1]] and partitions[self.inputs[1][0]][self.inputs[1][1]]) or
                 (partitions[self.inputs[0][0]][self.inputs[0][1]] and partitions[self.inputs[2][0]][self.inputs[2][1]]) or
                 (partitions[self.inputs[1][0]][self.inputs[1][1]] and partitions[self.inputs[2][0]][self.inputs[2][1]])) and \
                partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.MAJ3:
            partitions[self.outputs[0][0]][self.outputs[0][1]] = \
                ((partitions[self.inputs[0][0]][self.inputs[0][1]] and partitions[self.inputs[1][0]][self.inputs[1][1]]) or
                 (partitions[self.inputs[0][0]][self.inputs[0][1]] and partitions[self.inputs[2][0]][self.inputs[2][1]]) or
                 (partitions[self.inputs[1][0]][self.inputs[1][1]] and partitions[self.inputs[2][0]][self.inputs[2][1]])) and \
                partitions[self.outputs[0][0]][self.outputs[0][1]]
        elif self.type == GateType.INIT0:
            for output in self.outputs:
                partitions[output[0]][output[1]] = False
        elif self.type == GateType.INIT1:
            for output in self.outputs:
                partitions[output[0]][output[1]] = True

    def minPartitionUsed(self) -> int:
        """
        Computes the minimal index of used partitions
        :return: the minimal index
        """
        return min([input[0] for input in self.inputs] + [output[0] for output in self.outputs])

    def maxPartitionUsed(self) -> int:
        """
        Computes the maximal index of used partitions
        :return: the maximal index
        """
        return max([input[0] for input in self.inputs] + [output[0] for output in self.outputs])


def collides(first: Gate, second: Gate) -> bool:
    """
    Returns whether or not the gates "collide" with each-other in terms of partition resources
    :param first: the first gate
    :param second: the second gate
    :return: whether or not the gates collides
    """
    minFirst = first.minPartitionUsed()
    maxFirst = first.maxPartitionUsed()
    minSecond = second.minPartitionUsed()
    maxSecond = second.maxPartitionUsed()
    return (minFirst <= maxSecond) and (maxFirst >= minSecond)


class Operation:
    """
    Represents (possibly) multiple gates occurring in parallel
    """

    def __init__(self, gates: List[Gate]):
        """
        Initializes the operation with a list of gates.
        :param gates: the list of gates.
        """
        self.gates = gates
        for gate1 in gates:
            for gate2 in gates:
                if gate1 != gate2:
                    assert(not collides(gate1, gate2))

    def perform(self, partitions: List[List[bool]]):
        """
        Performs the operation on the partitions array
        :param partitions: an array of the partitions and their values
        """
        for gate in self.gates:
            gate.perform(partitions)


class Crossbar:
    """
    Represents a single-row crossbar
    """

    def __init__(self, partition_sizes: List[int]):
        """
        Initializes the row with the partition sizes
        :param partition_sizes: A list containing the size of each partition
        """
        self.partitions = [[False] * size for size in partition_sizes]
        self.op_counter = 0

    def perform(self, op: Operation):
        op.perform(self.partitions)
        self.op_counter += 1
