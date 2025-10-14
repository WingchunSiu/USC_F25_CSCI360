# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
from heapq import heappush, heappop

import numpy as np


class TextbookStack(object):
    """A class that tracks the"""

    def __init__(self, initial_order, initial_orientations):
        assert len(initial_order) == len(initial_orientations)
        self.num_books = len(initial_order)

        for i, a in enumerate(initial_orientations):
            assert i in initial_order
            assert a == 1 or a == 0

        self.order = np.array(initial_order)
        self.orientations = np.array(initial_orientations)

    def flip_stack(self, position):
        assert position <= self.num_books

        self.order[:position] = self.order[:position][::-1]
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[
            ::-1
        ]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx != self.order[idx]) or (front_matter != 1):
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)

    def __eq__(self, other):
        assert isinstance(
            other, TextbookStack
        ), "equality comparison can only ba made with other __TextbookStacks__"
        return all(self.order == other.order) and all(
            self.orientations == other.orientations
        )

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack

def heauritic(stack):
    h = 0
    # Iterate through consecutive pairs of books
    for i in range(stack.num_books - 1):
        book1 = stack.order[i]
        book2 = stack.order[i+1]
        orient1 = stack.orientations[i]
        orient2 = stack.orientations[i+1]
        
        # Check if they met any of the 4 conditions
        satify = False
        
        # condition 1: Not adjacent in goal stack
        if abs(book1 - book2) != 1:
            satify = True
        
        # conditions 2: Different orientations
        if orient1 != orient2:
            satify = True
        
        # condition 3: wrong order but both face up
        if orient1 == 1 and orient2 == 1 and book1 == book2 + 1:
            satify = True

        # condition 4: correct order but both face down 
        if orient1 == 0 and orient2 == 0 and book1 + 1 == book2:
            satify = True

        if satify:
            h += 1
    return h

def a_star_search(stack, return_stats=False):
    flip_sequence = []
    # --- v ADD YOUR CODE HERE v --- #

    # initialize the priority queue with the initial stack and an empty sequence
    # a visited set to keep track of visited states so we don't revisit them
    visited = set()
    pq = []
    h_start = heauritic(stack)
    counter = 0 # to keep track of number of iterations for breaking ties
    nodes_visited = 0  # Track number of nodes expanded
    heappush(pq, (h_start, 0, counter, stack, flip_sequence)) # (cost, stack, sequence) in the binary heap


    while pq:
        # we always expand the node with the lowest cost in A star search
        curr_cost, curr_g, _, curr_stack, curr_sequence = heappop(pq)

        # goal test
        state = (tuple(curr_stack.order), tuple(curr_stack.orientations))
        if state in visited:
            continue
        visited.add(state)
        nodes_visited += 1  # Count this node as visited

        if curr_stack.check_ordered():
            if return_stats:
                return curr_sequence, nodes_visited
            return curr_sequence

        for flip_position in range(1, curr_stack.num_books + 1):
            # push a copy so later mutations don't alias
            new_stack = curr_stack.copy()
            new_stack.flip_stack(flip_position)
            new_sequence = curr_sequence + [flip_position]
            g = len(new_sequence) # cost function g(n) is the number of flips so far
            h = heauritic(new_stack) # heuristic cost
            f = g + h # total cost
            # add the new state to the priority queue
            heappush(pq, (f, g, counter, new_stack, new_sequence))
            counter += 1

    if return_stats:
        return flip_sequence, nodes_visited
    return flip_sequence
    # ---------------------------- #


def weighted_a_star_search(stack, epsilon=None, N=1, return_stats=False):
    # Weighted A* is extra credit

    flip_sequence = []
    # --- v ADD YOUR CODE HERE v --- #
    # epsilon is the weight factor and is defualt to 1 for now
    if epsilon is None:
        epsilon = 1
    visited = set()
    pq = []
    counter = 0
    nodes_visited = 0  # Track number of nodes expanded

    # Initial state push
    h_start = heauritic(stack)
    d_start = 0
    w_start = 1 + (epsilon - 1) * (1 - d_start / N) if N > 0 else 1
    f_start = 0 + w_start * h_start
    heappush(pq, (f_start, 0, counter, stack, flip_sequence))
    counter += 1

    while pq:
        curr_f, curr_g, _, curr_stack, curr_sequence = heappop(pq)

        state = (tuple(curr_stack.order), tuple(curr_stack.orientations))
        if state in visited:
            continue
        visited.add(state)
        nodes_visited += 1

        if curr_stack.check_ordered():
            if return_stats:
                return curr_sequence, nodes_visited
            return curr_sequence

        for flip_position in range(1, curr_stack.num_books + 1):
            new_stack = curr_stack.copy()
            new_stack.flip_stack(flip_position)
            new_sequence = curr_sequence + [flip_position]

            g = len(new_sequence)
            d = g  # depth = g
            h = heauritic(new_stack)

            # Dynamic weighting
            w = 1 + (epsilon - 1) * (1 - d / N) if N > 0 else 1
            f = g + w * h

            heappush(pq, (f, g, counter, new_stack, new_sequence))
            counter += 1

    if return_stats:
        return flip_sequence, nodes_visited
    return flip_sequence

    # ---------------------------- #


if __name__ == "__main__":
    test = TextbookStack(initial_order=[3, 2, 1, 0], initial_orientations=[0, 0, 0, 0])
    output_sequence = a_star_search(test)
    correct_sequence = int(output_sequence == [4])

    new_stack = apply_sequence(test, output_sequence)
    stack_ordered = new_stack.check_ordered()

    print(f"Stack is {'' if stack_ordered else 'not '}ordered")
    print(f"Comparing output to expected traces  - \t{'PASSED' if correct_sequence else 'FAILED'}")
