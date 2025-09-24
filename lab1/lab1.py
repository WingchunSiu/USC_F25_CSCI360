# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
import numpy as np

class TextbookStack(object):
    """ A class that tracks the """
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
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[::-1]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx != self.order[idx]) or (front_matter != 1):
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)
    
    def __eq__(self, other):
        assert isinstance(other, TextbookStack), "equality comparison can only ba made with other __TextbookStacks__"
        return all(self.order == other.order) and all(self.orientations == other.orientations)

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack

def breadth_first_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #
    if stack.check_ordered():
        return flip_sequence
    
    # initialize the queue with the initial stack and an empty sequence (a node in the search tree)
    # FIFO structure
    queue = deque([(stack, [])])
    visited = set()
    visited.add((tuple(stack.order), tuple(stack.orientations)))
    
    while queue:
        current_stack, current_sequence = queue.popleft()
        for flip_position in range(1, stack.num_books + 1):
            new_stack = current_stack.copy()
            new_stack.flip_stack(flip_position)
            new_sequence = current_sequence + [flip_position]
            
            if new_stack.check_ordered():
                return new_sequence
            
            state_key = (tuple(new_stack.order), tuple(new_stack.orientations))
            if state_key not in visited:
                visited.add(state_key)
                queue.append((new_stack, new_sequence))
    

    return flip_sequence
    # ---------------------------- #


def depth_first_search(stack):
    flip_sequence = []
   
    # --- v ADD YOUR CODE HERE v --- #
    if stack.check_ordered():
        return flip_sequence
    # initialize the stack with the initial stack and an empty sequence (a node in the search tree)
    # LIFO structure
    stack_dfs = [(stack, [])]
    visited = set()
    visited.add((tuple(stack.order), tuple(stack.orientations)))
    while stack_dfs:
        current_stack, current_sequence = stack_dfs.pop()
        # iterate through possible flip positions in reverse order to maintain correct DFS order
        for filp_position in range(stack.num_books, 0,
   -1):
            new_stack = current_stack.copy()
            new_stack.flip_stack(filp_position)
            
            if new_stack.check_ordered():
                return current_sequence + [filp_position]
            
            state_key = (tuple(new_stack.order), tuple(new_stack.orientations))
            if state_key not in visited:
                visited.add(state_key)
                stack_dfs.append((new_stack, current_sequence + [filp_position]))
        
        
    return flip_sequence
    # ---------------------------- #
    

test = TextbookStack(initial_order=[3, 2, 1, 0], initial_orientations=[0, 0, 0, 0])
output_sequence = breadth_first_search(test)
print(output_sequence) # Should give you [4]

new_stack = apply_sequence(test, output_sequence)
stack_ordered = new_stack.check_ordered()
print(stack_ordered) # Should give you True