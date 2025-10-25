import numpy as np
import heapq

# Directions for moving in the grid (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# Define terrain costs
TERRAIN_COSTS = {
    'R': (1, 2),  # (normal_battery_cost, low_battery_cost)
    'H': (3, 4),
    'C': (5, 10),
    'E': (2, 2),
    'S': (1, 2),
    'G': (1, 2)
}


def a_star_search(grid, K=1):
    """
    Implement A* Search for the Autonomous Delivery Robot problem.

    Args:
        grid (List[List[str]]): 2D grid map containing:
            'S' - Start
            'G' - Goal
            'R' - Road
            'H' - Highway
            'C' - Construction Zone
            'E' - Charging Station
            'X' - Blocked / Impassable
        K (int): battery consumption per move

    Returns:
        path (List[Tuple[int, int]]): sequence of coordinates from S to G (inclusive)
        total_cost (float): total traversal cost of the found path
    """
    n = len(grid)

    # Locate Start (S) and Goal (G)
    sx, sy = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 'S'][0]
    gx, gy = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 'G'][0]

    # ----- WRITE YOUR CODE BELOW -----

    # ----- WRITE YOUR CODE ABOVE -----
    
    # If the open list becomes empty and the goal was not reached, no path exists.
    return [], float('inf')


if __name__ == "__main__":
    grid = [
        ['S','R','R','R','X','R'],
        ['C','X','E','R','C','R'],
        ['R','R','H','R','X','E'],
        ['X','C','R','H','R','R'],
        ['E','X','R','C','R','R'],
        ['R','R','R','X','H','G']
    ]

    path1, cost1 = a_star_search(grid, K=1)
    print("\nCase 1 (K=1):")
    if path1:
        print("  Optimal Path:", path1)
        print("  Minimum Cost:", cost1)
    else:
        print("  No path found.")

    path2, cost2 = a_star_search(grid, K=10)
    print("\nCase 2 (K=10):")
    if path2:
        print("  Optimal Path:", path2)
        print("  Minimum Cost:", cost2)
    else:
        print("  No path found.")

    path3, cost3 = a_star_search(grid, K=20)
    print("\nCase 3 (K=20):")
    if path3:
        print("  Optimal Path:", path3)
        print("  Minimum Cost:", cost3)
    else:
        print("  No path found.")
