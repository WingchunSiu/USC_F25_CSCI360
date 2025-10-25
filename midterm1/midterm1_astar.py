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
    
    # Manhattan distance heuristic
    def heuristic(x, y):
        return abs(x - gx) + abs(y - gy)
    
    # Get terrain cost based on current battery level
    def get_terrain_cost(terrain, battery):
        if terrain not in TERRAIN_COSTS:
            return float('inf')
        normal_cost, high_cost = TERRAIN_COSTS[terrain]
        
        return high_cost if battery < 50 else normal_cost

    INITIAL_BATTERY = 100

    initial_h = heuristic(sx, sy)
    initial_cost = get_terrain_cost(grid[sx][sy], INITIAL_BATTERY)

    # Counter for tie-breaking to ensure consistent ordering
    counter = 0

    # Priority queue: (f_score, h_score, counter, g_score, x, y, battery)
    # f_score = g_score + heuristic for priority ordering
   
    open_set = [(initial_cost + initial_h, initial_h, counter, initial_cost, sx, sy, INITIAL_BATTERY)]
    
    # Track visited states with best g_score (for expanded nodes)
    visited = {}
    
    # Best g_score discovered for any state (including frontier)
    best_cost = {(sx, sy, INITIAL_BATTERY): initial_cost}
    
    # Track parent for path reconstruction
    parent = {}
    
    while open_set:
        f_score, h_score, _, g_score, x, y, battery = heapq.heappop(open_set)
        state = (x, y, battery)
        
        # Skip if an improved path reached this state already
        if g_score > best_cost.get(state, float('inf')):
            continue
        
        # Skip if already visited with better or equal cost
        if state in visited and visited[state] <= g_score:
            continue
        
        visited[state] = g_score
        
        # if reached goal with battery remaining
        if (x, y) == (gx, gy) and battery > 0:
            path_list = []
            current_state = state
            while current_state in parent:
                back_x, back_y, _ = current_state
                path_list.append((back_x, back_y))
                current_state = parent[current_state]
            start_x, start_y, _ = current_state
            path_list.append((start_x, start_y))
            path_list.reverse()  # Reverse to get start -> goal order
            return path_list, g_score

        # Explore all neighbors of current state
        # Use custom direction order: down, right, up, left (to match expected path exploration)
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            
            terrain = grid[nx][ny]
            
            # Skip blocked cells
            if terrain == 'X':
                continue
                
            # Calculate battery after movement
            new_battery = battery - K
            # If battery goes below 0 path is invalid
            if new_battery < 0:
                continue

            # If battery is 0 can only reach goal
            if new_battery == 0 and (nx, ny) != (gx, gy):
                continue
            
            # Get terrain cost based on battery level when entering the cell
            terrain_cost = get_terrain_cost(terrain, new_battery)
            
            # Skip
            if terrain_cost == float('inf'):
                continue
            
            # Calculate the new g_score
            new_g_score = g_score + terrain_cost
            
            # Handle charging station by restoring battery to 100
            if terrain == 'E':
                final_battery = 100
            else:
                final_battery = new_battery
            
            # Create new state
            new_state = (nx, ny, final_battery)

            # Check if this neighbor is the goal with battery remaining
            if (nx, ny) == (gx, gy) and final_battery > 0:
                if new_g_score < best_cost.get(new_state, float('inf')):
                    best_cost[new_state] = new_g_score
                    parent[new_state] = state
                    path_list = []
                    current_state = new_state
                    while current_state in parent:
                        back_x, back_y, _ = current_state
                        path_list.append((back_x, back_y))
                        current_state = parent[current_state]
                    start_x, start_y, _ = current_state
                    path_list.append((start_x, start_y))
                    path_list.reverse()  # Reverse to get start -> goal order
                    return path_list, new_g_score
                else:
                    continue

            # Only process if this is a better path to this state
            if new_g_score < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_g_score
                
                # Calculate f_score for priority queue
                h_score = heuristic(nx, ny)
                new_f_score = new_g_score + h_score

                # Increment counter for each state added
                counter += 1

                # Add to openset with tie-breaking: when f is same, prefer smaller h (closer to goal)
                # counter ensures FIFO ordering when both f and h are equal
                heapq.heappush(open_set, (new_f_score, h_score, counter, new_g_score, nx, ny, final_battery))

                # Track parent for path reconstruction
                parent[new_state] = state
        

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
