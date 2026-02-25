import heapq
import itertools

def a_star(agent, nodes, start_cell, goal_cell, max_iterations=2000):
    if not nodes or start_cell is None or goal_cell is None: 
        return None
        
    node_by_cell = {n.cell: n for n in nodes}
    if start_cell not in node_by_cell or goal_cell not in node_by_cell:
        return None

    if node_by_cell[goal_cell].blocked:
        return None

    # Chebyshev distance for 8-way grid admissibility + Vector cross-product tie-breaker
    def heuristic(c):
        dx = abs(c[0] - goal_cell[0])
        dy = abs(c[1] - goal_cell[1])
        
        # Tie-breaker strictly enforces preference for the direct line of sight
        cross = abs((c[0] - start_cell[0]) * (goal_cell[1] - start_cell[1]) - 
                    (goal_cell[0] - start_cell[0]) * (c[1] - start_cell[1]))
        
        return max(dx, dy) + (cross * 0.001)

    counter = itertools.count()
    start_state = (start_cell, 0) # Track tuple of (current_node, water_count)
    start_h = heuristic(start_cell)
    
    # Queue: (f_score, tie_breaker, current_g, current_cell, water_count)
    queue = [(start_h, next(counter), 0.0, start_cell, 0)]
    g_scores = {start_state: 0.0}
    
    # Track node ancestry for O(N) path reconstruction instead of O(N^2) list cloning
    came_from = {}

    iterations = 0
    while queue and iterations < max_iterations:
        iterations += 1
        f, _, current_g, current_node, water_count = heapq.heappop(queue)
        
        current_state = (current_node, water_count)

        if current_node == goal_cell:
            path = []
            curr = current_state
            while curr in came_from:
                path.append(curr[0])
                curr = came_from[curr]
            path.append(start_cell)
            return path[::-1] # Reverse the traced ancestry
        
        if current_g > g_scores.get(current_state, float('inf')):
            continue

        curr_n = node_by_cell[current_node]
        for neighbour in curr_n.neighbors:
            neighbor_node = node_by_cell[neighbour]
            
            if neighbor_node.blocked and neighbour != start_cell:
                continue
                
            new_water_count = water_count + (1 if neighbor_node.is_water else 0)
            if new_water_count > 2: 
                continue
            if neighbour == goal_cell and neighbor_node.is_water: 
                continue

            move_cost = 1.0
            
            if neighbor_node.dmg > 0:
                move_cost += 100.0  
            if neighbor_node.speed < 0.9:
                move_cost += 3.0 

            v_type = getattr(agent, "virtual_map", {}).get(neighbour, {}).get("type", 1)
            if v_type == 4: 
                move_cost += 15.0

            margin_penalty = 0.0
            for nn in neighbor_node.neighbors:
                if nn in node_by_cell and node_by_cell[nn].blocked:
                    margin_penalty += 2.0 
            move_cost += margin_penalty
            
            # Turning penalty lookup via ancestor map
            if current_state in came_from:
                prev_node = came_from[current_state][0]
                if (current_node[0]-prev_node[0], current_node[1]-prev_node[1]) != (neighbour[0]-current_node[0], neighbour[1]-current_node[1]):
                    move_cost += 0.2

            new_g = current_g + move_cost
            neighbor_state = (neighbour, new_water_count)

            if new_g < g_scores.get(neighbor_state, float('inf')):
                came_from[neighbor_state] = current_state
                g_scores[neighbor_state] = new_g
                new_f = new_g + heuristic(neighbour)
                heapq.heappush(queue, (new_f, next(counter), new_g, neighbour, new_water_count))
                
    return None