import heapq
import itertools

def a_star(agent, nodes_dummy, start_cell, goal_cell, max_iterations=2000):
    if start_cell is None or goal_cell is None: 
        return None
        
    vmap = getattr(agent, "virtual_map", {})
    
    def get_cost(cell):
        cell_data = vmap.get(cell)
        if not cell_data: return 0.1  # Unknown is priority
            
        ct = cell_data.get("type", 0)
        
        if ct == 3: return 10.0 if cell == start_cell else float('inf')
        if ct == 0: return 0.1 
            
        base = 1.0
        if ct == 5: base = 80.0   
        elif ct == 2: base = 60.0  
        elif ct == 6: base = 5.0   
        elif ct == 4: base = 1.0 
        
        return base + cell_data.get("penalty", 0.0)

    def heuristic(c):
        dx = abs(c[0] - goal_cell[0])
        dy = abs(c[1] - goal_cell[1])
        h = dx + dy
        cross = abs((c[0] - start_cell[0]) * (goal_cell[1] - start_cell[1]) - 
                    (goal_cell[0] - start_cell[0]) * (c[1] - start_cell[1]))
        return h + (cross * 0.001)

    counter = itertools.count()
    queue = [(heuristic(start_cell), next(counter), 0.0, start_cell)]
    g_scores = {start_cell: 0.0}
    came_from = {}
    
    DIRS = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

    iterations = 0
    while queue and iterations < max_iterations:
        iterations += 1
        _, _, current_g, current_node = heapq.heappop(queue)

        if current_node == goal_cell:
            path = []
            curr = current_node
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start_cell)
            return path[::-1]

        if current_g > g_scores.get(current_node, float('inf')):
            continue

        for dx, dy in DIRS:
            nx, ny = current_node[0] + dx, current_node[1] + dy
            if not (0 <= nx <= 199 and 0 <= ny <= 199): continue
                
            # CRITICAL FIX: Prevent diagonal corner-cutting through solid walls
            if dx != 0 and dy != 0:
                if vmap.get((current_node[0]+dx, current_node[1]), {}).get("type", 0) == 3 or \
                   vmap.get((current_node[0], current_node[1]+dy), {}).get("type", 0) == 3:
                    continue
                
            neighbor = (nx, ny)
            step_cost = get_cost(neighbor)
            if step_cost == float('inf'): continue
                
            actual_step = step_cost * (1.414 if (dx and dy) else 1.0)
            new_g = current_g + actual_step
            
            if new_g < g_scores.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                g_scores[neighbor] = new_g
                f_score = new_g + heuristic(neighbor)
                heapq.heappush(queue, (f_score, next(counter), new_g, neighbor))
                
    return None