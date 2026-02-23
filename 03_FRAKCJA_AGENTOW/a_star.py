import heapq
import random

DMG_PENALTY = 4000   #koszt za 1 pkt obrazen
BASE_MOVE_COST = 1  #koszt ruchu o 1 kratke
SLOW_TERRAIN_PENALTY_WEIGHT = 75
STRAIGHT_PENALTY = 2  #koszt za jazdę prosto
DANGEROUS_NEIGHBOUR_PENALTY = 20 #Kara za to ze jestesmy bezposrednio obok niebezpiecznej kratki

def a_star(agent, nodes, goal_cell):
    if not nodes:
        return None
        
    node_by_cell = {n.cell: n for n in nodes}
    
    if goal_cell not in node_by_cell:
        # print(f"[A*] goal {goal_cell} not in graph (nodes={len(node_by_cell)})")
        return None
        
    # Znajdź start
    my_pos = agent.dynamic_info.get("position", {"x": 0.0, "y": 0.0})
    sx, sy = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
    
    # O(1) OPTIMIZATION: Direct lookup using spatial mapping
    start_cell = agent._cell_from_xy(sx, sy)
    
    # Fallback to O(N) linear search ONLY if the exact cell is missing from the active graph
    if start_cell not in node_by_cell:
        best_dist = float('inf')
        for cell, node in node_by_cell.items():
            wx, wy = node.world
            d2 = (wx-sx)**2 + (wy-sy)**2
            if d2 < best_dist:
                best_dist = d2
                start_cell = cell
                
    if start_cell is None:
        return None

    # Funkcja heurystyki
    def heuristic(c):
        return max(abs(c[0] - goal_cell[0]), abs(c[1] - goal_cell[1]))

    # f_score = g_score + h_score
    start_h = heuristic(start_cell)
    # OPTIMIZATION: Queue no longer stores the list, just the node
    queue = [(start_h, 0.0, start_cell)]
    
    g_scores = {start_cell: 0.0}
    came_from = {} # OPTIMIZATION: Pointer dictionary
    
    noise_map = {cell: random.uniform(0.0, 0.5) for cell in node_by_cell}

    while queue:
        f, current_g, current_node = heapq.heappop(queue)

        if current_node == goal_cell:
            # OPTIMIZATION: Reconstruct path only once at the end
            path = []
            curr = current_node
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start_cell)
            path.reverse()
            return path
        
        if current_g > g_scores.get(current_node, float('inf')):
            continue

        for neighbour in node_by_cell[current_node].neighbors:
            neighbor_node = node_by_cell[neighbour]
            
            obst_penalty = 100000.0 if neighbor_node.blocked else 0.0
            
            speed_loss = max(0.0, 1.0 - neighbor_node.speed)
            move_cost = BASE_MOVE_COST + (speed_loss * SLOW_TERRAIN_PENALTY_WEIGHT)
            move_cost += float(neighbor_node.dmg) * DMG_PENALTY
            move_cost += obst_penalty
            
            if neighbor_node.is_risk:
                move_cost += DANGEROUS_NEIGHBOUR_PENALTY

            # Straight Line Penalty using came_from
            if current_node in came_from:
                prev = came_from[current_node]
                curr = current_node
                nxt = neighbour
                if (curr[0]-prev[0], curr[1]-prev[1]) == (nxt[0]-curr[0], nxt[1]-curr[1]):
                    move_cost += STRAIGHT_PENALTY

            new_g = current_g + move_cost

            if new_g < g_scores.get(neighbour, float('inf')):
                came_from[neighbour] = current_node # Update pointer
                g_scores[neighbour] = new_g
                new_f = new_g + heuristic(neighbour)
                heapq.heappush(queue, (new_f, new_g, neighbour))
                
    return None