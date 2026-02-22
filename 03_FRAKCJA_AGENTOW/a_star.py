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
        print(f"[A*] goal {goal_cell} not in graph (nodes={len(node_by_cell)})")

        return None
    
        
    # Znajdź start
    my_pos = agent.dynamic_info.get("position", {"x": 0.0, "y": 0.0})
    sx, sy = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
    
    start_cell = None
    best_dist = float('inf')
    
    for cell, node in node_by_cell.items():
        wx, wy = node.world
        d2 = (wx-sx)**2 + (wy-sy)**2
        if d2 < best_dist:
            best_dist = d2
            start_cell = cell
            
    if start_cell is None:
        return None

    print("start_cell", start_cell)
    print("goal_cell", goal_cell)
    
    
    print("[A*] nodes:", len(node_by_cell))
    print("[A*] start deg:", len(node_by_cell[start_cell].neighbors))
    print("[A*] goal deg:", len(node_by_cell[goal_cell].neighbors))

    # Funkcja heurystyki
    def heuristic(c):
        return abs(c[0] - goal_cell[0]) + abs(c[1] - goal_cell[1])

    # f_score = g_score + h_score
    start_h = heuristic(start_cell)
    queue = [(start_h, 0.0, [start_cell])]
    
    # Słownik najlepszych kosztów dotarcia do pola (g_score)
    g_scores = {start_cell: 0.0}
    
    # Cache dla szumu (żeby nie generować w pętli)
    noise_map = {cell: random.uniform(0.0, 0.5) for cell in node_by_cell}

    while queue:
        # heapq.heappop jest O(1) - wyciąga element o najniższym f_score
        f, current_g, path = heapq.heappop(queue)
        current_node = path[-1]

        if current_node == goal_cell:
            return path
        
        # Jeśli znaleźliśmy już szybszą drogę do tego węzła w międzyczasie -> skip
        if current_g > g_scores.get(current_node, float('inf')):
            continue

        # Sprawdzanie sąsiadów
        for neighbour in node_by_cell[current_node].neighbors:
            neighbor_node = node_by_cell[neighbour]
            
            # --- Logika Kosztów ---
            
            # Soft Block dla ścian (umożliwia ucieczkę z inflacji)
            obst_penalty = 100000.0 if neighbor_node.blocked else 0.0
            
            # Teren i obrażenia
            speed_loss = max(0.0, 1.0 - neighbor_node.speed)
            move_cost = BASE_MOVE_COST + (speed_loss * SLOW_TERRAIN_PENALTY_WEIGHT)
            move_cost += float(neighbor_node.dmg) * DMG_PENALTY
            # move_cost += noise_map.get(neighbour, 0.0)
            move_cost += obst_penalty
            
            if neighbor_node.is_risk:
                move_cost += DANGEROUS_NEIGHBOUR_PENALTY

            # Straight Line Penalty
            if len(path) >= 2:
                prev = path[-2]
                curr = current_node
                nxt = neighbour
                if (curr[0]-prev[0], curr[1]-prev[1]) == (nxt[0]-curr[0], nxt[1]-curr[1]):
                    move_cost += STRAIGHT_PENALTY

            new_g = current_g + move_cost

            # Relaksacja krawędzi
            if new_g < g_scores.get(neighbour, float('inf')):
                g_scores[neighbour] = new_g
                new_f = new_g + heuristic(neighbour)
                heapq.heappush(queue, (new_f, new_g, path + [neighbour]))
                
    return None

