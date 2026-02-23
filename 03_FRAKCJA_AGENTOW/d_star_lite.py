import heapq
import math

# Traversal Modifiers
DMG_PENALTY = 4000.0   
BASE_MOVE_COST = 1.0  
SLOW_TERRAIN_PENALTY_WEIGHT = 200.0  # INCREASED: Heavily penalize water/slow terrain
DANGEROUS_NEIGHBOUR_PENALTY = 4.0 

class DStarLite:
    def __init__(self):
        self.s_start = None
        self.s_goal = None
        
        self.U = []          
        self.k_m = 0.0       
        self.rhs = {}        
        self.g = {}          
        
        self.nodes_data = {} 
        self.last_start = None

    def _heuristic(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _calculate_key(self, s):
        g_rhs_min = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (
            g_rhs_min + self._heuristic(self.s_start, s) + self.k_m,
            g_rhs_min
        )

    def _get_cost(self, u, v):
        if v not in self.nodes_data:
            return float('inf')
        
        node_v = self.nodes_data[v]
        
        if node_v.blocked:
            return float('inf')
            
        speed_loss = max(0.0, 1.0 - node_v.speed)
        cost = BASE_MOVE_COST + (speed_loss * SLOW_TERRAIN_PENALTY_WEIGHT)
        
        if node_v.dmg > 0:
            cost += (node_v.dmg * DMG_PENALTY)
        
        if getattr(node_v, 'is_risk', False):
            cost += DANGEROUS_NEIGHBOUR_PENALTY
            
        return cost

    def _update_vertex(self, u):
        if u != self.s_goal:
            min_rhs = float('inf')
            if u in self.nodes_data:
                for v in self.nodes_data[u].neighbors:
                    c = self._get_cost(u, v)
                    min_rhs = min(min_rhs, c + self.g.get(v, float('inf')))
            self.rhs[u] = min_rhs

        self.U = [item for item in self.U if item[2] != u]
        heapq.heapify(self.U)

        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, (*self._calculate_key(u), u))

    def _compute_shortest_path(self):
        while self.U:
            U_top_key = (self.U[0][0], self.U[0][1])
            u = self.U[0][2]
            
            if U_top_key >= self._calculate_key(self.s_start) and \
               self.rhs.get(self.s_start, float('inf')) == self.g.get(self.s_start, float('inf')):
                break
                
            k_old = U_top_key
            heapq.heappop(self.U)
            k_new = self._calculate_key(u)

            if k_old < k_new:
                heapq.heappush(self.U, (*k_new, u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                if u in self.nodes_data:
                    for s in self.nodes_data[u].neighbors:
                        self._update_vertex(s)
            else:
                self.g[u] = float('inf')
                self._update_vertex(u)
                if u in self.nodes_data:
                    for s in self.nodes_data[u].neighbors:
                        self._update_vertex(s)

    def update_graph_and_path(self, current_start, goal_cell, nodes_list):
        new_nodes_data = {n.cell: n for n in nodes_list}
        
        if self.s_goal != goal_cell:
            self.s_start = current_start
            self.s_goal = goal_cell
            self.U = []
            self.k_m = 0.0
            self.rhs = {self.s_goal: 0.0}
            self.g = {self.s_goal: float('inf')}
            self.nodes_data = new_nodes_data
            self.last_start = current_start
            
            heapq.heappush(self.U, (*self._calculate_key(self.s_goal), self.s_goal))
            self._compute_shortest_path()
            return self._extract_path()

        self.s_start = current_start
        self.k_m += self._heuristic(self.last_start, self.s_start)
        self.last_start = self.s_start

        changed_nodes = []
        for cell, new_node in new_nodes_data.items():
            if cell in self.nodes_data:
                old_node = self.nodes_data[cell]
                if (old_node.dmg != new_node.dmg or 
                    old_node.speed != new_node.speed or 
                    old_node.blocked != new_node.blocked or
                    getattr(old_node, 'is_risk', False) != getattr(new_node, 'is_risk', False)):
                    changed_nodes.append(cell)
            else:
                changed_nodes.append(cell)

        self.nodes_data = new_nodes_data

        if changed_nodes:
            for u in changed_nodes:
                self._update_vertex(u)
                for s in self.nodes_data[u].neighbors:
                    self._update_vertex(s)
            
            self._compute_shortest_path()

        return self._extract_path()

    def _extract_path(self):
        if self.g.get(self.s_start, float('inf')) == float('inf'):
            return None

        curr = self.s_start
        path = [curr]
        
        max_steps = 500 
        steps = 0
        
        while curr != self.s_goal and steps < max_steps:
            best_next = None
            min_cost = float('inf')
            
            if curr in self.nodes_data:
                for v in self.nodes_data[curr].neighbors:
                    c = self._get_cost(curr, v)
                    val = c + self.g.get(v, float('inf'))
                    if val < min_cost:
                        min_cost = val
                        best_next = v
                        
            if best_next is None or best_next in path:
                break
                
            curr = best_next
            path.append(curr)
            steps += 1
            
        return path