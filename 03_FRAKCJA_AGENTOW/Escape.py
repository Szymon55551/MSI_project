ESCAPE_REPLAN_EVERY = 100
from scan_strategy import scan_strategy
import math
import random
from pydantic import BaseModel
from a_star import a_star
from collections import deque


class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

def plan_farthest_escape_path_each_tick(agent, enemy_cell, max_candidates=None):
    if enemy_cell is None: return None, None, None
    ex, ey = enemy_cell
    
    my_pos = agent.dynamic_info.get("position", {})
    start_cell = agent._cell_from_xy(float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0)))
    vmap = getattr(agent, "virtual_map", {})

    queue = deque([start_cell])
    visited = {start_cell}
    came_from = {start_cell: None}
    
    # 4-way expansion is much faster for BFS grid searches
    DIRS = [(1,0), (-1,0), (0,1), (0,-1)] 
    
    best_goal = None
    best_score = -1.0
    
    # Cap to prevent lag. 600 nodes is plenty for a fast tactical retreat.
    max_bfs_nodes = 600 
    nodes_searched = 0

    while queue and nodes_searched < max_bfs_nodes:
        current = queue.popleft()
        nodes_searched += 1
        
        ct = vmap.get(current, {}).get("type", 0)
        
        # Track the node that maximizes distance to the enemy
        if ct in [1, 6]: # Safe terrains
            dist_sq = (current[0] - ex)**2 + (current[1] - ey)**2
            if dist_sq > best_score:
                best_score = dist_sq
                best_goal = current
                
        # Expand neighbors
        for dx, dy in DIRS:
            nb = (current[0] + dx, current[1] + dy)
            # Bound expansion strictly to index 19
            if nb not in visited and (0 <= nb[0] <= 19 and 0 <= nb[1] <= 19):
                nb_ct = vmap.get(nb, {}).get("type", 0)
                # Avoid walls, water, and danger zones during escape
                if nb_ct not in [2, 3, 5]: 
                    visited.add(nb)
                    came_from[nb] = current
                    queue.append(nb)

    # Reconstruct the path directly from BFS without using A*
    if best_goal is None:
        return None, None, None

    path = []
    curr = best_goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    
    # Reverse to go from start to goal
    path = path[::-1]
    
    return path, best_goal, best_score

def _escape_target_reached(agent) -> bool:
    goal = getattr(agent, "escape_target_cell", None)
    if goal is None: return True
    if agent.path_to_follow and agent.path_index >= len(agent.path_to_follow) - 1: return True
    pos = agent.dynamic_info.get("position") or {}
    mx, my = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
    gx, gy = agent._cell_center(goal)
    return math.hypot(gx - mx, gy - my) <= (1.5 * agent.CELL_SIZE)

def mode_escape(agent, enemy_now) -> ActionCommand:
    barrel_rot = scan_strategy(agent)
    top_speed = float(agent.static_info.get("top_speed", 0.0))

    if not hasattr(agent, "escape_target_locked"): agent.escape_target_locked = False
    if enemy_now is not None: agent.escape_until_tick = max(agent.escape_until_tick, agent.current_tick + agent.escape_commit_ticks)

    if agent.current_tick > agent.escape_until_tick:
        agent.escape_target_locked = False
        agent.escape_target_cell = agent.escape_planned_path = agent.escape_planned_goal = None
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=0.0, move_speed=0.0, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))

    if agent.escape_target_locked and _escape_target_reached(agent):
        agent.escape_target_locked = False
        agent.escape_target_cell = agent.escape_planned_path = agent.escape_planned_goal = None

    enemy = enemy_now if enemy_now is not None else agent._closest_visible_enemy()
    enemy_cell = agent._cell_from_xy(float(agent._get(agent._get(enemy, "position", {}), "x", 0.0)), float(agent._get(agent._get(enemy, "position", {}), "y", 0.0))) if enemy else None
    in_back_phase = (agent.escape_enter_tick >= 0) and ((agent.current_tick - agent.escape_enter_tick) < agent.escape_back_ticks)

    if in_back_phase:
        if not agent.escape_target_locked and enemy_cell is not None:
            best_path, best_goal, best_score = plan_farthest_escape_path_each_tick(agent, enemy_cell, max_candidates=5)
            if best_goal is not None:
                agent.escape_target_cell = best_goal
                agent.escape_target_last_pick_tick = int(agent.current_tick)
                agent.escape_planned_path, agent.escape_planned_goal, agent.escape_planned_score, agent.escape_planned_tick = best_path, best_goal, best_score, int(agent.current_tick)
                agent.escape_target_locked = True
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=0.0, move_speed=-top_speed, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))

    if agent.escape_target_locked and getattr(agent, "escape_planned_path", None) and getattr(agent, "escape_planned_goal", None):
        if agent.current_goal_cell != agent.escape_planned_goal:
            agent.path_to_follow, agent.current_goal_cell, agent.path_index, agent.path_stuck_ticks = agent.escape_planned_path, agent.escape_planned_goal, 0, 0

    if not agent.escape_target_locked and agent.escape_target_cell is None:
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=0.0, move_speed=-top_speed, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))

    if not agent.escape_target_locked and (agent.current_tick - agent.escape_target_last_pick_tick) >= ESCAPE_REPLAN_EVERY and enemy is not None:
        agent._start_escape(enemy)

    hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=agent.escape_target_cell)
    return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))