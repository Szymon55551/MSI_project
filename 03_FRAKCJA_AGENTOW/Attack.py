from scan_strategy import scan_strategy
import math
import random
from pydantic import BaseModel

class ActionCommand(BaseModel):
    """Output action from agent to engine."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

def select_orbit_cell_based_on_ammo(agent, nodes, enemy_cell, r_ammo_world):
    """
    Selects a safe orbital target using the persistent D* Lite planner.
    Uses squared Euclidean distances for map-invariant efficiency.
    """
    if not nodes or enemy_cell is None or r_ammo_world is None:
        return None

    node_by_cell = {n.cell: n for n in nodes}
    r_cells = float(r_ammo_world) / float(agent.CELL_SIZE)
    band = max(1.0, 0.1 * r_cells)  
    r_min_sq = (r_cells - band) ** 2
    r_max_sq = (r_cells + band) ** 2

    ex, ey = enemy_cell
    candidates = []

    for c, n in node_by_cell.items():
        if n.blocked or n.dmg > 0 or getattr(n, "is_risk", False):
            continue
        d2 = (c[0] - ex)**2 + (c[1] - ey)**2
        if r_min_sq <= d2 <= r_max_sq:
            candidates.append(c)

    if not candidates:
        return None

    # Shuffle to prevent deterministic corner-seeking behavior
    random.shuffle(candidates)
    
    pos = agent.dynamic_info.get("position") or {}
    start_cell = agent._cell_from_xy(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))

    # Evaluate reachability for the top 5 candidates to preserve D* memory
    for goal in candidates[:5]:
        path = agent.d_star_planner.update_graph_and_path(start_cell, goal, nodes)
        if path and len(path) >= 2:
            return goal
    return None

def mode_attack(agent, enemy_now):
    enemy = enemy_now if enemy_now is not None else agent._closest_visible_enemy()
    now = int(agent.current_tick)

    if enemy is None:
        barrel_rot = scan_strategy(agent)
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(
            override_goal_cell=agent.enemy_target_cell
        )
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, 
                             move_speed=move_speed, should_fire=False, ammo_to_load="LONG_DISTANCE")

    # Aiming and ammo selection logic
    dist = math.hypot(enemy["position"]["x"] - agent.dynamic_info["position"]["x"], 
                      enemy["position"]["y"] - agent.dynamic_info["position"]["y"])
    enemy_cell = agent._cell_from_xy(enemy["position"]["x"], enemy["position"]["y"])
    agent.enemy_target_cell = enemy_cell

    inv = agent._ammo_inventory()
    desired = next((a for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"] if inv.get(a, 0) > 0 and agent._ammo_range_world(a) >= dist), "LIGHT")
    barrel_rot = agent._aim_barrel_at_enemy(enemy)

    if agent._loaded_ammo_name() != desired:
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, 
                             move_speed=move_speed, should_fire=False, ammo_to_load=desired)

    # Orbital movement logic
    r_orbit_world = 0.65 * agent._ammo_range_world(desired)
    
    if not hasattr(agent, "current_orbit_goal") or (now - getattr(agent, "orbit_goal_tick", 0)) > 40:
        nodes = agent._divide_seen_area(agent.dynamic_info.get("visible_obstacles"), 
                                         agent.dynamic_info.get("visible_terrains"))
        agent.current_orbit_goal = select_orbit_cell_based_on_ammo(agent, nodes, enemy_cell, r_orbit_world)
        agent.orbit_goal_tick = now

    hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=agent.current_orbit_goal)
    ready_to_shoot = agent._can_fire_at_enemy_with_range(enemy, desired)

    return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, 
                         move_speed=move_speed, should_fire=ready_to_shoot, ammo_to_load=desired)