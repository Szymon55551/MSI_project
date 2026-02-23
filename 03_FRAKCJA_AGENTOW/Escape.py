from scan_strategy import scan_strategy
import math
import random
from pydantic import BaseModel

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

def plan_farthest_escape_path_each_tick(agent, enemy_cell, max_candidates=10):
    nodes = agent._divide_seen_area(agent.dynamic_info.get("visible_obstacles"), 
                                     agent.dynamic_info.get("visible_terrains"))
    if not nodes or enemy_cell is None:
        return None, None

    # Filter safe nodes and sort by distance from enemy
    ex, ey = agent._cell_center(enemy_cell)
    safe = [n for n in nodes if not n.blocked and n.dmg == 0 and not getattr(n, "is_risk", False)]
    if not safe: safe = [n for n in nodes if not n.blocked]
    
    safe.sort(key=lambda n: (n.world[0] - ex) ** 2 + (n.world[1] - ey) ** 2, reverse=True)
    
    pos = agent.dynamic_info.get("position") or {}
    start_cell = agent._cell_from_xy(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))

    for n in safe[:max_candidates]:
        path = agent.d_star_planner.update_graph_and_path(start_cell, n.cell, nodes)
        if path and len(path) > 5:
            return path, n.cell
    return None, None

def mode_escape(agent, enemy_now) -> ActionCommand:
    barrel_rot = scan_strategy(agent)
    now = agent.current_tick
    
    # Refresh escape timer if enemy is visible
    if enemy_now:
        agent.escape_until_tick = max(agent.escape_until_tick, now + agent.escape_commit_ticks)

    enemy = enemy_now or agent._closest_visible_enemy()
    enemy_cell = agent._cell_from_xy(enemy["position"]["x"], enemy["position"]["y"]) if enemy else None

    # Re-plan if goal reached or periodically
    if not getattr(agent, "escape_target_cell", None) or (now - getattr(agent, "escape_target_last_pick_tick", 0)) > 100:
        path, goal = plan_farthest_escape_path_each_tick(agent, enemy_cell)
        if goal:
            agent.escape_target_cell = goal
            agent.escape_target_last_pick_tick = now

    hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=agent.escape_target_cell)
    return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, 
                         move_speed=move_speed, should_fire=False, ammo_to_load="LIGHT")