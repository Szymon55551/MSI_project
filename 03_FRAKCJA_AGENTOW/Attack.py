from scan_strategy import scan_strategy
import math
import random
from pydantic import BaseModel
from a_star import a_star

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

def select_orbit_cell_based_on_ammo(agent, nodes, enemy_cell, r_ammo_world, band_ratio=0.1, max_candidates=60):
    if not nodes or enemy_cell is None or r_ammo_world is None: return None
    node_by_cell = {n.cell: n for n in nodes}
    if not node_by_cell: return None

    if enemy_cell not in node_by_cell:
        exw, eyw = agent._cell_center(enemy_cell)
        best, best_d2 = None, 1e30
        for c, n in node_by_cell.items():
            d2 = (n.world[0] - exw) ** 2 + (n.world[1] - eyw) ** 2
            if d2 < best_d2: best_d2, best = d2, c
        enemy_cell = best
    if enemy_cell is None: return None

    r_cells = float(r_ammo_world) / float(agent.CELL_SIZE)
    band = max(1.0, band_ratio * r_cells)  
    r_min, r_max = max(0.0, r_cells - band), r_cells + band
    ex, ey = enemy_cell
    candidates = []

    for c, n in node_by_cell.items():
        if n.blocked or n.hole or n.water: continue
        cx, cy = c
        d = math.hypot(cx - ex, cy - ey)
        if r_min <= d <= r_max: candidates.append(c)

    if not candidates: return None
    random.shuffle(candidates)
    
    my_pos = agent.dynamic_info.get("position", {})
    start_cell = agent._cell_from_xy(float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0)))
    
    for goal in candidates[:max_candidates]:
        path = a_star(agent, nodes, start_cell, goal)
        if path and len(path) >= 2: return goal
    return None

def mode_attack(agent, enemy_now):
    enemy = enemy_now if enemy_now is not None else agent._closest_visible_enemy()

    if enemy is None:
        barrel_rot = scan_strategy(agent)
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=agent.enemy_target_cell)
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=False, ammo_to_load="LONG_DISTANCE")

    now = int(agent.current_tick)
    if not hasattr(agent, "close_backoff_until_tick"): agent.close_backoff_until_tick = -10_000   
    if not hasattr(agent, "post_shot_backoff_until_tick"): agent.post_shot_backoff_until_tick = -10_000

    pos_my, pos_enemy = agent.dynamic_info.get("position") or {}, agent._get(enemy, "position", {}) or {}
    mx, my = float(agent._get(pos_my, "x", 0.0)), float(agent._get(pos_my, "y", 0.0))
    ex, ey = float(agent._get(pos_enemy, "x", 0.0)), float(agent._get(pos_enemy, "y", 0.0))

    dist = math.hypot(ex - mx, ey - my)
    enemy_cell = agent._cell_from_xy(ex, ey)
    agent.enemy_target_cell = enemy_cell

    inv, loaded = agent._ammo_inventory(), agent._loaded_ammo_name()
    desired = next((a for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"] if inv.get(a, 0) > 0 and agent._ammo_range_world(a) >= dist), next((a for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"] if inv.get(a, 0) > 0), None))

    barrel_rot = agent._aim_barrel_at_enemy(enemy)
    if desired is None: return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=0.0, move_speed=0.0, should_fire=False, ammo_to_load="LIGHT")
    if loaded != desired:
        r_desired = agent._ammo_range_world(desired)
        if dist > r_desired: hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)
        else:
            hull_rot = 0.0
            move_speed = 0.0 if random.random() < 0.7 else (0.15 * float(agent.static_info.get("top_speed", 0.0)))
        return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=False, ammo_to_load=desired)

    r_ammo_world = agent._ammo_range_world(desired)
    r_orbit_world = 0.65 * r_ammo_world
    ring_band_world = max(8.0, 0.10 * r_orbit_world)

    force_close_in = (desired != "LONG_DISTANCE") and (dist > 0.85 * r_ammo_world)
    too_close = dist < (r_orbit_world - ring_band_world)
    ready_to_shoot = dist <= r_ammo_world and agent._can_fire_at_enemy_with_range(enemy, desired, aim_tolerance_deg=5.0)
    in_post_shot_backoff = (now < agent.post_shot_backoff_until_tick)

    if too_close and not in_post_shot_backoff and now >= agent.close_backoff_until_tick: agent.close_backoff_until_tick = now + 50
    in_close_backoff = (now < agent.close_backoff_until_tick)
    do_backoff = in_post_shot_backoff or in_close_backoff
    if ready_to_shoot and not in_post_shot_backoff: do_backoff = False

    if not hasattr(agent, "attack_jitter_until_tick"):
        agent.attack_jitter_until_tick = -10_000
        agent.attack_jitter_mode = "orbit"  

    if agent.current_tick >= agent.attack_jitter_until_tick:
        agent.attack_jitter_until_tick = agent.current_tick + random.randint(20, 60)
        agent.attack_jitter_mode = random.choices(["orbit", "strafe", "pause"], weights=[0.60, 0.30, 0.10], k=1)[0]
        agent.attack_jitter_sign = random.choice([-1, 1])
        agent.attack_speed_jitter = random.uniform(1,5)

    hull_rot, move_speed = 0.0, 0.0
    if dist > r_ammo_world or force_close_in:
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)
    elif do_backoff:
        err = agent._angle_diff((math.degrees(math.atan2(ey - my, ex - mx)) + 360.0) % 360.0, float(agent.dynamic_info.get("heading", 0.0)))
        heading_spin = float(agent.static_info.get("heading_spin_rate", 0.0))
        hull_rot = agent._clamp(err, -heading_spin, heading_spin)
        top_speed = float(agent.static_info.get("top_speed", 0.0))

        if in_post_shot_backoff and not in_close_backoff: base_back = -0.65 * top_speed
        else: base_back = -(0.55 + 0.35 * max(0.0, min(1.0, (r_orbit_world - dist) / max(1e-6, r_orbit_world)))) * top_speed

        move_speed = base_back * (1.0 + agent.attack_speed_jitter)
        if agent.attack_jitter_mode == "strafe": hull_rot = agent._clamp(hull_rot + agent.attack_jitter_sign * 0.25 * heading_spin, -heading_spin, heading_spin)
    else:
        if agent.attack_jitter_mode != "pause":
            nodes = agent._build_graph()
            orbit_goal = select_orbit_cell_based_on_ammo(agent, nodes, enemy_cell, r_ammo_world=r_orbit_world, band_ratio=0.1, max_candidates=60)
            if orbit_goal is not None and agent.attack_jitter_mode in ("orbit", "strafe"):
                hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=orbit_goal)
                move_speed *= (1.0 + agent.attack_speed_jitter)
            else:
                err = agent._angle_diff((math.degrees(math.atan2(ey - my, ex - mx)) + 360.0) % 360.0, float(agent.dynamic_info.get("heading", 0.0)))
                heading_spin = float(agent.static_info.get("heading_spin_rate", 0.0))
                hull_rot = agent._clamp(err, -heading_spin, heading_spin)
                top_speed = float(agent.static_info.get("top_speed", 0.0))
                if agent.attack_jitter_mode == "strafe":
                    move_speed = -0.20 * top_speed * (1.0 + agent.attack_speed_jitter)
                    hull_rot = agent._clamp(hull_rot + agent.attack_jitter_sign * 0.30 * heading_spin, -heading_spin, heading_spin)
                else: move_speed = -0.15 * top_speed * (1.0 + agent.attack_speed_jitter)

    should_fire = bool(ready_to_shoot)
    if should_fire:
        agent.post_shot_backoff_until_tick = max(agent.post_shot_backoff_until_tick, now + 60)
        agent.close_backoff_until_tick = -10_000

    return ActionCommand(barrel_rotation_angle=barrel_rot, heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=should_fire, ammo_to_load=desired)