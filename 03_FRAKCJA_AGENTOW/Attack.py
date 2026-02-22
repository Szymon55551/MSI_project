from scan_strategy import scan_strategy
import math
import random
from pydantic import BaseModel
from a_star import a_star


class ActionCommand(BaseModel):
    """Output action from agent to engine."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False


def select_orbit_cell_based_on_ammo(
    agent,
    nodes,
    enemy_cell,
    r_ammo_world,
    band_ratio=0.1,
    max_candidates=60,
):
    """
    Wybiera losową bezpieczną komórkę w pierścieniu wokół enemy_cell:
      radius ≈ r_ammo_world / CELL_SIZE  (w sub-komórkach)
      ring = [r - band, r + band]
    Warunki "bezpieczne": not blocked, dmg==0, not is_risk.
    Dodatkowo sprawdza osiągalność przez A*.
    """
    if not nodes or enemy_cell is None or r_ammo_world is None:
        return None

    node_by_cell = {n.cell: n for n in nodes}
    if not node_by_cell:
        return None

    if enemy_cell not in node_by_cell:
        exw, eyw = agent._cell_center(enemy_cell)
        best = None
        best_d2 = 1e30
        for c, n in node_by_cell.items():
            wx, wy = n.world
            d2 = (wx - exw) ** 2 + (wy - eyw) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = c
        enemy_cell = best

    if enemy_cell is None:
        return None

    r_cells = float(r_ammo_world) / float(agent.CELL_SIZE)
    band = max(1.0, band_ratio * r_cells)  
    r_min = max(0.0, r_cells - band)
    r_max = r_cells + band

    ex, ey = enemy_cell
    candidates = []

    for c, n in node_by_cell.items():
        if n.blocked:
            continue
        if int(getattr(n, "dmg", 0)) != 0:
            continue
        if bool(getattr(n, "is_risk", False)):
            continue

        cx, cy = c
        d = math.hypot(cx - ex, cy - ey)  # dystans w subgridzie
        if r_min <= d <= r_max:
            candidates.append(c)

    if not candidates:
        return None

    random.shuffle(candidates)
    for goal in candidates[:max_candidates]:
        path = a_star(agent, nodes, goal)
        if path and len(path) >= 2:
            return goal

    return None


def mode_attack(agent, enemy_now):
    enemy = enemy_now if enemy_now is not None else agent._closest_visible_enemy()

    # brak wroga idź do ostatniej znanej komórki i skanuj
    if enemy is None:
        barrel_rot = scan_strategy(agent)
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(
            override_goal_cell=agent.enemy_target_cell if agent.enemy_target_cell is not None else None
        )
        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            should_fire=False,
            ammo_to_load="LONG_DISTANCE",
        )

    now = int(agent.current_tick)

    # init backoff timers
    if not hasattr(agent, "close_backoff_until_tick"):
        agent.close_backoff_until_tick = -10_000   
    if not hasattr(agent, "post_shot_backoff_until_tick"):
        agent.post_shot_backoff_until_tick = -10_000

    # pozycje i dystans 
    pos_my = agent.dynamic_info.get("position") or {}
    pos_enemy = agent._get(enemy, "position", {}) or {}

    mx = float(agent._get(pos_my, "x", 0.0))
    my = float(agent._get(pos_my, "y", 0.0))
    ex = float(agent._get(pos_enemy, "x", 0.0))
    ey = float(agent._get(pos_enemy, "y", 0.0))

    dist = math.hypot(ex - mx, ey - my)
    enemy_cell = agent._cell_from_xy(ex, ey)
    agent.enemy_target_cell = enemy_cell  # last known

    # wybór amunicji 
    inv = agent._ammo_inventory()
    loaded = agent._loaded_ammo_name()

    desired = None
    for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"]:
        if inv.get(a, 0) > 0 and agent._ammo_range_world(a) >= dist:
            desired = a
            break
    if desired is None:
        for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"]:
            if inv.get(a, 0) > 0:
                desired = a
                break

    barrel_rot = agent._aim_barrel_at_enemy(enemy)

    # brak amunicji - stań, celuj  --- do zmiany 
    if desired is None:
        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=0.0,
            move_speed=0.0,
            should_fire=False,
            ammo_to_load="LIGHT",
        )
    if loaded != desired:
        r_desired = agent._ammo_range_world(desired)

        if dist > r_desired:
            hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)
        else:
            hull_rot = 0.0
            top_speed = float(agent.static_info.get("top_speed", 0.0))
            move_speed = 0.0 if random.random() < 0.7 else (0.15 * top_speed)

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            should_fire=False,
            ammo_to_load=desired,
        )

    # właściwa amunicja załadowana
    r_ammo_world = agent._ammo_range_world(desired)

    # POLITYKA DYSTANSU 
    r_orbit_world = 0.65 * r_ammo_world
    ring_band_world = max(8.0, 0.10 * r_orbit_world)

    force_close_in = (desired != "LONG_DISTANCE") and (dist > 0.85 * r_ammo_world)
    too_close = dist < (r_orbit_world - ring_band_world)

    ready_to_shoot = False
    if dist <= r_ammo_world:
        ready_to_shoot = bool(agent._can_fire_at_enemy_with_range(enemy, desired, aim_tolerance_deg=5.0))

    in_post_shot_backoff = (now < agent.post_shot_backoff_until_tick)

    ### Here
    # if too_close and (not ready_to_shoot) and (not in_post_shot_backoff):  - źle, cofa się zawsze jak za blisko 
    if too_close and (not in_post_shot_backoff):
        if now >= agent.close_backoff_until_tick:
            agent.close_backoff_until_tick = now + 50

    in_close_backoff = (now < agent.close_backoff_until_tick)

    do_backoff = in_post_shot_backoff or in_close_backoff

    if ready_to_shoot and (not in_post_shot_backoff):
        do_backoff = False

    if not hasattr(agent, "attack_jitter_until_tick"):
        agent.attack_jitter_until_tick = -10_000
        agent.attack_jitter_mode = "orbit"  

    if agent.current_tick >= agent.attack_jitter_until_tick:
        agent.attack_jitter_until_tick = agent.current_tick + random.randint(20, 60)
        agent.attack_jitter_mode = random.choices(
            population=["orbit", "strafe", "pause"],
            weights=[0.60, 0.30, 0.10],
            k=1
        )[0]
        agent.attack_jitter_sign = random.choice([-1, 1])
        agent.attack_speed_jitter = random.uniform(1,5)  # +/-10%

    # ruch
    hull_rot = 0.0
    move_speed = 0.0

    if dist > r_ammo_world or force_close_in:
        hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)

    elif do_backoff:
        # MANUAL BACKOFF: kadłub na wroga, prędkość wstecz
        dx = ex - mx
        dy = ey - my
        desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        current_heading = float(agent.dynamic_info.get("heading", 0.0))
        err = agent._angle_diff(desired_heading, current_heading)

        heading_spin = float(agent.static_info.get("heading_spin_rate", 0.0))
        hull_rot = agent._clamp(err, -heading_spin, heading_spin)

        top_speed = float(agent.static_info.get("top_speed", 0.0))


        if in_post_shot_backoff and (not in_close_backoff):
            base_back = -0.65 * top_speed
        else:
            closeness = (r_orbit_world - dist) / max(1e-6, r_orbit_world)
            closeness = max(0.0, min(1.0, closeness))
            base_back = -(0.55 + 0.35 * closeness) * top_speed  # [-0.55..-0.90]

        move_speed = base_back * (1.0 + agent.attack_speed_jitter)

        if agent.attack_jitter_mode == "strafe":
            hull_rot = agent._clamp(
                hull_rot + agent.attack_jitter_sign * 0.25 * heading_spin,
                -heading_spin, heading_spin
            )

    else:
        # w pierścieniu: orbit / strafe / pause
        if agent.attack_jitter_mode == "pause":
            hull_rot, move_speed = 0.0, 0.0
        else:
            visible_obstacles = agent.dynamic_info.get("visible_obstacles") or []
            visible_terrains = agent.dynamic_info.get("visible_terrains") or []
            nodes = agent._divide_seen_area(visible_obstacles, visible_terrains)

            orbit_goal = select_orbit_cell_based_on_ammo(
                agent,
                nodes,
                enemy_cell,
                r_ammo_world=r_orbit_world,
                band_ratio=0.1,
                max_candidates=60,
            )

            if orbit_goal is not None and agent.attack_jitter_mode in ("orbit", "strafe"):
                hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=orbit_goal)
                move_speed *= (1.0 + agent.attack_speed_jitter)
            else:
                dx = ex - mx
                dy = ey - my
                desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                current_heading = float(agent.dynamic_info.get("heading", 0.0))
                err = agent._angle_diff(desired_heading, current_heading)

                heading_spin = float(agent.static_info.get("heading_spin_rate", 0.0))
                hull_rot = agent._clamp(err, -heading_spin, heading_spin)

                top_speed = float(agent.static_info.get("top_speed", 0.0))

                if agent.attack_jitter_mode == "strafe":
                    move_speed = -0.20 * top_speed * (1.0 + agent.attack_speed_jitter)
                    hull_rot = agent._clamp(
                        hull_rot + agent.attack_jitter_sign * 0.30 * heading_spin,
                        -heading_spin, heading_spin
                    )
                else:
                    move_speed = -0.15 * top_speed * (1.0 + agent.attack_speed_jitter)

    #  strzał 
    should_fire = bool(ready_to_shoot)

    if should_fire:
        agent.post_shot_backoff_until_tick = max(agent.post_shot_backoff_until_tick, now + 60)
        agent.close_backoff_until_tick = -10_000

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=hull_rot,
        move_speed=move_speed,
        should_fire=should_fire,
        ammo_to_load=desired,
    )