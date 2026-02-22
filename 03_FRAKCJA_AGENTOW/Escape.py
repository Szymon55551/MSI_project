ESCAPE_REPLAN_EVERY = 100

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


def _astar(agent, nodes, goal_cell):
    fn = getattr(agent, "_a_star", None)
    if callable(fn):
        return fn(nodes, goal_cell)
    from a_star import a_star
    return a_star(agent, nodes, goal_cell)


def plan_farthest_escape_path_each_tick(agent, enemy_cell, max_candidates=60):
    visible_obstacles = agent.dynamic_info.get("visible_obstacles") or []
    visible_terrains = agent.dynamic_info.get("visible_terrains") or []
    nodes = agent._divide_seen_area(visible_obstacles, visible_terrains)
    if not nodes:
        return None, None, None

    node_by_cell = {n.cell: n for n in nodes}
    if not node_by_cell:
        return None, None, None

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
            return None, None, None

    ex, ey = enemy_cell

    safe = [
        n for n in nodes
        if (not n.blocked) and int(getattr(n, "dmg", 0)) == 0 and (not bool(getattr(n, "is_risk", False)))
    ]
    if not safe:
        safe = [n for n in nodes if not n.blocked]
        if not safe:
            return None, None, None

    safe.sort(key=lambda n: (n.cell[0] - ex) ** 2 + (n.cell[1] - ey) ** 2, reverse=True)

    best_path = None
    best_goal = None
    best_score = -1.0
    best_len = -1

    for n in safe[:max_candidates]:
        goal = n.cell
        path = _astar(agent, nodes, goal)
        if not path or len(path) < 10:
            continue

        d2 = (goal[0] - ex) ** 2 + (goal[1] - ey) ** 2
        if (d2 > best_score) or (d2 == best_score and len(path) > best_len):
            best_score = float(d2)
            best_len = int(len(path))
            best_path = path
            best_goal = goal

    return best_path, best_goal, best_score


def _escape_target_reached(agent) -> bool:
    """Uznaj cel za osiągnięty, jeśli jesteś blisko środka komórki albo ścieżka się skończyła."""
    goal = getattr(agent, "escape_target_cell", None)
    if goal is None:
        return True

    if agent.path_to_follow and agent.path_index >= len(agent.path_to_follow) - 1:
        return True

    # dystans do środka komórki
    pos = agent.dynamic_info.get("position") or {}
    mx = float(pos.get("x", 0.0))
    my = float(pos.get("y", 0.0))
    gx, gy = agent._cell_center(goal)
    return math.hypot(gx - mx, gy - my) <= (0.75 * agent.CELL_SIZE)


def mode_escape(agent, enemy_now) -> ActionCommand:
    barrel_rot = scan_strategy(agent)
    top_speed = float(agent.static_info.get("top_speed", 0.0))

    # init lock state
    if not hasattr(agent, "escape_target_locked"):
        agent.escape_target_locked = False

    # jeśli wróg widoczny – odśwież commit escape
    if enemy_now is not None:
        agent.escape_until_tick = max(agent.escape_until_tick, agent.current_tick + agent.escape_commit_ticks)

    # jeśli escape się skończył -> odblokuj cel
    if agent.current_tick > agent.escape_until_tick:
        agent.escape_target_locked = False
        agent.escape_target_cell = None
        agent.escape_planned_path = None
        agent.escape_planned_goal = None
        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=0.0,
            move_speed=0.0,
            should_fire=False,
            ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]),
        )

    # jeżeli cel osiągnięty  odblokuj 
    if agent.escape_target_locked and _escape_target_reached(agent):
        agent.escape_target_locked = False
        agent.escape_target_cell = None
        agent.escape_planned_path = None
        agent.escape_planned_goal = None

    # enemy_cell (do planowania)
    enemy = enemy_now if enemy_now is not None else agent._closest_visible_enemy()
    enemy_cell = None
    if enemy is not None:
        epos = agent._get(enemy, "position", {}) or {}
        ex = float(agent._get(epos, "x", 0.0))
        ey = float(agent._get(epos, "y", 0.0))
        enemy_cell = agent._cell_from_xy(ex, ey)

    # back-phase: cofanie  
    in_back_phase = (agent.escape_enter_tick >= 0) and ((agent.current_tick - agent.escape_enter_tick) < agent.escape_back_ticks)

    if in_back_phase:
        if (not agent.escape_target_locked) and (enemy_cell is not None):
            best_path, best_goal, best_score = plan_farthest_escape_path_each_tick(agent, enemy_cell, max_candidates=5)
            print(best_goal)
            if best_goal is not None:
                agent.escape_target_cell = best_goal
                agent.escape_target_last_pick_tick = int(agent.current_tick)
                agent.escape_planned_path = best_path
                agent.escape_planned_goal = best_goal
                agent.escape_planned_score = best_score
                agent.escape_planned_tick = int(agent.current_tick)

                #
                agent.escape_target_locked = True

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=0.0,
            move_speed=-top_speed,
            should_fire=False,
            ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]),
        )


    if agent.escape_target_locked and getattr(agent, "escape_planned_path", None) and getattr(agent, "escape_planned_goal", None):
        if agent.current_goal_cell != agent.escape_planned_goal:
            agent.path_to_follow = agent.escape_planned_path
            agent.current_goal_cell = agent.escape_planned_goal
            agent.current_path_cost = None
            agent.path_index = 0
            agent.path_stuck_ticks = 0

    if (not agent.escape_target_locked) and (agent.escape_target_cell is None):
        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=0.0,
            move_speed=-top_speed,
            should_fire=False,
            ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]),
        )

    if (not agent.escape_target_locked) and ((agent.current_tick - agent.escape_target_last_pick_tick) >= ESCAPE_REPLAN_EVERY):
        if enemy is not None:
            agent._start_escape(enemy)

    hull_rot, move_speed = agent.Follow_Path_With_Modifiers(override_goal_cell=agent.escape_target_cell)

    return ActionCommand(
        barrel_rotation_angle=barrel_rot,
        heading_rotation_angle=hull_rot,
        move_speed=move_speed,
        should_fire=False,
        ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]),
    )