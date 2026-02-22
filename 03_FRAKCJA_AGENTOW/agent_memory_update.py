import math
import logging


def update_internal_state(agent, status, sensors):
    update_static_info(agent, status)
    update_dynamic_info(agent, status, sensors)
    update_map_memory(agent, sensors)
    update_obstacle_memory(agent, sensors)
    update_meta_info(agent)
    update_enemy_memory(agent)


def update_static_info(agent, status):
    agent.static_info = {
        "id": agent._get(status, "_id", "MISSING"),
        "team": agent._get(status, "_team", "MISSING"),
        "tank_type": agent._get(status, "_tank_type", "MISSING"),
        "vision_angle": agent._get(status, "_vision_angle", "MISSING"),
        "vision_range": agent._get(status, "_vision_range", "MISSING"),
        "top_speed": agent._get(status, "_top_speed", "MISSING"),
        "barrel_spin_rate": agent._get(status, "_barrel_spin_rate", 30),   # czasem brak w payload
        "heading_spin_rate": agent._get(status, "_heading_spin_rate", 2),  # czasem brak w payload
        "max_hp": agent._get(status, "_max_hp", "MISSING"),
        "max_shield": agent._get(status, "_max_shield", "MISSING"),
    }
    
def update_dynamic_info(agent, status, sensors):
    seen_tanks = agent._get(sensors, "seen_tanks", []) or []
    my_team = agent.static_info.get("team", None)

    agent.dynamic_info = {
        "hp": agent._get(status, "hp", "MISSING"),
        "shield": agent._get(status, "shield", "MISSING"),
        "position": agent._get(status, "position", "MISSING"),
        "move_speed": agent._get(status, "move_speed", "MISSING"),
        "barrel_angle": agent._get(status, "barrel_angle", "MISSING"),
        "heading": agent._get(status, "heading", "MISSING"),
        "ammo": agent._get(status, "ammo", "MISSING"),
        "ammo_loaded": agent._get(status, "ammo_loaded", "MISSING"),
        "is_overcharged": agent._get(status, "is_overcharged", "MISSING"),
        "size": agent._get(status, "size", "MISSING"),
        "reload_timer": agent._get(status, "_reload_timer", "MISSING"),
        "enemies_remaining": agent.enemies_remaining,

        "visible_tanks": seen_tanks,
        "visible_enemies": [t for t in seen_tanks if t.get("team") != my_team],
        "visible_friends": [t for t in seen_tanks if t.get("team") == my_team],

        "visible_obstacles": agent._get(sensors, "seen_obstacles", []) or [],
        "visible_terrains": agent._get(sensors, "seen_terrains", []) or [],
        "visible_powerups": agent._get(sensors, "seen_powerups", []) or [],
    }


def update_map_memory(agent, sensors):
    now = int(agent.current_tick)

    for t in (agent._get(sensors, "seen_terrains", []) or []):
        pos = t.get("position", {}) or {}
        cx = float(pos.get("x", 0.0))
        cy = float(pos.get("y", 0.0))
        dmg = int(t.get("dmg", 0) or 0)
        speed = float(t.get("speed_modifier", 1.0) or 1.0)

        for cell in agent._stamp_tile_center_to_subcells(cx, cy):
            agent.map_memory[cell] = {"dmg": dmg, "speed": speed, "last_seen_tick": now}

def update_obstacle_memory(agent, sensors):
    now = int(agent.current_tick)

    for ob in (agent._get(sensors, "seen_obstacles", []) or []):
        pos = ob.get("position", {}) or {}
        cx = float(pos.get("x", 0.0))
        cy = float(pos.get("y", 0.0))

        for cell in agent._stamp_tile_center_to_subcells(cx, cy):
            agent.obstacle_memory[cell] = now

def update_enemy_memory(agent):
    now = int(agent.current_tick)
    visible_enemies = agent.dynamic_info.get("visible_enemies", []) or []

    for enemy in visible_enemies:
        eid = agent._get(enemy, "id", "MISSING")
        agent.memory[eid] = {
            "id": eid,
            "last_seen_pos": agent._get(enemy, "position", "MISSING"),
            "last_seen_tick": now,
            "tank_type": agent._get(enemy, "tank_type", "MISSING"),
            "team": agent._get(enemy, "team", "MISSING"),
        }

def update_meta_info(agent):
    my_pos = agent.dynamic_info.get("position", None)
    visible_enemies = agent.dynamic_info.get("visible_enemies", []) or []

    closest_angle = 10.0
    closest_dist = 9999.0
    target_id = None
    is_targeted = False

    if not my_pos or not visible_enemies:
        agent.meta_info = {
            "closest_enemy_angle": closest_angle,
            "closest_enemy_dist": closest_dist,
            "target_id": target_id,
            "is_aimed_at": is_targeted,
        }
        return

    my_x = float(agent._get(my_pos, "x", 0.0))
    my_y = float(agent._get(my_pos, "y", 0.0))
    my_angle = float(agent.dynamic_info.get("barrel_angle", 0.0) or 0.0)

    try:
        closest = min(visible_enemies, key=lambda t: float(agent._get(t, "distance", 9999.0)))
        target_id = agent._get(closest, "id", None)
        closest_dist = float(agent._get(closest, "distance", 9999.0))

        cpos = agent._get(closest, "position", {}) or {}
        dx = float(agent._get(cpos, "x", 0.0)) - my_x
        dy = float(agent._get(cpos, "y", 0.0)) - my_y

        desired_angle = math.degrees(math.atan2(dy, dx))
        diff = (desired_angle - my_angle + 180.0) % 360.0 - 180.0
        closest_angle = diff
    except Exception as e:
        logging.error(f"[ERROR] Failed to calculate closest enemy angle: {e}")

    for enemy in visible_enemies:
        try:
            epos = agent._get(enemy, "position", {}) or {}
            ex = float(agent._get(epos, "x", 0.0))
            ey = float(agent._get(epos, "y", 0.0))

            dx = my_x - ex
            dy = my_y - ey
            angle_to_me = math.degrees(math.atan2(dy, dx))

            aim_diff = (angle_to_me - float(agent._get(enemy, "barrel_angle", 0.0) or 0.0) + 180.0) % 360.0 - 180.0
            if abs(aim_diff) < 15.0:
                is_targeted = True
                break
        except Exception as e:
            logging.error(f"[ERROR] Failed to check if enemy is aiming: {e}")

    agent.meta_info = {
        "closest_enemy_angle": float(closest_angle),
        "closest_enemy_dist": float(closest_dist),
        "target_id": target_id,
        "is_aimed_at": bool(is_targeted),
    }

