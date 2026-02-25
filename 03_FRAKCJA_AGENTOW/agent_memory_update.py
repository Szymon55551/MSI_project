import math
import logging

def update_internal_state(agent, status, sensors):
    update_static_info(agent, status)
    update_dynamic_info(agent, status, sensors)
    update_map_memory(agent, sensors)
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
        "barrel_spin_rate": agent._get(status, "_barrel_spin_rate", 30),
        "heading_spin_rate": agent._get(status, "_heading_spin_rate", 2),
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
        "enemies_remaining": getattr(agent, "enemies_remaining", 0),

        "visible_tanks": seen_tanks,
        "visible_enemies": [t for t in seen_tanks if t.get("team") != my_team],
        "visible_friends": [t for t in seen_tanks if t.get("team") == my_team],

        "visible_obstacles": agent._get(sensors, "seen_obstacles", []) or [],
        "visible_terrains": agent._get(sensors, "seen_terrains", []) or [],
        "visible_powerups": agent._get(sensors, "seen_powerups", []) or [],
    }

def update_map_memory(agent, sensors):
    MAX_CELL = 19
    
    if not hasattr(agent, "virtual_map") or len(agent.virtual_map) != 400:
        agent.virtual_map = {(x, y): {"type": 0, "tick": 0, "penalty": 0.0} for x in range(20) for y in range(20)}

    def set_cell(x, y, c_type):
        if 0 <= x <= MAX_CELL and 0 <= y <= MAX_CELL:
            agent.virtual_map[(x, y)]["type"] = c_type
            agent.virtual_map[(x, y)]["tick"] = agent.current_tick

    # Clear previous dynamic obstacles (Type 9) back to safe terrain (Type 1)
    for coord, data in agent.virtual_map.items():
        if data["type"] == 9:
            data["type"] = 1

    # 1. Update Static Terrains
    for t in (agent._get(sensors, "seen_terrains", []) or []):
        pos = t.get("position", {}) or {}
        bx, by = int(float(pos.get("x", 0.0)) // 10), int(float(pos.get("y", 0.0)) // 10)
        
        t_raw = t.get("type", t.get("terrain_type", t.get("_terrain_type", "Grass")))
        if isinstance(t_raw, dict): t_raw = t_raw.get("name", "Grass")
        t_str = str(t_raw).upper()
        
        speed = float(t.get("speed_modifier", t.get("movement_speed_modifier", 1.0)))
        dmg = int(t.get("dmg", t.get("deal_damage", 0)))
        
        existing = agent.virtual_map.get((bx, by), {}).get("type", 0)
        if existing not in [3, 4]: 
            if "WATER" in t_str or (speed <= 0.75 and dmg > 0): set_cell(bx, by, 2)
            elif "SWAMP" in t_str or "MUD" in t_str or speed <= 0.75: set_cell(bx, by, 6)
            elif "POTHOLE" in t_str or "DANGER" in t_str or dmg > 0: set_cell(bx, by, 5)
            else: set_cell(bx, by, 1)

    # 2. Update Obstacles
    seen_obs_tiles = set()
    for ob in (agent._get(sensors, "seen_obstacles", []) or []):
        pos = ob.get("position", {}) or {}
        bx, by = int(float(pos.get("x", 0.0)) // 10), int(float(pos.get("y", 0.0)) // 10)
        seen_obs_tiles.add((bx, by))
        
        is_tree = "TREE" in str(ob.get("type", ob.get("obstacle_type", "WALL"))).upper()
        set_cell(bx, by, 4 if is_tree else 3)

    # 3. Inject Dynamic Obstacles (Other Tanks)
    my_id = agent.static_info.get("id")
    for tank in (agent.dynamic_info.get("visible_tanks", []) or []):
        if agent._get(tank, "id") != my_id:
            pos = agent._get(tank, "position", {})
            tx, ty = int(float(agent._get(pos, "x", 0.0)) // 10), int(float(agent._get(pos, "y", 0.0)) // 10)
            # Do not overwrite hard walls, but overwrite passable terrain
            if agent.virtual_map.get((tx, ty), {}).get("type", 0) != 3:
                set_cell(tx, ty, 9)

    # 4. Clean up destructibles (Trees) out of FOV
    my_pos = agent.dynamic_info.get("position", {})
    mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
    vr = float(agent.static_info.get("vision_range", 0.0))
    
    if vr > 0:
        min_x, max_x = max(0, int((mx - vr)//10)), min(19, int((mx + vr)//10))
        min_y, max_y = max(0, int((my - vr)//10)), min(19, int((my + vr)//10))
        
        for px in range(min_x, max_x + 1):
            for py in range(min_y, max_y + 1):
                if agent.virtual_map.get((px, py), {}).get("type") == 4: 
                    tcx, tcy = (px * 10) + 5.0, (py * 10) + 5.0
                    if agent._is_in_vision(tcx, tcy) and (px, py) not in seen_obs_tiles:
                        set_cell(px, py, 1)

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

    closest_angle, closest_dist, target_id, is_targeted = 10.0, 9999.0, None, False

    if not my_pos or not visible_enemies:
        agent.meta_info = {
            "closest_enemy_angle": closest_angle, "closest_enemy_dist": closest_dist,
            "target_id": target_id, "is_aimed_at": is_targeted,
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
        dx, dy = float(agent._get(cpos, "x", 0.0)) - my_x, float(agent._get(cpos, "y", 0.0)) - my_y
        desired_angle = math.degrees(math.atan2(dy, dx))
        closest_angle = (desired_angle - my_angle + 180.0) % 360.0 - 180.0
    except Exception: pass

    for enemy in visible_enemies:
        try:
            epos = agent._get(enemy, "position", {}) or {}
            ex, ey = float(agent._get(epos, "x", 0.0)), float(agent._get(epos, "y", 0.0))
            angle_to_me = math.degrees(math.atan2(my_y - ey, my_x - ex))
            aim_diff = (angle_to_me - float(agent._get(enemy, "barrel_angle", 0.0) or 0.0) + 180.0) % 360.0 - 180.0
            if abs(aim_diff) < 15.0:
                is_targeted = True
                break
        except Exception: pass

    agent.meta_info = {
        "closest_enemy_angle": float(closest_angle), "closest_enemy_dist": float(closest_dist),
        "target_id": target_id, "is_aimed_at": bool(is_targeted),
    }