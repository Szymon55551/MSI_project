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
    # Initialize with default penalty of 0.0
    if not hasattr(agent, "virtual_map") or len(agent.virtual_map) < 40000:
        agent.virtual_map = {(x, y): {"type": 0, "tick": 0, "penalty": 0.0} for x in range(200) for y in range(200)}
        
    MAX_CELL = 199
    
    # --- NEW: Precomputation helper ---
    def apply_penalty(cx, cy, is_danger):
        penalty_value = 5.0 if is_danger else 10.0
        for p_dx in range(-2, 3):
            for p_dy in range(-2, 3):
                if p_dx == 0 and p_dy == 0: continue
                pen_x, pen_y = cx + p_dx, cy + p_dy
                if 0 <= pen_x <= MAX_CELL and 0 <= pen_y <= MAX_CELL:
                    target = agent.virtual_map.get((pen_x, pen_y))
                    # Apply penalty only to traversable terrain
                    if target and target.get("type", 0) not in [3, 5]:
                        current_penalty = target.get("penalty", 0.0)
                        target["penalty"] = min(30.0, current_penalty + penalty_value)

    def set_cell(x, y, c_type):
        if 0 <= x <= MAX_CELL and 0 <= y <= MAX_CELL:
            agent.virtual_map[(x, y)]["type"] = c_type
            agent.virtual_map[(x, y)]["tick"] = agent.current_tick
            # --- NEW: Trigger penalty for danger ---
            if c_type == 5:
                apply_penalty(x, y, is_danger=True)

    # 1. Update Terrains
    for t in (agent._get(sensors, "seen_terrains", []) or []):
        pos = t.get("position", {}) or {}
        cx, cy = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        
        bx, by = int(cx // 10) * 10, int(cy // 10) * 10
        t_raw = t.get("type", t.get("terrain_type", t.get("_terrain_type", "Grass")))
        if isinstance(t_raw, dict): t_raw = t_raw.get("name", "Grass")
        t_str = str(t_raw).upper()
        
        speed = float(t.get("speed_modifier", t.get("movement_speed_modifier", t.get("_movement_speed_modifier", 1.0))))
        dmg = int(t.get("dmg", t.get("deal_damage", t.get("_deal_damage", 0))))
        
        is_water, is_danger, is_mud = False, False, False
        if "WATER" in t_str: is_water = True
        elif "SWAMP" in t_str or "MUD" in t_str: is_mud = True
        elif "POTHOLE" in t_str or "DANGER" in t_str: is_danger = True
        elif speed <= 0.75 and dmg > 0: is_water = True
        elif dmg > 0: is_danger = True
        elif speed <= 0.75: is_mud = True

        for dx in range(10):
            for dy in range(10):
                px, py = bx + dx, by + dy
                if 0 <= px <= MAX_CELL and 0 <= py <= MAX_CELL:
                    existing = agent.virtual_map.get((px, py), {}).get("type", 0)
                    if existing not in [3, 4]: 
                        if is_water:
                            set_cell(px, py, 2)
                        elif is_mud:
                            set_cell(px, py, 6) # Mud type
                        elif is_danger:
                            # 4x4 red core, otherwise green safe zone
                            if 3 <= dx <= 6 and 3 <= dy <= 6:
                                set_cell(px, py, 5) 
                            else:
                                set_cell(px, py, 1)
                        else:
                            set_cell(px, py, 1)

    # 2. Update Obstacles (Persistent Walls)
    seen_obs_tiles = set()
    for ob in (agent._get(sensors, "seen_obstacles", []) or []):
        pos = ob.get("position", {}) or {}
        cx, cy = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        bx, by = int(cx // 10) * 10, int(cy // 10) * 10
        seen_obs_tiles.add((bx, by))
        
        o_raw = ob.get("type", ob.get("obstacle_type", "WALL"))
        is_tree = "TREE" in str(o_raw).upper()

        if is_tree:
            for dx in range(10):
                for dy in range(10):
                    set_cell(bx + dx, by + dy, 4)
        else:
            # WALLS: Persistent update. Do not clear once set.
            for dx in range(-3, 13): # Buffer inflation
                for dy in range(-3, 13):
                    px, py = bx + dx, by + dy
                    if 0 <= px <= MAX_CELL and 0 <= py <= MAX_CELL:
                        # Only update if not already a wall to save cycles
                        if agent.virtual_map[(px, py)].get("type", 0) != 3:
                            agent.virtual_map[(px, py)]["type"] = 3
                            agent.virtual_map[(px, py)]["tick"] = agent.current_tick
                            # --- NEW: Trigger penalty for wall ---
                            apply_penalty(px, py, is_danger=False)

    # 3. Clean up destructibles
    my_pos = agent.dynamic_info.get("position", {})
    mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
    vr = float(agent.static_info.get("vision_range", 0.0))
    
    if vr > 0:
        cells_to_clear = []
        min_x = max(0, int(mx - vr))
        max_x = min(199, int(mx + vr))
        min_y = max(0, int(my - vr))
        max_y = min(199, int(my + vr))
        
        for px in range(min_x, max_x + 1):
            for py in range(min_y, max_y + 1):
                if agent.virtual_map.get((px, py), {}).get("type") == 4: 
                    bx, by = (px // 10) * 10, (py // 10) * 10
                    tcx, tcy = bx + 5.0, by + 5.0
                    if agent._is_in_vision(tcx, tcy) and (bx, by) not in seen_obs_tiles:
                        cells_to_clear.append((px, py))
        
        for (px, py) in cells_to_clear:
            agent.virtual_map[(px, py)]["type"] = 1
            agent.virtual_map[(px, py)]["tick"] = agent.current_tick

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