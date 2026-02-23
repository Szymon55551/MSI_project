import math
import logging

def update_internal_state(agent, status, sensors):
    update_static_info(agent, status)
    update_dynamic_info(agent, status, sensors)
    update_map_memory(agent, sensors)
    update_obstacle_memory(agent, sensors)
    update_enemy_memory(agent)
    prune_stale_memory(agent)

def update_static_info(agent, status):
    agent.static_info = {
        "id": agent._get(status, "_id", "MISSING"),
        "team": agent._get(status, "_team", "MISSING"),
        "vision_range": float(agent._get(status, "_vision_range", 120.0)),
        "top_speed": float(agent._get(status, "_top_speed", 1.0)),
        "max_hp": float(agent._get(status, "_max_hp", 100.0)),
        "barrel_spin_rate": float(agent._get(status, "_barrel_spin_rate", 30.0)),
        "heading_spin_rate": float(agent._get(status, "_heading_spin_rate", 2.0)),
    }

def update_dynamic_info(agent, status, sensors):
    seen_tanks = agent._get(sensors, "seen_tanks", []) or []
    my_team = agent.static_info.get("team")

    agent.dynamic_info = {
        "hp": agent._get(status, "hp", 0),
        "position": agent._get(status, "position", {"x": 0, "y": 0}),
        "heading": agent._get(status, "heading", 0),
        "barrel_angle": agent._get(status, "barrel_angle", 0),
        "ammo": agent._get(status, "ammo", {}),
        "visible_enemies": [t for t in seen_tanks if t.get("team") != my_team],
        "visible_friends": [t for t in seen_tanks if t.get("team") == my_team],
        "visible_obstacles": agent._get(sensors, "seen_obstacles", []),
        "visible_terrains": agent._get(sensors, "seen_terrains", []),
        "visible_powerups": agent._get(sensors, "seen_powerups", []),
    }

def update_map_memory(agent, sensors):
    now = int(agent.current_tick)
    for t in (agent._get(sensors, "seen_terrains", []) or []):
        cx, cy = t["position"]["x"], t["position"]["y"]
        for cell in agent._stamp_tile_center_to_subcells(cx, cy):
            agent.map_memory[cell] = {
                "dmg": int(t.get("dmg", 0)), 
                "speed": float(t.get("speed_modifier", 1.0)), 
                "type": t.get("type", ""),
                "last_seen_tick": now
            }

def update_obstacle_memory(agent, sensors):
    now = int(agent.current_tick)
    for ob in (agent._get(sensors, "seen_obstacles", []) or []):
        cx, cy = ob["position"]["x"], ob["position"]["y"]
        is_destructible = ob.get("is_destructible", False)
        for cell in agent._stamp_tile_center_to_subcells(cx, cy):
            agent.obstacle_memory[cell] = {
                "tick": now, 
                "destructible": is_destructible
            }

def prune_stale_memory(agent):
    now = int(agent.current_tick)
    limit = 160
    
    agent.map_memory = {
        k: v for k, v in agent.map_memory.items() 
        if (now - v.get("last_seen_tick", 0)) <= limit
    }
    
    agent.obstacle_memory = {
        k: v for k, v in agent.obstacle_memory.items() 
        if isinstance(v, dict) and (now - v.get("tick", 0)) <= limit
    }

def update_enemy_memory(agent):
    now = int(agent.current_tick)
    for enemy in agent.dynamic_info.get("visible_enemies", []):
        eid = enemy.get("id")
        agent.memory[eid] = {"pos": enemy["position"], "tick": now}