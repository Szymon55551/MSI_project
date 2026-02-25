
import logging
import math
import json
import os 

################################################################################################
######################## DEBUG - I WIZULZIAJCA, FUNKCJE PISANE PRZEZ AI ########################
################################################################################################
def terrain_tiles_from_seen(agent):
    tiles = []
    seen_terrains = agent.dynamic_info.get("visible_terrains", [])
    for t in seen_terrains:
        pos = t.get("position", {})
        cx = float(pos.get("x", 0.0))
        cy = float(pos.get("y", 0.0))
        tile_x = int(cx // agent.TILE_SIZE)
        tile_y = int(cy // agent.TILE_SIZE)
        tiles.append((tile_x, tile_y))
    # unikalne
    tiles = list(dict.fromkeys(tiles))
    return tiles

def fov_debug_payload(agent):
    """
    Zwraca dane debug pola widzenia:
    - wedge (dwa promienie graniczne)
    - zbiór sub-komórek w FOV (w promieniu + w kącie)
    """
    pos = agent.dynamic_info.get("position") or {}
    mx = float(pos.get("x", 0.0))
    my = float(pos.get("y", 0.0))

    vr = float(agent.static_info.get("vision_range", 0.0))
    va = float(agent.static_info.get("vision_angle", 0.0))
    heading = float(agent.dynamic_info.get("heading", 0.0))

    if vr <= 0 or va <= 0:
        return {
            "tick": agent.current_tick,
            "origin": {"x": mx, "y": my},
            "vr": vr,
            "va": va,
            "heading": heading,
            "cells": [],
            "rays": [],
            "bbox": None,
        }

    # bbox w sub-grid ograniczony do vision_range
    cx, cy = agent._cell_from_xy(mx, my)
    r_cells = int(math.ceil(vr / agent.CELL_SIZE))

    minx, maxx = cx - r_cells, cx + r_cells
    miny, maxy = cy - r_cells, cy + r_cells

    # granice kąta
    half = va * 0.5
    ang_left = (heading - half) % 360.0
    ang_right = (heading + half) % 360.0

    def in_angle(err_deg: float) -> bool:
        return abs(err_deg) <= half

    cells = []
    vr2 = vr * vr

    # iteruj po sub-komórkach w bbox i filtruj dystans + kąt
    for ix in range(minx, maxx + 1):
        for iy in range(miny, maxy + 1):
            wx, wy = agent._cell_center((ix, iy))
            dx = wx - mx
            dy = wy - my
            d2 = dx*dx + dy*dy
            if d2 > vr2:
                continue

            ang_to = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            err = agent._angle_diff(ang_to, heading)
            if not in_angle(err):
                continue

            cells.append([ix, iy])

    # promienie graniczne (2 linie) — do rysowania klinu
    def ray_endpoint(angle_deg: float):
        rad = math.radians(angle_deg)
        return {"x": mx + math.cos(rad) * vr, "y": my + math.sin(rad) * vr}

    rays = [
        {"from": {"x": mx, "y": my}, "to": ray_endpoint(ang_left)},
        {"from": {"x": mx, "y": my}, "to": ray_endpoint(ang_right)},
    ]

    return {
        "tick": agent.current_tick,
        "origin": {"x": mx, "y": my},
        "vr": vr,
        "va": va,
        "heading": heading,
        "cells": cells,                 # lista sub-komórek w FOV
        "bbox": [minx, miny, maxx, maxy],
        "rays": rays,                   # 2 promienie graniczne
    }
    
def save_state_to_file(agent):
    logging.debug(f"save_state_to_file")
    """Save the current state of the agent to a JSON file."""

    state = {
        "current_tick": agent.current_tick,
        "static_info": agent.static_info,
        "dynamic_info": agent.dynamic_info,
        "meta_info": agent.meta_info,
        "memory": agent.memory,
        
        "agent_role": {
            "type": agent.tank_type,            
            "enabled": bool(agent.tank_type_used)
        },

        "debug": {
            "goal_cell": agent.debug_goal_cell,
            "path": agent.debug_path,
            "path_index": agent.debug_path_index,
            "seen_terrain_tiles": terrain_tiles_from_seen(agent),
            "my_pos": agent.dynamic_info.get("position", None),
            "fov": fov_debug_payload(agent),
            "graph_nodes": getattr(agent, "debug_graph_nodes", None),
            "start_cell": getattr(agent, "debug_start_cell", None),
            "goal_cell_candidate": getattr(agent, "debug_goal_cell_candidate", None),
        }
    }
    
    tmp_path = agent.state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    os.replace(tmp_path, agent.state_path)
    
################################################################################################
######################## DEBUG - I WIZULZIAJCA, FUNKCJE PISANE PRZEZ AI ########################
################################################################################################
