import random
import argparse
import sys
import os
import math
import json
import heapq
from scan_strategy import scan_strategy
from agent_memory_update import update_internal_state
from debug_agent import save_state_to_file
from Attack import mode_attack
from Escape import mode_escape
from Fuzzy_Controller import FuzzyCombatDecider
from d_star_lite import DStarLite

RECENT_TICKS_GRAPH = 150
MAX_PATH_STUCK_TICKS = 300
NO_MOVE_LIMIT_TICKS = 200

ESCAPE_COMMIT_TICKS = 300      
ESCAPE_BACK_TICKS   = 150       
ESCAPE_REPLAN_EVERY = 120     

current_dir = os.path.dirname(os.path.abspath(__file__))
controller_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA', 'controller')
sys.path.insert(0, controller_dir)

parent_dir = os.path.join(os.path.dirname(current_dir), '02_FRAKCJA_SILNIKA')
sys.path.insert(0, parent_dir)

from typing import Dict, Any
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

TILE_SIZE = 10.0
SUBDIV = 3
EVAL_EVERY = 1000

FORCE_REPLAN_EVERY = 1000
IMPROVEMENT_MARGIN = 0.3
MIN_ABS_IMPROVEMENT = 20.0

from dataclasses import dataclass, field
import logging
LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_debug.log")

def make_grid_helpers(tile_size, subdiv):
    if subdiv < 1:
        raise ValueError("subdiv must be >= 1")

    cell_size = tile_size / float(subdiv)
    tile_half = tile_size / 2.0

    def cell_from_xy(x, y):
        return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))

    def cell_center(cell):
        return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)

    def stamp_tile_center_to_subcells(cx, cy):
        cells = []
        start_x = cx - tile_half
        start_y = cy - tile_half

        for i in range(subdiv):
            sub_x = start_x + (i + 0.5) * cell_size
            for j in range(subdiv):
                sub_y = start_y + (j + 0.5) * cell_size
                cells.append(cell_from_xy(sub_x, sub_y))
        return cells

    return cell_size, cell_from_xy, stamp_tile_center_to_subcells, cell_center

class ActionCommand(BaseModel):
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

@dataclass
class GridNode:
    cell: tuple                     
    world: tuple                
    dmg: int
    speed: float
    blocked: bool
    dist_to_me: float
    neighbors: list = field(default_factory=list)  
    is_risk: bool = False

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

class RandomAgent:
    def __init__(self, name: str = "TestBot", modifier=None):
        self.name = name
        self.modifier = modifier
        self.is_destroyed = False
        self.current_tick = 0

        self.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, f"agent_state_{self.name}.json")

        self.mode = "search"              

        self.attack_commit_ticks = 100
        self.attack_until_tick = -10_000
        self.attack_orbit_cell = None
        self.attack_orbit_last_pick_tick = -10_000
        self.attack_orbit_repick_every = 50  

        self.enemy_target_id = None
        self.enemy_target_cell = None
        self.enemy_target_last_seen_tick = -10_000
        self.enemy_forget_after = 80
        self.attack_desired_range = 8.0

        self.escape_commit_ticks = ESCAPE_COMMIT_TICKS
        self.escape_until_tick = -10_000
        self.escape_back_ticks = ESCAPE_BACK_TICKS
        self.escape_enter_tick = -10_000 

        self.escape_target_cell = None
        self.escape_target_last_pick_tick = -10_000
        self.escape_forget_after = 120

        self.powerup_target_cell = None
        self.powerup_target_last_seen_tick = -10_000
        self.powerup_target_acquired_tick = -10_000
        self.powerup_forget_after = 200
        self.powerup_commit_ticks = 200
        self.powerup_switch_ratio = 0.80

        self.last_world_pos = None
        self.no_move_ticks = 0
        self.force_change_goal = False

        self.MIN_MOVE_EPS = 0.05
        self.NO_MOVE_LIMIT_TICKS = 100 

        self.last_eval_tick = -10_000
        self.last_forced_replan_tick = -10_000

        self.current_goal_cell = None
        self.current_path_cost = None

        self.path_to_follow = None
        self.path_index = 0
        self.path_stuck_ticks = 0
        self.reached_points = None

        self.debug_goal_cell = None
        self.debug_path = None
        self.debug_path_index = 0

        self.movement_list = []
        self.last_angle = 0

        self.static_info = {}
        self.dynamic_info = {}
        self.memory = {}
        self.map_memory = {}
        self.obstacle_memory = {}

        self.SECTOR_SIZE = 30.0
        self.visited_sectors = {}  

        self.TILE_SIZE = 10.0
        self.SUBDIV = SUBDIV
        (
            self.CELL_SIZE,
            self._cell_from_xy,
            self._stamp_tile_center_to_subcells,
            self._cell_center
        ) = make_grid_helpers(self.TILE_SIZE, self.SUBDIV)
        
        self.combat_decider = FuzzyCombatDecider()
        self.d_star_planner = DStarLite()
    
    def _get(self, d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)

    def _commit_mode(self, mode: str, enemy) -> None:
        mode = (mode or "").lower()
        if mode == "attack":
            self._enter_attack(enemy)
            self.mode = "attack"
        elif mode == "escape":
            self._enter_escape(enemy)
            self.mode = "escape"
        else:
            raise ValueError(f"Unknown mode to commit: {mode}")

    def _maybe_reconsider_combat_mode(self, enemy) -> bool:
        return False
    
    def _choose_combat_mode(self, enemy) -> str:
        return self.combat_decider.decide(self)
    
    def _enter_attack(self, enemy):
        now = self.current_tick
        self.attack_until_tick = max(self.attack_until_tick, now + self.attack_commit_ticks)
        return

    def _enter_escape(self, enemy):
        now = self.current_tick
        was_active = (now <= self.escape_until_tick)
        self.escape_until_tick = max(self.escape_until_tick, now + self.escape_commit_ticks)

        if not was_active:
            self.escape_enter_tick = now
            self._start_escape(enemy) 
        else:
            if self.escape_target_cell is None and enemy is not None:
                self._start_escape(enemy)

    def _select_escape_goal_cell(self, nodes, enemy_cell, max_candidates=5):
        if not nodes or enemy_cell is None:
            return None

        node_by_cell = {n.cell: n for n in nodes}
        if enemy_cell not in node_by_cell:
            ex, ey = self._cell_center(enemy_cell)
            best = None
            best_d2 = 1e30
            for c, n in node_by_cell.items():
                wx, wy = n.world
                d2 = (wx - ex) ** 2 + (wy - ey) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = c
            enemy_cell = best

        if enemy_cell is None:
            return None

        safe = [
            n for n in nodes
            if (not n.blocked) and (n.dmg == 0) and (not getattr(n, "is_risk", False))
        ]
        if not safe:
            safe = [n for n in nodes if not n.blocked]
            if not safe:
                return None

        ex, ey = self._cell_center(enemy_cell)
        safe.sort(key=lambda n: (n.world[0] - ex) ** 2 + (n.world[1] - ey) ** 2, reverse=True)

        pos = self.dynamic_info.get("position") or {}
        sx = float(pos.get("x", 0.0))
        sy = float(pos.get("y", 0.0))
        start_cell = self._cell_from_xy(sx, sy)

        for n in safe[:max_candidates]:
            path = self.d_star_planner.update_graph_and_path(start_cell, n.cell, nodes)
            if path and len(path) >= 2:
                return n.cell

        return None
    
    def _start_escape(self, enemy):
        now = self.current_tick
        self.escape_until_tick = max(self.escape_until_tick, now + self.escape_commit_ticks)

        pos = self._get(enemy, "position", {}) or {}
        ex = float(self._get(pos, "x", 0.0))
        ey = float(self._get(pos, "y", 0.0))
        enemy_cell = self._cell_from_xy(ex, ey)

        visible_obstacles = self.dynamic_info.get("visible_obstacles") or []
        visible_terrains  = self.dynamic_info.get("visible_terrains")  or []
        nodes = self._divide_seen_area(visible_obstacles, visible_terrains)

        goal = self._select_escape_goal_cell(nodes, enemy_cell)
        if goal is not None:
            self.escape_target_cell = goal
            self.escape_target_last_pick_tick = now

        return enemy_cell
    
    def _reload_ready(self) -> bool:
        rt = self.dynamic_info.get("reload_timer", None)
        if rt is None:
            rt = self.dynamic_info.get("current_reload_progress", 0)
        try:
            return int(rt) <= 0
        except:
            return True

    def _can_fire_at_enemy_with_range(self, enemy, ammo_name: str, aim_tolerance_deg=5.0, debug=False) -> bool:
        if enemy is None:
            return False
        if not self._reload_ready():
            return False

        dist_world = float(self._get(enemy, "distance", 1e9))  
        range_world = self._ammo_range_world(ammo_name)

        if dist_world > range_world:
            return False

        my_pos = self.dynamic_info.get("position") or {}
        enemy_pos = self._get(enemy, "position", {}) or {}
        mx = float(self._get(my_pos, "x", 0.0))
        my = float(self._get(my_pos, "y", 0.0))
        ex = float(self._get(enemy_pos, "x", 0.0))
        ey = float(self._get(enemy_pos, "y", 0.0))

        desired_abs = (math.degrees(math.atan2(ey - my, ex - mx)) + 360.0) % 360.0
        heading_abs = float(self.dynamic_info.get("heading", 0.0))
        barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
        barrel_abs = (heading_abs + barrel_rel) % 360.0

        err = self._angle_diff(desired_abs, barrel_abs)
        return abs(err) <= aim_tolerance_deg
    
    def _closest_visible_powerup(self):
        powerups = self.dynamic_info.get("visible_powerups", [])
        if not powerups:
            return None

        my_pos = self.dynamic_info.get("position") or {}
        mx = float(my_pos.get("x", 0.0))
        my = float(my_pos.get("y", 0.0))

        def dist2(p):
            pos = p.get("position", {}) or {}
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            return (x - mx) ** 2 + (y - my) ** 2

        pu = min(powerups, key=dist2)
        pos = pu.get("position", {}) or {}
        px = float(pos.get("x", 0.0))
        py = float(pos.get("y", 0.0))
        return pu, (px, py), dist2(pu)

    def _is_in_vision(self, x: float, y: float) -> bool:
        pos = self.dynamic_info.get("position") or {}
        mx = float(pos.get("x", 0.0))
        my = float(pos.get("y", 0.0))

        vr = float(self.static_info.get("vision_range", 0.0))
        va = float(self.static_info.get("vision_angle", 0.0))

        dx = x - mx
        dy = y - my
        dist = math.hypot(dx, dy)

        if vr > 0.0 and dist > vr:
            return False

        heading = float(self.dynamic_info.get("heading", 0.0))
        ang_to = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        err = self._angle_diff(ang_to, heading)

        return abs(err) <= (va * 0.5)
    
    def _is_target_powerup_visible(self, target_cell) -> bool:
        if target_cell is None:
            return False

        for p in (self.dynamic_info.get("visible_powerups", []) or []):
            pos = p.get("position", {}) or {}
            px = float(pos.get("x", 0.0))
            py = float(pos.get("y", 0.0))
            if self._cell_from_xy(px, py) == target_cell:
                return True
        return False
    
    def _select_powerup_goal_cell(self):
        now = self.current_tick
        if self._is_target_powerup_visible(self.powerup_target_cell):
            self.powerup_target_last_seen_tick = now

        self._maybe_forget_powerup_target()

        closest = self._closest_visible_powerup()
        if closest is None:
            return self.powerup_target_cell

        _, (px, py), d2_new = closest
        new_cell = self._cell_from_xy(px, py)

        if self.powerup_target_cell is None:
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = now 
            self.powerup_target_acquired_tick = now
            return self.powerup_target_cell

        if new_cell == self.powerup_target_cell:
            return self.powerup_target_cell

        my_pos = self.dynamic_info.get("position") or {}
        mx = float(my_pos.get("x", 0.0))
        my = float(my_pos.get("y", 0.0))

        cx, cy = self._cell_center(self.powerup_target_cell)
        d2_cur = (cx - mx) ** 2 + (cy - my) ** 2

        committed = (now - self.powerup_target_acquired_tick) < self.powerup_commit_ticks
        significantly_closer = (d2_new <= d2_cur * (self.powerup_switch_ratio ** 2))

        if (not committed) and significantly_closer:
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = now
            self.powerup_target_acquired_tick = now

        return self.powerup_target_cell
    
    def _path_cost(self, path, node_by_cell):
        if not path or len(path) < 2:
            return float("inf")
        cost = 0.0
        for cell in path[1:]:
            n = node_by_cell.get(cell)
            if n is None:
                return float("inf")  
            if n.blocked:
                return float("inf")
            cost += float(n.dmg)
        return cost
    
    def _extend_path_one_step(self, path, node_by_cell):
        if not path or len(path) < 2:
            return path

        last = path[-1]
        prev = path[-2]
        dx = last[0] - prev[0]
        dy = last[1] - prev[1]

        if (dx, dy) not in ((1,0), (-1,0), (0,1), (0,-1)):
            return path

        extra = (last[0] + dx, last[1] + dy)
        n = node_by_cell.get(extra)
        if n is None or n.blocked:
            return path
        if extra not in node_by_cell[last].neighbors:
            return path
        return path + [extra]
    
    def _plan_best_path(self, nodes, targets, max_targets=3):
        node_by_cell = {n.cell: n for n in nodes}

        pos = self.dynamic_info.get("position") or {}
        sx = float(pos.get("x", 0.0))
        sy = float(pos.get("y", 0.0))
        start_cell = self._cell_from_xy(sx, sy)

        best_path = None
        best_goal = None
        best_cost = float("inf")

        for goal in targets[:max_targets]:
            path = self.d_star_planner.update_graph_and_path(start_cell, goal, nodes)
            
            if not path:
                continue
            c = self._path_cost(path, node_by_cell)
            if c < best_cost:
                best_cost = c
                best_path = path
                best_goal = goal

        if best_path:
            best_path = self._extend_path_one_step(best_path, node_by_cell)

        return best_path, best_goal, best_cost
            
    def next_or_first(self, arr, value):
        try:
            index = arr.index(value)
        except ValueError:
            return 1  
        
        if index + 1 < len(arr):
            return arr[index + 1]
        else:
            return arr[0]

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        if self.movement_list:
            current_angle = self.next_or_first(self.movement_list, self.last_angle)
            self.last_angle = current_angle

        self.current_tick = current_tick
        self.enemies_remaining = enemies_remaining
        update_internal_state(self, my_tank_status, sensor_data)
        
        pos = self.dynamic_info.get("position") or {}
        px, py = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        
        self.visited_sectors[(int(px // self.SECTOR_SIZE), int(py // self.SECTOR_SIZE))] = self.current_tick

        if self.last_world_pos:
            lx, ly = self.last_world_pos
            if math.hypot(px - lx, py - ly) < self.MIN_MOVE_EPS:
                self.no_move_ticks += 1
            else:
                self.no_move_ticks = 0

        if self.no_move_ticks > self.NO_MOVE_LIMIT_TICKS:
            self.force_change_goal = True
            self.path_to_follow = None
            self.current_goal_cell = None
            self._generate_new_macro_goal(px, py)

        # Baseline Process Action Generation
        action = self._process_action()
        
        # --- TREE BLASTING LOGIC ---
        me_cell = self._cell_from_xy(px, py)
        near_tree = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                c = (me_cell[0] + dx, me_cell[1] + dy)
                info = self.obstacle_memory.get(c)
                if isinstance(info, dict) and info.get("destructible"):
                    near_tree = True
                    break
            if near_tree: break

        if self.no_move_ticks > 8 and near_tree and self.mode not in ["attack", "escape"]:
            heading = float(self.dynamic_info.get("heading", 0.0))
            barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
            err = self._angle_diff(0.0, barrel_rel)
            barrel_spin = float(self.static_info.get("barrel_spin_rate", 30.0))
            
            action.barrel_rotation_angle = self._clamp(err, -barrel_spin, barrel_spin)
            if abs(err) < 15.0:
                action.should_fire = True
                action.ammo_to_load = "LIGHT"

        # --- TACTICAL UNSTUCK OVERRIDE ---
        if not hasattr(self, "unstuck_until_tick"):
            self.unstuck_until_tick = -10_000
            self.current_unstuck_threshold = random.randint(90, 160)

        if self.no_move_ticks > getattr(self, "current_unstuck_threshold", 150):
            maneuver_duration = random.randint(150, 200)
            self.unstuck_until_tick = self.current_tick + maneuver_duration  
            self.no_move_ticks = 0 
            self.current_unstuck_threshold = random.randint(100, 180)
            
            top_speed = float(self.static_info.get("top_speed", 0.0))
            heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
            
            self.unstuck_move = random.choice([-1.0, 1.0]) * 0.7 * top_speed
            self.unstuck_rot = random.choice([-1.0, 1.0]) * heading_spin

        if self.current_tick <= self.unstuck_until_tick:
            action.move_speed = getattr(self, "unstuck_move", 0.0)
            action.heading_rotation_angle = getattr(self, "unstuck_rot", 0.0)

        self.last_world_pos = (px, py)
        self.last_commanded_speed = action.move_speed

        if self.current_tick < 20 or self.current_tick % 60 == 0:
            save_state_to_file(self)

        return action
        
    def _angle_diff(self, target_deg: float, current_deg: float) -> float:
        return (target_deg - current_deg + 180) % 360 - 180

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
    
    def _divide_seen_area(self, visible_obstacles, visible_terrains):
        now = int(self.current_tick)
        if hasattr(self, "_last_graph_tick") and hasattr(self, "_cached_nodes"):
            if now - self._last_graph_tick < 5:
                return self._cached_nodes
        
        my_pos = self.dynamic_info.get("position") or {"x": 0.0, "y": 0.0}
        me_x = float(my_pos.get("x", 0.0))
        me_y = float(my_pos.get("y", 0.0))
        recent_cutoff = now - int(RECENT_TICKS_GRAPH)

        terrain_info = {
            cell: info for cell, info in self.map_memory.items()
            if int(info.get("last_seen_tick", -10_000)) >= recent_cutoff
        }

        blocked_cells = set()
        destructible_cells = set()
        
        for (cx, cy), info in self.obstacle_memory.items():
            if not isinstance(info, dict):
                continue
            if int(info.get("tick", 0)) >= recent_cutoff:
                if info.get("destructible", False):
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            destructible_cells.add((cx + dx, cy + dy))
                else:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            blocked_cells.add((cx + dx, cy + dy))
                            
        for cell, info in terrain_info.items():
            spd = float(info.get("speed", 1.0))
            t_type = info.get("type", "")
            if spd < 0.2 or t_type == "Water": 
                blocked_cells.add(cell)

        visible_friends = self.dynamic_info.get("visible_friends", [])
        ally_repulsion_cells = {}

        for f in visible_friends:
            fpos = self._get(f, "position", {}) or {}
            fx, fy = float(self._get(fpos, "x", 0.0)), float(self._get(fpos, "y", 0.0))
            acell = self._cell_from_xy(fx, fy)
            
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    dist_sq = dx*dx + dy*dy
                    if dist_sq <= 9:
                        penalty = int(2500 / (dist_sq + 1)) 
                        cell = (acell[0] + dx, acell[1] + dy)
                        ally_repulsion_cells[cell] = ally_repulsion_cells.get(cell, 0) + penalty

        all_relevant_cells = set(terrain_info.keys()) | set(ally_repulsion_cells.keys()) | destructible_cells
        grid_radius = max(8, int((float(self.static_info.get("vision_range", 120.0)) / self.CELL_SIZE) * 0.7))
        me_cell = self._cell_from_xy(me_x, me_y)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                all_relevant_cells.add((me_cell[0] + dx, me_cell[1] + dy))

        nodes_by_cell = {}
        for cell in all_relevant_cells:
            info = terrain_info.get(cell, {"dmg": 0, "speed": 1.0})
            wx, wy = self._cell_center(cell)
            dist = math.hypot(wx - me_x, wy - me_y)

            base_dmg = int(info.get("dmg", 0))
            spd = float(info.get("speed", 1.0))
            
            is_start_node = (cell == me_cell)
            is_hard_block = (cell in blocked_cells or cell[0] < 0 or cell[1] < 0) and not is_start_node
            
            total_hazard = base_dmg + ally_repulsion_cells.get(cell, 0)
            if cell in destructible_cells:
                total_hazard += 5

            nodes_by_cell[cell] = GridNode(
                cell=cell,
                world=(wx, wy),
                dmg=total_hazard,
                speed=spd,
                blocked=is_hard_block,
                dist_to_me=dist,
                is_risk=(total_hazard > 0),
                neighbors=[]
            )

        DIRS = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for (x, y), node in nodes_by_cell.items():
            for dx, dy in DIRS:
                nb = (x + dx, y + dy)
                if nb in nodes_by_cell and not nodes_by_cell[nb].blocked:
                    node.neighbors.append(nb)

        self._last_graph_tick = now
        self._cached_nodes = list(nodes_by_cell.values())
        return self._cached_nodes
        
    def _select_target_point(self, divided_area, current_target=None, top_k=20, forbid_cells=None):
        forbid_cells = forbid_cells or set()

        candidates = [n for n in divided_area if (not n.blocked and n.cell not in forbid_cells)]
        if not candidates:
            return []

        safe_candidates = [n for n in candidates if n.dmg == 0 and not n.is_risk and n.speed >= 0.9]
        if not safe_candidates:
            safe_candidates = [n for n in candidates if n.dmg == 0]
            
        pool = safe_candidates if safe_candidates else candidates

        pool.sort(key=lambda n: n.dist_to_me, reverse=True)
        top = pool[:min(40, len(pool))]

        visible_friends = self.dynamic_info.get("visible_friends", [])
        friend_positions = []
        for f in visible_friends:
            fpos = self._get(f, "position", {}) or {}
            friend_positions.append((float(self._get(fpos, "x", 0.0)), float(self._get(fpos, "y", 0.0))))

        def sort_key(n):
            wx, wy = n.world
            sx = int(wx // getattr(self, "SECTOR_SIZE", 40.0))
            sy = int(wy // getattr(self, "SECTOR_SIZE", 40.0))
            
            visited_penalty = 3000.0 if (sx, sy) in getattr(self, "visited_sectors", {}) else 0.0
            
            hazard_penalty = 0.0
            if n.dmg > 0: hazard_penalty += 5000.0
            if getattr(n, "is_risk", False): hazard_penalty += 150.0 
            if n.speed < 0.9: hazard_penalty += 1000.0
            
            is_current = (n.cell == current_target)
            dist_bonus = 50.0 if is_current else 0.0
            
            ally_repulsion = 0.0
            for fx, fy in friend_positions:
                dist_sq = (wx - fx)**2 + (wy - fy)**2
                if dist_sq < 1.0: dist_sq = 1.0
                ally_repulsion += (12000.0 / dist_sq) 

            macro_penalty = 0.0
            macro_x, macro_y = getattr(self, "macro_target_world", (None, None))
            if macro_x is not None and macro_y is not None:
                dist_to_macro = math.hypot(wx - macro_x, wy - macro_y)
                macro_penalty = dist_to_macro * 4.0 

            return (hazard_penalty + visited_penalty + ally_repulsion + macro_penalty - (n.dist_to_me + dist_bonus))
        
        top.sort(key=sort_key)
        return [n.cell for n in top]

    def _Find_Target_and_Find_Path(self, override_goal_cell=None, forbid_cells=None):
        visible_obstacles = self.dynamic_info.get("visible_obstacles") or []
        visible_terrains  = self.dynamic_info.get("visible_terrains")  or []

        nodes = self._divide_seen_area(visible_obstacles, visible_terrains)
        if not nodes:
            return None, None, None, None

        if override_goal_cell is not None:
            targets = [override_goal_cell]
        else:
            targets = self._select_target_point(
                nodes,
                current_target=self.current_goal_cell,
                forbid_cells=(forbid_cells or set())
            )

        if not targets:
            return None, None, None, None

        path, goal, cost = self._plan_best_path(nodes, targets, max_targets=10)
        node_by_cell = {n.cell: n for n in nodes}
        return path, goal, cost, node_by_cell
  
    def _FollowPath(self):
        if self.path_to_follow and self.path_index >= len(self.path_to_follow) - 1:
            self.path_to_follow = None
            self.current_goal_cell = None
            self.current_path_cost = None
            return 0.0, 0.0
        
        if not self.path_to_follow:
            raise RuntimeError("_FollowPath: path_to_follow is empty/None")

        my_position = self.dynamic_info.get("position")
        my_x = float(my_position.get("x", 0.0))
        my_y = float(my_position.get("y", 0.0))
        
        if self.path_index < 0:
            self.path_index = 0
        if self.path_index >= len(self.path_to_follow) - 1:
            return 0.0, 0.0  
        next_cell = self.path_to_follow[self.path_index + 1]
        next_x, next_y = self._cell_center(next_cell)
        
        dist = math.hypot(next_x - my_x, next_y - my_y)

        reach_threshold = self.CELL_SIZE * 1
        reached_by_distance = dist <= reach_threshold
        reached_by_timeout = self.path_stuck_ticks >= MAX_PATH_STUCK_TICKS
        
        if reached_by_distance:
            self.no_move_ticks = 0
            self.last_dist_to_wp = None

        if reached_by_distance or reached_by_timeout:
            self.path_index += 1
            self.path_stuck_ticks = 0

            if self.path_index >= len(self.path_to_follow) - 1:
                return 0.0, 0.0

            next_cell = self.path_to_follow[self.path_index + 1]
            next_x, next_y = self._cell_center(next_cell)
            dist = math.hypot(next_x - my_x, next_y - my_y)

        self.path_stuck_ticks += 1

        dx = next_x - my_x
        dy = next_y - my_y
        desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        current_heading = float(self.dynamic_info.get("heading", 0.0))
        err = self._angle_diff(desired_heading, current_heading)

        heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
        heading_rotation_angle = self._clamp(err, -heading_spin, heading_spin)

        top_speed = float(self.static_info.get("top_speed", 0.0))

        move_speed = 0.0
        if abs(err) < 5:
            move_speed = top_speed
        else:
            move_speed = 0.5 * top_speed

        return heading_rotation_angle, move_speed
            
    def Follow_Path_With_Modifiers(self, override_goal_cell=None):
        def should_force_replan() -> bool:
            ticks_since_last = self.current_tick - self.last_forced_replan_tick
            if ticks_since_last < 50:
                return False

            if ticks_since_last >= FORCE_REPLAN_EVERY:
                return True

            if self.path_stuck_ticks > MAX_PATH_STUCK_TICKS:
                return True

            return False

        def should_eval() -> bool:
            return (self.current_tick - self.last_eval_tick) >= EVAL_EVERY

        def better_enough(new_cost: float, old_cost: float) -> bool:
            if old_cost is None or old_cost == float("inf"):
                return True
            if new_cost is None or new_cost == float("inf"):
                return False

            if new_cost == 0 and old_cost == 0:
                return False

            if new_cost <= old_cost * (1.0 - IMPROVEMENT_MARGIN):
                return True
            if (old_cost - new_cost) >= MIN_ABS_IMPROVEMENT:
                return True
            return False

        force = should_force_replan()

        if (self.path_to_follow is None) or (self.path_to_follow and self.path_index >= len(self.path_to_follow) - 1):
            force = True

        if override_goal_cell is not None and override_goal_cell != self.current_goal_cell:
            force = True

        forbid_cells = set()
        if self.force_change_goal and self.current_goal_cell is not None:
            forbid_cells.add(self.current_goal_cell)
            force = True

        eval_now = should_eval() or force

        node_by_cell = None
        new_path = None
        new_goal = None
        new_cost = None

        if eval_now:
            self.last_eval_tick = self.current_tick

            new_path, new_goal, new_cost, node_by_cell = self._Find_Target_and_Find_Path(
                override_goal_cell=override_goal_cell,
                forbid_cells=forbid_cells
            )

            self.debug_graph_nodes = None
            self.debug_start_cell = None
            self.debug_goal_cell_candidate = None

            if node_by_cell:
                self.debug_graph_nodes = [
                    {
                        "cell": [int(n.cell[0]), int(n.cell[1])],
                        "blocked": bool(n.blocked),
                        "dmg": int(n.dmg),
                        "speed": float(n.speed),
                        "is_risk": bool(getattr(n, "is_risk", False)),
                    }
                    for n in node_by_cell.values()
                ]

                pos = self.dynamic_info.get("position") or {}
                sx = float(pos.get("x", 0.0))
                sy = float(pos.get("y", 0.0))
                best = None
                best_d2 = 1e30
                for cell, n in node_by_cell.items():
                    wx, wy = n.world
                    d2 = (wx - sx) ** 2 + (wy - sy) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = cell
                if best is not None:
                    self.debug_start_cell = [int(best[0]), int(best[1])]

                if new_goal is not None:
                    self.debug_goal_cell_candidate = [int(new_goal[0]), int(new_goal[1])]
                elif override_goal_cell is not None:
                    self.debug_goal_cell_candidate = [int(override_goal_cell[0]), int(override_goal_cell[1])]

            if new_path and new_goal is not None:
                if force:
                    do_change = True
                    self.last_forced_replan_tick = self.current_tick
                else:
                    do_change = better_enough(new_cost, self.current_path_cost)

                if do_change:
                    self.path_to_follow = new_path
                    self.current_goal_cell = new_goal
                    self.current_path_cost = new_cost
                    self.path_index = 0
                    self.path_stuck_ticks = 0

                    self.force_change_goal = False
                    self.no_move_ticks = 0
                    self.last_world_pos = None  

        elif self.current_goal_cell is not None and self.path_to_follow is not None:
            pos = self.dynamic_info.get("position") or {}
            sx = float(pos.get("x", 0.0))
            sy = float(pos.get("y", 0.0))
            start_cell = self._cell_from_xy(sx, sy)

            vis_obs = self.dynamic_info.get("visible_obstacles") or []
            vis_ter = self.dynamic_info.get("visible_terrains") or []
            current_nodes = self._divide_seen_area(vis_obs, vis_ter)

            if current_nodes:
                updated_path = self.d_star_planner.update_graph_and_path(start_cell, self.current_goal_cell, current_nodes)
                if updated_path:
                    self.path_to_follow = updated_path
                    self.path_index = 0 

        hull_rot = 0.0
        move_speed = 0.0
        if self.path_to_follow is not None:
            hull_rot, move_speed = self._FollowPath()

        self.debug_goal_cell = self.current_goal_cell
        self.debug_path = self.path_to_follow
        self.debug_path_index = self.path_index

        return hull_rot, move_speed

    def _dist2_to_cell_center(self, cell):
        pos = self.dynamic_info.get("position") or {}
        mx = float(pos.get("x", 0.0))
        my = float(pos.get("y", 0.0))
        cx, cy = self._cell_center(cell)
        dx = cx - mx
        dy = cy - my
        return dx*dx + dy*dy

    def _is_powerup_still_visible_at_cell(self, cell) -> bool:
        if cell is None:
            return False
        for p in (self.dynamic_info.get("visible_powerups", []) or []):
            pos = p.get("position", {}) or {}
            px = float(pos.get("x", 0.0))
            py = float(pos.get("y", 0.0))
            if self._cell_from_xy(px, py) == cell:
                return True
        return False
    
    def _closest_visible_enemy(self):
        visible_enemies = self.dynamic_info.get("visible_enemies", [])
        if not visible_enemies:
            return None
        return min(visible_enemies, key=lambda enemy: self._get(enemy, "distance", 999999.0))
    
    def _update_attack_memory(self):
        now = self.current_tick
        enemy = self._closest_visible_enemy()
        if enemy is None:
            return None

        eid = self._get(enemy, "id", None)
        pos = self._get(enemy, "position", {}) or {}
        ex = float(self._get(pos, "x", 0.0))
        ey = float(self._get(pos, "y", 0.0))

        self.enemy_target_id = eid
        self.enemy_target_cell = self._cell_from_xy(ex, ey)
        self.enemy_target_last_seen_tick = now
        return enemy
        
    def _should_stay_in_escape(self):
        return self.current_tick <= self.escape_until_tick
    
    def _should_stay_in_attack(self):
        return self.current_tick <= self.attack_until_tick
    
    def _maybe_forget_enemy_target(self):
        if self.enemy_target_id is None:
            return
        if (self.current_tick - self.enemy_target_last_seen_tick) >= self.enemy_forget_after:
            self.enemy_target_id = None
            self.enemy_target_cell = None
            
    def _maybe_forget_powerup_target(self):
        if self.powerup_target_cell is None:
            return

        now = self.current_tick
        cx, cy = self._cell_center(self.powerup_target_cell)

        if self._is_in_vision(cx, cy):
            if (now - self.powerup_target_last_seen_tick) >= self.powerup_forget_after:
                self.powerup_target_cell = None
                
    def _ammo_range_world(self, ammo_name):
        ammo_name = ammo_name.upper()

        if ammo_name == "HEAVY":
            tiles = 25.0
        elif ammo_name == "LIGHT":
            tiles = 50.0
        elif ammo_name == "LONG_DISTANCE":
            tiles = 100.0
        else:
            tiles = 0.0

        return tiles 

    def _ammo_inventory(self) -> dict:
        inv = {"HEAVY": 0, "LIGHT": 0, "LONG_DISTANCE": 0}
        ammo_raw = self.dynamic_info.get("ammo")

        if not isinstance(ammo_raw, dict):
            return inv

        for ammo_name, slot in ammo_raw.items():
            name = str(ammo_name).upper()

            if isinstance(slot, dict):
                inv[name] = int(slot.get("count", 0) or 0)
            else:
                inv[name] = int(getattr(slot, "count", 0) or 0)

        return inv

    def _loaded_ammo_name(self) -> str | None:
        ammo_loaded = self.dynamic_info.get("ammo_loaded", None)
        if ammo_loaded is None:
            return None
        return getattr(ammo_loaded, "name", str(ammo_loaded)).upper()

    def _aim_barrel_at_enemy(self, enemy, aim_eps_deg=2.0):
        if enemy is None:
            return 0.0

        my_position = self.dynamic_info.get("position") or {}
        enemy_position = self._get(enemy, "position", {}) or {}

        my_x = float(self._get(my_position, "x", 0.0))
        my_y = float(self._get(my_position, "y", 0.0))
        enemy_x = float(self._get(enemy_position, "x", 0.0))
        enemy_y = float(self._get(enemy_position, "y", 0.0))

        desired_abs_angle = (math.degrees(math.atan2(enemy_y - my_y, enemy_x - my_x)) + 360.0) % 360.0

        heading_abs = float(self.dynamic_info.get("heading", 0.0))
        barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
        barrel_abs = (heading_abs + barrel_rel) % 360.0

        error_angle = self._angle_diff(desired_abs_angle, barrel_abs)

        if abs(error_angle) <= aim_eps_deg:
            return 0.0

        barrel_spin_rate = float(self.static_info.get("barrel_spin_rate", 0.0))
        return self._clamp(error_angle, -barrel_spin_rate, barrel_spin_rate)

    def _update_macro_goal(self):
        now = self.current_tick
        my_pos = self.dynamic_info.get("position", {})
        mx = float(self._get(my_pos, "x", 0.0))
        my = float(self._get(my_pos, "y", 0.0))
        
        if not hasattr(self, "macro_target_world") or self.macro_target_world[0] is None:
            self._generate_new_macro_goal(mx, my)
            
        mx_m, my_m = self.macro_target_world
        dist = math.hypot(mx_m - mx, my_m - my)
        
        timeout = (now - getattr(self, "macro_target_tick", 0)) > 800
        reached = dist < 30.0
        
        path_deadlock = (self.path_to_follow is None and self.no_move_ticks > 15)
        
        if timeout or reached or path_deadlock:
            self._generate_new_macro_goal(mx, my)

    def _generate_new_macro_goal(self, mx, my):
        if not hasattr(self, "spawn_pos"):
            self.spawn_pos = (mx, my)

        visited = set(self.visited_sectors.keys())
        
        if not visited:
            visited.add((int(mx // self.SECTOR_SIZE), int(my // self.SECTOR_SIZE)))

        frontiers = set()
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for vx, vy in visited:
            for dx, dy in DIRS:
                nx, ny = vx + dx, vy + dy
                if (nx, ny) not in visited: 
                    frontiers.add((nx, ny))

        best_frontier = None
        best_utility = -float('inf')
        
        friends = self.dynamic_info.get("visible_friends", [])
        friend_pos = [(float(self._get(f['position'], 'x')), float(self._get(f['position'], 'y'))) for f in friends]

        for fx, fy in frontiers:
            wx, wy = (fx + 0.5) * self.SECTOR_SIZE, (fy + 0.5) * self.SECTOR_SIZE
            
            if wx < 0 or wy < 0 or wx > 200 or wy > 200: 
                continue

            dist_from_spawn = math.hypot(wx - self.spawn_pos[0], wy - self.spawn_pos[1])
            
            social_penalty = 0
            for fx_p, fy_p in friend_pos:
                if math.hypot(wx - fx_p, wy - fy_p) < 30.0: 
                    social_penalty += 5000

            utility = dist_from_spawn * 3.0 - social_penalty + random.uniform(0, 15.0)
            
            if utility > best_utility:
                best_utility = utility
                best_frontier = (wx, wy)

        self.macro_target_world = best_frontier or (100.0, 100.0)
        self.macro_target_tick = self.current_tick

    def _process_action(self) -> ActionCommand:
        now = self.current_tick
        enemy_now = self._update_attack_memory()
        self._maybe_forget_enemy_target()

        in_attack = self._should_stay_in_attack()
        in_escape = self._should_stay_in_escape()

        if enemy_now is not None:
            if (in_attack or in_escape):
                switched = self._maybe_reconsider_combat_mode(enemy_now)  
                if switched:
                    in_attack = self._should_stay_in_attack()
                    in_escape = self._should_stay_in_escape()

            if (not in_attack) and (not in_escape):
                chosen = self._choose_combat_mode(enemy_now)  
                self._commit_mode(chosen, enemy_now)
                in_attack = self._should_stay_in_attack()
                in_escape = self._should_stay_in_escape()
            else:
                if in_escape:
                    self._commit_mode("escape", enemy_now)
                elif in_attack:
                    self._commit_mode("attack", enemy_now)

        if self._should_stay_in_escape():
            MODE = "escape"
        elif self._should_stay_in_attack():
            MODE = "attack"
        elif self.powerup_target_cell is not None or (self.dynamic_info.get("visible_powerups") or []):
            MODE = "power_up"
        else:
            MODE = "search"

        self.mode = MODE

        if MODE == "search":
            barrel_rot = scan_strategy(self)
            self._update_macro_goal()
            hull_rot, move_speed = self.Follow_Path_With_Modifiers()

            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=hull_rot,
                move_speed=move_speed,
                should_fire=False,
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        if MODE == "power_up":
            barrel_rot = scan_strategy(self)
            pu_goal = self._select_powerup_goal_cell()
            if pu_goal is not None:
                hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=pu_goal)
            else:
                hull_rot, move_speed = self.Follow_Path_With_Modifiers()

            if self.powerup_target_cell is not None:
                close2 = (self.CELL_SIZE * 0.35) ** 2
                if self._dist2_to_cell_center(self.powerup_target_cell) <= close2:
                    if not self._is_powerup_still_visible_at_cell(self.powerup_target_cell):
                        self.powerup_target_cell = None
                        self.current_goal_cell = None
                        self.current_path_cost = None
                        self.path_to_follow = None
                        self.path_index = 0
                        self.path_stuck_ticks = 0
                        self.last_forced_replan_tick = -10_000

            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=hull_rot,
                move_speed=move_speed,
                should_fire=False,
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        if MODE == "attack":
            return mode_attack(self, enemy_now)

        if MODE == "escape":
            return mode_escape(self, enemy_now)

    def destroy(self):
        self.is_destroyed = True
        logging.info(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        logging.info(f"[{self.name}] Game ended!")
        logging.info(f"[{self.name}] Damage dealt: {damage_dealt}")
        logging.info(f"[{self.name}] Tanks killed: {tanks_killed}")

app = FastAPI(
    title="Random Test Agent",
    description="Random walking and shooting agent for testing",
    version="1.0.0"
)

agent = RandomAgent()
from fastapi import HTTPException

@app.get("/")
async def root():
    return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}

@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    try:
        action = agent.get_action(
            current_tick=payload.get('current_tick', 0),
            my_tank_status=payload.get('my_tank_status', {}),
            sensor_data=payload.get('sensor_data', {}),
            enemies_remaining=payload.get('enemies_remaining', 0)
        )
        return action
    except Exception as e:
        logging.exception("Exception in /agent/action")
        raise HTTPException(status_code=500, detail=str(payload))
    
@app.post("/agent/destroy", status_code=204)
async def destroy():
    agent.destroy()

@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    agent.end(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random test agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    parser.add_argument("--modifier", type=str, default=None, help="Modifier")    
    args = parser.parse_args()
    
    agent.name = args.name if args.name else f"RandomBot_{args.port}"
    agent.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
    os.makedirs(agent.state_dir, exist_ok=True)
    agent.state_path = os.path.join(agent.state_dir, f"agent_state_{agent.name}.json")
        
    if args.modifier != None:
        try:
            agent.modifier = args.modifier
            modifier = agent.modifier.split("_")
            agent.movement_list = [int(element) for element in modifier]
        except:
            pass

    uvicorn.run(app, host=args.host, port=args.port)