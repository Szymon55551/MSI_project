import random
import argparse
import sys
import os
import math
import itertools
from scan_strategy import scan_strategy
from agent_memory_update import update_internal_state
from debug_agent import save_state_to_file
from Attack import mode_attack
from Escape import mode_escape
from Fuzzy_Controller import FuzzyCombatDecider
from a_star import a_star
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import uvicorn
from dataclasses import dataclass, field
import logging

TILE_SIZE = 10.0
SUBDIV = 10  # This enforces CELL_SIZE = 1.0
EVAL_EVERY = 1000

FORCE_REPLAN_EVERY = 1000
IMPROVEMENT_MARGIN = 0.3
MIN_ABS_IMPROVEMENT = 20.0

LOG_PATH = os.path.join(os.path.dirname(__file__), "agent_debug.log")

def make_grid_helpers(tile_size, subdiv):
    cell_size = tile_size / float(subdiv)
    def cell_from_xy(x, y): return (int(math.floor(x / cell_size)), int(math.floor(y / cell_size)))
    def cell_center(cell): return ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)
    return cell_size, cell_from_xy, cell_center

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
    is_water: bool = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)

class RandomAgent:
    def __init__(self, name: str = "TestBot", modifier=None):
        self.name = name
        self.modifier = modifier
        self.is_destroyed = False
        self.current_tick = 0
        
        self.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, f"agent_state_{self.name}.json")
        
        self.plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

        self.tank_type = None 
        self.tank_type_used = True
        self.friend_target_cell = None
        self.friend_target_last_seen_tick = -10_000
        self.friend_forget_after = 120

        self.mode = "search"              

        self.attack_commit_ticks = 100
        self.attack_until_tick = -10_000

        self.enemy_target_id = None
        self.enemy_target_cell = None
        self.enemy_target_last_seen_tick = -10_000
        self.enemy_forget_after = 80
        
        self.escape_commit_ticks = ESCAPE_COMMIT_TICKS
        self.escape_until_tick = -10_000
        self.escape_target_cell = None
        self.escape_target_last_pick_tick = -10_000

        self.powerup_target_cell = None
        self.powerup_target_last_seen_tick = -10_000
        self.powerup_target_acquired_tick = -10_000
        self.powerup_forget_after = 200
        self.powerup_commit_ticks = 200
        self.powerup_switch_ratio = 0.80

        self.last_world_pos = None
        self.last_commanded_speed = 0.0
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
        
        self.debug_goal_cell = None
        self.debug_path = None
        self.debug_path_index = 0

        self.movement_list = []
        self.last_angle = 0
        self.unstuck_should_fire = False

        self.static_info = {}
        self.dynamic_info = {}
        self.virtual_map = {} 
        self.virtual_map = {(x, y): {"type": 0, "tick": 0} for x in range(200) for y in range(200)}
        self.memory = {} # Fixed initialization for debug plotting

        self.TILE_SIZE = TILE_SIZE
        self.SUBDIV = SUBDIV
        self.CELL_SIZE, self._cell_from_xy, self._cell_center = make_grid_helpers(self.TILE_SIZE, self.SUBDIV)
        
        self.combat_decider = FuzzyCombatDecider()
    
    def _get(self, d, key, default=None): return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)

    def _dist2_to_cell_center(self, cell):
        """Helper to calculate squared distance to a cell's center, preventing the AttributeError."""
        cx, cy = self._cell_center(cell)
        mx = float(self.dynamic_info.get("position", {}).get("x", 0.0))
        my = float(self.dynamic_info.get("position", {}).get("y", 0.0))
        return (cx - mx)**2 + (cy - my)**2

    def _save_map_plot(self):
        img = np.full((200, 200, 3), 128, dtype=np.uint8) 
        colors = {
            0: [100, 100, 100], 
            1: [144, 238, 144], 
            2: [0, 0, 255],     
            3: [0, 0, 0],       
            4: [34, 139, 34],   
            5: [255, 0, 0],     
        }

        for (cx, cy), data in self.virtual_map.items():
            if 0 <= cx < 200 and 0 <= cy < 200:
                img[cy, cx] = colors.get(data["type"], [100, 100, 100])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, origin='lower', extent=[0, 200, 0, 200])

        if getattr(self, "path_to_follow", None):
            px = [self._cell_center(c)[0] for c in self.path_to_follow]
            py = [self._cell_center(c)[1] for c in self.path_to_follow]
            ax.plot(px, py, color='magenta', linewidth=1.5, marker='o', markersize=2, label='Path')

        my_pos = self.dynamic_info.get("position", {})
        mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        tank_rect = plt.Rectangle((mx - 2.5, my - 2.5), 5, 5, color='yellow', zorder=10, label='Tank')
        ax.add_patch(tank_rect)

        ax.set_xticks(np.arange(0, 201, 10))
        ax.set_yticks(np.arange(0, 201, 10))
        ax.set_xticks(np.arange(0, 201, 1), minor=True)
        ax.set_yticks(np.arange(0, 201, 1), minor=True)
        
        ax.grid(which='major', color='white', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.2, alpha=0.3)

        plt.title(f"Virtual Map - {self.name} (Tick {self.current_tick})")
        plt.legend(loc='upper right')
        
        plot_path = os.path.join(self.plot_dir, f"map_{self.name}_{self.current_tick}.png")
        plt.savefig(plot_path)
        plt.close(fig)

    def _commit_mode(self, mode: str, enemy) -> None:
        mode = (mode or "").lower()
        if mode == "attack":
            self._enter_attack(enemy)
            self.mode = "attack"
        elif mode == "escape":
            self._enter_escape(enemy)
            self.mode = "escape"

    def _maybe_reconsider_combat_mode(self, enemy) -> bool: return False
    def _choose_combat_mode(self, enemy) -> str: return self.combat_decider.decide(self)
    
    def _enter_attack(self, enemy): self.attack_until_tick = max(self.attack_until_tick, self.current_tick + self.attack_commit_ticks)

    def _enter_escape(self, enemy):
        was_active = (self.current_tick <= self.escape_until_tick)
        self.escape_until_tick = max(self.escape_until_tick, self.current_tick + self.escape_commit_ticks)
        if not was_active or self.escape_target_cell is None: self._start_escape(enemy) 

    def _select_escape_goal_cell(self, nodes, enemy_cell, max_candidates=5):
        if not nodes or enemy_cell is None: return None
        node_by_cell = {n.cell: n for n in nodes}
        
        if enemy_cell not in node_by_cell:
            ex, ey = self._cell_center(enemy_cell)
            best, best_d2 = None, 1e30
            for c, n in node_by_cell.items():
                d2 = (n.world[0] - ex) ** 2 + (n.world[1] - ey) ** 2
                if d2 < best_d2: best_d2, best = d2, c
            enemy_cell = best
        if enemy_cell is None: return None

        safe = [n for n in nodes if (not n.blocked) and (n.dmg == 0) and (not getattr(n, "is_risk", False))]
        if not safe: safe = [n for n in nodes if not n.blocked]
        if not safe: return None

        ex, ey = self._cell_center(enemy_cell)
        safe.sort(key=lambda n: (n.world[0] - ex) ** 2 + (n.world[1] - ey) ** 2, reverse=True)
        start_cell = self._cell_from_xy(float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0)))

        for n in safe[:max_candidates]:
            path = a_star(self, nodes, start_cell, n.cell)
            if path and len(path) >= 2: return n.cell
        return None
    
    def _start_escape(self, enemy):
        self.escape_until_tick = max(self.escape_until_tick, self.current_tick + self.escape_commit_ticks)
        enemy_cell = self._cell_from_xy(float(self._get(self._get(enemy, "position", {}), "x", 0.0)), float(self._get(self._get(enemy, "position", {}), "y", 0.0)))
        nodes = self._divide_seen_area()
        goal = self._select_escape_goal_cell(nodes, enemy_cell)
        if goal is not None:
            self.escape_target_cell = goal
            self.escape_target_last_pick_tick = self.current_tick
        return enemy_cell

    def _reload_ready(self) -> bool:
        rt = self.dynamic_info.get("reload_timer", self.dynamic_info.get("current_reload_progress", 0))
        try: return int(rt) <= 0
        except: return True

    def _can_fire_at_enemy_with_range(self, enemy, ammo_name: str, aim_tolerance_deg=5.0) -> bool:
        if enemy is None or not self._reload_ready(): return False
        if float(self._get(enemy, "distance", 1e9)) > self._ammo_range_world(ammo_name): return False

        my_pos, enemy_pos = self.dynamic_info.get("position", {}), self._get(enemy, "position", {})
        mx, my = float(self._get(my_pos, "x", 0.0)), float(self._get(my_pos, "y", 0.0))
        ex, ey = float(self._get(enemy_pos, "x", 0.0)), float(self._get(enemy_pos, "y", 0.0))

        desired_abs = (math.degrees(math.atan2(ey - my, ex - mx)) + 360.0) % 360.0
        barrel_abs = (float(self.dynamic_info.get("heading", 0.0)) + float(self.dynamic_info.get("barrel_angle", 0.0))) % 360.0
        return abs(self._angle_diff(desired_abs, barrel_abs)) <= aim_tolerance_deg
    
    def _closest_visible_powerup(self):
        powerups = self.dynamic_info.get("visible_powerups", [])
        if not powerups: return None
        my_pos = self.dynamic_info.get("position", {})
        mx, my = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        def dist2(p): return (float(p.get("position", {}).get("x", 0.0)) - mx) ** 2 + (float(p.get("position", {}).get("y", 0.0)) - my) ** 2
        pu = min(powerups, key=dist2)
        return pu, (float(pu.get("position", {}).get("x", 0.0)), float(pu.get("position", {}).get("y", 0.0))), dist2(pu)

    def _is_in_vision(self, x: float, y: float) -> bool:
        pos = self.dynamic_info.get("position", {})
        mx, my = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        vr, va = float(self.static_info.get("vision_range", 0.0)), float(self.static_info.get("vision_angle", 0.0))
        if vr > 0.0 and math.hypot(x - mx, y - my) > vr: return False
        return abs(self._angle_diff((math.degrees(math.atan2(y - my, x - mx)) + 360.0) % 360.0, float(self.dynamic_info.get("heading", 0.0)))) <= (va * 0.5)
    
    def _is_target_powerup_visible(self, target_cell) -> bool:
        if target_cell is None: return False
        for p in (self.dynamic_info.get("visible_powerups", []) or []):
            if self._cell_from_xy(float(p.get("position", {}).get("x", 0.0)), float(p.get("position", {}).get("y", 0.0))) == target_cell: return True
        return False
    
    def _select_powerup_goal_cell(self):
        if self._is_target_powerup_visible(self.powerup_target_cell): self.powerup_target_last_seen_tick = self.current_tick
        self._maybe_forget_powerup_target()
        closest = self._closest_visible_powerup()
        if closest is None: return self.powerup_target_cell

        _, (px, py), d2_new = closest
        new_cell = self._cell_from_xy(px, py)

        if self.powerup_target_cell is None or new_cell == self.powerup_target_cell:
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = self.powerup_target_acquired_tick = self.current_tick
            return self.powerup_target_cell

        cx, cy = self._cell_center(self.powerup_target_cell)
        d2_cur = (cx - float(self.dynamic_info.get("position", {}).get("x", 0.0))) ** 2 + (cy - float(self.dynamic_info.get("position", {}).get("y", 0.0))) ** 2

        if (self.current_tick - self.powerup_target_acquired_tick) >= self.powerup_commit_ticks and (d2_new <= d2_cur * (self.powerup_switch_ratio ** 2)):
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = self.powerup_target_acquired_tick = self.current_tick

        return self.powerup_target_cell
    
    def _path_cost(self, path, node_by_cell):
        if not path or len(path) < 2: return float("inf")
        cost = 0.0
        for cell in path[1:]:
            n = node_by_cell.get(cell)
            if n is None or n.blocked: return float("inf")  
            cost += float(n.dmg)
        return cost
    
    def _plan_best_path(self, nodes, targets, max_targets=3):
        node_by_cell = {n.cell: n for n in nodes}
        start_cell = self._cell_from_xy(float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0)))
        best_path, best_goal, best_cost = None, None, float("inf")

        for goal in targets[:max_targets]:
            path = a_star(self, nodes, start_cell, goal, max_iterations=2000)
            if not path: continue
            
            c = self._path_cost(path, node_by_cell)
            if c < best_cost:
                best_cost, best_path, best_goal = c, path, goal

        return best_path, best_goal, best_cost

    def get_action(self, current_tick: int, my_tank_status: Dict[str, Any], sensor_data: Dict[str, Any], enemies_remaining: int) -> ActionCommand:
        self.current_tick = current_tick
        self.enemies_remaining = enemies_remaining
        update_internal_state(self, my_tank_status, sensor_data)
        
        if self.current_tick > 0 and self.current_tick % 250 == 0: self._save_map_plot()

        pos = self.dynamic_info.get("position") or {}
        px, py = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        heading_rad = math.radians(float(self.dynamic_info.get("heading", 0.0)))
        top_speed = float(self.static_info.get("top_speed", 1.0))
        heading_spin = float(self.static_info.get("heading_spin_rate", 2.0))
        
        # STUCK / WALL BOUNCE DETECTION
        if self.last_world_pos:
            lx, ly = self.last_world_pos
            dist_moved = math.hypot(px - lx, py - ly)
            
            if dist_moved < self.MIN_MOVE_EPS: self.no_move_ticks += 1
            else: self.no_move_ticks = 0

            if dist_moved > (top_speed * 1.2) and self.last_commanded_speed > 0:
                front_cell = self._cell_from_xy(lx + math.cos(heading_rad) * self.CELL_SIZE * 1.5, ly + math.sin(heading_rad) * self.CELL_SIZE * 1.5)
                if not hasattr(self, "virtual_map"): self.virtual_map = {}
                if self.virtual_map.get(front_cell, {}).get("type", 0) not in [1, 2, 4, 5]:
                    self.virtual_map[front_cell] = {"type": 3, "tick": self.current_tick}
                self.force_change_goal, self.path_to_follow, self.current_goal_cell = True, None, None

        action = self._process_action()

        if getattr(self, "current_unstuck_threshold", 0) == 0:
            self.unstuck_until_tick = -10_000
            self.current_unstuck_threshold = random.randint(50, 90)

        if self.no_move_ticks > self.current_unstuck_threshold:
            front_cell = self._cell_from_xy(px + math.cos(heading_rad) * 12.0, py + math.sin(heading_rad) * 12.0)
            friend_blocking = any(math.hypot(float(self._get(f.get("position", {}), "x", 0.0)) - px, float(self._get(f.get("position", {}), "y", 0.0)) - py) < 22.0 and abs(self._angle_diff((math.degrees(math.atan2(float(self._get(f.get("position", {}), "y", 0.0)) - py, float(self._get(f.get("position", {}), "x", 0.0)) - px)) + 360) % 360, float(self.dynamic_info.get("heading", 0.0)))) < 60 for f in self.dynamic_info.get("visible_friends", []))
            front_type = self.virtual_map.get(front_cell, {}).get("type", 0)

            if friend_blocking:
                self.unstuck_until_tick, self.unstuck_move, self.unstuck_rot, self.unstuck_should_fire = self.current_tick + random.randint(40, 80), -0.8 * top_speed, random.choice([-1.0, 1.0]) * heading_spin, False
            elif front_type == 4:
                self.unstuck_until_tick, self.unstuck_move, self.unstuck_rot, self.unstuck_should_fire = self.current_tick + 20, 0.0, 0.0, True
            else:
                if not hasattr(self, "virtual_map"): self.virtual_map = {}
                if front_type not in [1, 2, 4, 5]: self.virtual_map[front_cell] = {"type": 3, "tick": self.current_tick}
                self.force_change_goal, self.path_to_follow, self.current_goal_cell = True, None, None
                self.unstuck_until_tick, self.unstuck_move, self.unstuck_rot, self.unstuck_should_fire = self.current_tick + random.randint(50, 90), -1.0 * top_speed, random.choice([-1.0, 1.0]) * heading_spin, False

            self.no_move_ticks, self.current_unstuck_threshold = 0, random.randint(50, 100)

        if self.current_tick <= getattr(self, "unstuck_until_tick", -10_000):
            action.move_speed, action.heading_rotation_angle = getattr(self, "unstuck_move", 0.0), getattr(self, "unstuck_rot", 0.0)
            if getattr(self, "unstuck_should_fire", False): action.should_fire = True
        else: self.unstuck_should_fire = False

        self.last_world_pos, self.last_commanded_speed = (px, py), action.move_speed
        if self.current_tick < 20 or self.current_tick % 60 == 0: save_state_to_file(self)
        return action
        
    def _angle_diff(self, target_deg: float, current_deg: float) -> float: return (target_deg - current_deg + 180) % 360 - 180
    def _clamp(self, x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
    
    def _divide_seen_area(self):
        now = int(self.current_tick)
        if hasattr(self, "_last_graph_tick") and hasattr(self, "_cached_nodes") and now - self._last_graph_tick < 5: 
            return self._cached_nodes
        
        me_x, me_y = float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0))
        me_cell = self._cell_from_xy(me_x, me_y)

        nodes_by_cell = {}
        for cell, data in self.virtual_map.items():
            v_type = data["type"]
            wx, wy = self._cell_center(cell)
            nodes_by_cell[cell] = GridNode(
                cell=cell, world=(wx, wy),
                dmg=1 if v_type == 5 else 0,
                speed=0.5 if v_type == 2 else 1.0,
                blocked=(v_type == 3) and not (cell == me_cell),
                dist_to_me=math.hypot(wx - me_x, wy - me_y),
                is_risk=(v_type == 5),
                is_water=(v_type == 2),
                neighbors=[]
            )

        DIRS = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for (x, y), node in nodes_by_cell.items():
            for dx, dy in DIRS:
                nb = (x + dx, y + dy)
                if nb in nodes_by_cell and not nodes_by_cell[nb].blocked: 
                    node.neighbors.append(nb)

        self._last_graph_tick, self._cached_nodes = now, list(nodes_by_cell.values())
        return self._cached_nodes
        
    def _select_target_point(self, divided_area, current_target=None, forbid_cells=None):
        candidates = [n for n in divided_area if (not n.blocked and n.cell not in (forbid_cells or set()))]
        if not candidates: return []

        safe_candidates = [n for n in candidates if n.speed >= 0.9 and not getattr(n, 'is_water', False) and n.dmg == 0]
        pool = safe_candidates if safe_candidates else candidates
        pool.sort(key=lambda n: n.dist_to_me, reverse=True)
        top = pool[:min(40, len(pool))]

        def sort_key(n):
            hazard_penalty = 1500.0 if n.dmg > 0 else 0.0
            if getattr(n, "is_risk", False): hazard_penalty += 5.0 
            if n.speed < 0.9: hazard_penalty += 1000.0
            macro_penalty = math.hypot(n.world[0] - self.macro_target_world[0], n.world[1] - self.macro_target_world[1]) * 4.0 if hasattr(self, "macro_target_world") and self.macro_target_world[0] is not None else 0.0
            return (hazard_penalty + macro_penalty - (n.dist_to_me + (50.0 if n.cell == current_target else 0.0)))
        
        top.sort(key=sort_key)
        return [n.cell for n in top]

    def _Find_Target_and_Find_Path(self, override_goal_cell=None, forbid_cells=None):
        nodes = self._divide_seen_area()
        if not nodes: return None, None, None, None
        targets = [override_goal_cell] if override_goal_cell is not None else self._select_target_point(nodes, self.current_goal_cell, forbid_cells)
        if not targets: return None, None, None, None
        # Increased iteration cap to accommodate 200x200 pixel resolution
        path, goal, cost = self._plan_best_path(nodes, targets, max_targets=10)
        return path, goal, cost, {n.cell: n for n in nodes}
  
    def _FollowPath(self):
        if self.path_to_follow and self.path_index >= len(self.path_to_follow) - 1:
            self.path_to_follow, self.current_goal_cell, self.current_path_cost = None, None, None
            return 0.0, 0.0
        
        my_x, my_y = float(self.dynamic_info.get("position", {}).get("x", 0.0)), float(self.dynamic_info.get("position", {}).get("y", 0.0))
        
        # 1. Advance the path index if we are close to the current target node
        while self.path_index < len(self.path_to_follow) - 1:
            nx, ny = self._cell_center(self.path_to_follow[self.path_index + 1])
            if math.hypot(nx - my_x, ny - my_y) <= 2.5: 
                self.path_index += 1
                self.path_stuck_ticks = 0
            else:
                break
                
        if self.path_index >= len(self.path_to_follow) - 1:
            return 0.0, 0.0
            
        # 2. Lookahead steering targeting (Pure Pursuit)
        lookahead_idx = self.path_index + 1
        while lookahead_idx < len(self.path_to_follow) - 1:
            lx, ly = self._cell_center(self.path_to_follow[lookahead_idx])
            if math.hypot(lx - my_x, ly - my_y) >= 8.0:
                break
            lookahead_idx += 1
            
        next_x, next_y = self._cell_center(self.path_to_follow[lookahead_idx])
        
        self.path_stuck_ticks += 1
        if self.path_stuck_ticks >= MAX_PATH_STUCK_TICKS:
            self.path_to_follow = None
            return 0.0, 0.0
            
        err = self._angle_diff((math.degrees(math.atan2(next_y - my_y, next_x - my_x)) + 360.0) % 360.0, float(self.dynamic_info.get("heading", 0.0)))
        heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
        top_speed = float(self.static_info.get("top_speed", 0.0))
        
        if abs(err) > 45: move_speed = 0.0
        elif abs(err) > 15: move_speed = 0.4 * top_speed
        else: move_speed = top_speed
            
        return self._clamp(err, -heading_spin, heading_spin), move_speed
    
    
    
    def Follow_Path_With_Modifiers(self, override_goal_cell=None):
        force = (self.current_tick - self.last_forced_replan_tick >= FORCE_REPLAN_EVERY) or self.path_stuck_ticks > MAX_PATH_STUCK_TICKS or self.path_to_follow is None or self.path_index >= len(self.path_to_follow) - 1 or (override_goal_cell is not None and override_goal_cell != self.current_goal_cell)
        forbid_cells = set()
        if self.force_change_goal and self.current_goal_cell is not None:
            forbid_cells.add(self.current_goal_cell)
            force = True

        if (self.current_tick - self.last_eval_tick >= EVAL_EVERY) or force:
            self.last_eval_tick = self.current_tick
            new_path, new_goal, new_cost, node_by_cell = self._Find_Target_and_Find_Path(override_goal_cell, forbid_cells)

            if new_path and new_goal is not None:
                if force or (new_cost is not None and self.current_path_cost is not None and (new_cost <= self.current_path_cost * (1.0 - IMPROVEMENT_MARGIN) or self.current_path_cost - new_cost >= MIN_ABS_IMPROVEMENT)):
                    self.path_to_follow, self.current_goal_cell, self.current_path_cost, self.path_index, self.path_stuck_ticks, self.force_change_goal, self.no_move_ticks, self.last_world_pos = new_path, new_goal, new_cost, 0, 0, False, 0, None
                    if force: self.last_forced_replan_tick = self.current_tick

        hull_rot, move_speed = self._FollowPath() if self.path_to_follow is not None else (0.0, 0.0)
        self.debug_goal_cell, self.debug_path, self.debug_path_index = self.current_goal_cell, self.path_to_follow, self.path_index
        return hull_rot, move_speed

    def _closest_visible_enemy(self):
        visible_enemies = self.dynamic_info.get("visible_enemies", [])
        return min(visible_enemies, key=lambda enemy: self._get(enemy, "distance", 999999.0)) if visible_enemies else None
    
    def _update_attack_memory(self):
        now = self.current_tick
        enemy = self._closest_visible_enemy()
        if enemy is None: return None
        self.enemy_target_id = self._get(enemy, "id", None)
        self.enemy_target_cell = self._cell_from_xy(float(self._get(self._get(enemy, "position", {}), "x", 0.0)), float(self._get(self._get(enemy, "position", {}), "y", 0.0)))
        self.enemy_target_last_seen_tick = now
        return enemy
        
    def _should_stay_in_escape(self): return self.current_tick <= self.escape_until_tick
    def _should_stay_in_attack(self): return self.current_tick <= self.attack_until_tick

    def _maybe_forget_enemy_target(self):
        if self.enemy_target_id and (self.current_tick - self.enemy_target_last_seen_tick) >= self.enemy_forget_after:
            self.enemy_target_id, self.enemy_target_cell = None, None
            
    def _maybe_forget_powerup_target(self):
        if self.powerup_target_cell is None: return
        if self._is_in_vision(*self._cell_center(self.powerup_target_cell)) and (self.current_tick - self.powerup_target_last_seen_tick) >= self.powerup_forget_after:
            self.powerup_target_cell = None
                
    def _ammo_range_world(self, ammo_name): return {"HEAVY": 25.0, "LIGHT": 50.0, "LONG_DISTANCE": 100.0}.get(ammo_name.upper(), 0.0)

    def _ammo_inventory(self) -> dict:
        inv = {"HEAVY": 0, "LIGHT": 0, "LONG_DISTANCE": 0}
        ammo_raw = self.dynamic_info.get("ammo")
        if isinstance(ammo_raw, dict):
            for ammo_name, slot in ammo_raw.items(): inv[str(ammo_name).upper()] = int(slot.get("count", 0) if isinstance(slot, dict) else getattr(slot, "count", 0) or 0)
        return inv

    def _loaded_ammo_name(self) -> str | None:
        ammo_loaded = self.dynamic_info.get("ammo_loaded", None)
        return getattr(ammo_loaded, "name", str(ammo_loaded)).upper() if ammo_loaded else None

    def _aim_barrel_at_enemy(self, enemy, aim_eps_deg=2.0):
        if enemy is None: return 0.0
        desired_abs_angle = (math.degrees(math.atan2(float(self._get(self._get(enemy, "position", {}), "y", 0.0)) - float(self._get(self.dynamic_info.get("position"), "y", 0.0)), float(self._get(self._get(enemy, "position", {}), "x", 0.0)) - float(self._get(self.dynamic_info.get("position"), "x", 0.0)))) + 360.0) % 360.0
        error_angle = self._angle_diff(desired_abs_angle, (float(self.dynamic_info.get("heading", 0.0)) + float(self.dynamic_info.get("barrel_angle", 0.0))) % 360.0)
        return 0.0 if abs(error_angle) <= aim_eps_deg else self._clamp(error_angle, -float(self.static_info.get("barrel_spin_rate", 0.0)), float(self.static_info.get("barrel_spin_rate", 0.0)))

    def _update_macro_goal(self):
        mx, my = float(self._get(self.dynamic_info.get("position", {}), "x", 0.0)), float(self._get(self.dynamic_info.get("position", {}), "y", 0.0))
        if not hasattr(self, "macro_target_world") or self.macro_target_world[0] is None or (self.current_tick - getattr(self, "macro_target_tick", 0)) > 800 or math.hypot(self.macro_target_world[0] - mx, self.macro_target_world[1] - my) < 20.0 or (self.path_to_follow is None and self.no_move_ticks > 15):
            
            frontiers = []
            for cell, data in self.virtual_map.items():
                if data["type"] == 0: 
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nb = (cell[0]+dx, cell[1]+dy)
                        if nb in self.virtual_map and self.virtual_map[nb]["type"] in [1, 2, 4]: 
                            frontiers.append((self._cell_center(cell), False))
                            break
            
            if not frontiers:
                grass_cells = [c for c, d in self.virtual_map.items() if d["type"] == 1]
                self.macro_target_world = self._cell_center(random.choice(grass_cells)) if grass_cells else (100.0, 100.0)
                self.macro_target_tick = self.current_tick
                return
                
            best_frontier, best_utility = None, -float('inf')
            friends = self.dynamic_info.get("visible_friends", [])
            for (wx, wy), near_tree in frontiers:
                social_penalty = sum(5000 for f in friends if math.hypot(wx - float(f.get("position", {}).get("x", 0)), wy - float(f.get("position", {}).get("y", 0))) < 40.0)
                utility = -math.hypot(wx - mx, wy - my) - social_penalty + random.uniform(0, 15.0)
                if utility > best_utility: best_utility, best_frontier = utility, (wx, wy)

            self.macro_target_world, self.macro_target_tick = best_frontier or (100.0, 100.0), self.current_tick

    def _process_action(self) -> ActionCommand:
        enemy_now = self._update_attack_memory()
        self._maybe_forget_enemy_target()
        in_attack, in_escape = self._should_stay_in_attack(), self._should_stay_in_escape()

        if enemy_now is not None:
            if in_attack or in_escape:
                if self._maybe_reconsider_combat_mode(enemy_now): in_attack, in_escape = self._should_stay_in_attack(), self._should_stay_in_escape()
            if not in_attack and not in_escape:
                self._commit_mode(self._choose_combat_mode(enemy_now), enemy_now)
            else:
                self._commit_mode("escape" if in_escape else "attack", enemy_now)

        if self._should_stay_in_escape(): MODE = "escape"
        elif self._should_stay_in_attack(): MODE = "attack"
        elif self.powerup_target_cell is not None or (self.dynamic_info.get("visible_powerups") or []): MODE = "power_up"
        else: MODE = "search"

        self.mode = MODE

        if MODE == "search":
            self._update_macro_goal()
            hull_rot, move_speed = self.Follow_Path_With_Modifiers()
            return ActionCommand(barrel_rotation_angle=scan_strategy(self), heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))

        if MODE == "power_up":
            pu_goal = self._select_powerup_goal_cell()
            hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=pu_goal) if pu_goal is not None else self.Follow_Path_With_Modifiers()
            if self.powerup_target_cell is not None and self._dist2_to_cell_center(self.powerup_target_cell) <= (self.CELL_SIZE * 0.35) ** 2 and not self._is_target_powerup_visible(self.powerup_target_cell):
                self.powerup_target_cell, self.current_goal_cell, self.current_path_cost, self.path_to_follow, self.path_index, self.path_stuck_ticks, self.last_forced_replan_tick = None, None, None, None, 0, 0, -10_000
            return ActionCommand(barrel_rotation_angle=scan_strategy(self), heading_rotation_angle=hull_rot, move_speed=move_speed, should_fire=False, ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"]))

        if MODE == "attack": return mode_attack(self, enemy_now)
        if MODE == "escape": return mode_escape(self, enemy_now)

    def destroy(self):
        self.is_destroyed = True
        logging.info(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        logging.info(f"[{self.name}] Game ended! Damage: {damage_dealt}, Kills: {tanks_killed}")

app = FastAPI(title="Random Test Agent", description="Random walking and shooting agent for testing", version="1.0.0")
agent = RandomAgent()

@app.get("/")
async def root(): return {"message": f"Agent {agent.name} is running", "destroyed": agent.is_destroyed}

@app.post("/agent/action", response_model=ActionCommand)
async def get_action(payload: Dict[str, Any] = Body(...)):
    try: return agent.get_action(current_tick=payload.get('current_tick', 0), my_tank_status=payload.get('my_tank_status', {}), sensor_data=payload.get('sensor_data', {}), enemies_remaining=payload.get('enemies_remaining', 0))
    except Exception as e:
        logging.exception("Exception in /agent/action")
        raise HTTPException(status_code=500, detail=str(payload))
    
@app.post("/agent/destroy", status_code=204)
async def destroy(): agent.destroy()

@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)): agent.end(damage_dealt=payload.get('damage_dealt', 0.0), tanks_killed=payload.get('tanks_killed', 0))

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
        try: agent.movement_list = [int(element) for element in agent.modifier.split("_")]
        except: pass

    uvicorn.run(app, host=args.host, port=args.port)