"""
Random Walking and Shooting Agent for Testing
Agent losowo chodzący i strzelający do testów

This agent implements IAgentController and performs random actions:
- Random barrel and heading rotation
- Random movement speed
- Random shooting

Usage:
    python random_agent.py --port 8001
    
To run multiple agents:
    python random_agent.py --port 8001  # Tank 1
    python random_agent.py --port 8002  # Tank 2
    ...
"""

import random
import argparse
import sys
import os
import math
import json
import heapq

RECENT_TICKS_GRAPH = 5
MAX_PATH_STUCK_TICKS = 300
NO_MOVE_LIMIT_TICKS = 300

ESCAPE_COMMIT_TICKS = 300      
ESCAPE_BACK_TICKS   = 150       
ESCAPE_REPLAN_EVERY = 120     

# Add paths for imports
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

#########################
#Params
#A* - modyfikatory kosztu
DMG_PENALTY = 4000   #koszt za 1 pkt obrazen
BASE_MOVE_COST = 1  #koszt ruchu o 1 kratke
SLOW_TERRAIN_PENALTY_WEIGHT = 75
STRAIGHT_PENALTY = 2  #koszt za jazdę prosto
DANGEROUS_NEIGHBOUR_PENALTY = 20 #Kara za to ze jestesmy bezposrednio obok niebezpiecznej kratki
FORCE_REPLAN_EVERY = 1000
IMPROVEMENT_MARGIN = 0.3
MIN_ABS_IMPROVEMENT = 20.0
#########################

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
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

# ============================================================================
# ACTION COMMAND MODEL
# ============================================================================

class ActionCommand(BaseModel):
    """Output action from agent to engine."""
    barrel_rotation_angle: float = 0.0
    heading_rotation_angle: float = 0.0
    move_speed: float = 0.0
    ammo_to_load: str = None
    should_fire: bool = False

@dataclass
class GridNode:
    cell: Tuple[int, int]                     # (ix, iy) w sub-grid
    world: Tuple[float, float]                # (x,y) środek sub-komórki w świecie
    dmg: int
    speed: float
    blocked: bool
    dist_to_me: float
    neighbors: List[Tuple[int, int]] = field(default_factory=list)  # 4-kierunkowo
    is_risk: bool = False


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,  # crucial: overrides uvicorn's logging config
)

logging.info(f"Logging to: {LOG_PATH}")

class RandomAgent:
    def __init__(self, name: str = "TestBot", modifier=None):

        # =========================================================
        # ID / META
        # =========================================================
        self.name = name
        self.modifier = modifier

        self.is_destroyed = False
        self.current_tick = 0

        # debug / state file
        self.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(self.state_dir, exist_ok=True)
        self.state_path = os.path.join(self.state_dir, f"agent_state_{self.name}.json")

        logging.info(f"[{self.name}] Agent initialized")
        logging.info(f"[modifier={self.modifier}] otrzymałem argument")

        # =========================================================
        # ROLE (Leader/Follower)
        # =========================================================
        self.tank_type = random.choices(population=["Leader", "Follower"], weights=[0.6, 0.4])[0]
        self.tank_type_used = True

        # follower memory
        self.friend_target_cell = None
        self.friend_target_last_seen_tick = -10_000
        self.friend_forget_after = 120

        # =========================================================
        # MODE / COMMIT STATE
        # =========================================================
        self.mode = "search"              # "search" | "power_up" | "attack" | "escape"

        # --- ATTACK ---
        self.attack_commit_ticks = 100
        self.attack_until_tick = -10_000

        self.enemy_target_id = None
        self.enemy_target_cell = None
        self.enemy_target_last_seen_tick = -10_000
        self.enemy_forget_after = 80
        self.attack_desired_range = 8.0

        # --- ESCAPE ---
        self.escape_commit_ticks = ESCAPE_COMMIT_TICKS
        self.escape_until_tick = -10_000

        self.escape_back_ticks = ESCAPE_BACK_TICKS
        self.escape_enter_tick = -10_000  # moment wejścia w escape (dla back-phase)

        self.escape_target_cell = None
        self.escape_target_last_pick_tick = -10_000
        self.escape_forget_after = 120

        # =========================================================
        # POWERUP TARGETING
        # =========================================================
        self.powerup_target_cell = None
        self.powerup_target_last_seen_tick = -10_000
        self.powerup_target_acquired_tick = -10_000
        self.powerup_forget_after = 200
        self.powerup_commit_ticks = 200
        self.powerup_switch_ratio = 0.80

        # =========================================================
        # MOVEMENT / PATHFOLLOW
        # =========================================================
        self.last_world_pos = None
        self.no_move_ticks = 0
        self.force_change_goal = False

        self.MIN_MOVE_EPS = 1
        self.NO_MOVE_LIMIT_TICKS = NO_MOVE_LIMIT_TICKS

        self.last_eval_tick = -10_000
        self.last_forced_replan_tick = -10_000

        self.current_goal_cell = None
        self.current_path_cost = None

        self.path_to_follow = None
        self.path_index = 0
        self.path_stuck_ticks = 0
        self.reached_points = None

        # debug path
        self.debug_goal_cell = None
        self.debug_path = None
        self.debug_path_index = 0

        # modifier-driven movement list
        self.movement_list = []
        self.last_angle = 0

        # =========================================================
        # SENSOR / MEMORY
        # =========================================================
        self.static_info = {}
        self.dynamic_info = {}
        self.memory = {}
        self.map_memory = {}
        self.obstacle_memory = {}

        self.meta_info = {
            "is_aimed_at": False,
            "closest_enemy_angle": 0.0,
            "closest_enemy_dist": 707,  # 500 * sqrt(2)
            "target_id": None
        }

        # =========================================================
        # SCANNING (barrel scan)
        # =========================================================
        self.scan_offset = 0.0
        self.scan_direction = 1.0
        self.scan_max_offset = 90.0

        self.full_scan_interval = 180
        self.full_scan_active = False
        self.full_scan_remaining = 0.0
        self.last_full_scan_tick = -10_000

        # =========================================================
        # GRID CONFIG
        # =========================================================
        self.TILE_SIZE = 10.0
        self.SUBDIV = SUBDIV
        (
            self.CELL_SIZE,
            self._cell_from_xy,
            self._stamp_tile_center_to_subcells,
            self._cell_center
        ) = make_grid_helpers(self.TILE_SIZE, self.SUBDIV)
        
        
        self.dodge_until_tick = -10_000
        self.dodge_heading_target = None
        self.dodge_dir = 1  # 1/-1
    
    def _get(self, d, key, default=None):
        return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)
    
    def _fallback_ammo_to_load(self) -> str:
        inv = self._ammo_inventory()
        loaded = self._loaded_ammo_name()

        if loaded and inv.get(loaded, 0) > 0:
            return loaded

        for a in ("LONG_DISTANCE", "LIGHT", "HEAVY"):
            if inv.get(a, 0) > 0:
                return a

        return "LIGHT"


    def _commit_mode(self, mode: str, enemy) -> None:
        """Ustawia commit + self.mode dla danego trybu."""
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
        """
        Miejsce na przyszłą zmianę trybu (np. gdy HP spadnie).
        Na razie ZABLOKOWANE: zawsze zwraca False.
        
        Docelowo: odpalane co N ticków (np. 50) i może przełączyć commit.
        """
        return False
    
    def _attack_micro_dodge(self, enemy, desired_range_world: float, band: float = 2.0):
        if enemy is None:
            return 0.0, 0.0

        pos_my = self.dynamic_info.get("position") or {}
        pos_enemy = self._get(enemy, "position", {}) or {}

        mx = float(self._get(pos_my, "x", 0.0))
        my = float(self._get(pos_my, "y", 0.0))
        ex = float(self._get(pos_enemy, "x", 0.0))
        ey = float(self._get(pos_enemy, "y", 0.0))

        dx = ex - mx
        dy = ey - my
        dist = math.hypot(dx, dy)

        angle_to_enemy = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        # losowe zniany żeby trudniej było odpwoiedzieć 
        if self.current_tick >= getattr(self, "dodge_until_tick", -10_000):
            self.dodge_dir = random.choice([-1, 1])
            self.dodge_offset = random.uniform(-20.0, 20.0)  # mały random
            self.dodge_until_tick = self.current_tick + random.randint(20, 60)

        base = (angle_to_enemy + 90.0 * self.dodge_dir + self.dodge_offset) % 360.0
        
        if dist < (desired_range_world - band):
            base = (base + random.uniform(20.0, 35.0) * self.dodge_dir) % 360.0
        elif dist > (desired_range_world + band):
            base = (base - random.uniform(20.0, 35.0) * self.dodge_dir) % 360.0

        current_heading = float(self.dynamic_info.get("heading", 0.0))
        err = self._angle_diff(base, current_heading)

        heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
        hull_rot = self._clamp(err, -heading_spin, heading_spin)

        top_speed = float(self.static_info.get("top_speed", 0.0))

        move = 0.9 * top_speed if abs(err) < 25 else 0.4 * top_speed
        return hull_rot, move


    def _choose_combat_mode(self, enemy) -> str:
        """
        TODO WSTAWIĆ STEROWNIK ROZMYTY DECYDUJĄCY CZY USICEKAC CZY AKAKOWAĆ
        """
        return random.choice(["attack", "escape"])

    def _update_attack_memory(self):
        """Jeśli widzimy wroga: zapamiętaj target (bez commitu trybu)."""
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
    
    def _enter_attack(self, enemy):
        now = self.current_tick
        self.attack_until_tick = max(self.attack_until_tick, now + self.attack_commit_ticks)
        # zapamiętaj target
        return

    def _enter_escape(self, enemy):
        now = self.current_tick

        was_active = (now <= self.escape_until_tick)
        self.escape_until_tick = max(self.escape_until_tick, now + self.escape_commit_ticks)

        # tylko nowe wejście w escape
        if not was_active:
            self.escape_enter_tick = now
            self._start_escape(enemy) 
        else:
            if self.escape_target_cell is None and enemy is not None:
                self._start_escape(enemy)

    def _should_stay_in_attack(self) -> bool:
        return self.current_tick <= self.attack_until_tick

    def _furthest_visible_friend(self):
            """Return furthest visible friendly tank."""
            visible = self.dynamic_info.get("visible_tanks", []) or []
            my_id = self.static_info.get("id")
            my_team = self.static_info.get("team")

            friends = []
            for t in visible:
                try:
                    if self._get(t, "team", None) != my_team:
                        continue
                    if self._get(t, "id", None) == my_id:
                        continue
                    friends.append(t)
                except:
                    continue

            if not friends:
                return None

            # choose max distance
            return max(friends, key=lambda t: float(self._get(t, "distance", -1e9)))

    def _update_friend_memory(self):
        """If we see any friendly tank, remember the furthest one's cell."""
        now = self.current_tick
        f = self._furthest_visible_friend()
        if f is None:
            return None

        pos = self._get(f, "position", {}) or {}
        fx = float(self._get(pos, "x", 0.0))
        fy = float(self._get(pos, "y", 0.0))

        self.friend_target_cell = self._cell_from_xy(fx, fy)
        self.friend_target_last_seen_tick = now
        return f
    
    def _mode_escape(self, enemy_now) -> ActionCommand:
        barrel_rot = self._scan_strategy()
        top_speed = float(self.static_info.get("top_speed", 0.0))

        # jeśli wróg jest widoczny – odśwież commit i  escape_target
        if enemy_now is not None:
            self.escape_until_tick = max(self.escape_until_tick, self.current_tick + self.escape_commit_ticks)

        # back-phase przez ESCAPE_BACK_TICKS od wejścia
        if self.escape_enter_tick >= 0 and (self.current_tick - self.escape_enter_tick) < self.escape_back_ticks:
            if enemy_now is None:
                return ActionCommand(
                    barrel_rotation_angle=barrel_rot,
                    heading_rotation_angle=0.0,
                    move_speed=-top_speed,
                    should_fire=False,
                    ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
                )

            my_pos = self.dynamic_info.get("position") or {}
            enemy_pos = self._get(enemy_now, "position", {}) or {}

            mx = float(self._get(my_pos, "x", 0.0))
            my = float(self._get(my_pos, "y", 0.0))
            ex = float(self._get(enemy_pos, "x", 0.0))
            ey = float(self._get(enemy_pos, "y", 0.0))

            dx = ex - mx
            dy = ey - my

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                hull_rot = 0.0
            else:
                desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                current_heading = float(self.dynamic_info.get("heading", 0.0))
                err = self._angle_diff(desired_heading, current_heading)

                heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
                hull_rot = self._clamp(err, -heading_spin, heading_spin)

            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=hull_rot,
                move_speed=-top_speed,
                should_fire=False,
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        # jeśli nie mamy celu ucieczki, fallback: dalej cofaj
        if self.escape_target_cell is None:
            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=0.0,
                move_speed=-top_speed,
                should_fire=False,
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        # odśwież escape goal co ESCAPE_REPLAN_EVERY podczas commitu (opcjonalnie)
        if (self.current_tick - self.escape_target_last_pick_tick) >= ESCAPE_REPLAN_EVERY:
            enemy = enemy_now if enemy_now is not None else self._closest_visible_enemy()
            if enemy is not None:
                self._start_escape(enemy)

        hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=self.escape_target_cell)

        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            should_fire=False,
            ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
        )
        
    
    def _select_escape_goal_cell(self, nodes, enemy_cell, max_candidates=40):
        """
        Wybiera cel ucieczki: bezpieczny + daleko od wroga + osiągalny (A*).
        - "bezpieczny" = dmg==0, not risk, not blocked
        - ranking = dystans od wroga (max)
        """
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

        # filtr bezpiecznych kandydatów
        safe = [
            n for n in nodes
            if (not n.blocked) and (n.dmg == 0) and (not getattr(n, "is_risk", False))
        ]
        if not safe:
            # fallback: cokolwiek nieblocked
            safe = [n for n in nodes if not n.blocked]
            if not safe:
                return None

        # sortuj: najdalej od enemy
        ex, ey = self._cell_center(enemy_cell)
        safe.sort(key=lambda n: (n.world[0] - ex) ** 2 + (n.world[1] - ey) ** 2, reverse=True)

        # bierz top i sprawdzaj osiągalność A*
        for n in safe[:max_candidates]:
            path = self._a_star(nodes, n.cell)
            if path and len(path) >= 2:
                return n.cell

        return None
    
    def _start_escape(self, enemy):
        now = self.current_tick

        # commit
        self.escape_until_tick = max(self.escape_until_tick, now + self.escape_commit_ticks)

        # pick enemy cell (last known)
        pos = self._get(enemy, "position", {}) or {}
        ex = float(self._get(pos, "x", 0.0))
        ey = float(self._get(pos, "y", 0.0))
        enemy_cell = self._cell_from_xy(ex, ey)

        # build graph from memory and pick escape goal
        visible_obstacles = self.dynamic_info.get("visible_obstacles") or []
        visible_terrains  = self.dynamic_info.get("visible_terrains")  or []
        nodes = self._divide_seen_area(visible_obstacles, visible_terrains)

        goal = self._select_escape_goal_cell(nodes, enemy_cell)
        if goal is not None:
            self.escape_target_cell = goal
            self.escape_target_last_pick_tick = now

        return enemy_cell
    
    def _should_stay_in_escape(self) -> bool:
        return self.current_tick <= self.escape_until_tick

    def _maybe_forget_friend_target(self):
        if self.friend_target_cell is None:
            return
        if (self.current_tick - self.friend_target_last_seen_tick) >= self.friend_forget_after:
            self.friend_target_cell = None

    def _select_friend_goal_cell(self):
        """
        Returns remembered friend cell if still valid.
        Priority:
        - refresh memory if visible
        - otherwise keep remembered until forget_after
        """
        self._update_friend_memory()
        self._maybe_forget_friend_target()
        return self.friend_target_cell 

    def _maybe_forget_enemy_target(self):
        if self.enemy_target_id is None:
            return
        if (self.current_tick - self.enemy_target_last_seen_tick) >= self.enemy_forget_after:
            self.enemy_target_id = None
            self.enemy_target_cell = None

    def _terrain_tiles_from_seen(self):
        tiles = []
        seen_terrains = self.dynamic_info.get("visible_terrains", [])
        for t in seen_terrains:
            pos = t.get("position", {})
            cx = float(pos.get("x", 0.0))
            cy = float(pos.get("y", 0.0))
            tile_x = int(cx // self.TILE_SIZE)
            tile_y = int(cy // self.TILE_SIZE)
            tiles.append((tile_x, tile_y))
        # unikalne
        tiles = list(dict.fromkeys(tiles))
        return tiles
    
    def _ammo_inventory(self) -> dict:
            """
            TODO Pobierać to z silnika 
            Zwraca dict: {"HEAVY": count, "LIGHT": count, "LONG_DISTANCE": count}
            """
            inv = {"HEAVY": 0, "LIGHT": 0, "LONG_DISTANCE": 0}
            ammo = self.dynamic_info.get("ammo", {}) or {}

            for k, slot in ammo.items():
                # k może być AmmoType enum albo string / coś z .name
                if hasattr(k, "name"):
                    name = k.name
                else:
                    name = str(k)

                # slot może mieć .count albo być dict
                if isinstance(slot, dict):
                    cnt = int(slot.get("count", 0))
                else:
                    cnt = int(getattr(slot, "count", 0))

                # normalizacja
                name = name.upper()
                if "LONG" in name:
                    inv["LONG_DISTANCE"] += cnt
                elif "HEAVY" in name:
                    inv["HEAVY"] += cnt
                elif "LIGHT" in name:
                    inv["LIGHT"] += cnt

            return inv

    def _ammo_range_world(self, ammo_name: str) -> float:
        """Ammo range in WORLD units. Input ranges are defined in TILES."""
        ammo_name = ammo_name.upper()

        # TODO - To też z silnika 
        if ammo_name == "HEAVY":
            tiles = 25.0
        elif ammo_name == "LIGHT":
            tiles = 50.0
        elif ammo_name == "LONG_DISTANCE":
            tiles = 100.0
        else:
            tiles = 0.0

        return tiles * float(self.TILE_SIZE)  # TILE_SIZE = 10.0 => convert to world

    def _reload_ready(self) -> bool:
        rt = self.dynamic_info.get("reload_timer", None)
        if rt is None:
            rt = self.dynamic_info.get("current_reload_progress", 0)
        try:
            return int(rt) <= 0
        except:
            return True

    def _loaded_ammo_name(self) -> str | None:
        ammo_loaded = self.dynamic_info.get("ammo_loaded", None)
        if ammo_loaded is None:
            return None
        return getattr(ammo_loaded, "name", str(ammo_loaded)).upper()


    def _can_fire_at_enemy_with_range(self, enemy, ammo_name: str, aim_tolerance_deg=5.0, debug=False) -> bool:
        if enemy is None:
            return False

        # reload gate
        if not self._reload_ready():
            if debug:
                rt = self.dynamic_info.get("reload_timer", None)
                print("[FIRE] reload not ready:", rt)
            return False

        # --- distance (WORLD) ---
        dist_world = float(self._get(enemy, "distance", 1e9))  # payload already world
        dist_tiles = dist_world / float(self.TILE_SIZE)

        # --- range (WORLD) ---
        range_world = self._ammo_range_world(ammo_name)
        range_tiles = range_world / float(self.TILE_SIZE)

        if debug:
            loaded = self._loaded_ammo_name()
            inv = self._ammo_inventory()
            print("=== ATTACK DEBUG ===")
            print("dist(world)=", dist_world, "tiles≈", dist_tiles)
            print("ammo_name=", ammo_name, "loaded=", loaded, "inv=", inv)
            print("range(world)=", range_world, "tiles=", range_tiles)
            print("reload_timer=", self.dynamic_info.get("reload_timer", None))
            print("====================")

        # range gate
        if dist_world > range_world:
            return False

        # --- aim gate ---
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

    
    def _fov_debug_payload(self):
        """
        Zwraca dane debug pola widzenia:
        - wedge (dwa promienie graniczne)
        - zbiór sub-komórek w FOV (w promieniu + w kącie)
        """
        pos = self.dynamic_info.get("position") or {}
        mx = float(pos.get("x", 0.0))
        my = float(pos.get("y", 0.0))

        vr = float(self.static_info.get("vision_range", 0.0))
        va = float(self.static_info.get("vision_angle", 0.0))
        heading = float(self.dynamic_info.get("heading", 0.0))

        if vr <= 0 or va <= 0:
            return {
                "tick": self.current_tick,
                "origin": {"x": mx, "y": my},
                "vr": vr,
                "va": va,
                "heading": heading,
                "cells": [],
                "rays": [],
                "bbox": None,
            }

        # bbox w sub-grid ograniczony do vision_range
        cx, cy = self._cell_from_xy(mx, my)
        r_cells = int(math.ceil(vr / self.CELL_SIZE))

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
                wx, wy = self._cell_center((ix, iy))
                dx = wx - mx
                dy = wy - my
                d2 = dx*dx + dy*dy
                if d2 > vr2:
                    continue

                ang_to = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                err = self._angle_diff(ang_to, heading)
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
            "tick": self.current_tick,
            "origin": {"x": mx, "y": my},
            "vr": vr,
            "va": va,
            "heading": heading,
            "cells": cells,                 # lista sub-komórek w FOV
            "bbox": [minx, miny, maxx, maxy],
            "rays": rays,                   # 2 promienie graniczne
        }

    
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

        # vision aligned with hull heading (change to barrel_abs if your engine uses barrel)
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
    
    def _maybe_forget_powerup_target(self):
        if self.powerup_target_cell is None:
            return

        now = self.current_tick
        cx, cy = self._cell_center(self.powerup_target_cell)

        if self._is_in_vision(cx, cy):
            if (now - self.powerup_target_last_seen_tick) >= self.powerup_forget_after:
                self.powerup_target_cell = None
    
    def _select_powerup_goal_cell(self):
        now = self.current_tick

        # 0) If we currently see our target, refresh last_seen
        if self._is_target_powerup_visible(self.powerup_target_cell):
            self.powerup_target_last_seen_tick = now

        # If target should be visible but isn't (for long enough), forget it
        self._maybe_forget_powerup_target()

        # If no powerups visible now: keep committed target if any
        closest = self._closest_visible_powerup()
        if closest is None:
            return self.powerup_target_cell

        _, (px, py), d2_new = closest
        new_cell = self._cell_from_xy(px, py)

        #  Acquire if none
        if self.powerup_target_cell is None:
            self.powerup_target_cell = new_cell
            self.powerup_target_last_seen_tick = now  # we definitely saw it now
            self.powerup_target_acquired_tick = now
            return self.powerup_target_cell

        # If same: refresh already handled above, just return
        if new_cell == self.powerup_target_cell:
            return self.powerup_target_cell

        # Switch policy (anti-ping-pong)
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
    
        
    def save_state_to_file(self):
        logging.debug(f"save_state_to_file")
        """Save the current state of the agent to a JSON file."""

        state = {
            "current_tick": self.current_tick,
            "static_info": self.static_info,
            "dynamic_info": self.dynamic_info,
            "meta_info": self.meta_info,
            "memory": self.memory,
            
            "agent_role": {
                "type": self.tank_type,            
                "enabled": bool(self.tank_type_used)
            },

            "debug": {
                "goal_cell": self.debug_goal_cell,
                "path": self.debug_path,
                "path_index": self.debug_path_index,
                "seen_terrain_tiles": self._terrain_tiles_from_seen(),
                "my_pos": self.dynamic_info.get("position", None),
                "fov": self._fov_debug_payload(),
                "graph_nodes": getattr(self, "debug_graph_nodes", None),
                "start_cell": getattr(self, "debug_start_cell", None),
                "goal_cell_candidate": getattr(self, "debug_goal_cell_candidate", None),
            }
        }
        
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        os.replace(tmp_path, self.state_path)

    
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

        # upewnij się, że to faktycznie sąsiad last
        if extra not in node_by_cell[last].neighbors:
            return path

        return path + [extra]
    
    def _plan_best_path(self, nodes, targets, max_targets=3):
        node_by_cell = {n.cell: n for n in nodes}

        best_path = None
        best_goal = None
        best_cost = float("inf")

        for goal in targets[:max_targets]:
            path = self._a_star(nodes, goal)
            print("path", path)
            if not path:
                continue
            c = self._path_cost(path, node_by_cell)
            if c < best_cost:
                best_cost = c
                best_path = path
                best_goal = goal

        # AUTO-EXTEND NA KONIEC (tylko dla wybranej ścieżki)
        if best_path:
            best_path = self._extend_path_one_step(best_path, node_by_cell)

        return best_path, best_goal, best_cost
            
    def  next_or_first(self, arr, value):
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
        else:
            print("No movement list provided")
            pass

        self.current_tick = current_tick
        self.enemies_remaining = enemies_remaining

        # Update internal state
        self._update_internal_state(my_tank_status, sensor_data)

        # --- update no-move / stuck state ---
        pos = self.dynamic_info.get("position") or {}
        px = float(pos.get("x", 0.0))
        py = float(pos.get("y", 0.0))

        if self.last_world_pos is None:
            self.no_move_ticks = 0
            self.force_change_goal = False
        else:
            lx, ly = self.last_world_pos
            moved_dist = math.hypot(px - lx, py - ly)

            if moved_dist < self.MIN_MOVE_EPS:
                self.no_move_ticks += 1
            else:
                self.no_move_ticks = 0
                self.force_change_goal = False

            if self.no_move_ticks >= self.NO_MOVE_LIMIT_TICKS:
                self.force_change_goal = True

        # ALWAYS update last_world_pos
        self.last_world_pos = (px, py)

        # Process action
        action = self._process_action()

        # Save state to file
        self.save_state_to_file()

        return action
        

    def _update_internal_state(self, status, sensors):

        self.static_info = {
            "id": self._get(status, '_id', 'MISSING'),
            "team": self._get(status, '_team', 'MISSING'),
            "tank_type": self._get(status, '_tank_type', 'MISSING'),
            "vision_angle": self._get(status, '_vision_angle', 'MISSING'),
            "vision_range": self._get(status, '_vision_range', 'MISSING'),
            "top_speed": self._get(status, '_top_speed', 'MISSING'),
            "barrel_spin_rate": self._get(status, '_barrel_spin_rate', 30),                  #### NIE OBECNE W PAYLOAD 
            "heading_spin_rate": self._get(status, '_heading_spin_rate', 2),                 #### NIE OBECNE W PAYLOAD 
            "max_hp": self._get(status, '_max_hp', 'MISSING'),
            "max_shield": self._get(status, '_max_shield', 'MISSING'),
        }
        # logging.debug(f"[DEBUG] static_info updated: {self.static_info}")

        # Safely extract dynamic info
        self.dynamic_info = {
            "hp": self._get(status, 'hp', 'MISSING'),
            "shield": self._get(status, 'shield', 'MISSING'),
            "position": self._get(status, 'position', 'MISSING'),
            "move_speed": self._get(status, 'move_speed', 'MISSING'),
            "barrel_angle": self._get(status, 'barrel_angle', 'MISSING'),
            "heading": self._get(status, 'heading', 'MISSING'),
            "ammo": self._get(status, 'ammo', 'MISSING'),
            "ammo_loaded": self._get(status, 'ammo_loaded', 'MISSING'),
            "is_overcharged": self._get(status, 'is_overcharged', 'MISSING'),
            "size": self._get(status, 'size', 'MISSING'),
            "reload_timer": self._get(status, '_reload_timer', 'MISSING'),
            "enemies_remaining": self.enemies_remaining,
            "visible_tanks": self._get(sensors, 'seen_tanks', []),
            "visible_enemies": [tank for tank in self._get(sensors, 'seen_tanks', []) if tank["team"] != self.static_info["team"]],
            "visible_obstacles": self._get(sensors, 'seen_obstacles', []),
            "visible_terrains": self._get(sensors, 'seen_terrains', []),
            "visible_powerups": self._get(sensors, "seen_powerups", []),
            
        }
        
        current_terrains = self._get(sensors, 'seen_terrains', [])
        for t in current_terrains:
            pos = t.get("position", {})
            cx, cy = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
            dmg = int(t.get("dmg", 0))
            speed = float(t.get("speed_modifier", 1.0))
            
            # Save every sub-cell to memory
            now = self.current_tick  
            
            for cell in self._stamp_tile_center_to_subcells(cx, cy):
                self.map_memory[cell] = {"dmg": dmg, "speed": speed, "last_seen_tick": now}
    
        # 2. Memorize Obstacles (Walls)
        current_obstacles = self._get(sensors, 'seen_obstacles', [])
        for ob in current_obstacles:
            pos = ob.get("position", {})
            cx, cy = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
            
            # Save walls to memory
            now = self.current_tick
            for cell in self._stamp_tile_center_to_subcells(cx, cy):
                self.obstacle_memory[cell] = now
        

        self._process_meta_info()
        self._process_memory()

    def _process_memory(self):
        visible_enemies = self.dynamic_info.get('visible_enemies', [])

        for enemy in visible_enemies:
            self.memory[self._get(enemy, 'id', 'MISSING')] = {
                "id": self._get(enemy, 'id', 'MISSING'),
                "last_seen_pos": self._get(enemy, 'position', 'MISSING'),
                "last_seen_tick": self.current_tick,
                "tank_type": self._get(enemy, 'tank_type', 'MISSING'),
                "team": self._get(enemy, 'team', 'MISSING'),
            }
        

    def _process_meta_info(self):
        my_pos = self.dynamic_info.get('position', None)
        my_angle = self.dynamic_info.get('barrel_angle', 0.0)
        visible_enemies = self.dynamic_info.get('visible_enemies', [])

        if not my_pos or not visible_enemies:
            return

        closest_angle = 10.0
        closest_dist = 9999.0
        target_id = None
        is_targeted = False

        if visible_enemies:
            try:
                closest = sorted(visible_enemies, key=lambda t: self._get(t, 'distance', 9999))[0]
                target_id = self._get(closest, 'id', None)
                closest_dist = self._get(closest, 'distance', 9999)

                dx = self._get(self._get(closest, 'position', None), 'x', 0) - self._get(my_pos, 'x', 0)
                dy = self._get(self._get(closest, 'position', None), 'y', 0) - self._get(my_pos, 'y', 0)
                angle_rad = math.atan2(dy, dx)
                desired_angle = math.degrees(angle_rad)
                diff = (desired_angle - my_angle + 180) % 360 - 180
                closest_angle = diff
            except Exception as e:
                logging.error(f"[ERROR] Failed to calculate closest enemy angle: {e}")

        for enemy in visible_enemies:
            try:
                dx = self._get(my_pos, 'x', 0) - self._get(self._get(enemy, 'position', None), 'x', 0)
                dy = self._get(my_pos, 'y', 0) - self._get(self._get(enemy, 'position', None), 'y', 0)
                angle_to_me = math.degrees(math.atan2(dy, dx))
                aim_diff = (angle_to_me - self._get(enemy, 'barrel_angle', 0) + 180) % 360 - 180
                if abs(aim_diff) < 15:
                    is_targeted = True
                    break
            except Exception as e:
                logging.error(f"[ERROR] Failed to check if enemy is aiming: {e}")

        self.meta_info = {
            "closest_enemy_angle": closest_angle,
            "closest_enemy_dist": closest_dist,
            "target_id": target_id,
            "is_aimed_at": is_targeted,
        }

    def _angle_diff(self, target_deg: float, current_deg: float) -> float:
        return (target_deg - current_deg + 180) % 360 - 180

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
    

    def _scan_strategy(self) -> float:
            """
            heading: aboslute to map
            barrel_angle: relative to tank
            barrel_abs = (heading + barrel_angle) % 360

            Modes (movement_list[0]):
            1: fast scan (continuous spin)
            2: slow scan (continuous spin, half speed)
            3: random look
            4: look in heading direction always (barrel_abs == heading)
            5: look around heading with sweep -90..+90 (memory-based)
            6: as 5, plus occasional full 360 scan (memory-based)
            """
            if not self.movement_list:
                return 0.0

            mode = self.movement_list[0]

            barrel_spin = float(self.static_info.get("barrel_spin_rate", 0.0))
            if barrel_spin <= 0.0:
                return 0.0

            heading_abs = float(self.dynamic_info.get("heading", 0.0))
            barrel_rel = float(self.dynamic_info.get("barrel_angle", 0.0))
            barrel_abs = (heading_abs + barrel_rel) % 360.0

            def clamp_step(step: float) -> float:
                return self._clamp(step, -barrel_spin, barrel_spin)

            # 1) Fast scan: constant rotation
            if mode == 1:
                return clamp_step(barrel_spin * self.scan_direction)

            # 2) Slow scan: constant rotation, slower
            if mode == 2:
                return clamp_step((barrel_spin * 0.5) * self.scan_direction)

            # 3) Random look
            if mode == 3:
                return random.uniform(-barrel_spin, barrel_spin)

            # 4) Lock barrel to heading direction
            if mode == 4:
                desired_abs = heading_abs
                err = self._angle_diff(desired_abs, barrel_abs)
                # Reset scan memory to avoid drift when switching modes
                self.scan_offset = 0.0
                self.scan_direction = 1.0
                self.full_scan_active = False
                return clamp_step(err)

            # Modes 5/6: partial sweep around heading
            scan_center_abs = heading_abs
            self.scan_max_offset = 90.0

            # 6) Occasionally do a full 360 scan (relative rotation)
            if mode == 6:
                if (not self.full_scan_active) and (self.current_tick - self.last_full_scan_tick >= self.full_scan_interval):
                    self.full_scan_active = True
                    self.full_scan_remaining = 360.0
                    self.last_full_scan_tick = self.current_tick

                if self.full_scan_active:
                    step = barrel_spin * self.scan_direction
                    self.full_scan_remaining -= abs(step)
                    if self.full_scan_remaining <= 0.0:
                        self.full_scan_active = False
                        self.scan_offset = 0.0
                    return clamp_step(step)

            # 5) (and 6 when not in full scan): oscillate offset and aim to (center + offset)
            if mode in (5, 6):
                # advance offset in world terms
                step = barrel_spin * self.scan_direction
                self.scan_offset += step

                if self.scan_offset > self.scan_max_offset:
                    self.scan_offset = self.scan_max_offset
                    self.scan_direction = -1.0
                elif self.scan_offset < -self.scan_max_offset:
                    self.scan_offset = -self.scan_max_offset
                    self.scan_direction = 1.0

                desired_abs = (scan_center_abs + self.scan_offset) % 360.0
                err = self._angle_diff(desired_abs, barrel_abs)
                return clamp_step(err)

            return 0.0
    
    def _divide_seen_area(self, visible_obstacles, visible_terrains):
        # current position
        my_pos = self.dynamic_info.get("position") or {"x": 0.0, "y": 0.0}
        me_x = float(my_pos.get("x", 0.0))
        me_y = float(my_pos.get("y", 0.0))

        # cutoff for "recent"
        recent_cutoff = int(self.current_tick) - int(RECENT_TICKS_GRAPH)

        # only terrain seen recently
        terrain_info = {
            cell: info for cell, info in self.map_memory.items()
            if int(info.get("last_seen_tick", -10_000)) >= recent_cutoff
        }

        # classify bad terrain (damage or slow)
        bad_terrain_cells = set()
        for cell, info in terrain_info.items():
            dmg = int(info.get("dmg", 0))
            spd = float(info.get("speed", 1.0))
            if dmg > 0 or spd < 0.9:
                bad_terrain_cells.add(cell)

        # only obstacles seen recently 
        recent_obstacle_cells = {
            cell for cell, t in self.obstacle_memory.items()
            if int(t) >= recent_cutoff
        }

        # blocked cells (optionally inflated) 
        blocked_cells = set()
        inflate_walls = 0  # TODO Odblokować i zobsvzyć co się wydarzy
        for (cx, cy) in recent_obstacle_cells:
            for dx in range(-inflate_walls, inflate_walls + 1):
                for dy in range(-inflate_walls, inflate_walls + 1):
                    blocked_cells.add((cx + dx, cy + dy))

        # risk buffer around bad terrain 
        risk_cells = set()
        risk_radius = 2  # "safety buffer" size in SUBCELLS (i.e., sub-grid cells)
        for (cx, cy) in bad_terrain_cells:
            for dx in range(-risk_radius, risk_radius + 1):
                for dy in range(-risk_radius, risk_radius + 1):
                    risk_cells.add((cx + dx, cy + dy))

        # relevant cells we will build nodes for 
        all_relevant_cells = set(terrain_info.keys()) | risk_cells

        # fallback window if we have too few recent cells (prevents empty/disconnected graph)
        if len(all_relevant_cells) < 50:
            me_cell = self._cell_from_xy(me_x, me_y)
            R = 5  # radius in sub-cells (=> (2R+1)^2 nodes)
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    all_relevant_cells.add((me_cell[0] + dx, me_cell[1] + dy))

        # default terrain for cells in risk buffer that weren't directly seen 
        default_info = {"dmg": 0, "speed": 1.0, "last_seen_tick": -10_000}

        # build nodes 
        nodes_by_cell = {}
        for cell in all_relevant_cells:
            info = terrain_info.get(cell, default_info)

            wx, wy = self._cell_center(cell)
            dist = math.hypot(wx - me_x, wy - me_y)

            dmg = int(info.get("dmg", 0))
            spd = float(info.get("speed", 1.0))

            nodes_by_cell[cell] = GridNode(
                cell=cell,
                world=(wx, wy),
                dmg=dmg,
                speed=spd,
                blocked=(cell in blocked_cells),
                dist_to_me=dist,
                is_risk=(cell in risk_cells and cell not in bad_terrain_cells),
                neighbors=[]
            )

        # connectivity 
        DIRS8 = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
        for (x, y), node in nodes_by_cell.items():
            for dx, dy in DIRS8:
                nb = (x + dx, y + dy)
                if nb in nodes_by_cell:
                    node.neighbors.append(nb)

        return list(nodes_by_cell.values())

        
    def _select_target_point(self, divided_area, current_target=None, top_k=20, forbid_cells=None):
        forbid_cells = forbid_cells or set()

        # odrzuć zablokowane + zabronione
        candidates = [n for n in divided_area if (not n.blocked and n.cell not in forbid_cells)]
        if not candidates:
            return []

        # Podział na bezpieczne i niebezpieczne
        safe_candidates = [n for n in candidates if n.dmg == 0]
        pool = safe_candidates if safe_candidates else candidates

        # top_k najdalszych
        pool.sort(key=lambda n: n.dist_to_me, reverse=True)
        top = pool[:min(top_k, len(pool))]

        def sort_key(n):
            is_current = (n.cell == current_target)
            dist_bonus = 50.0 if is_current else 0.0
            return (n.dmg, n.is_risk, -(n.dist_to_me + dist_bonus))

        top.sort(key=sort_key)
        print("Targets: ", [n.cell for n in top])
        return [n.cell for n in top]
    
    def _a_star(self, nodes, goal_cell):
        if not nodes:
            return None
            
        node_by_cell = {n.cell: n for n in nodes}
        
        
        if goal_cell not in node_by_cell:
            print(f"[A*] goal {goal_cell} not in graph (nodes={len(node_by_cell)})")

            return None
        
            
        # Znajdź start
        my_pos = self.dynamic_info.get("position", {"x": 0.0, "y": 0.0})
        sx, sy = float(my_pos.get("x", 0.0)), float(my_pos.get("y", 0.0))
        
        start_cell = None
        best_dist = float('inf')
        
        for cell, node in node_by_cell.items():
            wx, wy = node.world
            d2 = (wx-sx)**2 + (wy-sy)**2
            if d2 < best_dist:
                best_dist = d2
                start_cell = cell
                
        if start_cell is None:
            return None

        print("start_cell", start_cell)
        print("goal_cell", goal_cell)
        
        
        print("[A*] nodes:", len(node_by_cell))
        print("[A*] start deg:", len(node_by_cell[start_cell].neighbors))
        print("[A*] goal deg:", len(node_by_cell[goal_cell].neighbors))

        # Funkcja heurystyki
        def heuristic(c):
            return abs(c[0] - goal_cell[0]) + abs(c[1] - goal_cell[1])

        # f_score = g_score + h_score
        start_h = heuristic(start_cell)
        queue = [(start_h, 0.0, [start_cell])]
        
        # Słownik najlepszych kosztów dotarcia do pola (g_score)
        g_scores = {start_cell: 0.0}
        
        # Cache dla szumu (żeby nie generować w pętli)
        noise_map = {cell: random.uniform(0.0, 0.5) for cell in node_by_cell}

        while queue:
            # heapq.heappop jest O(1) - wyciąga element o najniższym f_score
            f, current_g, path = heapq.heappop(queue)
            current_node = path[-1]

            if current_node == goal_cell:
                return path
            
            # Jeśli znaleźliśmy już szybszą drogę do tego węzła w międzyczasie -> skip
            if current_g > g_scores.get(current_node, float('inf')):
                continue

            # Sprawdzanie sąsiadów
            for neighbour in node_by_cell[current_node].neighbors:
                neighbor_node = node_by_cell[neighbour]
                
                # --- Logika Kosztów ---
                
                # Soft Block dla ścian (umożliwia ucieczkę z inflacji)
                obst_penalty = 100000.0 if neighbor_node.blocked else 0.0
                
                # Teren i obrażenia
                speed_loss = max(0.0, 1.0 - neighbor_node.speed)
                move_cost = BASE_MOVE_COST + (speed_loss * SLOW_TERRAIN_PENALTY_WEIGHT)
                move_cost += float(neighbor_node.dmg) * DMG_PENALTY
                # move_cost += noise_map.get(neighbour, 0.0)
                move_cost += obst_penalty
                
                if neighbor_node.is_risk:
                    move_cost += DANGEROUS_NEIGHBOUR_PENALTY

                # Straight Line Penalty
                if len(path) >= 2:
                    prev = path[-2]
                    curr = current_node
                    nxt = neighbour
                    if (curr[0]-prev[0], curr[1]-prev[1]) == (nxt[0]-curr[0], nxt[1]-curr[1]):
                        move_cost += STRAIGHT_PENALTY

                new_g = current_g + move_cost

                # Relaksacja krawędzi
                if new_g < g_scores.get(neighbour, float('inf')):
                    g_scores[neighbour] = new_g
                    new_f = new_g + heuristic(neighbour)
                    heapq.heappush(queue, (new_f, new_g, path + [neighbour]))
                    
        return None
    
    def _Find_Target_and_Find_Path(self, override_goal_cell=None, forbid_cells=None):
        visible_obstacles = self.dynamic_info.get("visible_obstacles") or []
        visible_terrains  = self.dynamic_info.get("visible_terrains")  or []

        # nie blokuj replanu na braku widoczności — użyj pamięci
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

        path, goal, cost = self._plan_best_path(nodes, targets, max_targets=5)
        node_by_cell = {n.cell: n for n in nodes}
        return path, goal, cost, node_by_cell
    
##############################################################################################   
    def _FollowPath(self):
        if self.path_to_follow and self.path_index >= len(self.path_to_follow) - 1:
            # koniec ścieżki => wymuś nowy cel w następnym ticku
            self.path_to_follow = None
            self.current_goal_cell = None
            self.current_path_cost = None
            return 0.0, 0.0
        
        if not self.path_to_follow:
            raise RuntimeError("_FollowPath: path_to_follow is empty/None")

        my_position = self.dynamic_info.get("position")
        my_x = float(my_position.get("x", 0.0))
        my_y = float(my_position.get("y", 0.0))
        my_id = self.static_info.get("id")
        
        if self.path_index < 0:
            self.path_index = 0
        if self.path_index >= len(self.path_to_follow) - 1:
            return 0.0, 0.0  
        next_cell = self.path_to_follow[self.path_index + 1]
        next_x, next_y = self._cell_center(next_cell)
        
        dist = math.hypot(next_x - my_x, next_y - my_y)

        # warunki "reached"
        reach_threshold = self.CELL_SIZE * 1
        reached_by_distance = dist <= reach_threshold
        reached_by_timeout = self.path_stuck_ticks >= MAX_PATH_STUCK_TICKS
        
        if reached_by_distance:
            self.no_move_ticks = 0
            self.last_dist_to_wp = None

        if reached_by_distance or reached_by_timeout:
            if reached_by_timeout:
                print(f"[STUCK] skipping point {next_cell} after 50 ticks")
            self.path_index += 1
            self.path_stuck_ticks = 0

            if self.path_index >= len(self.path_to_follow) - 1:
                return 0.0, 0.0

            next_cell = self.path_to_follow[self.path_index + 1]
            next_x, next_y = self._cell_center(next_cell)
            dist = math.hypot(next_x - my_x, next_y - my_y)

        self.path_stuck_ticks += 1

        # kąt docelowy
        dx = next_x - my_x
        dy = next_y - my_y
        desired_heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        current_heading = float(self.dynamic_info.get("heading", 0.0))
        err = self._angle_diff(desired_heading, current_heading)

        heading_spin = float(self.static_info.get("heading_spin_rate", 0.0))
        heading_rotation_angle = self._clamp(err, -heading_spin, heading_spin)

        top_speed = float(self.static_info.get("top_speed", 0.0))

        # ruch zależny od tego jak bardzo jesteśmy odchyleni od kierunku
        
        #########
        move_speed = 0.0
        if abs(err) < 5:
            move_speed = top_speed
        else:
            # eśli kąt jest duży  jedzie wolno i się obraca
            move_speed = 0.5 * top_speed

        return heading_rotation_angle, move_speed
            
##############################################################################################  
    
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

            # Jeśli oba 0 (częste na płaskiej mapie), nie przełączaj
            if new_cost == 0 and old_cost == 0:
                return False

            if new_cost <= old_cost * (1.0 - IMPROVEMENT_MARGIN):
                return True
            if (old_cost - new_cost) >= MIN_ABS_IMPROVEMENT:
                return True
            return False

        # force logic
        force = should_force_replan()

        if (self.path_to_follow is None) or (self.path_to_follow and self.path_index >= len(self.path_to_follow) - 1):
            force = True

        if override_goal_cell is not None and override_goal_cell != self.current_goal_cell:
            force = True

        # forbid cells when no-move detected 
        forbid_cells = set()
        if self.force_change_goal and self.current_goal_cell is not None:
            forbid_cells.add(self.current_goal_cell)
            force = True

        # eval_now after force adjustments
        eval_now = should_eval() or force

        node_by_cell = None
        new_path = None
        new_goal = None
        new_cost = None

        if eval_now:
            print(f"[EVAL] tick={self.current_tick} force={force} "
            f"stuck={self.path_stuck_ticks} no_move={self.no_move_ticks} "
            f"path_end={(self.path_to_follow is None) or (self.path_index >= len(self.path_to_follow)-1)} "
            f"override_changed={(override_goal_cell is not None and override_goal_cell != self.current_goal_cell)}")
            self.last_eval_tick = self.current_tick

            new_path, new_goal, new_cost, node_by_cell = self._Find_Target_and_Find_Path(
                override_goal_cell=override_goal_cell,
                forbid_cells=forbid_cells
            )

            # DEBUG: always dump current graph snapshot 
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

                # start_cell: closest node to current position
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

                # goal candidate: even if path None
                if new_goal is not None:
                    self.debug_goal_cell_candidate = [int(new_goal[0]), int(new_goal[1])]
                elif override_goal_cell is not None:
                    self.debug_goal_cell_candidate = [int(override_goal_cell[0]), int(override_goal_cell[1])]

            # decyzja o zmianie ścieżki 
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

                    print(
                        f"[REPLAN] tick={self.current_tick} "
                        f"goal={self.current_goal_cell} cost={self.current_path_cost:.2f} "
                        f"force={force}"
                    )

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
    
##############################################################
    def _closest_visible_enemy(self):
        visible_enemies = self.dynamic_info.get("visible_enemies", [])
        if not visible_enemies:
            return None
        return min(visible_enemies, key=lambda enemy: self._get(enemy, "distance", 999999.0))


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

##############################################################
#############################
    def _process_action(self) -> ActionCommand:
        print(self.no_move_ticks)
        now = self.current_tick

        # update enemy memory (memory-only)
        enemy_now = self._update_attack_memory()
        self._maybe_forget_enemy_target()

        # commit flags
        in_attack = self._should_stay_in_attack()
        in_escape = self._should_stay_in_escape()

        # COMBAT EVENT: enemy visible -> choose/extend exactly ONE mode
        if enemy_now is not None:

            # (future hook) allow switching while commit active (currently disabled)
            if (in_attack or in_escape):
                switched = self._maybe_reconsider_combat_mode(enemy_now)  # returns False now
                if switched:
                    in_attack = self._should_stay_in_attack()
                    in_escape = self._should_stay_in_escape()

            # FIRST sighting (no active commit) -> pick ONE mode and commit it
            if (not in_attack) and (not in_escape):
                chosen = self._choose_combat_mode(enemy_now)  # attack/escape (for now random)
                self._commit_mode(chosen, enemy_now)
                # refresh flags after commit
                in_attack = self._should_stay_in_attack()
                in_escape = self._should_stay_in_escape()

            # Enemy still visible and commit active -> extend ONLY the active one
            else:
                if in_escape:
                    self._commit_mode("escape", enemy_now)
                elif in_attack:
                    self._commit_mode("attack", enemy_now)

        # MODE selection (priority)
        if self._should_stay_in_escape():
            MODE = "escape"
        elif self._should_stay_in_attack():
            MODE = "attack"
        elif self.powerup_target_cell is not None or (self.dynamic_info.get("visible_powerups") or []):
            MODE = "power_up"
        else:
            MODE = "search"

        self.mode = MODE
        print("MODE"); print(MODE); print("MODE")

        # ---------------------------
        # SEARCH
        # ---------------------------
        if MODE == "search":
            barrel_rot = self._scan_strategy()

            # FOLLOWER: override search target with furthest known friendly tank
            if self.tank_type_used and self.tank_type == "Follower":
                friend_goal = self._select_friend_goal_cell()
                if friend_goal is not None:
                    hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=friend_goal)
                else:
                    hull_rot, move_speed = self.Follow_Path_With_Modifiers()
            else:
                hull_rot, move_speed = self.Follow_Path_With_Modifiers()

            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=hull_rot,
                move_speed=move_speed,
                should_fire=False,
                ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
            )

        # ---------------------------
        # POWER UP
        # ---------------------------
        if MODE == "power_up":
            barrel_rot = self._scan_strategy()

            pu_goal = self._select_powerup_goal_cell()
            if pu_goal is not None:
                hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=pu_goal)
            else:
                hull_rot, move_speed = self.Follow_Path_With_Modifiers()

            # clear powerup target if we reached and it's gone
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

        # ---------------------------
        # ATTACK
        # ---------------------------
        if MODE == "attack":
            enemy = enemy_now if enemy_now is not None else self._closest_visible_enemy()

            # no visible enemy -> go to last known cell and scan
            if enemy is None:
                barrel_rot = self._scan_strategy()
                hull_rot, move_speed = self.Follow_Path_With_Modifiers(
                    override_goal_cell=self.enemy_target_cell if self.enemy_target_cell is not None else None
                )
                return ActionCommand(
                    barrel_rotation_angle=barrel_rot,
                    heading_rotation_angle=hull_rot,
                    move_speed=move_speed,
                    should_fire=False,
                    ammo_to_load="LONG_DISTANCE"
                )

            # distance
            pos_my = self.dynamic_info.get("position") or {}
            pos_enemy = self._get(enemy, "position", {}) or {}
            mx = float(self._get(pos_my, "x", 0.0))
            my = float(self._get(pos_my, "y", 0.0))
            ex = float(self._get(pos_enemy, "x", 0.0))
            ey = float(self._get(pos_enemy, "y", 0.0))
            dist = math.hypot(ex - mx, ey - my)

            # pick ammo
            inv = self._ammo_inventory()
            loaded = self._loaded_ammo_name()

            desired = None
            for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"]:
                if inv.get(a, 0) > 0 and self._ammo_range_world(a) >= dist:
                    desired = a
                    break
            if desired is None:
                for a in ["LONG_DISTANCE", "LIGHT", "HEAVY"]:
                    if inv.get(a, 0) > 0:
                        desired = a
                        break

            barrel_rot = self._aim_barrel_at_enemy(enemy)

            if desired is None:
                return ActionCommand(
                    barrel_rotation_angle=barrel_rot,
                    heading_rotation_angle=0.0,
                    move_speed=0.0,
                    should_fire=False,
                    ammo_to_load=self._fallback_ammo_to_load(),
                )

            if loaded != desired:
                # If enemy is out of range for the ammo we want, chase while loading.
                if dist > self._ammo_range_world(desired):
                    enemy_cell = self._cell_from_xy(ex, ey)
                    self.enemy_target_cell = enemy_cell
                    hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)
                else:
                    hull_rot, move_speed = 0.0, 0.0  # in-range: stop to stabilize aim

                return ActionCommand(
                    barrel_rotation_angle=barrel_rot,
                    heading_rotation_angle=hull_rot,
                    move_speed=move_speed,
                    should_fire=False,
                    ammo_to_load=desired
                )
        
            in_range = (self._ammo_range_world(desired) >= dist)
            if in_range:
                can_fire = self._can_fire_at_enemy_with_range(enemy, desired, aim_tolerance_deg=5.0)

                # desired standoff range: np. 0.6 zasięgu amunicji
                desired_range = 0.60 * self._ammo_range_world(desired)

                # tylko gdy wróg celuje w nas albo losowo co jakiś czas
                if self.meta_info.get("is_aimed_at", False) or random.random() < 0.25:
                    hull_rot, move_speed = self._attack_micro_dodge(enemy, desired_range_world=desired_range, band=2.0)
                else:
                    hull_rot, move_speed = 0.0, 0.0

                return ActionCommand(
                    barrel_rotation_angle=barrel_rot,
                    heading_rotation_angle=hull_rot,
                    move_speed=move_speed,
                    should_fire=bool(can_fire),
                    ammo_to_load=desired
    )
                

            enemy_cell = self._cell_from_xy(ex, ey)
            self.enemy_target_cell = enemy_cell
            hull_rot, move_speed = self.Follow_Path_With_Modifiers(override_goal_cell=enemy_cell)

            return ActionCommand(
                barrel_rotation_angle=barrel_rot,
                heading_rotation_angle=hull_rot,
                move_speed=move_speed,
                should_fire=False,
                ammo_to_load=desired
            )

        # ---------------------------
        # ESCAPE
        # ---------------------------
        if MODE == "escape":
            return self._mode_escape(enemy_now)

        # fallback (should not happen)
        barrel_rot = self._scan_strategy()
        hull_rot, move_speed = self.Follow_Path_With_Modifiers()
        return ActionCommand(
            barrel_rotation_angle=barrel_rot,
            heading_rotation_angle=hull_rot,
            move_speed=move_speed,
            should_fire=False,
            ammo_to_load=random.choice(["LIGHT", "HEAVY", "LONG_DISTANCE"])
        )

############################
    def destroy(self):
        self.is_destroyed = True
        logging.info(f"[{self.name}] Tank destroyed!")

    def end(self, damage_dealt: float, tanks_killed: int):
        logging.info(f"[{self.name}] Game ended!")
        logging.info(f"[{self.name}] Damage dealt: {damage_dealt}")
        logging.info(f"[{self.name}] Tanks killed: {tanks_killed}")

# ============================================================================
# FASTAPI SERVER
# ============================================================================

app = FastAPI(
    title="Random Test Agent",
    description="Random walking and shooting agent for testing",
    version="1.0.0"
)

# Global agent instance
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
    """Called when the tank is destroyed."""
    agent.destroy()


@app.post("/agent/end", status_code=204)
async def end(payload: Dict[str, Any] = Body(...)):
    """Called when the game ends."""
    agent.end(
        damage_dealt=payload.get('damage_dealt', 0.0),
        tanks_killed=payload.get('tanks_killed', 0)
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random test agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--name", type=str, default=None, help="Agent name")
    parser.add_argument("--modifier", type=str, default=None, help="Modifier")    #### ACHTUNG - sprawdzić czy modifier działa 
    args = parser.parse_args()
    
    if args.name:
        agent.name = args.name
        agent.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(agent.state_dir, exist_ok=True)
        agent.state_path = os.path.join(agent.state_dir, f"agent_state_{agent.name}.json")
        print("State path:", agent.state_path)
    else:
        agent.name = f"RandomBot_{args.port}"
        agent.state_dir = os.path.join(os.path.dirname(__file__), "agent_states")
        os.makedirs(agent.state_dir, exist_ok=True)
        agent.state_path = os.path.join(agent.state_dir, f"agent_state_{agent.name}.json")
        print("State path:", agent.state_path)
        
    if args.modifier != None:
        try:
            agent.modifier = args.modifier
            modifier = agent.modifier.split("_")
            movement_list = [int(element) for element in modifier]
            agent.movement_list = movement_list
            print("agent.movement_list: ", agent.movement_list)
        except:
            print("No modifier except")
    else:
        print("No mdofier")

    print(f"Starting {agent.name} on {args.host}:{args.port}, with modifier {args.modifier}")
    
    uvicorn.run(app, host=args.host, port=args.port)
