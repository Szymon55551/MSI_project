"""
Skrypt do uruchamiania pełnej symulacji gry w trybie graficznym (headful).

Ten skrypt automatycznie:
1. Uruchamia wymaganą liczbę serwerów agentów w osobnych procesach.
2. Inicjalizuje Pygame i ładuje zasoby graficzne.
3. Uruchamia główną pętlę gry, która łączy logikę silnika z renderowaniem w Pygame.
4. Wyświetla na bieżąco stan gry: pozycje czołgów, strzały, power-upy.
5. Po zakończeniu gry zamyka okno i serwery agentów.
"""

prev_pos = {}  # tank_id -> (x, y)
fov_dbg = True

import json
TILE_SIZE = 10
SUBDIV = 3
CELL_SIZE = TILE_SIZE / SUBDIV

def cell_center(cell):
    return ((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)


import ctypes
try:
    # To naprawia problem skalowania DPI w Windows
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass  # Ignoruj, jeśli to nie Windows lub starsza wersja

import subprocess
import sys
import os
import time
import random
from pygame.math import Vector2
import pygame
import math
from typing import Dict, Any, List

# --- Konfiguracja Ścieżek ---
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_file_dir)
    AGENT_STATE_DIR = os.path.join(main_dir, "03_FRAKCJA_AGENTOW", "agent_states")

    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    from backend.engine.game_loop import GameLoop, TEAM_A_NBR, TEAM_B_NBR, AGENT_BASE_PORT, TankScoreboard
    from backend.utils.logger import set_log_level
    from backend.engine.physics import process_physics_tick
    from controller.api import ActionCommand, AmmoType
    from backend.tank.base_tank import Tank
    from backend.tank.light_tank import LightTank
    from backend.structures.position import Position

except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że skrypt jest uruchamiany z katalogu '02_FRAKCJA_SILNIKA' lub że struktura projektu jest poprawna.")
    sys.exit(1)

# --- DEBUG: predicted shot rays (max range) ---
PREDICTED_SHOTS = []   # list of dicts: {"start": (sx,sy), "end": (ex,ey), "life": int, "dist_world": float, "ammo": str}

# Jeśli chcesz "zostajemy w pixelach/world units", to range też musi być w world units.
# Ustaw to na to, co Twoim zdaniem jest specem silnika.
AMMO_RANGE_WORLD = {
    "HEAVY": 25.0,
    "LIGHT": 50.0,
    "LONG_DISTANCE": 100.0
    }


# --- Stałe Konfiguracyjne Grafiki ---
LOG_LEVEL = "DEBUG"
#MAP_SEED = "road_trees.csv"
MAP_SEED = "road.csv"
TARGET_FPS = 60
SCALE = 3.5 # Współczynnik skalowania grafiki (wszystko będzie 4x większe)
TILE_SIZE = 10  # To MUSI być zgodne z domyślną wartością w map_loader.py
AGENT_NAME = "random_agent.py" # Nazwa pliku agenta

AGENT_FILES = [
    "random_agent_seba.py",
    "random_agent.py",
]

ARGUMENTS = [
    "2",
    None,

]

ASSETS_BASE_PATH = os.path.join(current_file_dir, 'frontend', 'assets')
TILE_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tiles')
POWERUP_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'power-ups')
TANK_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'tanks')
ICONS_ASSETS_PATH = os.path.join(ASSETS_BASE_PATH, 'icons')

BACKGROUND_COLOR = (20, 20, 30)
TEAM_COLORS = {
    1: (50, 150, 255),  # Niebieski
    2: (255, 50, 50)    # Czerwony
}

TANK_ASSET_MAP = {
    "LIGHT": "light_tank",
    "HEAVY": "heavy_tank",
    "Sniper": "sniper_tank"
}

POWERUP_ASSET_MAP = {
    "MEDKIT": "Medkit",
    "SHIELD": "Shield",
    "OVERCHARGE": "Overcharge",
    "AMMO_HEAVY": "AmmoBox_Heavy",
    "AMMO_LIGHT": "AmmoBox_Light",
    "AMMO_LONG_DISTANCE": "AmmoBox_Sniper",
}

def _normalize_angle_180(a: float) -> float:
    """[-180, 180]"""
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

def draw_shot_debug_cone_and_hitdot(
    map_surface: pygame.Surface,
    shooter: Tank,
    all_tanks: dict,             # game_loop.tanks
    agent_actions: dict,         # game_loop.last_actions
    scale: float,
    map_h: int,
    angle_eps_deg: float = 5.0,
):
    """
    Wizualizacja dokładnie pod aktualne fire_projectile():
    - strzał trafia gdy abs(angle_to_target - shoot_direction) <= angle_eps_deg
    - wybieramy najbliższy w zasięgu
    - celowanie jest do CENTER (Position) celu
    """

    # tylko jeśli agent kazał strzelać w tym ticku
    act = agent_actions.get(shooter._id)
    if not act or not getattr(act, "should_fire", False):
        return

    # ammo + range (WORLD units!) — dopasuj tu, jeśli w silniku Range jest w innych jednostkach
    ammo_loaded = getattr(shooter, "ammo_loaded", None)
    ammo_name = getattr(ammo_loaded, "name", str(ammo_loaded)).upper() if ammo_loaded else "NONE"

    # Uwaga: w Twoim silniku ammo_range bierzesz z ammo.value["Range"].
    # Tu (renderer) używamy Twojej mapy AMMO_RANGE_WORLD.
    r_world = float(AMMO_RANGE_WORLD.get(ammo_name, 0.0))
    if r_world <= 0.0:
        return

    # kierunek strzału jak w fizyce: heading + barrel_angle
    shoot_dir_deg = _normalize_angle_180(shooter.heading + shooter.barrel_angle)

    # pozycja i “koniec lufy” (tak jak u Ciebie w efektach)
    shooter_center = Vector2(shooter.position.x * scale, map_h - (shooter.position.y * scale))
    visual_dir = Vector2(1, 0).rotate(-shoot_dir_deg)  # screen-space (minus, bo Y flip)
    barrel_len = (TILE_SIZE * scale) * 0.8
    barrel_tip = shooter_center + visual_dir * barrel_len

    # narysuj dwa promienie graniczne ±5°
    left_dir  = Vector2(1, 0).rotate(-(shoot_dir_deg - angle_eps_deg))
    right_dir = Vector2(1, 0).rotate(-(shoot_dir_deg + angle_eps_deg))

    left_end  = barrel_tip + left_dir  * (r_world * scale)
    right_end = barrel_tip + right_dir * (r_world * scale)

    pygame.draw.line(map_surface, (255, 255, 0), (int(barrel_tip.x), int(barrel_tip.y)), (int(left_end.x), int(left_end.y)), 2)
    pygame.draw.line(map_surface, (255, 255, 0), (int(barrel_tip.x), int(barrel_tip.y)), (int(right_end.x), int(right_end.y)), 2)

    # znajdź “trafiony” cel wg tej samej logiki co fire_projectile()
    best_target = None
    best_dist = r_world  # closest_hit_distance init

    sx, sy = shooter.position.x, shooter.position.y

    for tid, target in all_tanks.items():
        if target._id == shooter._id:
            continue
        if not target.is_alive():
            continue

        dx = target.position.x - sx
        dy = target.position.y - sy
        dist = math.hypot(dx, dy)
        if dist >= best_dist:
            continue

        angle_to_target = math.degrees(math.atan2(dy, dx))
        if abs(_normalize_angle_180(angle_to_target - shoot_dir_deg)) <= angle_eps_deg:
            best_dist = dist
            best_target = target

    # duża czerwona kropka NAD celem (center + offset w screen-space)
    if best_target is not None:
        tx = best_target.position.x * scale
        ty = map_h - (best_target.position.y * scale)

        dot_pos = (int(tx), int(ty))
        
        pygame.draw.circle(map_surface, (255, 0, 0), dot_pos, 12)      # duża kropka
        pygame.draw.circle(map_surface, (0, 0, 0), dot_pos, 12, 2)     # obrys dla czytelności

        # opcjonalnie: podpis dystansu
        font = pygame.font.Font(None, 18)
        label = font.render(f"HIT? {best_dist:.1f}", True, (255, 0, 0))
        map_surface.blit(label, (dot_pos[0] + 14, dot_pos[1] - 10))

def draw_tank_weapon_range(map_surface: pygame.Surface, tank: Tank, scale: float, map_h: int):
    """
    Rysuje ciągłą linię zasięgu od końca lufy do max range aktualnie załadowanej amunicji.
    Jednostki range traktujemy jako WORLD (czyli te same co pozycje x/y).
    """
    # ammo name
    ammo_loaded = getattr(tank, "ammo_loaded", None)
    ammo_name = getattr(ammo_loaded, "name", str(ammo_loaded)).upper() if ammo_loaded else "NONE"

    # zasięgi WORLD (jak ustaliłeś: zostajemy w pixel/world, nic nie zmieniamy)
    AMMO_RANGE_WORLD = {
        "HEAVY": 25.0,
        "LIGHT": 50.0,
        "LONG_DISTANCE": 100.0,
    }
    r_world = float(AMMO_RANGE_WORLD.get(ammo_name, 0.0))
    if r_world <= 0.0:
        return

    # kąt jak w Twoich efektach strzału (spójnie z wieżą)
    final_turret_angle = tank.heading + tank.barrel_angle

    # kierunek w screen-space
    visual_dir = Vector2(1, 0).rotate(-final_turret_angle)  # 0° = w prawo
    
    # środek tanka w screen coords
    tank_center = Vector2(tank.position.x * scale, map_h - (tank.position.y * scale))

    # koniec lufy (heurystyka jak wcześniej)
    barrel_len = (TILE_SIZE * scale) * 0.8
    barrel_tip = tank_center + visual_dir * barrel_len

    # punkt końcowy zasięgu
    end_pt = barrel_tip + visual_dir * (r_world * scale)

    # kolor wg drużyny (albo stały, tu: biały + lekki outline)
    pygame.draw.line(map_surface, (255, 255, 255), (int(barrel_tip.x), int(barrel_tip.y)), (int(end_pt.x), int(end_pt.y)), 2)

    # podpis w połowie
    mx = int((barrel_tip.x + end_pt.x) * 0.5)
    my = int((barrel_tip.y + end_pt.y) * 0.5)
    font = pygame.font.Font(None, 18)
    label = f"{ammo_name} {r_world:.1f}"
    surf = font.render(label, True, (255, 255, 255))
    map_surface.blit(surf, (mx + 6, my + 6))

def draw_fov_overlay(map_surface, fov_dbg, scale, map_h, alpha=80):
    if not fov_dbg:
        return

    origin = fov_dbg.get("origin")
    cells = fov_dbg.get("cells", [])
    rays = fov_dbg.get("rays", [])

    if not origin:
        return

    overlay = pygame.Surface(map_surface.get_size(), pygame.SRCALPHA)

    # 1) wypełnienie sub-komórek w FOV (zielony półprzezroczysty)
    # rysujemy jako małe recty w skali SUBDIV
    cell_px = int(CELL_SIZE * scale)

    for c in cells:
        ix, iy = int(c[0]), int(c[1])
        # lewy-dolny róg komórki w świecie:
        world_left = ix * CELL_SIZE
        world_bottom = iy * CELL_SIZE

        px_left = int(world_left * scale)
        px_top  = int(map_h - ((world_bottom + CELL_SIZE) * scale))
        # pygame.draw.rect(overlay, (0, 255, 0, alpha), (px_left, px_top, cell_px, cell_px))

    # 2) promienie graniczne (żółte)
    ox = float(origin["x"]); oy = float(origin["y"])
    sox = int(ox * scale); soy = int(map_h - (oy * scale))

    for r in rays:
        to = r.get("to")
        if not to:
            continue
        tx = float(to["x"]); ty = float(to["y"])
        stx = int(tx * scale); sty = int(map_h - (ty * scale))
        pygame.draw.line(overlay, (255, 255, 0, 200), (sox, soy), (stx, sty), 2)

    # 3) punkt origin (biały)
    pygame.draw.circle(overlay, (255, 255, 255, 220), (sox, soy), 4)

    map_surface.blit(overlay, (0, 0))

def draw_graph_nodes(map_surface, debug, scale, map_h, fov_dbg=None, alpha=120):
    if not debug:
        return

    nodes = debug.get("graph_nodes")
    if not nodes:
        return

    # FOV musi iść "z silnika"/agenta -> bierzemy cells z debug["fov"] (albo z fov_dbg przekazanego z zewnątrz)
    fov = fov_dbg or debug.get("fov")
    fov_cells = set()
    if fov and fov.get("cells"):
        fov_cells = set((int(c[0]), int(c[1])) for c in fov["cells"])

    overlay = pygame.Surface(map_surface.get_size(), pygame.SRCALPHA)
    cell_px = int(CELL_SIZE * scale)

    for n in nodes:
        cx, cy = int(n["cell"][0]), int(n["cell"][1])

        in_fov = (cx, cy) in fov_cells
        blocked = bool(n.get("blocked", False))

        if blocked and not in_fov:
            col = (255, 0, 0, alpha)          # czerwone = blocked poza FOV
        elif (not blocked) and not in_fov:
            col = (0, 120, 255, alpha)        # niebieskie = free poza FOV
        elif blocked and in_fov:
            col = (255, 165, 0, alpha)       # pomarańczowe = blocked w FOV
        else:
            col = (0, 120, 255, alpha) 

        world_left = cx * CELL_SIZE
        world_bottom = cy * CELL_SIZE
        px_left = int(world_left * scale)
        px_top  = int(map_h - ((world_bottom + CELL_SIZE) * scale))

        pygame.draw.rect(overlay, col, (px_left, px_top, cell_px, cell_px))

    map_surface.blit(overlay, (0, 0))

    
    
    
def draw_start_goal(map_surface, debug, scale, map_h):
    if not debug:
        return

    start = debug.get("start_cell")
    goal  = debug.get("goal_cell_candidate")  # używamy “kandydata”

    def draw_cell(cell, color, r=7):
        if not cell: 
            return
        wx, wy = cell_center((int(cell[0]), int(cell[1])))
        sx = int(wx * scale)
        sy = int(map_h - (wy * scale))
        pygame.draw.circle(map_surface, color, (sx, sy), r)
        pygame.draw.circle(map_surface, (0,0,0), (sx, sy), r, 1)

    draw_cell(start, (0, 255, 0), r=7)     # start = zielony
    draw_cell(goal,  (255, 0, 255), r=7)   # goal  = magenta


def draw_agent_debug_path(map_surface, debug, scale, map_h):
    if not debug:
        return

    path = debug.get("path")
    goal = debug.get("goal_cell")
    path_index = debug.get("path_index", 0)

    if path:
        pts = []
        for i, cell in enumerate(path):
            wx, wy = cell_center(cell)
            sx = int(wx * scale)
            sy = int(map_h - (wy * scale))
            pts.append((sx, sy))

            if i <= path_index:
                color = (80, 80, 80)      
                r = 3
            else:
                color = (0, 255, 255)    
                r = 4

            pygame.draw.circle(map_surface, color, (sx, sy), r)

        if len(pts) >= 2:
            pygame.draw.lines(map_surface, (0, 255, 255), False, pts, 1)

    if goal:
        wx, wy = cell_center(goal)
        sx = int(wx * scale)
        sy = int(map_h - (wy * scale))
        pygame.draw.circle(map_surface, (255, 0, 255), (sx, sy), 7)   # magenta
        pygame.draw.circle(map_surface, (0, 0, 0), (sx, sy), 7, 1)    # outline


def draw_seen_terrain_tiles(map_surface, seen_tiles, scale, map_h, alpha=0.5):
    if not seen_tiles:
        return
    overlay = pygame.Surface(map_surface.get_size(), pygame.SRCALPHA)
    a = int(255 * alpha)
    w = int(TILE_SIZE * scale)
    h = int(TILE_SIZE * scale)

    for (tx, ty) in seen_tiles:
        world_left = tx * TILE_SIZE
        world_bottom = ty * TILE_SIZE
        px_left = int(world_left * scale)
        px_top  = int(map_h - ((world_bottom + TILE_SIZE) * scale))
        pygame.draw.rect(overlay, (0, 255, 0, a), (px_left, px_top, w, h))

    map_surface.blit(overlay, (0, 0))


# --- Funkcje Pomocnicze Renderowania ---

class ExplosionParticle:
    """Prosta klasa do zarządzania cząsteczkami eksplozji."""
    def __init__(self, pos, velocity, start_size, lifetime):
        self.pos = list(pos)
        self.velocity = list(velocity)
        self.size = start_size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        # Każda cząsteczka losuje swój kolor z palety eksplozji
        self.color = random.choice([
            (255, 0, 0),      # Czerwony
            (255, 100, 0),    # Pomarańczowy
            (255, 215, 0),    # Złoty/Żółty
            (139, 0, 0)       # Ciemnoczerwony
        ])

    def update(self):
        """Aktualizuje pozycję i czas życia cząsteczki."""
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.lifetime -= 1
        # Dodaj losowość do ruchu, aby dym się rozpraszał
        self.velocity[0] += random.uniform(-0.05, 0.05)
        self.velocity[1] += random.uniform(-0.05, 0.05)

    def draw(self, surface):
        """Rysuje cząsteczkę na podanej powierzchni."""
        if self.lifetime > 0:
            lerp_factor = self.lifetime / self.max_lifetime
            current_size = int(self.size * lerp_factor)
            if current_size > 0:
                r, g, b = self.color
                # Przyciemnianie koloru w miarę upływu życia cząsteczki
                final_color = (
                    int(r * lerp_factor),
                    int(g * lerp_factor),
                    int(b * lerp_factor)
                )
                pygame.draw.circle(surface, final_color, self.pos, current_size)

def generate_radial_explosion(particles_list: List[ExplosionParticle], position: tuple, num_particles: int):
    """Generuje promienisty "wybuch" cząsteczek w danym punkcie."""
    for _ in range(num_particles):
        angle = random.uniform(0, 360)
        speed = random.uniform(0.5, 2.0)
        velocity = Vector2(1, 0).rotate(angle) * speed

        particles_list.append(ExplosionParticle(
            pos=position, velocity=velocity,
            start_size=random.randint(3, 4), lifetime=random.randint(20, 40)
        ))

def generate_cone_explosion(particles_list: List[ExplosionParticle], position: tuple, num_particles: int, base_direction_vector: Vector2, cone_angle: float):
    """Generuje stożek cząsteczek eksplozji."""
    for _ in range(num_particles):
        # Losowy kąt wewnątrz stożka
        angle_offset = random.uniform(-cone_angle / 2, cone_angle / 2)
        # Losowa prędkość
        speed = random.uniform(1.5, 3.5)
        # Obróć wektor kierunku i pomnóż przez prędkość
        velocity = base_direction_vector.rotate(angle_offset) * speed
        
        particles_list.append(ExplosionParticle(
            pos=position, velocity=velocity, 
            start_size=random.randint(3, 5), lifetime=random.randint(20, 40)
        ))

def load_assets():
    """Ładuje wszystkie potrzebne zasoby graficzne."""
    assets = {
        'tiles': {},
        'powerups': {},
        'tanks': {},
        'icons': {}
    }
    print("--- Ładowanie zasobów graficznych ---")

    # Kafelki
    tile_names = ['Wall', 'Tree', 'AntiTankSpike', 'Grass', 'Road', 'Swamp', 'PotholeRoad', 'Water']
    for name in tile_names:
        try:
            path = os.path.join(TILE_ASSETS_PATH, f"{name}.png")
            img = pygame.image.load(path).convert_alpha()
            # Skalujemy asset do docelowego rozmiaru
            assets['tiles'][name] = pygame.transform.scale(img, (TILE_SIZE * SCALE, TILE_SIZE * SCALE))
        except pygame.error:
            print(f"[!] Nie znaleziono assetu dla kafelka: {name}")

    # Power-upy
    powerup_names = ['Medkit', 'Shield', 'Overcharge', 'AmmoBox_Heavy', 'AmmoBox_Light', 'AmmoBox_Sniper']
    powerup_render_size = (int(TILE_SIZE * SCALE * 0.8), int(TILE_SIZE * SCALE * 0.8))
    for name in powerup_names:
        try:
            path = os.path.join(POWERUP_ASSETS_PATH, f"{name}.png")
            img = pygame.image.load(path).convert_alpha()
            assets['powerups'][name] = pygame.transform.scale(img, powerup_render_size)
        except pygame.error:
            print(f"[!] Nie znaleziono assetu dla power-upa: {name}")

    # Czołgi
    tank_render_size = (TILE_SIZE * SCALE, TILE_SIZE * SCALE)
    for tank_type, folder_name in TANK_ASSET_MAP.items():
        try:
            base_path = os.path.join(TANK_ASSETS_PATH, folder_name)
            # Grafiki czołgów są domyślnie skierowane w lewo.
            assets['tanks'][tank_type] = {
                'body': pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'tnk1.png')).convert_alpha(), tank_render_size),
                'mask_body': pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'msk1.png')).convert_alpha(), tank_render_size),
                'turret': pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'tnk2.png')).convert_alpha(), tank_render_size),
                'mask_turret': pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'msk2.png')).convert_alpha(), tank_render_size),
            }
        except pygame.error:
            print(f"[!] Nie znaleziono assetów dla czołgu: {tank_type}")
            
    # Ikony
    icon_render_size = (128, 64)
    for tank_type, folder_name in TANK_ASSET_MAP.items():
        # folder_name to 'light_tank', 'heavy_tank', etc.
        icon_filename = f"{folder_name}.png"
        try:
            path = os.path.join(ICONS_ASSETS_PATH, icon_filename)
            img = pygame.image.load(path).convert_alpha()
            assets['icons'][tank_type] = pygame.transform.scale(img, icon_render_size)
        except pygame.error:
            # Jeśli nie ma ikony, stwórz pusty placeholder, żeby uniknąć błędów
            print(f"[!] Nie znaleziono assetu dla ikony: {icon_filename}")
            assets['icons'][tank_type] = pygame.Surface(icon_render_size, pygame.SRCALPHA)

    print("--- Ładowanie zakończone ---")
    return assets

def draw_tank(
    surface: pygame.Surface,
    tank: Tank,
    assets: Dict,
    scale: int,
    map_height: int,
    role: str = None,  # NEW: "Leader"/"Follower" or None
):
    """Rysuje pojedynczy czołg (żywy lub wrak) na ekranie z uwzględnieniem skali i odwróconej osi Y."""
    tank_assets = assets['tanks'].get(tank._tank_type)
    if not tank_assets:
        return

    is_alive = tank.is_alive()
    team_color = TEAM_COLORS.get(tank.team, (255, 255, 255))

    # Przeskalowana i odwrócona pozycja środka czołgu
    center_pos = (tank.position.x * scale, map_height - (tank.position.y * scale))

    # --- Kadłub ---
    body_img = tank_assets['body'].copy()
    if not is_alive:
        body_img.set_alpha(100)  # Półprzezroczysty wrak

    # Obrót: Kąty w silniku rosną zgodnie z zegarem, a w Pygame przeciwnie.
    # Dodatkowe -180 stopni, bo assety są skierowane w lewo.
    BODY_OFFSET = 0
    rotated_body = pygame.transform.rotate(body_img, tank.heading + BODY_OFFSET)
    body_rect = rotated_body.get_rect(center=center_pos)
    surface.blit(rotated_body, body_rect.topleft)

    # Maska koloru kadłuba
    mask_body_img = tank_assets['mask_body'].copy()
    if not is_alive:
        mask_body_img.set_alpha(100)

    color_layer = pygame.Surface(mask_body_img.get_size())
    color_layer.fill(team_color)
    color_layer.blit(mask_body_img, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    color_layer.set_colorkey((0, 0, 0))
    rotated_mask = pygame.transform.rotate(color_layer, -tank.heading - 180)
    surface.blit(rotated_mask, body_rect.topleft)

    # Wieżę rysujemy tylko dla żywych czołgów
    if is_alive:
        # --- Wieża ---
        turret_img = tank_assets['turret']
        total_turret_angle = tank.heading + tank.barrel_angle
        rotated_turret = pygame.transform.rotate(turret_img, -total_turret_angle - 180)
        turret_rect = rotated_turret.get_rect(center=center_pos)
        surface.blit(rotated_turret, turret_rect.topleft)

        # Maska koloru wieży
        mask_turret_img = tank_assets['mask_turret']
        turret_color_layer = pygame.Surface(mask_turret_img.get_size())
        turret_color_layer.fill(team_color)
        turret_color_layer.blit(mask_turret_img, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        turret_color_layer.set_colorkey((0, 0, 0))
        rotated_turret_mask = pygame.transform.rotate(turret_color_layer, -total_turret_angle - 180)
        surface.blit(rotated_turret_mask, turret_rect.topleft)

    # --- Pasek HP ---
    if is_alive:
        hp_bar_width = 40
        hp_bar_height = 5
        hp_ratio = max(0, tank.hp / tank._max_hp)

        hp_bar_x = center_pos[0] - hp_bar_width / 2
        hp_bar_y = center_pos[1] - (body_img.get_height() / 2) - 15

        pygame.draw.rect(surface, (50, 50, 50), (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        pygame.draw.rect(surface, (0, 255, 0), (hp_bar_x, hp_bar_y, hp_bar_width * hp_ratio, hp_bar_height))

    # --- NEW: Leader/Follower badge ---
    # role expected: "Leader" / "Follower" (or None)
    if role in ("Leader", "Follower"):
        badge_char = "L" if role == "Leader" else "F"

        # position above tank
        bx = int(center_pos[0])
        by = int(center_pos[1] - 35)

        # colors (simple + readable)
        bg = (0, 0, 0)
        fg = (255, 255, 255)
        ring = team_color  # ring matches team

        pygame.draw.circle(surface, bg, (bx, by), 10)
        pygame.draw.circle(surface, ring, (bx, by), 10, 2)

        badge_font = pygame.font.Font(None, 20)
        txt = badge_font.render(badge_char, True, fg)
        rect = txt.get_rect(center=(bx, by))
        surface.blit(txt, rect)

    # --- DEBUG: heading vector vs velocity vector ---
    cx, cy = center_pos

    # heading vector (world -> screen: y flipped)
    heading_len = 100
    theta = math.radians(-tank.heading)  # screen-space
    hx = cx + math.cos(theta) * heading_len
    hy = cy + math.sin(theta) * heading_len
    pygame.draw.line(surface, (255, 255, 255), (cx, cy), (hx, hy), 2)  # white = heading

    # velocity vector from last frame
    pid = tank._id
    p = prev_pos.get(pid)
    if p is not None:
        lastx, lasty = p
        vx = tank.position.x - lastx
        vy = tank.position.y - lasty

        # convert velocity to screen (flip y)
        vxs = vx
        vys = -vy

        vlen = math.hypot(vxs, vys)
        if vlen > 1e-6:
            scale_v = 60.0  # visual scale
            ex = cx + (vxs / vlen) * scale_v
            ey = cy + (vys / vlen) * scale_v
            pygame.draw.line(surface, (255, 255, 0), (cx, cy), (ex, ey), 2)  # yellow = motion

    prev_pos[pid] = (tank.position.x, tank.position.y)



def draw_shot_effect(surface: pygame.Surface, start_pos: Dict, end_pos: Dict, life: int, scale: int, map_height: int):
    """Rysuje linię symbolizującą strzał z uwzględnieniem skali."""
    if life > 0:
        alpha = int(255 * (life / 10.0)) # Efekt zanikania
        color = (255, 255, 0, alpha)
        line_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        # Skalowanie i odwracanie pozycji
        scaled_start = (start_pos.x * scale, map_height - (start_pos.y * scale))
        scaled_end = (end_pos.x * scale, map_height - (end_pos.y * scale))
        pygame.draw.line(line_surface, color, scaled_start, scaled_end, 2)
        surface.blit(line_surface, (0, 0))

def create_background_surface(map_info: Any, assets: Dict, scale: int, width: int, height: int) -> pygame.Surface:
    """Tworzy i zwraca powierzchnię z narysowaną statyczną mapą (teren + przeszkody)."""
    print("--- Tworzenie pre-renderowanego tła mapy ---")
    background = pygame.Surface((width, height))
    background.fill(BACKGROUND_COLOR)

    # Rysowanie terenu i przeszkód
    all_map_objects = map_info.terrain_list + map_info.obstacle_list
    for obj in all_map_objects:
        obj_class_name = obj.__class__.__name__
        asset = assets['tiles'].get(obj_class_name)
        if asset:
            # Pozycja obiektu to jego środek. Skalujemy ją i odwracamy oś Y.
            pos_x = obj._position.x * scale
            pos_y = height - (obj._position.y * scale)
            # Obliczamy lewy górny róg na podstawie przeskalowanego środka i rozmiaru assetu
            top_left = (pos_x - asset.get_width() / 2, pos_y - asset.get_height() / 2)
            background.blit(asset, top_left)
    
    print("--- Tło mapy utworzone ---")
    return background

def draw_ui(screen: pygame.Surface, font: pygame.font.Font, game_loop: GameLoop, window_width: int, map_rect: pygame.Rect, assets: Dict):
    """Rysuje interfejs użytkownika na bocznych panelach."""
    
    # Mniejsza czcionka dla szczegółów czołgów
    detail_font = pygame.font.Font(None, 22)

    # Statystyki drużyn
    team1_tanks = sorted([t for t in game_loop.tanks.values() if t.team == 1], key=lambda t: t._id)
    team1_alive = sum(1 for t in team1_tanks if t.is_alive())
    team1_kills = sum(s.tanks_killed for s in game_loop.scoreboards.values() if s.team == 1)
    
    team2_tanks = sorted([t for t in game_loop.tanks.values() if t.team == 2], key=lambda t: t._id)
    team2_alive = sum(1 for t in team2_tanks if t.is_alive())
    team2_kills = sum(s.tanks_killed for s in game_loop.scoreboards.values() if s.team == 2)

    # --- Panel lewy (Team 1) ---
    panel1_x = map_rect.left / 2
    current_y = 100
    
    title1_surf = font.render("TEAM 1", True, TEAM_COLORS[1])
    title1_rect = title1_surf.get_rect(center=(panel1_x, current_y))
    screen.blit(title1_surf, title1_rect)
    current_y += 50

    alive1_surf = font.render(f"Alive: {team1_alive}", True, (200, 200, 200))
    alive1_rect = alive1_surf.get_rect(center=(panel1_x, current_y))
    screen.blit(alive1_surf, alive1_rect)
    current_y += 30

    kills1_surf = font.render(f"Kills: {team1_kills}", True, (200, 200, 200))
    kills1_rect = kills1_surf.get_rect(center=(panel1_x, current_y))
    screen.blit(kills1_surf, kills1_rect)
    current_y += 30

    # Szczegóły czołgów drużyny 1
    for tank in team1_tanks:
        if not tank.is_alive():
            continue

        # Wyświetlanie ikony typu czołg
        icon_surf = assets['icons'].get(tank._tank_type)
        icon_rect = icon_surf.get_rect(centerx=panel1_x, top=current_y)
        screen.blit(icon_surf, icon_rect)
        current_y += 70

        # --- HP Bar ---
        hp_bar_width = 120
        hp_bar_height = 12
        hp_ratio = max(0, tank.hp / tank._max_hp)
        bar_x = panel1_x - hp_bar_width / 2
        bar_y = current_y
        pygame.draw.rect(screen, (100, 0, 0), (bar_x, bar_y, hp_bar_width, hp_bar_height))
        pygame.draw.rect(screen, TEAM_COLORS[1], (bar_x, bar_y, hp_bar_width * hp_ratio, hp_bar_height))
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, hp_bar_width, hp_bar_height), 1)
        current_y += hp_bar_height + 10

        # --- HP Text ---
        hp_text = f"{round(tank.hp,1)} / {tank._max_hp}"
        hp_surf = detail_font.render(hp_text, True, (255, 255, 255))
        hp_rect = hp_surf.get_rect(center=(panel1_x, current_y))
        screen.blit(hp_surf, hp_rect)
        current_y += 20

        # Wyświetlanie wszystkich typów amunicji
        for ammo_type in AmmoType:
            ammo_slot = tank.ammo.get(ammo_type)
            count = ammo_slot.count if ammo_slot else 0
            
            prefix = "> " if tank.ammo_loaded == ammo_type else "  "
            ammo_text = f"{prefix}{ammo_type.name}: {count}"
            
            ammo_surf = detail_font.render(ammo_text, True, (200, 200, 200))
            ammo_rect = ammo_surf.get_rect(center=(panel1_x, current_y))
            screen.blit(ammo_surf, ammo_rect)
            current_y += 20

        current_y += 3 # Dodatkowy odstęp między czołgami

    # --- Panel prawy (Team 2) ---
    panel2_x = map_rect.right + (window_width - map_rect.right) / 2
    current_y = 100

    title2_surf = font.render("TEAM 2", True, TEAM_COLORS[2])
    title2_rect = title2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(title2_surf, title2_rect)
    current_y += 50

    alive2_surf = font.render(f"Alive: {team2_alive}", True, (200, 200, 200))
    alive2_rect = alive2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(alive2_surf, alive2_rect)
    current_y += 30

    kills2_surf = font.render(f"Kills: {team2_kills}", True, (200, 200, 200))
    kills2_rect = kills2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(kills2_surf, kills2_rect)
    current_y += 30

    # Szczegóły czołgów drużyny 2
    for tank in team2_tanks:
        if not tank.is_alive():
            continue

        # Wyświetlanie ikony typu czołgu
        icon_surf = assets['icons'].get(tank._tank_type)
        icon_rect = icon_surf.get_rect(centerx=panel2_x, top=current_y)
        screen.blit(icon_surf, icon_rect)
        current_y += 70

        # --- HP Bar ---
        hp_bar_width = 120
        hp_bar_height = 12
        hp_ratio = max(0, tank.hp / tank._max_hp)
        bar_x = panel2_x - hp_bar_width / 2
        bar_y = current_y
        pygame.draw.rect(screen, (100, 0, 0), (bar_x, bar_y, hp_bar_width, hp_bar_height))
        pygame.draw.rect(screen, TEAM_COLORS[2], (bar_x, bar_y, hp_bar_width * hp_ratio, hp_bar_height))
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, hp_bar_width, hp_bar_height), 1)
        current_y += hp_bar_height + 10

        # --- HP Text ---
        hp_text = f"{round(tank.hp,1)} / {tank._max_hp}"
        hp_surf = detail_font.render(hp_text, True, (255, 255, 255))
        hp_rect = hp_surf.get_rect(center=(panel2_x, current_y))
        screen.blit(hp_surf, hp_rect)
        current_y += 20

        # Wyświetlanie wszystkich typów amunicji
        for ammo_type in AmmoType:
            ammo_slot = tank.ammo.get(ammo_type)
            count = ammo_slot.count if ammo_slot else 0
            
            prefix = "> " if tank.ammo_loaded == ammo_type else "  "
            ammo_text = f"{prefix}{ammo_type.name}: {count}"
            
            ammo_surf = detail_font.render(ammo_text, True, (200, 200, 200))
            ammo_rect = ammo_surf.get_rect(center=(panel2_x, current_y))
            screen.blit(ammo_surf, ammo_rect)
            current_y += 20

        current_y += 5 # Dodatkowy odstęp między czołgami


    # --- Panel prawy (Team 2) ---
    panel2_x = map_rect.right + (window_width - map_rect.right) / 2
    current_y = 100

    title2_surf = font.render("TEAM 2", True, TEAM_COLORS[2])
    title2_rect = title2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(title2_surf, title2_rect)
    current_y += 50

    alive2_surf = font.render(f"Alive: {team2_alive}", True, (200, 200, 200))
    alive2_rect = alive2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(alive2_surf, alive2_rect)
    current_y += 30

    kills2_surf = font.render(f"Kills: {team2_kills}", True, (200, 200, 200))
    kills2_rect = kills2_surf.get_rect(center=(panel2_x, current_y))
    screen.blit(kills2_surf, kills2_rect)
    current_y += 30

    # Szczegóły czołgów drużyny 2
    for tank in team2_tanks:
        if not tank.is_alive():
            continue

        # Wyświetlanie ikony typu czołgu
        icon_surf = assets['icons'].get(tank._tank_type)
        icon_rect = icon_surf.get_rect(centerx=panel2_x, top=current_y)
        screen.blit(icon_surf, icon_rect)
        current_y += 70

        # --- HP Bar ---
        hp_bar_width = 120
        hp_bar_height = 12
        hp_ratio = max(0, tank.hp / tank._max_hp)
        bar_x = panel2_x - hp_bar_width / 2
        bar_y = current_y
        pygame.draw.rect(screen, (100, 0, 0), (bar_x, bar_y, hp_bar_width, hp_bar_height))
        pygame.draw.rect(screen, TEAM_COLORS[2], (bar_x, bar_y, hp_bar_width * hp_ratio, hp_bar_height))
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, hp_bar_width, hp_bar_height), 1)
        current_y += hp_bar_height + 10

        # --- HP Text ---
        hp_text = f"{round(tank.hp,1)} / {tank._max_hp}"
        hp_surf = detail_font.render(hp_text, True, (255, 255, 255))
        hp_rect = hp_surf.get_rect(center=(panel2_x, current_y))
        screen.blit(hp_surf, hp_rect)
        current_y += 20

        # Wyświetlanie wszystkich typów amunicji
        for ammo_type in AmmoType:
            ammo_slot = tank.ammo.get(ammo_type)
            count = ammo_slot.count if ammo_slot else 0
            
            prefix = "> " if tank.ammo_loaded == ammo_type else "  "
            ammo_text = f"{prefix}{ammo_type.name}: {count}"
            
            ammo_surf = detail_font.render(ammo_text, True, (200, 200, 200))
            ammo_rect = ammo_surf.get_rect(center=(panel2_x, current_y))
            screen.blit(ammo_surf, ammo_rect)
            current_y += 20

        current_y += 5 # Dodatkowy odstęp między czołgami

def draw_debug_info(screen: pygame.Surface, font: pygame.font.Font, clock: pygame.time.Clock, current_tick: int):
    """Rysuje informacje debugowe (FPS, Tick) w lewym górnym rogu."""
    # Użyj mniejszej czcionki dla informacji debugowych
    debug_font = pygame.font.Font(None, 24)
    
    fps_text = f"FPS: {clock.get_fps():.1f}"
    tick_text = f"Tick: {current_tick}"
    
    fps_surf = debug_font.render(fps_text, True, (255, 255, 0))
    tick_surf = debug_font.render(tick_text, True, (255, 255, 0))
    
    screen.blit(fps_surf, (10, 10))
    screen.blit(tick_surf, (10, 30))



def main():
    """Główna funkcja uruchamiająca symulację z grafiką."""
    print("--- Uruchamianie symulacji w trybie graficznym ---")
    set_log_level(LOG_LEVEL)

    agent_processes = []
    total_tanks = TEAM_A_NBR + TEAM_B_NBR
    agent_script_path = os.path.join(main_dir, '03_FRAKCJA_AGENTOW', AGENT_NAME)

    if not os.path.exists(agent_script_path):
        print(f"BŁĄD: Nie znaleziono skryptu agenta w: {agent_script_path}")
        return

    # --- Inicjalizacja Gry ---
    game_loop = GameLoop(headless=False)

    try:
        # 1. Uruchomienie serwerów agentów (teraz używamy random_agent.py)
        print(f"Uruchamianie {total_tanks} serwerów agentów...")
        for i in range(total_tanks):
            port = AGENT_BASE_PORT + i
            name = f"Bot_{i+1}"

            agent_file = AGENT_FILES[i % len(AGENT_FILES)]
            agent_script_path = os.path.join(main_dir, '03_FRAKCJA_AGENTOW', agent_file)
            argument = ARGUMENTS[i % len(AGENT_FILES)]

            if argument is not None:
                command = [sys.executable, agent_script_path, "--port", str(port), "--name", name,  "--modifier", argument]
            else:
                command = [sys.executable, agent_script_path, "--port", str(port), "--name", name]

            proc = subprocess.Popen(command)
            agent_processes.append(proc)
            print(f"  -> Agent '{name}' uruchomiony na porcie {port} (PID: {proc.pid})")

        print("\nOczekiwanie 3 sekundy na start serwerów agentów...")
        time.sleep(3)

        # 2. Inicjalizacja silnika gry
        if not game_loop.initialize_game(map_seed=MAP_SEED):
            raise RuntimeError("Inicjalizacja pętli gry nie powiodła się!")

        # 3. Inicjalizacja Pygame i okna 16:9
        pygame.init()
        map_engine_width, map_engine_height = game_loop.map_info._size
        map_render_width = map_engine_width * SCALE
        map_render_height = map_engine_height * SCALE

        # Ustaw okno na pełny ekran
        # screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen = pygame.display.set_mode((1280, 720))  # bez pygame.FULLSCREEN
        window_width, window_height = screen.get_size()
        pygame.display.set_caption("Symulator Walk Czołgów")
        clock = pygame.time.Clock()
        assets = load_assets()
        font = pygame.font.Font(None, 42)
        start_font = pygame.font.Font(None, 72)

        # Utworzenie powierzchni do rysowania samej mapy
        map_surface = pygame.Surface((map_render_width, map_render_height))
        map_rect = map_surface.get_rect(center=(window_width / 2, window_height / 2))

        # OPTYMALIZACJA: Pre-renderowanie statycznego tła mapy
        background_surface = create_background_surface(game_loop.map_info, assets, SCALE, map_render_width, map_render_height)

        # --- Wyświetlanie informacji o spawnie ---
        print("\n--- Informacje o Spawnie ---")
        print("Zespawnowane czołgi:")
        if game_loop.tanks:
            # Sortowanie dla czytelności
            sorted_tanks = sorted(game_loop.tanks.values(), key=lambda t: t._id)
            for tank in sorted_tanks:
                print(f"  - Czołg: {tank._id} (Team: {tank.team}, Typ: {tank._tank_type}) na pozycji ({tank.position.x:.1f}, {tank.position.y:.1f})")
        else:
            print("  Brak czołgów.")

        print("\nZespawnowane power-upy:")
        if game_loop.map_info and game_loop.map_info.powerup_list:
            for powerup in game_loop.map_info.powerup_list:
                print(f"  - Power-up: {powerup.powerup_type.name} na pozycji ({powerup.position.x:.1f}, {powerup.position.y:.1f})")
        else:
            print("  Brak power-upów na mapie.")

        # --- TEST DIAGNOSTYCZNY: Wyświetlenie zamrożonej mapy i UI ---
        print("\n--- TEST: Wyświetlanie statycznej mapy i interfejsu ---")
        print("--- Naciśnij SPACJĘ, aby rozpocząć symulację ---")

        running = True
        waiting_for_start = True
        while waiting_for_start:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_start = False
                    running = False  # Ustaw flagę wyjścia z gry
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting_for_start = False

            if not running:  # Jeśli użytkownik zamknął okno, wyjdź z pętli oczekiwania
                break

            screen.fill(BACKGROUND_COLOR)
            map_surface.blit(background_surface, (0, 0)) # Narysuj tło na powierzchni mapy
            screen.blit(map_surface, map_rect) # Narysuj powierzchnię mapy na ekranie
            draw_ui(screen, font, game_loop, window_width, map_rect, assets) # Narysuj UI
            draw_debug_info(screen, font, clock, 0) # Pokaż info debugowe

            # --- DODANE: Pulsujący napis "Press SPACE to start" ---
            # Używamy sinusa do uzyskania płynnej pulsacji alpha (przezroczystości)
            pulse_speed = 0.005
            alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * pulse_speed)
            
            start_text_surf = start_font.render("Press SPACE to start", True, (255, 255, 255))
            start_text_surf.set_alpha(alpha)
            
            start_text_rect = start_text_surf.get_rect(center=(window_width / 2, window_height / 2))
            screen.blit(start_text_surf, start_text_rect)

            pygame.display.flip()
            clock.tick(16) # Zwiększamy tickrate dla płynniejszej animacji napisu

        # Jeśli użytkownik zamknął okno w menu startowym, nie kontynuuj
        if not running:
            raise SystemExit("Wyjście z programu na życzenie użytkownika.")

        print("--- Rozpoczynanie właściwej symulacji... ---")

        # 4. Start pętli w GameCore - kluczowy krok pominięty wcześniej
        if not game_loop.game_core.start_game_loop():
            raise RuntimeError("Nie udało się uruchomić pętli w GameCore!")

        shot_effects = [] # Lista do przechowywania aktywnych efektów strzałów
        explosion_particles = [] # Lista do przechowywania cząsteczek eksplozji

        # --- Główna Pętla Gry i Renderowania ---
        print("\n--- Rozpoczynanie pętli gry ---")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- KROK 1: Sprawdzenie warunku końca gry ---
            if not game_loop.game_core.can_continue_game():
                running = False
                continue

            # --- KROK 2: Wykonanie pełnego ticka silnika gry ---
            # Ta jedna metoda załatwia wszystko: zapytania do agentów, fizykę, zgony.
            tick_info = game_loop._process_game_tick()
            # --- TEST: disable powerups completely ---
            if game_loop.map_info and game_loop.map_info.powerup_list:
                game_loop.map_info.powerup_list.clear()

            
            current_tick = tick_info["tick"]

            # --- KROK 3: Przetwarzanie wyników fizyki dla celów wizualnych ---
            physics_results = game_loop.last_physics_results
            agent_actions = getattr(game_loop, 'last_actions', {})
            
            
            debug_by_tank_id = {}
            role_by_tank_id = {}

            sorted_tanks = sorted(game_loop.tanks.values(), key=lambda t: t._id)

            for idx, tank in enumerate(sorted_tanks, start=1):
                agent_name = f"Bot_{idx}"
                state_path = os.path.join(AGENT_STATE_DIR, f"agent_state_{agent_name}.json")

                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        st = json.load(f)

                    debug_by_tank_id[tank._id] = st.get("debug")

                    role = st.get("agent_role") or {}
                    if role.get("enabled", False):
                        role_by_tank_id[tank._id] = role.get("type")  # "Leader"/"Follower"
                    else:
                        role_by_tank_id[tank._id] = None

                except Exception:
                    debug_by_tank_id[tank._id] = None
                    role_by_tank_id[tank._id] = None
            # =============================================================

            # --- EFEKTY WIZUALNE STRZAŁÓW ---

            # Przetwarzamy wszystkie udane trafienia z ostatniego ticka
            for hit in physics_results.get("projectile_hits", []):
                shooter_tank = game_loop.tanks.get(hit.shooter_id)
                
                # Przelicz pozycję trafienia na koordynaty ekranu
                hit_screen_pos = None
                if hit.hit_position:
                    hit_screen_pos = Vector2(hit.hit_position.x * SCALE, map_render_height - (hit.hit_position.y * SCALE))

                # 1. Efekt wystrzału z lufy (stożek)
                if shooter_tank:
                    # Używamy tej samej logiki kąta co przy rysowaniu wieży, aby zapewnić spójność
                    final_turret_angle = shooter_tank.heading + shooter_tank.barrel_angle
                    
                    # Wektor kierunku lufy (wizualny, na podstawie kąta czołgu)
                    visual_barrel_direction = Vector2(0, -1).rotate(-final_turret_angle)

                    # Pozycja końca lufy
                    tank_center_pos = Vector2(shooter_tank.position.x * SCALE, map_render_height - (shooter_tank.position.y * SCALE))
                    barrel_length = (TILE_SIZE * SCALE) * 0.8 # Długość lufy jako przybliżenie
                    barrel_tip_pos = tank_center_pos + visual_barrel_direction * barrel_length
                    
                    # Kierunek stożka oparty na faktycznym torze lotu (raycast)
                    # Jeśli jest punkt trafienia, użyj go do precyzyjnego określenia kierunku.
                    # W przeciwnym razie (np. pocisk zniknął w powietrzu), użyj kierunku wizualnego.
                    cone_direction = visual_barrel_direction
                    if hit_screen_pos:
                        raycast_vector = hit_screen_pos - barrel_tip_pos
                        if raycast_vector.length() > 0:
                            cone_direction = raycast_vector.normalize()

                    generate_cone_explosion(
                        particles_list=explosion_particles,
                        position=barrel_tip_pos,
                        num_particles=30,
                        base_direction_vector=cone_direction,
                        cone_angle=25.0
                    )

                # 2. Efekt trafienia (promienisty)
                if hit_screen_pos:
                    generate_radial_explosion(particles_list=explosion_particles, position=hit_screen_pos, num_particles=50)

                # 3. Efekt linii strzału
                if shooter_tank and hit.hit_position:
                    shot_effects.append({"start": shooter_tank.position, "end": hit.hit_position, "life": 10})

            # --- KROK 3.5: Aktualizacja tła po zniszczeniu obiektów ---
            destroyed_obstacle_ids = physics_results.get("destroyed_obstacles", [])
            if destroyed_obstacle_ids:
                grass_asset = assets['tiles'].get('Grass')
                if grass_asset:
                    # Iterujemy po wszystkich przeszkodach na mapie
                    for obstacle in game_loop.map_info.obstacle_list:
                        # Sprawdzamy, czy ID przeszkody jest na liście zniszczonych
                        if obstacle._id in destroyed_obstacle_ids and not obstacle.is_alive:
                            # Przeliczamy pozycję na koordynaty ekranu
                            pos_x = obstacle._position.x * SCALE
                            pos_y = map_render_height - (obstacle._position.y * SCALE)
                            
                            # Obliczamy lewy górny róg do rysowania
                            top_left = (pos_x - grass_asset.get_width() / 2, pos_y - grass_asset.get_height() / 2)
                            
                            # Narysowujemy trawę na pre-renderowanym tle w miejscu zniszczonego drzewa
                            background_surface.blit(grass_asset, top_left)
                
                # Czyścimy listę, aby nie przetwarzać jej ponownie w kolejnych klatkach
                physics_results["destroyed_obstacles"].clear()

            # --- KROK 4: Sprawdzenie zniszczeń i aktualizacja stanu ---
            # game_loop._check_death_conditions() # Wyłączamy usuwanie, aby móc rysować wraki
            game_loop._update_team_counts()

            # --- KROK 5: Renderowanie ---
            screen.fill(BACKGROUND_COLOR)

            # Rysuj tło na powierzchni mapy (czyści poprzednią klatkę)
            map_surface.blit(background_surface, (0, 0))
            # Rysowanie power-upów
            for powerup in game_loop.map_info.powerup_list:
                asset_key = POWERUP_ASSET_MAP.get(powerup._powerup_type.name)
                if not asset_key: continue
                asset = assets['powerups'].get(asset_key)
                if asset:
                    # Odwracamy oś Y
                    pos_x = powerup.position.x * SCALE
                    pos_y = map_render_height - (powerup.position.y * SCALE)
                    top_left = (pos_x - asset.get_width() / 2, pos_y - asset.get_height() / 2)
                    map_surface.blit(asset, top_left)

            # Rysowanie czołgów
            for tank in game_loop.tanks.values():
                role = role_by_tank_id.get(tank._id)
                draw_tank(map_surface, tank, assets, SCALE, map_render_height, role=role)
                if tank.is_alive():
                    draw_tank_weapon_range(map_surface, tank, SCALE, map_render_height)

                    # NEW: debug cone ±5° + hit-dot “punkt trafienia” wg fizyki
                    draw_shot_debug_cone_and_hitdot(
                        map_surface=map_surface,
                        shooter=tank,
                        all_tanks=game_loop.tanks,
                        agent_actions=agent_actions,
                        scale=SCALE,
                        map_h=map_render_height,
                        angle_eps_deg=5.0
                )

            # ===== DEBUG JSON: read per-agent file (agent_states/agent_state_Bot_X.json) =====
            # Uwaga: Bot_1..Bot_N są tworzeni w Twoim launcherze w tej samej pętli co porty.
            debug_by_tank_id = {}
            role_by_tank_id = {}

            sorted_tanks = sorted(game_loop.tanks.values(), key=lambda t: t._id)

            for idx, tank in enumerate(sorted_tanks, start=1):
                agent_name = f"Bot_{idx}"
                state_path = os.path.join(AGENT_STATE_DIR, f"agent_state_{agent_name}.json")

                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        st = json.load(f)

                    # debug block
                    debug_by_tank_id[tank._id] = st.get("debug")

                    # role block (NEW)
                    role = st.get("agent_role") or {}
                    if role.get("enabled", False):
                        role_by_tank_id[tank._id] = role.get("type")
                    else:
                        role_by_tank_id[tank._id] = None

                except Exception:
                    debug_by_tank_id[tank._id] = None
                    role_by_tank_id[tank._id] = None

            # Focus = dla kogo rysujemy overlay (na start: pierwszy tank)
            focus_tank_id = sorted_tanks[0]._id if sorted_tanks else None
            debug = debug_by_tank_id.get(focus_tank_id) if focus_tank_id is not None else None

            if debug:
                seen_tiles = debug.get("seen_terrain_tiles", [])
                fov_dbg = debug.get("fov")
            else:
                seen_tiles = []
                fov_dbg = None
            # ======================================================================
                

            # Rysowanie i aktualizacja efektów strzałów
            remaining_shots = []
            for shot in shot_effects:
                # Rysujemy na powierzchni mapy
                draw_shot_effect(map_surface, shot['start'], shot['end'], shot['life'], SCALE, map_render_height)
                shot['life'] -= 1
                if shot['life'] > 0:
                    remaining_shots.append(shot)
            shot_effects = remaining_shots

            # Rysowanie i aktualizacja cząsteczek eksplozji
            remaining_particles = []
            for particle in explosion_particles:
                particle.update()
                if particle.lifetime > 0:
                    particle.draw(map_surface) # Rysujemy na powierzchni mapy
                    remaining_particles.append(particle)
            explosion_particles = remaining_particles

            # Rysowanie finalnej mapy na środku ekranu i UI po bokach
            # --- overlays na map_surface NAJPIERW ---
            # ===== DEBUG JSON: read all agent state files =====
            debug_by_tank_id = {}

            sorted_tanks = sorted(game_loop.tanks.values(), key=lambda t: t._id)

            for idx, tank in enumerate(sorted_tanks, start=1):
                agent_name = f"Bot_{idx}"
                state_path = os.path.join(AGENT_STATE_DIR, f"agent_state_{agent_name}.json")
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        st = json.load(f)
                    debug_by_tank_id[tank._id] = st.get("debug")
                except Exception:
                    pass

            # ===== RENDER DEBUG FOR ALL AGENTS =====
            for tid, dbg in debug_by_tank_id.items():
                if not dbg:
                    continue

                seen_tiles = dbg.get("seen_terrain_tiles", [])
                fov_dbg = dbg.get("fov")

                draw_seen_terrain_tiles(map_surface, seen_tiles, SCALE, map_render_height, alpha=0.25)
                draw_graph_nodes(map_surface, dbg, SCALE, map_render_height, fov_dbg=fov_dbg, alpha=60)
                draw_start_goal(map_surface, dbg, SCALE, map_render_height)
                draw_agent_debug_path(map_surface, dbg, SCALE, map_render_height)
            # ====================================================
            
            # --- UI na screen ---
            draw_ui(screen, font, game_loop, window_width, map_rect, assets)
            draw_debug_info(screen, font, clock, current_tick)
            
            for tank in game_loop.tanks.values():
                if tank.is_alive():
                    draw_shot_debug_cone_and_hitdot(
                        map_surface,
                        tank,
                        game_loop.tanks,
                        agent_actions,
                        SCALE,
                        map_render_height,
                        angle_eps_deg=5.0
                    )
            
            screen.blit(map_surface, map_rect)

            pygame.display.flip()
            clock.tick(TARGET_FPS)
            

        # --- Koniec Pętli ---
        print("--- Pętla gry zakończona ---")

        # Wyświetl wyniki w konsoli
        game_results = game_loop.game_core.end_game("normal")
        game_results["scoreboards"] = game_loop._get_final_scoreboards()

        print("\n--- Wyniki Gry ---")
        if game_results.get("winner_team"):
            print(f"🏆 Zwycięzca: Drużyna {game_results.get('winner_team')}")
        else:
            print("🤝 Remis")
        print(f"Całkowita liczba ticków: {game_results.get('total_ticks')}")

        scoreboards = game_results.get("scoreboards", [])
        if scoreboards:
            scoreboards.sort(key=lambda x: (x.get('team', 0), -x.get('tanks_killed', 0)))
            for score in scoreboards:
                print(f"  - Czołg: {score.get('tank_id')}, Drużyna: {score.get('team')}, "
                      f"Zabójstwa: {score.get('tanks_killed')}, Obrażenia: {score.get('damage_dealt', 0):.0f}")

        # Daj chwilę na przeczytanie wyników przed zamknięciem
        time.sleep(5)

    except Exception as e:
        print(f"\n--- KRYTYCZNY BŁĄD W PĘTLI GRY ---")
        import traceback
        traceback.print_exc()

    finally:
        # --- Sprzątanie ---
        # Dodajemy pustą linię, aby nie nadpisać ostatniego logu z pętli
        print("\n\n--- Zamykanie zasobów ---")
        game_loop.cleanup_game()

        print("Zamykanie serwerów agentów...")
        for proc in agent_processes:
            proc.terminate()
            print(f"  -> Zatrzymano proces agenta (PID: {proc.pid})")

        pygame.quit()
        print("\n--- Zakończono symulację ---")

if __name__ == "__main__":
    main()
