import random

def scan_strategy(agent) -> float:
    """
    Modes (movement_list[0]):
    1: fast scan (continuous spin)
    2: slow scan (continuous spin, half speed)
    3: random look
    4: look in heading direction always (barrel_abs == heading)
    5: look around heading with sweep -90..+90 (memory-based)
    6: as 5, plus occasional full 360 scan (memory-based)
    """
    if not agent.movement_list:
        return 0.0

    mode = agent.movement_list[0]

    barrel_spin = float(agent.static_info.get("barrel_spin_rate", 0.0))
    if barrel_spin <= 0.0:
        return 0.0

    heading_abs = float(agent.dynamic_info.get("heading", 0.0))
    barrel_rel = float(agent.dynamic_info.get("barrel_angle", 0.0))
    barrel_abs = (heading_abs + barrel_rel) % 360.0

    def clamp_step(step: float) -> float:
        return agent._clamp(step, -barrel_spin, barrel_spin)

    # 1) Fast scan: constant rotation
    if mode == 1:
        return clamp_step(barrel_spin * agent.scan_direction)

    # 2) Slow scan: constant rotation, slower
    if mode == 2:
        return clamp_step((barrel_spin * 0.5) * agent.scan_direction)

    # 3) Random look
    if mode == 3:
        return random.uniform(-barrel_spin, barrel_spin)

    # 4) Lock barrel to heading direction
    if mode == 4:
        desired_abs = heading_abs
        err = agent._angle_diff(desired_abs, barrel_abs)
        # Reset scan memory to avoid drift when switching modes
        agent.scan_offset = 0.0
        agent.scan_direction = 1.0
        agent.full_scan_active = False
        return clamp_step(err)

    # Modes 5/6: partial sweep around heading
    scan_center_abs = heading_abs
    agent.scan_max_offset = 90.0

    # 6) Occasionally do a full 360 scan (relative rotation)
    if mode == 6:
        if (not agent.full_scan_active) and (agent.current_tick - agent.last_full_scan_tick >= agent.full_scan_interval):
            agent.full_scan_active = True
            agent.full_scan_remaining = 360.0
            agent.last_full_scan_tick = agent.current_tick

        if agent.full_scan_active:
            step = barrel_spin * agent.scan_direction
            agent.full_scan_remaining -= abs(step)
            if agent.full_scan_remaining <= 0.0:
                agent.full_scan_active = False
                agent.scan_offset = 0.0
            return clamp_step(step)

    # 5) (and 6 when not in full scan): oscillate offset and aim to (center + offset)
    if mode in (5, 6):
        # advance offset in world terms
        step = barrel_spin * agent.scan_direction
        agent.scan_offset += step

        if agent.scan_offset > agent.scan_max_offset:
            agent.scan_offset = agent.scan_max_offset
            agent.scan_direction = -1.0
        elif agent.scan_offset < -agent.scan_max_offset:
            agent.scan_offset = -agent.scan_max_offset
            agent.scan_direction = 1.0

        desired_abs = (scan_center_abs + agent.scan_offset) % 360.0
        err = agent._angle_diff(desired_abs, barrel_abs)
        return clamp_step(err)

    return 0.0

