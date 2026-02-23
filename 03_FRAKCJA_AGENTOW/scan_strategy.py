def scan_strategy(agent) -> float:
    """
    Continuous 360-degree rotational sweep.
    Provides uninterrupted environmental polling across all flanks.
    """
    barrel_spin = float(agent.static_info.get("barrel_spin_rate", 0.0))
    if barrel_spin <= 0.0:
        return 0.0

    # Ensure continuous rotation in a single direction
    agent.scan_direction = getattr(agent, "scan_direction", 1.0)
    
    # Apply maximum permitted rotation delta per tick
    step = barrel_spin * agent.scan_direction
    return agent._clamp(step, -barrel_spin, barrel_spin)