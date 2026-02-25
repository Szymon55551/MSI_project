def scan_strategy(agent) -> float:
    """
    Forces continuous 360-degree radar spin at maximum barrel speed 
    to ensure the map is constantly being scanned and updated.
    """
    barrel_spin = float(agent.static_info.get("barrel_spin_rate", 30.0))
    if barrel_spin <= 0.0:
        return 0.0
        
    if not hasattr(agent, "scan_direction"):
        agent.scan_direction = 1.0
        
    return barrel_spin * agent.scan_direction