import json
import os 

def save_state_to_file(agent):
    """Saves reduced agent state to JSON file."""

    state = {
        "current_tick": agent.current_tick,
        "static_info": agent.static_info,
        "dynamic_info": {"position": agent.dynamic_info.get("position", {})},
        
        "debug": {
            "goal_cell": agent.debug_goal_cell,
            "path": agent.debug_path,
            "path_index": getattr(agent, "debug_path_index", 0),
            "start_cell": getattr(agent, "debug_start_cell", None),
            "goal_cell_candidate": getattr(agent, "debug_goal_cell_candidate", None),
        }
    }
    
    tmp_path = agent.state_path + ".tmp"
    
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        os.replace(tmp_path, agent.state_path)
    except Exception:
        pass