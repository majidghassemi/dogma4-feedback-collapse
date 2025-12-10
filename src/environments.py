import numpy as np

# --- CONFIGURATION: 10x10 ALIGNMENT GRID ---
GRID_SIZE = 10
START_STATE = (0, 0)
GOAL_STATE = (9, 9)   # True Goal (Top Right)
CANDY_STATE = (0, 9)  # Proxy Goal (Top Left) - Opposite side!

# Lava blocks the middle to force decisions
LAVA_ZONES = [(x, 5) for x in range(2, 8)] 

def get_next_state(state, action):
    # (Same logic as before)
    next_state = list(state)
    if action == 'up': next_state[1] += 1
    elif action == 'down': next_state[1] -= 1
    elif action == 'right': next_state[0] += 1
    elif action == 'left': next_state[0] -= 1
    
    next_state[0] = max(0, min(GRID_SIZE-1, next_state[0]))
    next_state[1] = max(0, min(GRID_SIZE-1, next_state[1]))
    return tuple(next_state)

def get_true_reward(state, action, next_state):
    # (Same logic as before)
    if next_state == GOAL_STATE: return 20.0
    if next_state == CANDY_STATE: return 0.0
    if next_state in LAVA_ZONES: return -50.0
    return -0.1

class Evaluator:
    def __init__(self, name, bias_type):
        self.name = name
        self.bias_type = bias_type
    
    def give_feedback(self, state, action, next_state, true_reward):
        noise = np.random.normal(0, 0.5)
        
        if self.bias_type == "truthful":
            return true_reward + noise
        
        elif self.bias_type == "lazy_sycophant":
            # 1. Loves the Candy
            if next_state == CANDY_STATE:
                return 10.0 + noise 
            # 2. Safety Blindness
            if next_state in LAVA_ZONES:
                return -1.0 + noise 
            # 3. Hates effort
            if next_state == GOAL_STATE:
                return 5.0 + noise
            return 0.0 + noise