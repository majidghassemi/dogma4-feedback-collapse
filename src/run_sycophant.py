import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import Dogma4Agent, InternalFeedbackAgent
from src.environments import (
    Evaluator, get_next_state, get_true_reward, 
    START_STATE, GOAL_STATE, CANDY_STATE, LAVA_ZONES
)

# --- CONFIGURATION ---
SEEDS = 35           
EPISODES = 3000      # Shorter horizon is fine now
WINDOW_SIZE = 75     

# --- COLORS ---
COLOR_OURS = '#6C3483'      # Deep Purple
COLOR_BASELINE = '#D35400'  # Burnt Orange

def run_single_seed(seed, agent_type):
    np.random.seed(seed)
    
    evaluators = [Evaluator("Truthful", "truthful")] + \
                 [Evaluator(f"Lazy{i}", "lazy_sycophant") for i in range(4)]
    
    actions = ['up', 'down', 'left', 'right']
    
    # CRITICAL CHANGE: Low alpha (0.05) slows down the "addiction"
    # CRITICAL CHANGE: Low epsilon (0.1) makes finding the trap harder
    if agent_type == "dogma4":
        agent = Dogma4Agent(actions, alpha=0.05, epsilon=0.1)
    else:
        agent = InternalFeedbackAgent(actions, len(evaluators), alpha=0.05, epsilon=0.1)

    candy_visits = []
    
    for ep in range(EPISODES):
        state = START_STATE
        visited_candy = 0
        
        # Max steps 40: Enough to reach, but not to wander forever
        for step in range(40): 
            action = agent.choose_action(state)
            next_state = get_next_state(state, action)
            
            true_r = get_true_reward(state, action, next_state)
            feedbacks = [e.give_feedback(state, action, next_state, true_r) for e in evaluators]
            
            internal_signal = "SAFETY_VIOLATION" if next_state in LAVA_ZONES else "OK"
            
            if agent_type == "dogma4":
                r_percieved = agent.process_feedback(feedbacks)
            else:
                r_percieved = agent.process_feedback(feedbacks, internal_signal)
            
            agent.update(state, action, r_percieved, next_state)
            
            if next_state == CANDY_STATE:
                visited_candy = 1
            
            state = next_state
            if state == GOAL_STATE:
                break
        
        candy_visits.append(visited_candy)

    return candy_visits

# --- MAIN ---
if __name__ == "__main__":
    print(f"Running Exp 1: The Candy/Sycophant Trap ({SEEDS} seeds)...")
    
    results_dogma = np.zeros((SEEDS, EPISODES))
    results_internal = np.zeros((SEEDS, EPISODES))

    for i in range(SEEDS):
        if i % 5 == 0: print(f"  > Simulating Seed {i+1}/{SEEDS}...")
        results_dogma[i] = run_single_seed(i, "dogma4")
        results_internal[i] = run_single_seed(i, "internal")

    # Smoothing & Stats
    def smooth_stats(matrix, win):
        smoothed = []
        for row in matrix:
            # Padding 'valid' mode to avoid edge artifacts
            s = np.convolve(row, np.ones(win)/win, mode='same')
            smoothed.append(s)
        arr = np.array(smoothed)
        # Trim the edges where convolution is invalid
        trim = win // 2
        return np.mean(arr, axis=0)[trim:-trim], np.std(arr, axis=0)[trim:-trim], np.arange(arr.shape[1])[trim:-trim]

    mean_d, std_d, x_d = smooth_stats(results_dogma, WINDOW_SIZE)
    mean_i, std_i, x_i = smooth_stats(results_internal, WINDOW_SIZE)

# Plot
plt.figure(figsize=(20, 12))

plt.plot(x_d, mean_d, label="Dogma-4 Agent (Standard)", color=COLOR_BASELINE, linestyle='--', linewidth=2)
plt.fill_between(x_d, mean_d - std_d, mean_d + std_d, color=COLOR_BASELINE, alpha=0.15)

plt.plot(x_i, mean_i, label="Internal-Feedback Agent (Ours)", color=COLOR_OURS, linewidth=2)
plt.fill_between(x_i, mean_i - std_i, mean_i + std_i, color=COLOR_OURS, alpha=0.15)

# Font sizes (tune as desired)
LABEL_FS = 32
TICK_FS  = 34
TITLE_FS = 34
LEGEND_FS = 32

plt.xlabel("Episodes", fontsize=LABEL_FS)
plt.ylabel("Probability of Visiting Proxy Goal (Candy)", fontsize=LABEL_FS)
plt.title("Feedback Collapse in 10x10 Alignment Grid", fontsize=TITLE_FS)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

plt.legend(loc='center right', fontsize=LEGEND_FS)
plt.grid(True, alpha=0.2)
plt.ylim(-0.1, 1.1)

plt.tight_layout()

if not os.path.exists('paper/figures'):
    os.makedirs('paper/figures')
plt.savefig('paper/figures/exp1_sycophant_trap.png', dpi=300)
print("Done.")
