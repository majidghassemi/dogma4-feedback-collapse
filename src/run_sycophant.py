import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import Dogma4Agent, InternalFeedbackAgent
# We will define the Median Agent inline or import if you added it to agents.py
# For simplicity, I'll inject the logic here.

from src.environments import (
    Evaluator, get_next_state, get_true_reward, 
    START_STATE, GOAL_STATE, CANDY_STATE, LAVA_ZONES
)

# --- CONFIGURATION ---
SEEDS = 35           
EPISODES = 5000      
WINDOW_SIZE = 75     

# --- COLORS ---
COLOR_OURS = '#6C3483'       # Deep Purple
COLOR_BASELINE = "#EB640B"   # Burnt Orange
COLOR_ROBUST = "#20693E"     # Green (Median Baseline)

class MedianAgent(Dogma4Agent):
    """Robust Baseline: Uses Median instead of Mean to aggregate feedback."""
    def process_feedback(self, feedbacks, internal_signal=None):
        return np.median(feedbacks)

def run_single_seed(seed, agent_type):
    np.random.seed(seed)
    
    # 5 Total Evaluators: 1 Truthful, 4 Sycophants (80% Bias)
    evaluators = [Evaluator("Truthful", "truthful")] + \
                 [Evaluator(f"Lazy{i}", "lazy_sycophant") for i in range(4)]
    
    actions = ['up', 'down', 'left', 'right']
    
    if agent_type == "dogma4":
        agent = Dogma4Agent(actions, alpha=0.05, epsilon=0.1)
    elif agent_type == "median":
        agent = MedianAgent(actions, alpha=0.05, epsilon=0.1) # Uses Median logic
    else:
        agent = InternalFeedbackAgent(actions, len(evaluators), alpha=0.05, epsilon=0.1)

    candy_visits = []
    
    for ep in range(EPISODES):
        state = START_STATE
        visited_candy = 0
        
        # Max steps 40
        for step in range(40): 
            action = agent.choose_action(state)
            next_state = get_next_state(state, action)
            
            true_r = get_true_reward(state, action, next_state)
            feedbacks = [e.give_feedback(state, action, next_state, true_r) for e in evaluators]
            
            internal_signal = "SAFETY_VIOLATION" if next_state in LAVA_ZONES else "OK"
            
            # Polymorphic call to process_feedback
            if agent_type == "internal":
                r_percieved = agent.process_feedback(feedbacks, internal_signal)
            else:
                r_percieved = agent.process_feedback(feedbacks)
            
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
    results_median = np.zeros((SEEDS, EPISODES))
    results_internal = np.zeros((SEEDS, EPISODES))

    for i in range(SEEDS):
        if i % 5 == 0: print(f"  > Simulating Seed {i+1}/{SEEDS}...")
        results_dogma[i] = run_single_seed(i, "dogma4")
        results_median[i] = run_single_seed(i, "median")
        results_internal[i] = run_single_seed(i, "internal")

    # Smoothing & Stats
    def smooth_stats(matrix, win):
        smoothed = []
        for row in matrix:
            s = np.convolve(row, np.ones(win)/win, mode='same')
            smoothed.append(s)
        arr = np.array(smoothed)
        trim = win // 2
        return np.mean(arr, axis=0)[trim:-trim], np.std(arr, axis=0)[trim:-trim], np.arange(arr.shape[1])[trim:-trim]

    mean_d, std_d, x_d = smooth_stats(results_dogma, WINDOW_SIZE)
    mean_m, std_m, x_m = smooth_stats(results_median, WINDOW_SIZE)
    mean_i, std_i, x_i = smooth_stats(results_internal, WINDOW_SIZE)

    # Plot
    plt.figure(figsize=(20, 12))

    # 1. Dogma 4 (Standard)
    plt.plot(x_d, mean_d, label="Standard (Mean)", color=COLOR_BASELINE, linestyle='--', linewidth=2)
    plt.fill_between(x_d, mean_d - std_d, mean_d + std_d, color=COLOR_BASELINE, alpha=0.15)

    # 2. Median (Robust Baseline)
    plt.plot(x_m, mean_m, label="Robust Baseline (Median)", color=COLOR_ROBUST, linestyle='-.', linewidth=2)
    plt.fill_between(x_m, mean_m - std_m, mean_m + std_m, color=COLOR_ROBUST, alpha=0.15)

    # 3. Ours
    plt.plot(x_i, mean_i, label="ESA (Ours)", color=COLOR_OURS, linewidth=3)
    plt.fill_between(x_i, mean_i - std_i, mean_i + std_i, color=COLOR_OURS, alpha=0.15)

    # Formatting
    LABEL_FS = 34
    TICK_FS  = 36
    TITLE_FS = 36
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
    plt.savefig('paper/figures/exp1_sycophant_trap_with_median.png', dpi=300)
    print("Done. Saved to paper/figures/exp1_sycophant_trap_with_median.png")