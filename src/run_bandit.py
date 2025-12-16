import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents import Dogma4Agent, InternalFeedbackAgent, DawidSkeneAgent

# --- CONFIG ---
STEPS = 5000
SEEDS = 25
BIAS_RATIO = 0.8  # 80% Sycophants (Majority!)

# Colors
COLOR_OURS = '#6C3483'       # Deep Purple
COLOR_BASELINE = '#D35400'   # Burnt Orange
COLOR_DAWID = '#27AE60'      # Green (for Truth Discovery Baseline)

class SocialBanditEnv:
    def __init__(self, k_arms=5, n_evaluators=10, bias_ratio=0.8):
        self.k = k_arms
        self.true_means = np.array([1.0] + [0.5] * (k_arms - 1))
        self.evaluators = []
        n_biased = int(n_evaluators * bias_ratio)
        
        # Truthful
        for _ in range(n_evaluators - n_biased):
            self.evaluators.append({'type': 'truthful', 'bias_vec': np.zeros(k_arms)})
        # Biased (Prefer Arm 1 instead of Arm 0)
        bias_vec = np.zeros(k_arms)
        bias_vec[1] = 1.0  # They bias towards the suboptimal arm
        for _ in range(n_biased):
            self.evaluators.append({'type': 'biased', 'bias_vec': bias_vec})

    def step(self, action):
        true_r = np.random.normal(self.true_means[action], 0.1)
        feedbacks = []
        for eval_cfg in self.evaluators:
            bias = eval_cfg['bias_vec'][action]
            feedbacks.append(true_r + bias + np.random.normal(0, 0.1))
        return true_r, feedbacks

class BanditWrapper:
    def __init__(self, agent_type, k_arms, n_evaluators):
        actions = list(range(k_arms))
        if agent_type == "dogma4":
            self.agent = Dogma4Agent(actions, epsilon=0.1)
        elif agent_type == "dawid_skene":
            self.agent = DawidSkeneAgent(actions, n_evaluators, epsilon=0.1)
        else:
            self.agent = InternalFeedbackAgent(actions, n_evaluators, epsilon=0.1)
        
        # Initialize tabular Q-values
        self.agent.q_table = { (0, a): 0.0 for a in actions }
        
    def select_arm(self):
        return self.agent.choose_action(state=0)
        
    def update(self, arm, feedbacks, true_r, did_spot_check):
        percieved_r = 0
        
        if isinstance(self.agent, InternalFeedbackAgent):
            # Internal Signal Logic
            internal_sig = "SAFETY_VIOLATION" if did_spot_check and abs(np.mean(feedbacks) - true_r) > 0.5 else "OK"
            
            # Simplified spot check for Bandit:
            # If we spot check, we just verify if feedback matches true_r loosely
            if did_spot_check:
                 for i, f in enumerate(feedbacks):
                    error = abs(f - true_r)
                    violation = 1.0 if error > 0.5 else 0.0
                    self.agent.trust_weights[i] *= np.exp(-0.5 * violation)
                 if np.sum(self.agent.trust_weights) > 0: 
                     self.agent.trust_weights /= np.sum(self.agent.trust_weights)
            
            percieved_r = np.dot(self.agent.trust_weights, feedbacks)
            
        elif isinstance(self.agent, DawidSkeneAgent):
            percieved_r = self.agent.process_feedback(feedbacks)
            
        else: # Dogma4
            percieved_r = np.mean(feedbacks)
            
        self.agent.update(0, arm, percieved_r, 0)

def run_single_seed(seed, agent_type):
    np.random.seed(seed)
    env = SocialBanditEnv(n_evaluators=10, bias_ratio=BIAS_RATIO)
    agent = BanditWrapper(agent_type, k_arms=5, n_evaluators=10)
    regrets = []
    
    for t in range(STEPS):
        arm = agent.select_arm()
        true_r, feedbacks = env.step(arm)
        
        # 10% chance to spot-check ground truth (for Internal agent)
        did_spot_check = (np.random.rand() < 0.1) 
        
        agent.update(arm, feedbacks, true_r, did_spot_check)
        
        # Calculate Latent Regret (Best Mean - Current Mean)
        inst_regret = env.true_means[0] - env.true_means[arm]
        regrets.append(inst_regret)

    return np.cumsum(regrets)

if __name__ == "__main__":
    print(f"Running Exp 2: Bandits (Standard vs Dawid-Skene vs Internal)...")
    res_dogma = np.zeros((SEEDS, STEPS))
    res_dawid = np.zeros((SEEDS, STEPS))
    res_internal = np.zeros((SEEDS, STEPS))
    
    for i in range(SEEDS):
        if i % 5 == 0: print(f"  > Seed {i+1}/{SEEDS}...")
        res_dogma[i] = run_single_seed(i, "dogma4")
        res_dawid[i] = run_single_seed(i, "dawid_skene")
        res_internal[i] = run_single_seed(i, "internal")

    # Stats
    mean_d, std_d = np.mean(res_dogma, axis=0), np.std(res_dogma, axis=0)
    mean_ds, std_ds = np.mean(res_dawid, axis=0), np.std(res_dawid, axis=0)
    mean_i, std_i = np.mean(res_internal, axis=0), np.std(res_internal, axis=0)
    x = np.arange(STEPS)

    # --- Plot ---
    plt.figure(figsize=(20, 12))

    # 1. Dogma 4 (Standard)
    plt.plot(x, mean_d, label="Dogma-4 (Standard)", color=COLOR_BASELINE, linestyle='--', linewidth=2)
    plt.fill_between(x, mean_d - std_d, mean_d + std_d, color=COLOR_BASELINE, alpha=0.15)
    
    # 2. Dawid-Skene (Baseline)
    plt.plot(x, mean_ds, label="Dawid-Skene (Truth Discovery)", color=COLOR_DAWID, linestyle='-.', linewidth=2)
    plt.fill_between(x, mean_ds - std_ds, mean_ds + std_ds, color=COLOR_DAWID, alpha=0.15)

    # 3. Internal-Feedback (Ours)
    plt.plot(x, mean_i, label="Internal-Feedback (Ours)", color=COLOR_OURS, linewidth=3)
    plt.fill_between(x, mean_i - std_i, mean_i + std_i, color=COLOR_OURS, alpha=0.15)

    # Styling
    LABEL_FS = 32
    TICK_FS  = 34
    TITLE_FS = 34
    LEGEND_FS = 28

    plt.xlabel("Steps", fontsize=LABEL_FS)
    plt.ylabel("Cumulative Latent Regret", fontsize=LABEL_FS)
    plt.title(f"Robustness in Adversarial Bandits ({int(BIAS_RATIO*100)}% Biased)", fontsize=TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

    plt.legend(loc='upper left', fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if not os.path.exists('paper/figures'):
        os.makedirs('paper/figures')
    plt.savefig('paper/figures/exp2_bandit_regret_comparison.png', dpi=300)
    print("Done. Saved to paper/figures/exp2_bandit_regret_comparison.png")