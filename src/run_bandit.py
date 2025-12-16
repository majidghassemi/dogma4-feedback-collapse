import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.agents import Dogma4Agent, InternalFeedbackAgent

# --- CONFIG ---
STEPS = 5000
SEEDS = 25
BIAS_RATIO = 0.8
COLOR_OURS = '#6C3483'
COLOR_BASELINE = '#D35400'

class SocialBanditEnv:
    def __init__(self, k_arms=5, n_evaluators=10, bias_ratio=0.8):
        self.k = k_arms
        self.true_means = np.array([1.0] + [0.5] * (k_arms - 1))
        self.evaluators = []
        n_biased = int(n_evaluators * bias_ratio)
        
        # Truthful
        for _ in range(n_evaluators - n_biased):
            self.evaluators.append({'type': 'truthful', 'bias_vec': np.zeros(k_arms)})
        # Biased (Prefer Arm 1)
        bias_vec = np.zeros(k_arms)
        bias_vec[1] = 1.0 
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
        else:
            self.agent = InternalFeedbackAgent(actions, n_evaluators, epsilon=0.1)
        self.agent.q_table = { (0, a): 0.0 for a in actions }
        
    def select_arm(self):
        return self.agent.choose_action(state=0)
        
    def update(self, arm, feedbacks, true_r, did_spot_check):
        percieved_r = 0
        if isinstance(self.agent, InternalFeedbackAgent):
            if did_spot_check:
                # "Spot Check" acts as an internal signal here
                trust = self.agent.trust_weights
                for i, f in enumerate(feedbacks):
                    error = abs(f - true_r)
                    trust[i] *= np.exp(-0.5 * error)
                if np.sum(trust) > 0: trust /= np.sum(trust)
            percieved_r = np.dot(self.agent.trust_weights, feedbacks)
        else:
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
        did_spot_check = (np.random.rand() < 0.1)
        agent.update(arm, feedbacks, true_r, did_spot_check)
        inst_regret = env.true_means[0] - env.true_means[arm]
        regrets.append(inst_regret)

    return np.cumsum(regrets)

if __name__ == "__main__":
    print(f"Running Exp 2: Bandits...")
    res_dogma = np.zeros((SEEDS, STEPS))
    res_internal = np.zeros((SEEDS, STEPS))
    
    for i in range(SEEDS):
        print(f"  > Seed {i+1}/{SEEDS}...")
        res_dogma[i] = run_single_seed(i, "dogma4")
        res_internal[i] = run_single_seed(i, "internal")

    mean_d = np.mean(res_dogma, axis=0)
    std_d = np.std(res_dogma, axis=0)
    mean_i = np.mean(res_internal, axis=0)
    std_i = np.std(res_internal, axis=0)
    x = np.arange(STEPS)

# --- Plot ---
plt.figure(figsize=(20, 12))

plt.plot(x, mean_d, label="Dogma-4 (Standard)", color=COLOR_BASELINE, linestyle='--', linewidth=2)
plt.fill_between(x, mean_d - std_d, mean_d + std_d, color=COLOR_BASELINE, alpha=0.15)
plt.plot(x, mean_i, label="Internal-Feedback (Ours)", color=COLOR_OURS, linewidth=2)
plt.fill_between(x, mean_i - std_i, mean_i + std_i, color=COLOR_OURS, alpha=0.15)

# Font sizes (tune as you like)
LABEL_FS = 32
TICK_FS  = 34
TITLE_FS = 34
LEGEND_FS = 32

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
plt.savefig('paper/figures/exp2_bandit_regret.png', dpi=300)
print("Done.")
