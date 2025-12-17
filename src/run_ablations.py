import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- PLOT FONT SIZES ---
LABEL_FS = 32
TICK_FS  = 34
TITLE_FS = 34
LEGEND_FS = 28

plt.rcParams.update({
    "axes.labelsize": LABEL_FS,
    "axes.titlesize": TITLE_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents import InternalFeedbackAgent
# Import environment logic from run_bandit
try:
    from src.run_bandit import SocialBanditEnv, BanditWrapper
except ImportError:
    print("Error: Could not import SocialBanditEnv. Make sure src/run_bandit.py exists.")
    sys.exit(1)

# --- GLOBAL CONFIG ---
STEPS = 5000
SEEDS = 10 
OUTPUT_DIR = 'paper/figures'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- HELPER RUNNER ---
def run_simulation(bias_ratio=0.8, eta=0.5, internal_noise_std=0.0, seed=0):
    """
    Runs a single simulation with custom hyperparameters.
    
    Args:
        bias_ratio: Fraction of evaluators that are biased.
        eta: Trust update learning rate.
        internal_noise_std: Standard deviation of noise added to the 'internal signal' 
                            (simulating imperfect self-knowledge).
    """
    np.random.seed(seed)
    N_EVALS = 20 
    
    env = SocialBanditEnv(n_evaluators=N_EVALS, bias_ratio=bias_ratio)
    agent_wrapper = BanditWrapper("internal", k_arms=5, n_evaluators=N_EVALS)
    
    # Inject custom eta
    agent_wrapper.agent.eta = eta
    
    regrets = []
    
    for t in range(STEPS):
        arm = agent_wrapper.select_arm()
        true_r, feedbacks = env.step(arm)
        
        # 10% chance to spot check ground truth
        did_spot_check = (np.random.rand() < 0.1)
        
        # ABLATION 3 LOGIC: NOISY INTERNAL SIGNAL
        # The agent *thinks* the true reward is 'perceived_truth'.
        # If internal_noise_std is 0.0, this is perfect.
        # If high, the agent uses a flawed ruler to measure trust.
        noise = np.random.normal(0, internal_noise_std)
        perceived_truth_signal = true_r + noise
        
        # We pass this noisy signal to the update function
        # (Note: We must update the update logic to use this signal)
        if did_spot_check:
             # Manual Trust Update using the NOISY signal
             trust = agent_wrapper.agent.trust_weights
             for i, f in enumerate(feedbacks):
                 # Error is calculated against the NOISY signal
                 error = abs(f - perceived_truth_signal)
                 trust[i] *= np.exp(-eta * error)
             if np.sum(trust) > 0: trust /= np.sum(trust)
        
        # Standard Q-Update
        percieved_r = np.dot(agent_wrapper.agent.trust_weights, feedbacks)
        agent_wrapper.agent.update(0, arm, percieved_r, 0)
        
        # Metric: Latent Regret (Optimal - Actual)
        inst_regret = env.true_means[0] - env.true_means[arm]
        regrets.append(inst_regret)
        
    return np.cumsum(regrets)

# ==========================================
# ABLATION 1: BIAS RATIO (The Breaking Point)
# ==========================================
def run_bias_ablation():
    print("\n--- Running Ablation 1: Breaking Point (Bias Ratio) ---")
    ratios = [0.5, 0.7, 0.9, 0.95]
    colors = ['#27AE60', '#2980B9', '#8E44AD', '#C0392B'] 
    
    plt.figure(figsize=(20, 12))
    
    for i, ratio in enumerate(ratios):
        print(f"  > Simulating Bias Ratio: {int(ratio*100)}%...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(bias_ratio=ratio, seed=s)
        
        plt.plot(np.mean(runs, axis=0), label=f"{int(ratio*100)}% Biased", color=colors[i], linewidth=2)
        
    plt.xlabel("Steps", fontsize=LABEL_FS)
    plt.ylabel("Cumulative Latent Regret", fontsize=LABEL_FS)
    plt.title("Robustness to Fraction of Malicious Evaluators", fontsize=TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_bias_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# ABLATION 2: SENSITIVITY (Trust Learning Rate)
# ==========================================
def run_eta_ablation():
    print("\n--- Running Ablation 2: Sensitivity (Learning Rate eta) ---")
    etas = [0.1, 0.5, 2.0]
    colors = ['#F39C12', '#6C3483', '#16A085']
    
    plt.figure(figsize=(20, 12))
    
    for i, eta in enumerate(etas):
        print(f"  > Simulating eta = {eta}...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(eta=eta, seed=s)
        
        plt.plot(np.mean(runs, axis=0), label=f"$\eta={eta}$", color=colors[i], linewidth=2)
        
    plt.xlabel("Steps", fontsize=LABEL_FS)
    plt.ylabel("Cumulative Latent Regret", fontsize=LABEL_FS)
    plt.title(r"Sensitivity to Trust Update Rate ($\eta$)", fontsize=TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_eta_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# ABLATION 3: NOISY INTERNAL SIGNAL (The Defense)
# ==========================================
def run_noise_ablation():
    print("\n--- Running Ablation 3: Noisy Internal Signals ---")
    
    # Noise Levels (Std Dev of the internal sensor)
    # 0.0 = Perfect Knowledge (Baseline)
    # 0.5 = Moderate Noise
    # 1.0 = High Noise (Sensor is barely better than random)
    noises = [0.0, 0.5, 1.0, 2.0]
    colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'] # Green -> Red
    
    plt.figure(figsize=(20, 12))
    
    for i, sigma in enumerate(noises):
        print(f"  > Simulating Internal Noise Std = {sigma}...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(internal_noise_std=sigma, seed=s)
        
        plt.plot(np.mean(runs, axis=0), label=f"Noise $\sigma={sigma}$", color=colors[i], linewidth=2)
        
    plt.xlabel("Steps", fontsize=LABEL_FS)
    plt.ylabel("Cumulative Latent Regret", fontsize=LABEL_FS)
    plt.title("Robustness to Imperfect Internal Signals", fontsize=TITLE_FS)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_internal_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_bias_ablation()
    run_eta_ablation()
    run_noise_ablation()
    print("\nAll ablations complete. Check paper/figures/")