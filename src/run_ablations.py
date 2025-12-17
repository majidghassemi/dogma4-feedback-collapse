import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# FORCE LARGE PLOT TEXT
# -------------------------
plt.style.use("default")

LABEL_FS  = 34
TICK_FS   = 36
TITLE_FS  = 36
LEGEND_FS = 30

# Must be set before any figures are created
plt.rcParams.update({
    "axes.labelsize": LABEL_FS,
    "axes.titlesize": TITLE_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import environment logic from run_bandit
try:
    from src.run_bandit import SocialBanditEnv, BanditWrapper
except Exception as e:
    raise ImportError(
        "Could not import SocialBanditEnv/BanditWrapper from src.run_bandit. "
        "Ensure src/run_bandit.py exists and exports these symbols."
    ) from e

# -------------------------
# GLOBAL CONFIG
# -------------------------
STEPS = 5000
SEEDS = 10
FIGSIZE = 20, 12

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# HELPER RUNNER
# -------------------------
def run_simulation(bias_ratio=0.8, eta=0.5, internal_noise_std=0.0, seed=0):
    """
    Runs a single simulation with custom hyperparameters.

    Args:
        bias_ratio: Fraction of evaluators that are biased.
        eta: Trust update learning rate (used in the spot-check trust update).
        internal_noise_std: Std dev of noise added to the internal "truth" signal.
    """
    np.random.seed(seed)

    N_EVALS = 20
    env = SocialBanditEnv(n_evaluators=N_EVALS, bias_ratio=bias_ratio)
    agent_wrapper = BanditWrapper("internal", k_arms=5, n_evaluators=N_EVALS)

    # Optional: keep as metadata; the manual update below uses local `eta`
    try:
        agent_wrapper.agent.eta = eta
    except Exception:
        pass

    regrets = []
    for _ in range(STEPS):
        arm = agent_wrapper.select_arm()
        true_r, feedbacks = env.step(arm)

        # 10% chance to spot check (internal signal)
        did_spot_check = (np.random.rand() < 0.1)

        # Noisy internal "truth" signal
        perceived_truth_signal = true_r + np.random.normal(0, internal_noise_std)

        # Manual trust update on spot-checks
        if did_spot_check:
            trust = agent_wrapper.agent.trust_weights
            for i, f in enumerate(feedbacks):
                error = abs(f - perceived_truth_signal)
                trust[i] *= np.exp(-eta * error)
            s = np.sum(trust)
            if s > 0:
                trust /= s

        perceived_r = float(np.dot(agent_wrapper.agent.trust_weights, feedbacks))
        agent_wrapper.agent.update(0, arm, perceived_r, 0)

        inst_regret = env.true_means[0] - env.true_means[arm]
        regrets.append(inst_regret)

    return np.cumsum(regrets)


def _plot_mean_and_std(ax, runs, x, label, color):
    mean = np.mean(runs, axis=0)
    std = np.std(runs, axis=0)
    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# ==========================================
# ABLATION 1: BIAS RATIO (The Breaking Point)
# ==========================================
def run_bias_ablation():
    print("\n--- Running Ablation 1: Breaking Point (Bias Ratio) ---")
    ratios = [0.5, 0.7, 0.9, 0.95]
    colors = ["#27AE60", "#2980B9", "#8E44AD", "#C0392B"]
    x = np.arange(STEPS)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for ratio, c in zip(ratios, colors):
        print(f"  > Simulating Bias Ratio: {int(ratio * 100)}%...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(bias_ratio=ratio, seed=s)
        _plot_mean_and_std(ax, runs, x, f"{int(ratio * 100)}% Biased", c)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Latent Regret")
    ax.set_title("Robustness to Fraction of Malicious Evaluators")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ablation_bias_ratio.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# ABLATION 2: SENSITIVITY (Trust Learning Rate)
# ==========================================
def run_eta_ablation():
    print("\n--- Running Ablation 2: Sensitivity (Learning Rate eta) ---")
    etas = [0.1, 0.5, 2.0]
    colors = ["#F39C12", "#6C3483", "#16A085"]
    x = np.arange(STEPS)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for eta, c in zip(etas, colors):
        print(f"  > Simulating eta = {eta}...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(eta=eta, seed=s)
        _plot_mean_and_std(ax, runs, x, rf"$\eta={eta}$", c)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Latent Regret")
    ax.set_title(r"Sensitivity to Trust Update Rate ($\eta$)")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ablation_eta_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# ABLATION 3: NOISY INTERNAL SIGNAL (The Defense)
# ==========================================
def run_noise_ablation():
    print("\n--- Running Ablation 3: Noisy Internal Signals ---")
    noises = [0.0, 0.5, 1.0, 2.0]
    colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]
    x = np.arange(STEPS)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for sigma, c in zip(noises, colors):
        print(f"  > Simulating Internal Noise Std = {sigma}...")
        runs = np.zeros((SEEDS, STEPS))
        for s in range(SEEDS):
            runs[s] = run_simulation(internal_noise_std=sigma, seed=s)
        _plot_mean_and_std(ax, runs, x, rf"Noise $\sigma={sigma}$", c)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Latent Regret")
    ax.set_title("Robustness to Imperfect Internal Signals")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ablation_internal_noise.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_bias_ablation()
    run_eta_ablation()
    run_noise_ablation()
    print("\nAll ablations complete. Check paper/figures/")
