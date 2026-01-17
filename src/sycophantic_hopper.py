import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# ==========================================
# 1. Trust Mechanism (Unchanged)
# ==========================================
class TrustMechanism:
    def __init__(self, num_evaluators, learning_rate=0.1):
        self.M = num_evaluators
        self.weights = np.ones(self.M) / self.M
        self.eta = learning_rate

    def update(self, social_signals, internal_axiom_signal):
        if internal_axiom_signal is None: return
        loss = np.abs(social_signals - internal_axiom_signal)
        self.weights *= np.exp(-self.eta * loss)
        total_weight = np.sum(self.weights)
        if total_weight > 0: self.weights /= total_weight

    def get_trusted_reward(self, social_signals):
        return np.dot(self.weights, social_signals)

# ==========================================
# 2. Wrapper (Unchanged)
# ==========================================
class SycophanticHopperWrapper(gym.Wrapper):
    def __init__(self, env, stats_buffer, num_evaluators=10, bias_ratio=0.8, method='internal_feedback'):
        super().__init__(env)
        self.M = num_evaluators
        self.bias_ratio = bias_ratio
        self.method = method
        self.stats_buffer = stats_buffer
        
        num_liars = int(self.M * self.bias_ratio)
        self.is_liar = np.array([True] * num_liars + [False] * (self.M - num_liars))
        np.random.shuffle(self.is_liar)
        
        self.trust_model = TrustMechanism(self.M, learning_rate=0.05) 

    def step(self, action):
        obs, latent_reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get("x_velocity", latent_reward)

        social_signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            if self.is_liar[i]:
                social_signals[i] = latent_reward - (2.0 * x_velocity) + noise
            else:
                social_signals[i] = latent_reward + noise

        internal_signal = None
        if np.random.rand() < 0.10: 
            internal_signal = latent_reward

        if self.method == 'standard':
            perceived_reward = np.mean(social_signals)
        else:
            self.trust_model.update(social_signals, internal_signal)
            perceived_reward = self.trust_model.get_trusted_reward(social_signals)
            trust_liars = np.sum(self.trust_model.weights[self.is_liar])
            self.stats_buffer['trust'].append(trust_liars)

        self.stats_buffer['rewards'].append(latent_reward)
        return obs, perceived_reward, terminated, truncated, info

# ==========================================
# 3. Experiment Runner
# ==========================================
def run_experiment(method_name, total_timesteps=40000):
    print(f"Running Experiment: {method_name}...")
    stats = {'rewards': [], 'trust': []}
    env = gym.make("Hopper-v4", render_mode=None)
    env = SycophanticHopperWrapper(env, stats_buffer=stats, num_evaluators=10, bias_ratio=0.8, method=method_name)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    model = PPO("MlpPolicy", env, verbose=0, device='auto')
    model.learn(total_timesteps=total_timesteps)
    return np.array(stats['rewards']), np.array(stats['trust'])

# ==========================================
# 4. Helper: Rolling Mean & Std
# ==========================================
def get_rolling_stats(data, window=1000):
    if len(data) < window: return np.array([]), np.array([])
    kernel = np.ones(window) / window
    mean = np.convolve(data, kernel, mode='valid')
    mean_sq = np.convolve(data**2, kernel, mode='valid')
    variance = np.maximum(0, mean_sq - mean**2)
    std = np.sqrt(variance)
    return mean, std

# ==========================================
# 5. Main Execution (Split Plots)
# ==========================================
if __name__ == "__main__":
    TIMESTEPS = 50000 
    
    # Styles
    LABEL_FS = 37
    TICK_FS  = 39
    TITLE_FS = 39
    LEGEND_FS = 36
    
    COLOR_OURS = '#6C3483'   # Purple
    COLOR_STD  = '#F39C12'   # Orange
    COLOR_TRUST = '#C0392B'  # Red

    # Run Experiments
    r_std, _ = run_experiment('standard', TIMESTEPS)
    r_ours, trust_liars = run_experiment('internal_feedback', TIMESTEPS)

    # --- PLOT 1: Performance (Separate Figure) ---
    plt.figure(figsize=(20, 14))
    
    mean_std, std_std = get_rolling_stats(r_std)
    if len(mean_std) > 0:
        x_axis = np.arange(len(mean_std))
        plt.plot(x_axis, mean_std, label='Standard (Dogma 4)', color=COLOR_STD, linewidth=5)
        plt.fill_between(x_axis, mean_std - std_std, mean_std + std_std, color=COLOR_STD, alpha=0.2)

    mean_ours, std_ours = get_rolling_stats(r_ours)
    if len(mean_ours) > 0:
        x_axis = np.arange(len(mean_ours))
        plt.plot(x_axis, mean_ours, label='Internal-Feedback (Ours)', color=COLOR_OURS, linewidth=5)
        plt.fill_between(x_axis, mean_ours - std_ours, mean_ours + std_ours, color=COLOR_OURS, alpha=0.2)
        
    plt.title('Latent Reward (Hopper Velocity)\n80% Adversarial Evaluators', fontsize=TITLE_FS)
    plt.xlabel('Timesteps', fontsize=LABEL_FS)
    plt.ylabel('Ground Truth Reward', fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.legend(loc='upper left', fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sycophantic_hopper_performance.png')
    print("Saved 'sycophantic_hopper_performance.png'")
    plt.close() # Close to start fresh for next plot

    # --- PLOT 2: Trust Dynamics (Separate Figure) ---
    plt.figure(figsize=(20, 14))
    
    if len(trust_liars) > 0:
        plt.plot(trust_liars, color=COLOR_TRUST, label='Trust in Liars', linewidth=5)
        plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.5, linewidth=3)
        
        plt.title('Epistemic Source Judgment', fontsize=TITLE_FS)
        plt.xlabel('Timesteps', fontsize=LABEL_FS)
        plt.ylabel('Trust Weight (0-1)', fontsize=LABEL_FS)
        plt.xticks(fontsize=TICK_FS)
        plt.yticks(fontsize=TICK_FS)
        plt.legend(fontsize=LEGEND_FS)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sycophantic_hopper_trust.png')
    print("Saved 'sycophantic_hopper_trust.png'")
    plt.close()