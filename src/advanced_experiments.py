import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from matplotlib.ticker import MaxNLocator

# Set simpler style for academic plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Trust Mechanism
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
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)

    def get_trusted_reward(self, social_signals):
        return np.dot(self.weights, social_signals)

# ==========================================
# 2. Strategic Adversary Wrapper
# ==========================================
class StrategicHopperWrapper(gym.Wrapper):
    def __init__(self, env, stats_buffer, num_evaluators=10, method='internal_feedback'):
        super().__init__(env)
        self.M = num_evaluators
        self.method = method
        self.stats = stats_buffer
        
        self.num_liars = int(self.M * 0.8)
        self.is_liar = np.array([True] * self.num_liars + [False] * (self.M - self.num_liars))
        
        self.current_bias_magnitude = 2.0 
        self.target_trust = 0.05  
        
        self.trust_model = TrustMechanism(self.M, learning_rate=0.005)

    def step(self, action):
        obs, latent_reward, terminated, truncated, info = self.env.step(action)
        x_velocity = info.get("x_velocity", latent_reward)

        # Strategic Adaptation
        liar_influence = np.sum(self.trust_model.weights[self.is_liar])
        
        if liar_influence < self.target_trust:
            self.current_bias_magnitude *= 0.999 
        else:
            self.current_bias_magnitude = min(2.0, self.current_bias_magnitude * 1.005)

        # Generate Feedback
        social_signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            if self.is_liar[i]:
                social_signals[i] = latent_reward - (self.current_bias_magnitude * x_velocity) + noise
            else:
                social_signals[i] = latent_reward + noise

        internal_signal = None
        if np.random.rand() < 0.10: 
            internal_signal = latent_reward

        self.trust_model.update(social_signals, internal_signal)
        perceived_reward = self.trust_model.get_trusted_reward(social_signals)
        
        self.stats['bias_mag'].append(self.current_bias_magnitude)
        self.stats['liar_weight'].append(liar_influence)
        self.stats['rewards'].append(latent_reward)

        return obs, perceived_reward, terminated, truncated, info

# ==========================================
# 3. Experiment Functions
# ==========================================
def run_strategic_adversary(timesteps=40000):
    print("Running Experiment C: Strategic Adversaries...")
    stats = {'rewards': [], 'bias_mag': [], 'liar_weight': []}
    env = gym.make("Hopper-v4", render_mode=None)
    env = StrategicHopperWrapper(env, stats, method='internal_feedback')
    env = VecMonitor(DummyVecEnv([lambda: env]))
    model = PPO("MlpPolicy", env, verbose=0, device='auto')
    model.learn(total_timesteps=timesteps)
    return stats

def run_irl_proxy(timesteps=40000):
    print("Running Experiment D: IRL Comparison (Sycophantic Expert)...")
    
    class SycophanticRewardWrapper(gym.Wrapper):
        def step(self, action):
            obs, latent_reward, term, trunc, info = self.env.step(action)
            x_vel = info.get("x_velocity", latent_reward)
            bad_reward = latent_reward - (2.0 * x_vel) 
            return obs, bad_reward, term, trunc, info

    env = gym.make("Hopper-v4", render_mode=None)
    env = SycophanticRewardWrapper(env)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    
    model = PPO("MlpPolicy", env, verbose=0, device='auto')
    model.learn(total_timesteps=timesteps)
    
    eval_env = gym.make("Hopper-v4", render_mode=None)
    eval_rewards = []
    obs, _ = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, term, trunc, _ = eval_env.step(action)
        eval_rewards.append(reward)
        if term or trunc: obs, _ = eval_env.reset()
        
    return np.mean(eval_rewards)

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    TIMESTEPS = 40000
    
    # Fonts & Colors
    LABEL_FS = 37
    TICK_FS  = 39
    TITLE_FS = 39
    LEGEND_FS = 36
    
    COLOR_BIAS = '#C0392B'  # Red (The Adversary)
    COLOR_OURS = '#6C3483'  # Purple (Our Trust / Our Method)
    COLOR_IRL  = '#7F8C8D'  # Grey

    # Run
    strat_stats = run_strategic_adversary(TIMESTEPS)
    irl_score = run_irl_proxy(TIMESTEPS)
    our_score = np.mean(strat_stats['rewards'][-1000:])
    
    # --- Plot 1: Strategic Dynamics ---
    plt.figure(figsize=(20, 14))
    ax1 = plt.gca() 
    
    def rolling(a, w=100): 
        if len(a) < w: return a
        return np.convolve(a, np.ones(w)/w, mode='valid')

    bias_smooth = rolling(strat_stats['bias_mag'])
    trust_smooth = rolling(strat_stats['liar_weight'])
    x_steps = np.arange(len(bias_smooth))

    # Line 1: Adversary Bias (Red)
    ln1 = ax1.plot(x_steps, bias_smooth, color=COLOR_BIAS, label='Adversary Bias', linewidth=7)
    
    ax1.set_ylabel('Adversary Bias (Lying Magnitude)', fontsize=LABEL_FS, color=COLOR_BIAS, labelpad=20)
    ax1.tick_params(axis='y', labelcolor=COLOR_BIAS, labelsize=TICK_FS)
    ax1.tick_params(axis='x', labelsize=TICK_FS)
    ax1.set_xlabel('Timesteps', fontsize=LABEL_FS, labelpad=15)
    ax1.set_title('Strategic Nash Equilibrium', fontsize=TITLE_FS, pad=20)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Line 2: Agent Trust (Now PURPLE to match "Ours")
    ax2 = ax1.twinx()
    ln2 = ax2.plot(x_steps, trust_smooth, color=COLOR_OURS, label='Agent\'s Trust (Ours)', linestyle='--', linewidth=7)
    
    ax2.set_ylabel('Agent Trust (Weight)', fontsize=LABEL_FS, color=COLOR_OURS, labelpad=20)
    ax2.tick_params(axis='y', labelcolor=COLOR_OURS, labelsize=TICK_FS)
    
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', fontsize=LEGEND_FS, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.grid(True, alpha=0.1, linestyle='--')
    plt.tight_layout()
    plt.savefig('advanced_experiments_strategic.png')
    print("Saved 'advanced_experiments_strategic.png'")
    plt.close()

    # --- Plot 2: IRL Comparison ---
    plt.figure(figsize=(20, 14))
    ax3 = plt.gca()
    
    methods = ['IRL / GAIL\n(Mimics Majority)', 'ESA\n(Ours)']
    scores = [irl_score, our_score]
    # Bar 2 is already using COLOR_OURS (Purple)
    colors = [COLOR_IRL, COLOR_OURS]
    
    bars = ax3.bar(methods, scores, color=colors, width=0.6)
    
    ax3.set_title('Robustness vs. Imitation', fontsize=TITLE_FS, pad=20)
    ax3.set_ylabel('Latent Ground Truth Reward', fontsize=LABEL_FS, labelpad=20)
    ax3.tick_params(axis='y', labelsize=TICK_FS)
    ax3.tick_params(axis='x', labelsize=TICK_FS)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=40, fontweight='bold')
    plt.grid(True, alpha=0.1, linestyle='--')
    plt.tight_layout()
    plt.savefig('advanced_experiments_irl.png')
    print("Saved 'advanced_experiments_irl.png'")
    plt.close()