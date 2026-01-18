import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# ==========================================
# 1. Trust Mechanism (Reused)
# ==========================================
class TrustMechanism:
    def __init__(self, num_evaluators, learning_rate=0.1):
        self.M = num_evaluators
        self.weights = np.ones(self.M) / self.M
        self.eta = learning_rate

    def update(self, social_signals, internal_axiom_signal):
        if internal_axiom_signal is None: return
        loss = np.abs(social_signals - internal_axiom_signal)
        # Update: Strong penalty for lies
        self.weights *= np.exp(-self.eta * loss)
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)

    def get_trusted_reward(self, social_signals):
        return np.dot(self.weights, social_signals)

# ==========================================
# 2. Social Topology Wrapper
# ==========================================
class SocialGraphWrapper(gym.Wrapper):
    def __init__(self, env, stats_buffer, num_evaluators=20, method='internal_feedback'):
        super().__init__(env)
        self.M = num_evaluators
        self.method = method
        self.stats = stats_buffer
        
        # --- A. Build Social Graph (Scale-Free) ---
        self.G = nx.barabasi_albert_graph(n=self.M, m=2, seed=42)
        
        # Identify "Patient Zero" (Highest Degree Influencer)
        degrees = dict(self.G.degree())
        self.patient_zero = max(degrees, key=degrees.get)
        
        # Infection State: 0 = Truthful, 1 = Sycophantic
        self.infection_status = np.zeros(self.M)
        self.infection_status[self.patient_zero] = 1.0 
        
        # Dynamics parameters
        self.infection_prob = 0.05 
        self.spread_interval = 1000 
        self.total_steps = 0
        
        self.trust_model = TrustMechanism(self.M, learning_rate=0.2)

    def step(self, action):
        obs, latent_reward, term, trunc, info = self.env.step(action)
        self.total_steps += 1
        x_velocity = info.get("x_velocity", latent_reward)

        # --- B. Contagion Dynamics ---
        if self.total_steps % self.spread_interval == 0:
            self._spread_infection()

        # --- C. Generate Signals ---
        social_signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            if self.infection_status[i] == 1.0:
                # INFECTED: Sycophantic Lie
                social_signals[i] = latent_reward - (2.0 * x_velocity) + noise
            else:
                # HEALTHY: Truth
                social_signals[i] = latent_reward + noise

        # --- D. Agent Update ---
        internal_signal = None
        if np.random.rand() < 0.10: 
            internal_signal = latent_reward

        self.trust_model.update(social_signals, internal_signal)
        
        if self.method == 'standard':
            perceived_reward = np.mean(social_signals)
        else:
            perceived_reward = self.trust_model.get_trusted_reward(social_signals)

        # Log Metrics
        self.stats['rewards'].append(latent_reward)
        self.stats['infection_rate'].append(np.mean(self.infection_status))
        
        self.stats['trust_zero'].append(self.trust_model.weights[self.patient_zero])
        
        healthy_mask = (self.infection_status == 0)
        if np.sum(healthy_mask) > 0:
            avg_healthy_trust = np.mean(self.trust_model.weights[healthy_mask])
            self.stats['trust_healthy'].append(avg_healthy_trust)
        else:
            self.stats['trust_healthy'].append(0.0)

        return obs, perceived_reward, term, trunc, info

    def _spread_infection(self):
        new_infections = self.infection_status.copy()
        for node in self.G.nodes():
            if self.infection_status[node] == 0: 
                infected_neighbors = sum([self.infection_status[n] for n in self.G.neighbors(node)])
                if np.random.rand() < (1 - (1 - self.infection_prob)**infected_neighbors):
                    new_infections[node] = 1.0
        self.infection_status = new_infections

# ==========================================
# 3. Experiment Runner
# ==========================================
def run_graph_experiment(method_name, timesteps=30000):
    print(f"Running Topology Experiment: {method_name}...")
    stats = {'rewards': [], 'infection_rate': [], 'trust_zero': [], 'trust_healthy': []}
    
    env = gym.make("Hopper-v4", render_mode=None)
    env = SocialGraphWrapper(env, stats, num_evaluators=20, method=method_name)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    
    graph_wrapper = env.envs[0]
    return stats, graph_wrapper

# ==========================================
# 4. Visualization
# ==========================================
if __name__ == "__main__":
    STEPS = 40000
    
    # Styles
    LABEL_FS = 37
    TICK_FS  = 39
    TITLE_FS = 39
    LEGEND_FS = 36
    
    COLOR_PATIENT_ZERO = '#C0392B' # Red
    COLOR_HEALTHY_TRUST = '#6C3483' # Purple (Our Approach/Trust in Good Nodes)
    COLOR_INFECTION_AREA = '#E74C3C' 

    # Run
    stats, wrapper = run_graph_experiment('internal_feedback', STEPS)
    
    # --- Plot 1: Dynamics (Updated Legend Location) ---
    plt.figure(figsize=(20, 14))
    
    x = np.arange(len(stats['infection_rate']))
    
    plt.fill_between(x, stats['infection_rate'], color=COLOR_INFECTION_AREA, alpha=0.1, label='Network Infection Rate')
    plt.plot(stats['trust_zero'], color=COLOR_PATIENT_ZERO, linestyle='--', label='Trust in Patient Zero', linewidth=5)
    plt.plot(stats['trust_healthy'], color=COLOR_HEALTHY_TRUST, label='Avg Trust in Healthy Nodes', linewidth=5)
    
    plt.title("Dynamic Quarantine: Isolating the Contagion", fontsize=TITLE_FS)
    plt.xlabel("Timesteps", fontsize=LABEL_FS)
    plt.ylabel("Trust Weight / Infection %", fontsize=LABEL_FS)
    plt.xticks(fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    
    # FIX: Moved legend to 'upper left' to avoid overlap
    plt.legend(loc='upper left', fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('social_topology_dynamics.png')
    print("Saved 'social_topology_dynamics.png'")
    plt.close()

    # --- Plot 2: Graph Snapshot ---
    plt.figure(figsize=(20, 14))
    G = wrapper.G
    final_trust = wrapper.trust_model.weights
    
    pos = nx.spring_layout(G, seed=42)
    norm_trust = final_trust / np.max(final_trust)
    node_colors = plt.cm.RdYlGn(norm_trust) 
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors='black', linewidths=2)
    
    px, py = pos[wrapper.patient_zero]
    plt.text(px, py+0.1, "Patient Zero", ha='center', fontweight='bold', color=COLOR_PATIENT_ZERO, fontsize=LEGEND_FS)
    
    plt.title(f"Final Trust State (T={STEPS})\n(Green=Trusted, Red=Blocked)", fontsize=TITLE_FS)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('social_topology_graph.png')
    print("Saved 'social_topology_graph.png'")
    plt.close()