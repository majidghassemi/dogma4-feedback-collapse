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
        # m=2 means each new node attaches to 2 existing nodes (preferential attachment)
        self.G = nx.barabasi_albert_graph(n=self.M, m=2, seed=42)
        
        # Identify "Patient Zero" (Highest Degree Influencer)
        degrees = dict(self.G.degree())
        self.patient_zero = max(degrees, key=degrees.get)
        
        # Infection State: 0 = Truthful, 1 = Sycophantic
        self.infection_status = np.zeros(self.M)
        self.infection_status[self.patient_zero] = 1.0 # Patient Zero starts infected
        
        # Dynamics parameters
        self.infection_prob = 0.05 # Chance to infect neighbor per "spread event"
        self.spread_interval = 1000 # Spread bias every N steps
        self.total_steps = 0
        
        self.trust_model = TrustMechanism(self.M, learning_rate=0.2)

    def step(self, action):
        obs, latent_reward, term, trunc, info = self.env.step(action)
        self.total_steps += 1
        x_velocity = info.get("x_velocity", latent_reward)

        # --- B. Contagion Dynamics (The Spread) ---
        if self.total_steps % self.spread_interval == 0:
            self._spread_infection()

        # --- C. Generate Signals based on Infection ---
        social_signals = np.zeros(self.M)
        for i in range(self.M):
            noise = np.random.normal(0, 0.1)
            if self.infection_status[i] == 1.0:
                # INFECTED: Sycophantic Lie (Invert Goal)
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
        
        # Log Trust in Patient Zero vs Trust in Truthful
        self.stats['trust_zero'].append(self.trust_model.weights[self.patient_zero])
        # Average trust of currently healthy nodes
        healthy_mask = (self.infection_status == 0)
        if np.sum(healthy_mask) > 0:
            avg_healthy_trust = np.mean(self.trust_model.weights[healthy_mask])
            self.stats['trust_healthy'].append(avg_healthy_trust)
        else:
            self.stats['trust_healthy'].append(0.0)

        return obs, perceived_reward, term, trunc, info

    def _spread_infection(self):
        """Standard SIR-like spread: Infected neighbors infect you."""
        new_infections = self.infection_status.copy()
        for node in self.G.nodes():
            if self.infection_status[node] == 0: # If healthy
                # Check neighbors
                infected_neighbors = sum([self.infection_status[n] for n in self.G.neighbors(node)])
                # Probability of infection increases with exposure
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
    
    # Extract the wrapper to get the graph for plotting later
    graph_wrapper = env.envs[0]
    return stats, graph_wrapper

# ==========================================
# 4. Visualization
# ==========================================
if __name__ == "__main__":
    STEPS = 40000
    
    # Run Internal Feedback Agent
    stats, wrapper = run_graph_experiment('internal_feedback', STEPS)
    
    # Create Figure
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2)
    
    # --- Plot 1: Dynamics over Time ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    # X-axis
    x = np.arange(len(stats['infection_rate']))
    
    # Plot Infection Spread (Area)
    ax1.fill_between(x, stats['infection_rate'], color='red', alpha=0.1, label='Network Infection Rate')
    
    # Plot Trust Dynamics
    ax1.plot(stats['trust_zero'], color='red', linestyle='--', label='Trust in Patient Zero')
    ax1.plot(stats['trust_healthy'], color='green', label='Avg Trust in Healthy Nodes')
    
    ax1.set_title("Dynamic Quarantine: Isolating the Contagion")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Trust Weight / Infection %")
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Social Topology Snapshot (The Graph) ---
    ax2 = fig.add_subplot(gs[0, 1])
    G = wrapper.G
    final_trust = wrapper.trust_model.weights
    final_infection = wrapper.infection_status
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by Trust (Green=Trusted, Red=Untrusted)
    # We normalize trust for coloring 0..1 (relative to max trust)
    norm_trust = final_trust / np.max(final_trust)
    node_colors = plt.cm.RdYlGn(norm_trust) 
    
    # Draw
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, edgecolors='black', ax=ax2)
    
    # Label Patient Zero
    px, py = pos[wrapper.patient_zero]
    ax2.text(px, py+0.1, "Patient Zero", ha='center', fontweight='bold', color='red')
    
    ax2.set_title(f"Final Trust State (T={STEPS})\n(Green=Trusted, Red=Blocked)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('social_topology_results.png')
    print("Saved to 'social_topology_results.png'")
    plt.show()