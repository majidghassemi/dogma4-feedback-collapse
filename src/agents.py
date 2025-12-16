import numpy as np
import random

class BaseAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_max_q = max([self.get_q(next_state, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q

class Dogma4Agent(BaseAgent):
    """Standard RL: Averages all social feedback blindly."""
    def process_feedback(self, feedbacks, internal_signal=None):
        return np.mean(feedbacks)

class DawidSkeneAgent(BaseAgent):
    """Baseline: Truth Discovery using Online EM.
    Fails under systematic bias because the 'majority vote' is wrong.
    """
    def __init__(self, actions, num_evaluators, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        self.trust_weights = np.ones(num_evaluators) / num_evaluators
        self.history_feedbacks = [] # Store history for EM steps

    def process_feedback(self, feedbacks, internal_signal=None):
        # 1. Store history
        self.history_feedbacks.append(feedbacks)
        
        # 2. Run EM periodically (e.g., every 50 steps) to update trust
        if len(self.history_feedbacks) >= 10 and len(self.history_feedbacks) % 50 == 0:
            self._run_em_step()

        # 3. Weighted Aggregation using learned trust
        return np.dot(self.trust_weights, feedbacks)

    def _run_em_step(self):
        """
        Expectation-Maximization:
        E-Step: Estimate 'truth' as weighted average of history.
        M-Step: Update trust based on distance from that estimated truth.
        """
        data = np.array(self.history_feedbacks) # Shape [T, M]
        
        # E-Step: What is the consensus 'truth' for each past step?
        # Shape [T]
        estimated_truth = np.average(data, axis=1, weights=self.trust_weights)
        
        # M-Step: Who disagrees with the consensus?
        # Calculate Mean Squared Error for each evaluator relative to consensus
        # Shape [M]
        errors = np.mean((data - estimated_truth[:, None])**2, axis=0)
        
        # Invert error to get trust (add epsilon to avoid div/0)
        new_trust = 1.0 / (errors + 1e-6)
        
        # Normalize
        if np.sum(new_trust) > 0:
            self.trust_weights = new_trust / np.sum(new_trust)

class InternalFeedbackAgent(BaseAgent):
    """Dogma-4 Compliant: Filters feedback based on internal axioms."""
    def __init__(self, actions, num_evaluators, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(actions, alpha, gamma, epsilon)
        self.trust_weights = np.ones(num_evaluators) / num_evaluators
        self.eta = 0.5 # Trust Learning rate

    def process_feedback(self, feedbacks, internal_signal):
        """
        internal_signal: "SAFETY_VIOLATION" means the agent detected an objective harm.
        """
        # 1. Update Trust
        if internal_signal == "SAFETY_VIOLATION":
            for i, r in enumerate(feedbacks):
                # If evaluator gave neutral/positive reward (r > -5.0) for a disaster, they are lying.
                # (Assuming True penalty is -50, so anything > -5 is suspicious)
                violation = 1.0 if r > -5.0 else 0.0
                self.trust_weights[i] *= np.exp(-self.eta * violation)
            
            # Renormalize
            if np.sum(self.trust_weights) > 0:
                self.trust_weights /= np.sum(self.trust_weights)
        
        # 2. Weighted Aggregation
        return np.dot(self.trust_weights, feedbacks)