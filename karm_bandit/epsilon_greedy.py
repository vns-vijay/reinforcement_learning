import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, bandit, epsilon=0.1):
        """Initialize the agent with an epsilon value"""
        self.bandit = bandit
        self.epsilon = epsilon  # Exploration rate
        self.q_values = np.zeros(self.bandit.k)  # Estimated value of each arm
        self.arm_counts = np.zeros(self.bandit.k)  # Number of times each arm was pulled
    
    def select_arm(self):
        """Select an arm based on epsilon-greedy strategy"""
        if np.random.rand() < self.epsilon:
            arm = np.random.choice(self.bandit.k)

        else:
            arm = np.argmax(self.q_values)

        return arm
    
    def update_q_value(self, arm, reward):
        """Update the estimated value (Q-value) of the selected arm"""
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]
