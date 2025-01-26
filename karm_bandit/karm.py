import numpy as np
import matplotlib.pyplot as plt


class KArmBandit:
    def __init__(self, k, means, stddevs):
        """Initialize the bandit problem with K arms"""
        self.k = k  # Number of arms
        self.means = means  # Means of the arms' reward distributions
        self.stddevs = stddevs  # Standard deviations of the arms' reward distributions
        self.best_arm = np.argmax(self.means)  # The arm with the highest mean reward
    
    def pull_arm(self, arm):
        """Simulate pulling an arm and receiving a reward"""
        reward = np.random.normal(self.means[arm], self.stddevs[arm])
        return reward


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


def simulate(bandit, agent, n_steps=1000):
    rewards = np.zeros(n_steps)  
    optimal_action_count = 0  
    
    for step in range(n_steps):
        arm = agent.select_arm()
        reward = bandit.pull_arm(arm)
        agent.update_q_value(arm, reward)
        
        rewards[step] = reward
        
        if arm == bandit.best_arm:
            optimal_action_count += 1
    
    return rewards, optimal_action_count


if __name__ == "__main__":
    k = 10  
    true_means = np.random.uniform(0, 1, k) 
    true_stddevs = np.ones(k) * 1.0 
    
    bandit = KArmBandit(k, true_means, true_stddevs)
    agent = EpsilonGreedyAgent(bandit, epsilon=0.1)
    
    rewards, optimal_action_count = simulate(bandit, agent, n_steps=1000)
    
    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / (np.arange(1, len(cumulative_rewards) + 1))
    
    print(f"Total reward after 1000 steps: {cumulative_rewards[-1]}")
    print(f"Optimal arm selected {optimal_action_count} times.")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Reward over Time")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(average_rewards)
    plt.title("Average Reward over Time")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    
    plt.tight_layout()
    plt.show()
