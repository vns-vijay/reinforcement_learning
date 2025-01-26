import numpy as np
import matplotlib.pyplot as plt

from karm_agent import KArmBandit 
from epsilon_greedy import EpsilonGreedyAgent


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
