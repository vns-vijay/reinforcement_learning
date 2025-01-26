import numpy as np


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
    