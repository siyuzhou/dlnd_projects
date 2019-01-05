import copy
import random
from collections import namedtuple, deque
import numpy as np
import torch


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)

        random.seed(seed)

        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu.)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * \
            np.array([np.random.rand() for i in range(len(self.state))])
        self.state += dx
        return self.state


class SumTree:
    """
    Left balanced sum tree.
    # Code borrowed and modified from https://github.com/rlcode/per.
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Max number of leaves.
        self.tree = np.zeros(2 * capacity - 1)  # Values of the tree nodes.

    @property
    def root(self):
        return self.tree[0]

    def update(self, idx, p):
        """UPdate the leaf node of index `idx` with value `p`."""
        if idx < 0 or idx > self.capacity - 1:
            raise ValueError('idx must be a positive integer small than capacity')
        tree_idx = idx + self.capacity - 1

        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        self._propagate(tree_idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent > 0:
            self._propagate(parent, change)

    def get(self, s):
        """
        Get the index and value of the leaf that is required to reach sum `s`, 
        if the leaves are summed from left to right. Also return the index of
        the leaf if the leaves are ordered from lower level to higher level and 
        from left to right.
        """
        if s > self.root:
            raise ValueError("required sum exceeds the root value")
        idx = 0

        left = 2 * idx + 1  # Index of left child.

        while left < len(self.tree):
            right = left + 1  # Index of right child.
            if s <= self.tree[left] or self.tree[right] == 0:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
            left = 2 * idx + 1

        return idx - self.capacity + 1, self.tree[idx]


class PrioritizedMemory:
    """
    Prioritized memory buffer.
    # Code borrowed and modified from https://github.com/rlcode/per.

    Priority is calculated as:
        p = (e + ϵ)ᵅ, where e is the error, α is a constant exponent, and ϵ is a small
        positive constant. 
    """

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment = 0.0001

    def __init__(self, capacity, seed):
        self.priority_sumtree = SumTree(capacity)
        self.experiences = np.zeros(capacity, dtype=object)

        self.capacity = capacity
        self.fill = 0
        self.write_idx = 0

        random.seed(seed)

    def __len__(self):
        return self.fill

    def _priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def update(self, idx, error):
        """Update the priority of index `idx` only."""
        p = self._priority(error)
        self.priority_sumtree.update(idx, p)

    def add(self, error, experience):
        """Add priority and experience to memory."""
        self.update(self.write_idx, error)
        self.experiences[self.write_idx] = experience

        self.fill = min(self.fill + 1, self.capacity)
        self.write_idx = (self.write_idx + 1) % self.capacity

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []

        bin_size = self.priority_sumtree.root / n  # Divide sum of priorities in to n bins.

        # Linearly increase beta until 1.
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(n):
            a = bin_size * i
            b = bin_size * (i + 1)

            # Sample an experience from each bin.
            s = random.uniform(a, b)
            idx, p = self.priority_sumtree.get(s)
            if idx >= self.fill:
                print('\n')
                print(idx, self.fill)
                print(s, self.priority_sumtree.root)
                print(self.experiences[idx])
                # import pickle
                # with open('temp', 'wb') as f:
                #     pickle.dump(self.priority_sumtree, f)
                raise ValueError

            idx = min(idx, self.fill - 1)
            experience = self.experiences[idx]

            priorities.append(p)
            idxs.append(idx)
            batch.append(experience)

        sampling_probabilities = priorities / self.priority_sumtree.root
        importance_weight = np.power(self.fill *
                                     sampling_probabilities, -self.beta)

        importance_weight /= importance_weight.max()  # Normalization.

        return batch, idxs, importance_weight
