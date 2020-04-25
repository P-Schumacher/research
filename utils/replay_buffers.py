import numpy as np
import tensorflow as tf
from pudb import set_trace
import random

class ReplayBuffer(object):
    '''Simple replay buffer class which samples tensorflow tensors.'''
    def __init__(self, state_dim, action_dim, buffer_cnf):
        if not cnf.offpolicy:
            c_step = 1
        else:
            c_step = cnf.c_step
        self.max_size = cnf.max_size
        self.ptr = 0
        self.size = 0 
        self.action_dim = action_dim
        self.state = np.zeros((cnf.max_size, state_dim), dtype = np.float32)
        self.action = np.zeros((cnf.max_size, action_dim), dtype = np.float32)
        self.next_state = np.zeros((cnf.max_size, state_dim), dtype = np.float32)
        self.reward = np.zeros((cnf.max_size, 1), dtype = np.float32)
        self.done = np.zeros((cnf.max_size, 1), dtype = np.float32)
        self.state_seq = np.zeros((cnf.max_size, c_step, state_dim - 3 * cnf.goal_smooth), dtype = np.float32)  
        self.action_seq = np.zeros((cnf.max_size, c_step, 8), dtype = np.float32)  
        self._prepare_parameters(buffer_cnf)

    def _prepare_parameters(self, buffer_cnf):
        self.alpha
        self.beta
        self.use_cer
        self.


    def add(self, state, action, reward, next_state, done, state_seq, action_seq):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.state_seq[self.ptr] = state_seq
        self.action_seq[self.ptr] = action_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = self._sample_idxs(batch_size)
        return (  
        tf.convert_to_tensor(self.state[ind]),
        tf.convert_to_tensor(self.action[ind]),
        tf.convert_to_tensor(self.reward[ind]),
        tf.convert_to_tensor(self.next_state[ind]),
        tf.convert_to_tensor(self.done[ind]),
        tf.convert_to_tensor(self.state_seq[ind]),
        tf.convert_to_tensor(self.action_seq[ind]))

    def reset(self):
        pass

    def _sample_idxs(self, batch_size):
        '''Uniformly samples idxs to sample from. This fct is overwritten by PER to
        sample non-uniformly.'''
        return  tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)


class PrioritizedReplay(ReplayBuffer):
    '''
    Prioritized Experience Replay
    Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" https://arxiv.org/pdf/1511.05952.pdf and is Jaromír Janisch's with minor adaptations.
    See memory_util.py for the license and link to Jaromír's excellent blog
    Stores agent experiences and samples from them for agent training according to each experience's priority
    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree to store and sample the priorities.
    e.g. memory_spec
    "memory": {
        "name": "PrioritizedReplay",
        "alpha": 1,
        "epsilon": 0,
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, state_dim, action_dim, c_step, offpolicy, max_size, goal_smooth=0):
        # TODO initialize all settings from normal buffer
        super().__init__(self, state_dim, action_dim, c_step, offpolicy, max_size, goal_smooth=0)

        self.epsilon = np.full((1,), self.epsilon)
        self.alpha = np.full((1,), self.alpha)
        # adds a 'priorities' scalar to the data_keys and call reset again
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        self.reset()

    def reset(self):
        super().reset()
        self.tree = SumTree(self.max_size)

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        super().add_experience(state, action, reward, next_state, done)
        priority = self.get_priority(error)
        self.priorities[self.head] = priority
        self.tree.add(priority, self.head)

    def _get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        return np.power(error + self.epsilon, self.alpha).squeeze()

    def _sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_idxs.size
        for idx, p in zip(self.batch_idxs, priorities):
            self.priorities[idx] = p
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)

class SumTree:
    '''
    Helper class for PrioritizedReplay
    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
    MIT License
    Copyright (c) 2018 Jaromír Janisch
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    In this implementation, a conventional FIFO buffer holds the data, while the tree 
    only holds the indices and the corresponding priorities.

    N.B. A full binary tree has 2^N - 1 nodes for N leaves. (c.f. Gaussian Summation Formula)
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')

