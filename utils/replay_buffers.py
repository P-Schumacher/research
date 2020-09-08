import numpy as np
import wandb
import tensorflow as tf
from pudb import set_trace
import random
from matplotlib import pyplot as plt
from collections import deque



class ReplayBuffer(object):
    '''Simple replay buffer class which samples tensorflow tensors.'''
    def __init__(self, state_dim, action_dim, buffer_cnf):
        self._prepare_parameters(state_dim, action_dim, buffer_cnf)
        self.reset()

    def add(self, state, action, reward, next_state, done, state_seq, action_seq, reversed_reward, unmod_state=None,
            stabe=False, unmod_next_state=None):
        self.state[self.ptr] = state.astype(np.float16)
        if stabe:
            self.unmod_state[self.ptr] = unmod_state.astype(np.float16)
            self.unmod_next_state[self.ptr] = unmod_next_state.astype(np.float16)
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state.astype(np.float16)
        self.reward[self.ptr] = reward
        self.reversed_reward[self.ptr] = reversed_reward
        self.done[self.ptr] = float(done)
        self.state_seq[self.ptr] = state_seq
        self.action_seq[self.ptr] = action_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        self.batch_idxs = self._sample_idxs(batch_size)
        return (  
        tf.convert_to_tensor(self.state[self.batch_idxs]),
        tf.convert_to_tensor(self.action[self.batch_idxs]),
        tf.convert_to_tensor(self.reward[self.batch_idxs]),
        tf.convert_to_tensor(self.next_state[self.batch_idxs]),
        tf.convert_to_tensor(self.done[self.batch_idxs]),
        tf.convert_to_tensor(self.state_seq[self.batch_idxs]),
        tf.convert_to_tensor(self.action_seq[self.batch_idxs]),
        tf.convert_to_tensor(self.reversed_reward[self.batch_idxs]),
        tf.convert_to_tensor(self.unmod_state[self.batch_idxs]),
        tf.convert_to_tensor(self.unmod_next_state[self.batch_idxs]))

    def save_data(self):
        '''Saves the buffer data to the disk. Only useful for analyzing them.'''
        for field in self.data_fields:
            np.save(f'{field}.npy', getattr(self, field))

    def load_data(self, directory):
        '''Loads buffer data from the disk. Remember to set the size and ptr manually.'''
        for field in self.data_fields:
            setattr(self, field, np.load(f'{directory}{field}.npy'))
        index = np.where(self.state[:,0] == 0)[0][0]
        self.size = index
        self.ptr = index

    def reset(self):
        self.state = np.zeros((self.max_size, self.state_dim), dtype = np.float32)
        self.unmod_state = np.zeros((self.max_size, self.state_dim+3), dtype = np.float32)
        self.unmod_next_state = np.zeros((self.max_size, self.state_dim+3), dtype = np.float32)
        self.action = np.zeros((self.max_size, self.action_dim), dtype = np.float32)
        self.next_state = np.zeros((self.max_size, self.state_dim), dtype = np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype = np.float32)
        self.done = np.zeros((self.max_size, 1), dtype = np.float32)
        self.state_seq = np.zeros((self.max_size, self.c_step, self.state_dim - 3 * self.goal_smooth), dtype =
                                  np.float32)
        self.action_seq = np.zeros((self.max_size, self.c_step, 8), dtype = np.float32)  
        self.reversed_reward = np.zeros((self.max_size, 1), dtype = np.float32)  
        self.ptr = 0
        self.size = 0

    def _sample_idxs(self, batch_size):
        '''Uniformly samples idxs to sample from. This fct is overwritten by PER to
        sample non-uniformly.'''
        return  tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)

    def _prepare_parameters(self, state_dim, action_dim, buffer_cnf):
        if not buffer_cnf.offpolicy:
            self.c_step = 1
        else:
            self.c_step = buffer_cnf.c_step
        self.max_size = buffer_cnf.max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_smooth = buffer_cnf.goal_smooth
        self.alpha = buffer_cnf.alpha
        self.epsilon = buffer_cnf.epsilon
        self.beta = buffer_cnf.beta
        self.beta_increment = buffer_cnf.beta_increment
        self.use_cer = buffer_cnf.use_cer
        self.is_weight = 1
        self.data_fields = ['state', 'action', 'next_state', 'reward', 'done', 'state_seq', 'action_seq']

class nstepbuffer(ReplayBuffer):
    '''This is a WRAPPER class for other ReplayBuffer type classes to enable n-step returns'''
    def __init__(self, replaybuffer, nstep=5):
        self._replay_buffer = replaybuffer
        self.nstep = nstep
        self.n_step_buffer = deque(maxlen=self.nstep)

    def sample(self, batch_size):
        return self._replay_buffer.sample(batch_size)

    def add(self, state, action, reward, next_state, done, state_seq, action_seq):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.nstep:
            state, action, reward, next_state, done = self._calc_multistep_transitions()
            self._replay_buffer.add(state, action, reward, next_state, done, 0, 0)

    def _calc_multistep_transitions(self):
        ret = 0
        for idx in range(self.nstep):
            ret += 0.99**idx * self.n_step_buffer[idx][2]
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], ret, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    @property
    def is_weight(self):
        return self._replay_buffer.is_weight

    def update_priorities(self, errors):
        self._replay_buffer.update_priorities(errors)

    def reset(self):
        self._replay_buffer.reset()
        
class PriorityBuffer(ReplayBuffer):
    '''
    Prioritized Experience Replay
    Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" https://arxiv.org/pdf/1511.05952.pdf and is Jaromír Janisch's with minor adaptations.
    Stores agent experiences and samples from them for agent training according to each experience's priority
    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree and an array to store and sample the priorities.
    '''

    def __init__(self, state_dim, action_dim, buffer_cnf):

        super().__init__(state_dim, action_dim, buffer_cnf)
        self.epsilon = np.full((1,), self.epsilon)
        self.alpha = np.full((1,), self.alpha)
        self.is_weight = tf.Variable(tf.zeros([buffer_cnf.batch_size, 1]), dtype=tf.float32) 
        self.reset()

    def reset(self):
        super().reset()
        self.priorities = np.zeros((self.max_size, 1), dtype = np.float32)
        self.tree = SumTree(self.max_size)

    def add(self, state, action, reward, next_state, done, state_seq, action_seq, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        priority = self._get_priority(error)
        self.priorities[self.ptr] = priority
        self.tree.add(priority, self.ptr)
        # Call this AFTER the others, it increments the *self.ptr* pointer
        # otherwise the priorities won't match the transitions
        super().add(state, action, reward, next_state, done, state_seq, action_seq)

    def save_data(self):
        '''Saves the buffer data to the disk. Only useful for analyzing them.'''
        super().save_data()
        np.save('priorities.npy', self.priorities)
        np.save('tree.npy', self.tree.tree)        
        np.save('indices.npy', self.tree.indices)        

    def load_data(self, directory):
        '''Loads buffer data from the disk. Remember to set the size and ptr manually.'''
        print('Loading priority buffer data...')
        super().load_data(directory)
        try:
            self.priorities = np.load(f'{directory}priorities.npy')
            self.tree.tree = np.load(f'{directory}tree.npy')
            self.tree.indices = np.load(f'{directory}indices.npy')
            print('Loading successfull')
        except:
            print('Loading priority buffer fields has failed, moving to manual priorities')
            self._update_manual_priorities()

    def _update_manual_priorities(self): 
        state, action, reward, next_state, done = self.get_buffer() 
        td_error = np.ones(shape=[state.shape[0],], dtype=np.float32) * 100000
        self.batch_idxs = np.arange(0, self.size, 1) 
        self.tree_idxs = self.batch_idxs + self.tree.capacity - 1 
        self.update_priorities(td_error) 


    def sample_uniformly(self, batch_size):
        self.batch_idxs = np.random.uniform(0, self.size, [batch_size,])
        self.batch_idxs = np.asarray(a=self.batch_idxs, dtype=np.int32)
        return (  
        tf.convert_to_tensor(self.state[self.batch_idxs]),
        tf.convert_to_tensor(self.action[self.batch_idxs]),
        tf.convert_to_tensor(self.reward[self.batch_idxs]),
        tf.convert_to_tensor(self.next_state[self.batch_idxs]),
        tf.convert_to_tensor(self.done[self.batch_idxs]),
        tf.convert_to_tensor(self.state_seq[self.batch_idxs]),
        tf.convert_to_tensor(self.action_seq[self.batch_idxs]))

    def _get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        #if np.isnan(np.power(error + self.epsilon, self.alpha)):
        #    set_trace()
        return np.power(error + self.epsilon, self.alpha).squeeze()

    def _sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)
        priorities = np.zeros([batch_size, 1]) 
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx
            priorities[i] = p

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.ptr

        #sampling_probabilities = priorities / self.tree.total()
        #self.is_weight.assign(np.power(self.tree.n_entries * sampling_probabilities, - self.beta))
        #self.is_weight.assign(self.is_weight /  tf.reduce_max(self.is_weight))
        return batch_idxs

    def _sample_idxs_without_index(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority, but without
        modifying the class state, such that we do not update priorities in subsequent steps.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)
        priorities = np.zeros([batch_size, 1]) 
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx
            priorities[i] = p

        batch_idxs = np.asarray(batch_idxs).astype(int)
        tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.ptr

        sampling_probabilities = priorities / self.tree.total()
        self.is_weight.assign(np.power(self.tree.n_entries * sampling_probabilities, - self.beta))
        self.is_weight.assign(self.is_weight /  tf.reduce_max(self.is_weight))
        return batch_idxs

    def sample_without_index(self, batch_size):
        batch_idxs = self._sample_idxs_without_index(batch_size)
        return (  
        tf.convert_to_tensor(self.state[batch_idxs]),
        tf.convert_to_tensor(self.action[batch_idxs]),
        tf.convert_to_tensor(self.reward[batch_idxs]),
        tf.convert_to_tensor(self.next_state[batch_idxs]),
        tf.convert_to_tensor(self.done[batch_idxs]),
        tf.convert_to_tensor(self.state_seq[batch_idxs]),
        tf.convert_to_tensor(self.action_seq[batch_idxs]))

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self._get_priority(errors)
        assert len(priorities) == self.batch_idxs.size
        for idx, p in zip(self.batch_idxs, priorities):
            self.priorities[idx] = p
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)
    
    def get_buffer(self):
        return (tf.convert_to_tensor(self.state[:self.size]),
                tf.convert_to_tensor(self.action[:self.size]),
                tf.convert_to_tensor(self.reward[:self.size]),
                tf.convert_to_tensor(self.next_state[:self.size]),
                tf.convert_to_tensor(self.done[:self.size]))

class SumTree:
    '''
    Taken from 'Foundations of deep reinforcement learning' Book:
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
    P.Schumacher:
    N.B. A full binary tree has 2*N - 1 nodes for N leaves. (c.f. Gaussian Summation Formula)
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences
        self.n_entries = 0.

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
        if self.n_entries < self.capacity:
            self.n_entries += 1

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

