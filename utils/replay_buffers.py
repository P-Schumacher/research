import numpy as np
import tensorflow as tf
from pudb import set_trace
import random

class ReplayBuffer(object):
    '''Simple replay buffer class which samples tensorflow tensors.'''
    def __init__(self, state_dim, action_dim, c_step, offpolicy, max_size, goal_smooth=0):
        if not offpolicy:
            c_step = 1
        self.max_size = max_size
        self.ptr = 0
        self.size = 0 
        self.action_dim = action_dim
        self.state = np.zeros((max_size, state_dim), dtype = np.float32)
        self.action = np.zeros((max_size, action_dim), dtype = np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype = np.float32)
        self.reward = np.zeros((max_size, 1), dtype = np.float32)
        self.done = np.zeros((max_size, 1), dtype = np.float32)
        self.state_seq = np.zeros((max_size, c_step, state_dim - 3 * goal_smooth), dtype = np.float32)  
        self.action_seq = np.zeros((max_size, c_step, 8), dtype = np.float32)  

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
        ind = tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)
        return (  
        tf.convert_to_tensor(self.state[ind]),
        tf.convert_to_tensor(self.action[ind]),
        tf.convert_to_tensor(self.next_state[ind]),
        tf.convert_to_tensor(self.reward[ind]),
        tf.convert_to_tensor(self.done[ind]),
        tf.convert_to_tensor(self.state_seq[ind]),
        tf.convert_to_tensor(self.action_seq[ind]))




class PriorityBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    pmax = 1e6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, state, action, reward, next_state, done, state_seq, action_seq):
        sample = (state, action, reward, next_state, done)
        error = self.pmax
        self._add(error, sample)

    def _add(self, error, sample):
        #p = self._getPriority(error)
        p = error
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return self._get_from_batch(batch, idxs, is_weight)

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


    def _get_from_batch(self, batch, idxs, is_weight):
        self.idxs = idxs
        self.is_weight = is_weight
        return(tf.convert_to_tensor([i[0] for i in batch], dtype=tf.float32),
        tf.convert_to_tensor([i[1] for i in batch], dtype=tf.float32),
        tf.convert_to_tensor([i[2] for i in batch], dtype=tf.float32),
        tf.convert_to_tensor([i[3] for i in batch], dtype=tf.float32),
        tf.convert_to_tensor([i[4] for i in batch], dtype=tf.float32),
        0,
        0)

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

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

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
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
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

