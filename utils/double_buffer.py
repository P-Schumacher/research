import numpy as np
import wandb
import random
import tensorflow as tf
from pudb import set_trace
from matplotlib import pyplot as plt

from utils.replay_buffers import PriorityBuffer
from utils.replay_buffers import SumTree
from utils.math_fns import huber_not_reduce

class DoubleBuffer(PriorityBuffer):
    def reset(self):
        super().reset()
        self.priorities_low = np.zeros((self.max_size, 1), dtype=np.float32)
        self.tree_low = SumTree(self.max_size)

    
    def add(self, state, action, reward, next_state, done, state_seq, action_seq, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        priority = self._get_priority2(error)
        self.priorities_low[self.ptr] = priority
        self.tree_low.add(priority, self.ptr)
        super().add(state, action, reward, next_state, done, state_seq, action_seq, error=100000)

    def _sample_idxs_low(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)
        priorities = np.zeros([batch_size, 1]) 
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(0, self.tree_low.total())
            (tree_idx, p, idx) = self.tree_low.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx
            priorities[i] = p

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.ptr

        sampling_probabilities = priorities / self.tree.total()
        #self.is_weight.assign(np.power(self.tree.n_entries * sampling_probabilities, - self.beta))
        #self.is_weight.assign(self.is_weight /  tf.reduce_max(self.is_weight))
        return batch_idxs

    def sample_low(self, batch_size):

        self.batch_idxs = self._sample_idxs_low(batch_size)
        return(
        tf.convert_to_tensor(self.state[self.batch_idxs]),
        tf.convert_to_tensor(self.action[self.batch_idxs]),
        tf.convert_to_tensor(self.reward[self.batch_idxs]),
        tf.convert_to_tensor(self.next_state[self.batch_idxs]),
        tf.convert_to_tensor(self.done[self.batch_idxs]),
        tf.convert_to_tensor(self.state_seq[self.batch_idxs]),
        tf.convert_to_tensor(self.action_seq[self.batch_idxs]))

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self._get_priority(errors)
        priorities_low = self._get_priority2(1/(tf.abs(errors)+0.00001))
        assert len(priorities) == self.batch_idxs.size
        for idx, p, p_low in zip(self.batch_idxs, priorities, priorities_low):
            self.priorities[idx] = p
            self.priorities_low[idx] = p_low 
        for i, p, p_low in zip(self.tree_idxs, priorities, priorities_low):
            self.tree.update(i, p)
            self.tree_low.update(i, p_low)

    def _get_priority2(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        return np.power(error + 10., 2.).squeeze()
