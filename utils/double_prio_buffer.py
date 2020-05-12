from utils.replay_buffers import PriorityBuffer, SumTree
import numpy as np
import tensorflow as tf
import random


class DoublePrioBuffer(PriorityBuffer):
    def __init__(self, state_dim, action_dim, buffer_cnf):
        super().__init__(state_dim, action_dim, buffer_cnf)
        self.alpha2 = np.full((1,), self.alpha)
        self.is_weight2 = tf.Variable(tf.zeros([buffer_cnf.batch_size,]), dtype=tf.float32)
        self.reset()

    def reset(self):
        super().reset()
        self.priorities2 = np.zeros((self.max_size, 1), dtype=np.float32)
        self.tree2 = SumTree(self.max_size)

    def sample2(self, batch_size):
        self.batch_idxs2 = self._sample_idxs2(batch_size)
        return (  
        tf.convert_to_tensor(self.state[self.batch_idxs2]),
        tf.convert_to_tensor(self.action[self.batch_idxs2]),
        tf.convert_to_tensor(self.reward[self.batch_idxs2]),
        tf.convert_to_tensor(self.next_state[self.batch_idxs2]),
        tf.convert_to_tensor(self.done[self.batch_idxs2]),
        tf.convert_to_tensor(self.state_seq[self.batch_idxs2]),
        tf.convert_to_tensor(self.action_seq[self.batch_idxs2]))

        
    def add2(self, state, action, reward, next_state, done, state_seq, action_seq, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        priority = self._get_priority(error)
        self.priorities2[self.ptr] = priority
        self.tree2.add(priority, self.ptr)
        # Call this AFTER the others, it increments the *self.ptr* pointer
        # otherwise the priorities won't match the transitions

    def _sample_idxs2(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)
        priorities = np.zeros(batch_size) 
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            s = random.uniform(0, self.tree2.total())
            (tree_idx, p, idx) = self.tree2.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx
            priorities[i] = p

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs2 = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.ptr

        sampling_probabilities = priorities / self.tree2.total()
        self.is_weight2.assign(np.power(self.tree2.n_entries * sampling_probabilities, - self.beta))
        self.is_weight2.assign(self.is_weight2 /  tf.reduce_max(self.is_weight2))
        return batch_idxs
    
    def update_priorities2(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self._get_priority(errors)
        assert len(priorities) == self.batch_idxs2.size
        for idx, p in zip(self.batch_idxs2, priorities):
            self.priorities2[idx] = p
        for p, i in zip(priorities, self.tree_idxs2):
            self.tree2.update(i, p)
