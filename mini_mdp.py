import numpy as np
from pudb import set_trace
N = 10
M = 100000
class Env:
    def __init__(self):
        self.state = np.zeros([N])
        self.state[0] = 1

    def step(self, x):
        if self.state[0]:
            if x == -1:
                rew = - (N - np.where(self.state == 1)[0]) 
                return self.state, rew, False

        if not self.state[-1]:
            self.state = np.roll(self.state, x)
            rew = - (N - np.where(self.state == 1)[0]) 
            return self.state, rew, False

        else: 
            rew = 0
            return self.state, rew, True
       

    def reset(self):
        self.state = np.zeros([N])
        self.state[0] = 1
        return self.state, False

def select_action(q, s):
    q_value = q[int(np.where(state == 1)[0]), :]
    index = np.argmax(q_value)
    if index == 0:
        return -1
    else: 
        return 1

def q_update(q, state, action, reward, next_state, done):
    lr = 0.001
    gamma = 0.99
    action_index = [1 if action == 1 else 0][0]
    state_index = int(np.where(state == 1)[0])
    next_state_index = int(np.where(next_state == 1)[0])
    done = False
    q[state_index, action_index] =  q[state_index, action_index] + lr  * (reward + gamma * select_action(q, next_state)
                                                                          * (1 - done))

q = np.random.normal(0., 1., size=[N, 2])
env = Env()
step_list = []
for i in range(M):
    state, done = env.reset()
    steps = 0
    while not done:
        steps += 1
        action = select_action(q, state)
        if np.random.uniform(0, 1) < 0.1:
            action = np.random.choice([-1 ,1])
        next_state, reward, done = env.step(action)
        print(f'Q TABLE: {q}')
        q_update(q, state, action, reward, next_state, done)
        state = next_state
        print('done!')
    step_list.append(steps)
print(np.sum(np.array(step_list)))
