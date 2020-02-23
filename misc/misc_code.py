
### MISC potentially useful code####
def debug_the_buffer(buffer, debug_buffer, counter):
	print("this has been step {}".format(counter))
	counter += 1
	for idx in range(debug_buffer.ptr):
		print("States")
		print("DEBUG: {}".format(debug_buffer.state[idx]))
		print("SUB: {}".format(buffer.state[idx]))
		print("Next States")
		print("DEBUG: {}".format(debug_buffer.next_state[idx]))
		print("SUB: {}".format(buffer.next_state[idx]))
		print("Rewards")
		print("DEBUG: {}".format(debug_buffer.reward[idx]))
		print("SUB: {}".format(buffer.reward[idx]))
		print("Actions")
		print("DEBUG: {}".format(debug_buffer.action[idx]))
		print("SUB: {}".format(buffer.action[idx]))


def print_fct(self, state, action, reward, next_state, done):
	print("States")
	print(state)
	print("Actions")
	print(action)
	print("Rewards")
	print(reward)
	print("Next_states")
	print(next_state)
	print()

def replay_buffer_plotter(replay_buffer):
	plt.subplot(231)
	state = replay_buffer.state[replay_buffer.ptr]
	plt.plot(state[0], 'r')
	plt.plot(state[0], 'b')
	plt.subplot(232)
	action = replay_buffer.action[replay_buffer.ptr]
	plt.plot(action[0], 'r')
	plt.subplot(233)
	plt.plot(replay_buffer.reward[ptr], 'r')


def debug_setup():
	metrics = np.zeros((50, int(3e6)))
	return metrics

def debug_save(metrics, state, goal, extr_reward, intr_reward, step):
	for i in range(state.shape[0]):
		metrics[i, step] = state[i]
	for i in range(goal.shape[0]):
		metrics[i + state.shape[0], step] = goal[i]
	metrics[0 + state.shape[0] + goal.shape[0], step ] = intr_reward
	metrics[1 + state.shape[0] + goal.shape[0], step ] = extr_reward


def buffer_print(buffer, t):
    print("sub agent")
    for i in range(t + 5):
        state = buffer.sub_replay_buffer.state[i, 0:3]
        subgoal = buffer.sub_replay_buffer.state[i, 30:32]
        action = buffer.sub_replay_buffer.action[i, :]
        next_state = buffer.sub_replay_buffer.next_state[i, 0:3]
        done = buffer.sub_replay_buffer.done[i]
        reward = buffer.sub_replay_buffer.reward[i]
        print(f"t: {i} State: {state}")
        print(f"subgoal: {subgoal}")
        print(f"action: {action}")
        print(f"next_state: {next_state}")
        print(f"done: {done}")
        print(f"reward:{reward}")

    print("meta agent")
    for i in range(t):
        state = buffer.meta_replay_buffer.state[i, 0:3]
        subgoal = buffer.meta_replay_buffer.state[i, 30:]
        action = buffer.meta_replay_buffer.action[i, 0:3]
        next_state = buffer.meta_replay_buffer.next_state[i, 0:3]
        done = buffer.meta_replay_buffer.done[i]
        reward = buffer.meta_replay_buffer.reward[i]
        print(f"t: {i} State: {state}")
        print(f"target: {subgoal}")
        print(f"action: {action}")
        print(f"next_state: {next_state}")
        print(f"done: {done}")
        print(f"reward:{reward}")


