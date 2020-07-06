import tensorflow as tf

@tf.function
def off_policy_correction(subgoal_ranges, target_dim, pi, goal_b, state_b, next_state_b, no_candidates, c_step, state_seq,
                          action_seq, zero_obs):
    # TODO Update docstring to real dimensions
    '''Computes the off-policy correction for the meta-agent.
    c = candidate_nbr; t = time; i = vector coordinate (e.g. action 0 of 8 dims) 
    Dim(candidates) = [b_size, g_dim, no_candidates+2 ]
    Dim(state_b)     = [b_size, state_dim]
    Dim(action_seq)    = []
    Dim(prop_goals) = [c_step, b_size, g_dim, no_candidates]'''
    b_size = state_b.shape[0] # Batch Size
    g_dim = goal_b[0].shape[0] # Subgoal dimension
    action_dim = action_seq.shape[-1] # action dimension
    # States contains state+target. Need to take out only the state.
    state_seq = state_seq[:,:, :-target_dim]
    state_b = state_b[:, :-target_dim]
    next_state_b = next_state_b[:, :-target_dim]
    # Get the sampled candidates
    candidates =  _get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b)
    # Take all the possible candidates and propagate them through time using h = st + gt - st+1 cf. HIRO Paper
    prop_goals = _multi_goal_transition(state_seq, candidates, c_step)
    # Zero out xy for sub agent, AFTER goals have been calculated from it.
    state_seq = tf.reshape(state_seq, [b_size * c_step, state_seq.shape[-1]])
    if zero_obs:
        state_seq *= tf.concat([tf.zeros([state_seq.shape[0], zero_obs]), tf.ones([state_seq.shape[0],
                                                                                         state_seq.shape[1] -
                                                                                         zero_obs])], axis=1)
    best_c = _get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi) 
    return _get_corrected_goal(b_size, candidates, best_c) 

def _multi_goal_transition(state_seq, candidates, c_step):
    # Realize that the multi timestep goal transition can be vectorized to a single calculation.
    b_size = candidates.shape[0]
    g_dim = candidates.shape[1]
    c_step = state_seq.shape[1] 
    no_candidates = candidates.shape[2]
    # In state_seq, equal batches are separated by c timesteps if we query a subogoal every c_steps 
    prop_goals = tf.broadcast_to(tf.expand_dims(candidates, axis=1), [b_size, c_step, g_dim, no_candidates])
    tmp = tf.broadcast_to(tf.expand_dims(state_seq[:,0,:g_dim], axis=1), [b_size, c_step, g_dim]) - state_seq[:, :, :g_dim] 
    prop_goals += tf.broadcast_to(tf.expand_dims(tmp, axis=3), [b_size, c_step, g_dim, no_candidates]) 
    return prop_goals 

def _get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b):
    # Original Goal
    orig_goal = tf.expand_dims(goal_b[:, :g_dim], axis=2)
    # Goal between the states s_t+1 - s_t
    diff_goal = tf.expand_dims(next_state_b[:, :g_dim] - state_b[:, :g_dim], axis=2)
    goal_mean = tf.broadcast_to(diff_goal, [b_size, g_dim, no_candidates])
    # Broadcast the subgoal_ranges to [b_size, g_dim, no_candidates] for use as clipping and as std
    clip_tensor = tf.expand_dims(tf.broadcast_to(subgoal_ranges, [b_size, subgoal_ranges.shape[0]]), axis=2)
    clip_tensor = tf.broadcast_to(clip_tensor, [b_size, subgoal_ranges.shape[0], no_candidates+2])
    goal_std = 0.25 * clip_tensor # cf. HIRO Paper
    candidates = tf.random.normal([b_size, g_dim, no_candidates], goal_mean, goal_std[:, :, :no_candidates])
    candidates = tf.concat([orig_goal, diff_goal, candidates], axis=2)
    candidates = tf.clip_by_value(candidates, -clip_tensor, clip_tensor)
    return candidates

def _get_corrected_goal(b_size, candidates, best_c): 
    corrected_goals = tf.TensorArray(dtype=tf.float32, size=b_size)
    for b in tf.range(b_size):
        corrected_goals = corrected_goals.write(b, candidates[b, :, best_c[b]])
    return corrected_goals.stack() 

def _get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi):
    # Compute the logpropabilities for different subgoal candidates
    max_logprob = tf.constant(-1e9, shape=[b_size,])
    best_c = tf.zeros([b_size,], dtype=tf.int32) # In Graph mode, elements need to be defined BEFORE the loop
    action_seq = tf.reshape(action_seq, [b_size * c_step, action_dim])
    for c in tf.range(no_candidates):
        goals = prop_goals[:, :, :, c]  # Pick one candidate 
        # Change dimension of goals so that it can be computed in one batch in the network [b_size * c_step, g_dim]
        goals = tf.reshape(goals, [b_size * c_step, g_dim])
        state_action = tf.concat([state_seq, goals], axis=1)
        pred_action =  pi(state_action)
        diff = action_seq - pred_action
        # Have padded the sequences where len(seq)<c_step to the max(c_step) with np.inf
        # This results in nan in the computed action differences. These are now set to 0
        # as 0 does not influence the sum over all actions that happens later.
        diff = tf.where(tf.math.is_nan(diff), 0., diff)
        # This reshape only works when rehaping consecutive dimension (e.g. [2560,8] to [256,10,8] or vice versa. 
        # Else elements will be thrown around.
        diffn = tf.reshape(diff, [b_size, c_step, action_dim])
        # cf. HIRO Paper
        logprob = - 0.5 * tf.reduce_sum(tf.square(tf.norm(diffn, axis=2)), axis=1)
        best_c = tf.where(logprob > max_logprob, c, best_c)
        max_logprob = tf.where(logprob > max_logprob, logprob, max_logprob)
    return best_c

