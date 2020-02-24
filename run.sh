#export LD_PRELOAD=libGL.so:libGLEW.so
export CUDA_VISIBLE_DEVICES=-1
python3 training_loop.py \
	--policy "TD3" \
	--env "Vrep" \
	--eval_freq 30000 \
	--time_limit 300 \
	--batch_size 128 \
	--goal_type Absolute\
	--meta_noise 0.5 \
	--sub_noise 50  \
	--max_timesteps 100000000 \
    	--policy_freq 2 \
    	--c_step 10 \
	--zero_obs 0  \
	--seed 2 \
	--offpolicy \
	--start_timesteps 1000 \
	--vrep \
	--force \
	--ee_pos \
	--random_target \
	--subgoal_ee_range 1 \
	--sparse_rew \
	--log \
	--ri_re
	
