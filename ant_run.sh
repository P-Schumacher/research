#export LD_PRELOAD=libGL.so:libGLEW.so
#export CUDA_VISIBLE_DEVICES=-1
python3 my_main.py \
	--policy "TD3" \
	--env "AntMaze" \
	--eval_freq 50000 \
	--time_limit 500 \
	--batch_size 128 \
	--goal_type Direction \
	--meta_noise 1 \
	--sub_noise 1  \
	--max_timesteps 5000000 \
    	--policy_freq 1 \
    	--c_step 10 \
	--zero_obs 2  \
	--offpolicy \
	--start_timesteps 2500 
