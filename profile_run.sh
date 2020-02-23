#export LD_PRELOAD=libGL.so:libGLEW.so
#export CUDA_VISIBLE_DEVICES=-1
python3 -m cProfile -s cumulative my_main.py \
	--policy "TD3" \
	--env "Vrep" \
	--eval_freq 20000 \
	--time_limit 1000 \
	--batch_size 128 \
	--goal_type Absolute\
	--meta_noise 0.10 \
	--sub_noise 50  \
	--max_timesteps 3000 \
    --policy_freq 2 \
    --c_step 10 \
	--zero_obs 0  \
    	--vrep \
	--force \
	--offpolicy \
	--start_timesteps 0 
