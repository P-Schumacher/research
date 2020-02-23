#export LD_PRELOAD=libGL.so:libGLEW.so
#export CUDA_VISIBLE_DEVICES=-1
python3 my_main.py \
	--policy "TD3" \
	--env "AntMaze" \
	--start_timesteps 2500 \
	--max_timesteps 1900000 \
	--eval_freq 50000 \
	--time_limit 500 \
	--batch_size 128 \
	--goal_type Direction \
	--meta_noise 1 \
	--sub_noise 1  \
    --policy_freq 1 \
    --goal_every_n 10 \
	--zero_obs  \
	--offpolicy 
