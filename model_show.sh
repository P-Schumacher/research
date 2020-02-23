export CUDA_VISIBLE_DEVICES=-1
export LD_PRELOAD=libGL.so:libGLEW.so
python show_model.py \
	--policy "TD3" \
	--env "Vrep" \
	--start_timesteps 2500 \
	--max_timesteps 1000000 \
	--eval_freq 10000 \
	--time_limit 10300 \
	--seed 25 \
	--batch_size 100\
	--c_step 50 \
	--vrep \
	--force \
	--render \
	--goal_type Absolute 
