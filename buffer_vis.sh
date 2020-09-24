# Loads up saved experience replay transitions, trains the meta agent on them for 1000 iterations and visualizes the sampled transitions.
# Good for visualizing different prioritization methods
python3 constant_buffer.py
mv -t per_exp/buffer_data m1.npy m2.npy errors.npy
cd per_exp 
python3 visualize_buffer.py
cd ..
