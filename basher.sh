for i in 1 2 3 4 
do
	python main.py config7 $i &
done
#TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" python main.py config8
