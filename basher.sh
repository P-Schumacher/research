#for i in 1 2 3 4 5 
#do
#	python3 main.py config8 $i &
#done
TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" time python main.py config8
