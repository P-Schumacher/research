for i in 1 2 3 
do
	python3 main.py --name=config8 --alpha=0.5 --beta=0.5 --seed=$i  &
done
