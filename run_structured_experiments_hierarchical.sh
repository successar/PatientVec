for yr in 1 2 3 0.25 0.5
do
    python "Discovery Experiments"/run_models.py --dataset $task"_"$yr"yr" --data_dir=. --output_dir='outputs/' \
    --exp_types hierarchical --display --bsize 8 --n_iters=20;
    python "Discovery Experiments"/run_models.py --dataset $task"_"$yr"yr" --data_dir=. --output_dir='outputs/' \
    --exp_types hierarchical --structured --display --bsize 8 --n_iters=20;
done
