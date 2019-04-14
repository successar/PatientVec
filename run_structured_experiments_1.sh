python "Discovery Experiments"/run_models.py --dataset $task"_"$yr"yr" --data_dir=. --output_dir='outputs/' \
--exp_types vanilla attention hierarchical --display --bsize 8 --n_iters=20;
python "Discovery Experiments"/run_models.py --dataset $task"_"$yr"yr" --data_dir=. --output_dir='outputs/' \
--exp_types vanilla attention hierarchical --structured --display --bsize 8 --n_iters=20;