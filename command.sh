rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --scale overfitting --temp_Gram_optimum --attention none

rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale overfitting

# parser.add_argument('--datapath', type=str, default='../../../../hdd/junhong/data/bbad_v2')