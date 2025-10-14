rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --scale overfitting --temp_Gram_optimum --attention none

rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale overfitting

rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale overfitting --use_opt_gram --visualize

# parser.add_argument('--datapath', type=str, default='../../../../hdd/junhong/data/bbad_v2')


rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale overfitting --use_opt_gram --visualize

rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale overfitting --use_opt_gram --visualize --multiplicity 100


rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --epochs 10 --scale overfitting --visualize --multiplicity 100 
rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --epochs 10 --scale overfitting --use_opt_gram --visualize --multiplicity 100 



rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale tiny --visualize --multiplicity 33 
rm -rf checkpoint/default_logpath/ && CUDA_VISIBLE_DEVICES=0 python main.py --scale tiny --visualize --multiplicity 33 --debugged_circle_loss