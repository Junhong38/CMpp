
:<<end
rm -rf checkpoint/TEST_shonan_2t2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_2t2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly shonan
rm -rf checkpoint/TEST_shonan_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly shonan
rm -rf checkpoint/TEST_shonan_RANSAC_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_RANSAC_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly shonan --use_RANSAC --use_predicted_normal
end



rm -rf checkpoint/TOP500_score_RANSAC_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TOP500_score_RANSAC_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly naive --infer_topk 500 --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --use_seg_result
rm -rf checkpoint/TOP500_score_RANSAC_pen_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TOP500_score_RANSAC_pen_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly naive --infer_topk 500 --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --use_seg_result --use_penetration



rm -rf checkpoint/TEST_default_2t2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_default_2t2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly none
rm -rf checkpoint/TEST_default_RANSAC_2t2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_default_RANSAC_2t2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly none --use_RANSAC --use_predicted_normal


rm -rf checkpoint/TEST_obo_2t2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_obo_2t2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly naive
rm -rf checkpoint/TEST_obo_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_obo_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly naive
rm -rf checkpoint/TEST_obo_RANSAC_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_obo_RANSAC_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly naive --use_RANSAC --use_predicted_normal



