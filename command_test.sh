

:<<end
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller

rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller


rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller

rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller



rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode none --move_smaller





rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller



rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller --min_part 3 --max_part 3




rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller




rm -rf checkpoint/TEST_multi/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST_multi --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller


rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller --min_part 3 --max_part 3 --multi_part_assembly
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller --min_part 3 --max_part 3 --use_RANSAC --use_predicted_normal

rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB2_F1_2/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv1_2 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly



rm -rf checkpoint/TA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --wandb --wandb_entity CMppProject --wandb_project CMpp_test &> TA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1.log
rm -rf checkpoint/MA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath MA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly naive --wandb --wandb_entity CMppProject --wandb_project CMpp_test &> MA2t2_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1.log


rm -rf checkpoint/MA2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath MA2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly naive --wandb --wandb_entity CMppProject --wandb_project CMpp_test &> MA2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1.log
rm -rf checkpoint/MARANSAC2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath MARANSAC2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --use_RANSAC --use_predicted_normal --multi_part_assembly naive --wandb --wandb_entity CMppProject --wandb_project CMpp_test &> MARANSAC2t20_LOAD_THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1.log




rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 5000 --r_knn 0.00
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --scale overfitting --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 5000 --r_knn 0.00





# rm -rf checkpoint/TEST_r000/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_r000 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 5000 --r_knn 0.00
# rm -rf checkpoint/TEST_r005/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_r005 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 5000 --r_knn 0.05


# rm -rf checkpoint/TEST_n10000_r005/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n10000_r005 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 10000 --r_knn 0.05
# rm -rf checkpoint/TEST_n20000_r005/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n20000_r005 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 20000 --r_knn 0.05

# rm -rf checkpoint/TEST_n10000_r000/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n10000_r000 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 10000 --r_knn 0.00
# rm -rf checkpoint/TEST_n20000_r000/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n20000_r000 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 20000 --r_knn 0.00



# rm -rf checkpoint/TEST_m1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_m1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 5000 --m_knn 1 --r_knn 0.00
# rm -rf checkpoint/TEST_n10000_m2V2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n10000_m2V2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 10000 --m_knn 2 --r_knn 0.00
# rm -rf checkpoint/TEST_n20000_m4V2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_n20000_m4V2 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --n_pts 20000 --m_knn 4 --r_knn 0.00




rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly shonan --visualize_mode light
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 3 --max_part 3 --multi_part_assembly shonan --visualize_mode light --use_RANSAC




rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST --gpus 0 --n_worker 1 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly shonan




rm -rf checkpoint/TEST_default/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_default --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly none
end



rm -rf checkpoint/TEST_shonan_new/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_new --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 2 --multi_part_assembly shonan
rm -rf checkpoint/TEST_shonan_new_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_new_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly shonan
rm -rf checkpoint/TEST_shonan_new_RANSAC_2t20/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1/models/last.ckpt --logpath TEST_shonan_new_RANSAC_2t20 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode rightv4 --seg_head_mode mlp --move_smaller --min_part 2 --max_part 20 --multi_part_assembly shonan --use_RANSAC --use_predicted_normal




