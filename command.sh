:<<end

## TEST
rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_DET_CM --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_DET_CMbyCMpp --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
end


## SMALL TEST
rm -rf checkpoint/DET_T9_S_AVNOON_NCD_NPM_NODOCC_OG_4/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVNOON_NCD_NPM_NODOCC_OG_4 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_NCD_NPM_NODOCC_OG_4/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath DET_T9_S_NCD_NPM_NODOCC_OG_4 --scale small --multiplicity 1 --epochs 0 --visualize --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_single_4/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_single_4 --scale small --multiplicity 1 --epochs 0 --visualize --scheduler_mode none --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &


# Sinkhorn
# Base: DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4

# Delete Sinkhorn, Delete Sinkhorn and remove exp from registration, Cosine similarity matching score mode, Delete Sinkhorn and Cosine similarity matching score mode
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --delete_Sinkhorn --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS_svdNE/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS_svdNE --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --delete_Sinkhorn --svd_no_exp --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --matching_score_mode cos --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS_DS/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS_DS --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --delete_Sinkhorn --matching_score_mode cos --n_worker 4 --wandb --wandb_project CMpp &


# Delete Sinkhorn, Delete Sinkhorn and remove exp from registration, Delete Sinkhorn and Cosine similarity matching score mode + Use Sinkhorn for inference
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DSUSI/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DSUSI --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --use_Sinkhorn_infer --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS_DSUSI/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS_DSUSI --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --use_Sinkhorn_infer --matching_score_mode cos --n_worker 4 --wandb --wandb_project CMpp &


# CM original
rm -rf checkpoint/DET_T9_S_CM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CM_equiassem --logpath DET_T9_S_CM --scale small --multiplicity 1 --epochs 0 --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T9_S_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath DET_T9_S_CMbyCMpp --scale small --multiplicity 1 --epochs 0 --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &




:<<end
rm -rf checkpoint/TEST_DET_T9_S_AVNOON_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVNOON_NCD_NPM_NODOCC_OG_4/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVNOON_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --only_one_norm --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_NCD_NPM_NODOCC_OG_4/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --n_avn 0 --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_single_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_single_4/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_single_6 --scale small --multiplicity 1 --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test


rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_DS/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_DS --scale small --multiplicity 1 --use_opt_gram --deterministic --delete_Sinkhorn --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_DS_svdNE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_DS_svdNE/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_DS_svdNE --scale small --multiplicity 1 --use_opt_gram --deterministic --delete_Sinkhorn --svd_no_exp  --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_MCOS/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_MCOS --scale small --multiplicity 1 --use_opt_gram --deterministic --matching_score_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_MCOS_DS/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T9_S_AVN_NCD_NPM_NODOCC_OG_4_MCOS_DS/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T9_S_AVN_NCD_NPM_NODOCC_OG_6_MCOS_DS --scale small --multiplicity 1 --use_opt_gram --deterministic --delete_Sinkhorn --matching_score_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
end



:<<end
rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath DEBUG --scale overfitting --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4
rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint/DET_T8_S_AVN_NCD_NPM_NODOCC_OG_4_DS/models/last.ckpt --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --use_opt_gram --deterministic --delete_Sinkhorn --gpus 0 --n_worker 6 --move_smaller
rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint/DET_T8_S_AVN_NCD_NPM_NODOCC_OG_4/models/last.ckpt --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --use_opt_gram --deterministic --gpus 0 --n_worker 6 --move_smaller
end




