:<<end

## TEST
rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_DET_CM --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_DET_CMbyCMpp --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
end

:<<end
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
end



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


rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath DEBUG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 --n_worker 6 --move_smaller


rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_DET_CM --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_DET_CMbyCMpp --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test




# CM original
rm -rf checkpoint/DET_T10_S_CM/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CM_equiassem --logpath DET_T10_S_CM --scale small --multiplicity 1 --epochs 0 --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T10_S_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath DET_T10_S_CMbyCMpp --scale small --multiplicity 1 --epochs 0 --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &


rm -rf checkpoint/DET_T10_S_NCD_NPM_NODOCC_OG_4/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DET_T10_S_NCD_NPM_NODOCC_OG_4 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T10_S_NCD_NPM_NODOCC_OG_FN_4/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath DET_T10_S_NCD_NPM_NODOCC_OG_FN_4 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T10_S_NCD_NPM_NODOCC_OG_UC_4/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath DET_T10_S_NCD_NPM_NODOCC_OG_UC_4 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --use_consistency_loss --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &
rm -rf checkpoint/DET_T10_S_NCD_NPM_NODOCC_OG_FN_UC_4/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath DET_T10_S_NCD_NPM_NODOCC_OG_FN_UC_4 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp &



rm -rf checkpoint/DET_T10_F_NCD_NPM_NODOCC_OG_4/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath DET_T10_F_NCD_NPM_NODOCC_OG_4 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 4 --wandb --wandb_project CMpp &


rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_entity CMppProject --wandb_project CMpp 

rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_project CMpp 


rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale overfitting --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4
end




:<<end
rm -rf checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_MT1_F_NCD_NPM_NODOCC_OG_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_FN_UC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_MT1_F_NCD_NPM_NODOCC_OG_FN_UC_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/NDET_MT1_F_AVN_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_MT1_F_AVN_NCD_NPM_NODOCC_OG_6 --scale full --multiplicity 1 --epochs 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_FN_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_MT1_F_NCD_NPM_NODOCC_OG_FN_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp

# Check deterministic
# rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --deterministic --gpus 0 --n_worker 4 --wandb --wandb_entity CMppProject --wandb_project CMpp 
end



:<<end
rm -rf checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_MT1_F_NCD_NPM_NODOCC_OG_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp

rm -rf checkpoint/TOP128_NDET_MT1_F_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath TOP128_NDET_MT1_F_NCD_NPM_NODOCC_OG_6 --scale full --multiplicity 1 --n_avn 0 --use_opt_gram --n_worker 6 --move_smaller --gpus 0 1 2 3 4 5 6 7 --wandb --wandb_entity CMppProject --wandb_project CMpp_test
rm -rf checkpoint/TOP128FN_NDET_MT1_F_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath TOP128FN_NDET_MT1_F_NCD_NPM_NODOCC_OG_6 --scale full --multiplicity 1 --n_avn 0 --use_opt_gram --flip_normal --n_worker 6 --move_smaller --gpus 0 1 2 3 4 5 6 7 --wandb --wandb_entity CMppProject --wandb_project CMpp_test




rm -rf checkpoint/DEBUG_TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath DEBUG_TEST --scale full --multiplicity 1 --n_avn 0 --use_opt_gram --n_worker 6 --move_smaller --infer_match_option soft_topk --infer_topk 5 --infer_score_threshold_ratio 0.01 --gpus 0 1 2 3 4 5 6 7 --wandb --wandb_entity CMppProject --wandb_project CMpp_test

rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_CM --scale full --multiplicity 1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test
rm -rf checkpoint/TEST_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_CMbyCMpp --scale full --multiplicity 1 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test



rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale overfitting --multiplicity 1 --epochs 1 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 --n_worker 6
rm -rf checkpoint/DEBUG_2nd/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_2nd --scale overfitting --multiplicity 1 --epochs 1 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 --n_worker 6 --load checkpoint/DEBUG/models/last.ckpt





# rm -rf checkpoint/DEBUG_prev/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath DEBUG_prev --scale small --multiplicity 1 --epochs 5 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp 
rm -rf checkpoint/NDET_S1_MT1_S_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath NDET_S1_MT1_S_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_project CMpp
# rm -rf checkpoint/DEBUG_2nd/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/DEBUG/models/last.ckpt --model CMpp_equiassem --logpath DEBUG_2nd --scale small --multiplicity 1 --epochs 1 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/DEBUG_2nd_FN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/DEBUG/models/last.ckpt --model CMpp_equiassem --logpath DEBUG_2nd_FN --scale small --multiplicity 1 --epochs 1 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


rm -rf checkpoint/NDET_S2_MT1_S_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/NDET_S1_MT1_S_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath NDET_S2_MT1_S_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --scheduler_mode onecycle --wandb --wandb_project CMpp
rm -rf checkpoint/NDET_RS2_MT1_S_NCD_NPM_NODOCC_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --resume checkpoint/NDET_S1_MT1_S_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath NDET_RS2_MT1_S_NCD_NPM_NODOCC_OG_6 --scale small --multiplicity 1 --epochs 180 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp




rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --resume checkpoint/NDET_S1_MT1_S_NCD_NPM_NODOCC_OG_6/models/last.ckpt --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --epochs 180 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 --n_worker 6
end


:<<end
# rm -rf checkpoint/G8ND_S_AVN0_OG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_S_AVN0_OG_FN_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FN_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_S_AVN0_OG_UC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_UC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_S_AVN0_OG_FNUC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FNUC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


rm -rf checkpoint/G8ND_S_AVN0_OG_S1_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S1_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_S2_FN_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S2_FN_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_S2_FN_onecycle_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S2_FN_onecycle_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --scheduler_mode onecycle --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_S2_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S2_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_S2_onecycle_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S2_onecycle_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --scheduler_mode onecycle --wandb --wandb_project CMpp


rm -rf checkpoint/G8ND_S_AVN0_OG_S2_FNUC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S2_FNUC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp




rm -rf checkpoint/DEBUG_G8ND_F_AVN0_OG_S1_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath DEBUG_G8ND_F_AVN0_OG_S1_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 --n_worker 6 --only_train_normal

rm -rf checkpoint/DEBUG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath DEBUG --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --freeze_normal_param --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp



rm -rf checkpoint/G8ND_S_AVN0_OG_S1UC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_S1UC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --use_consistency_loss --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FREEZE_S_AVN0_OG_UCS2_FN_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FREEZE_S_AVN0_OG_UCS2_FN_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --freeze_normal_param --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp

rm -rf checkpoint/G8ND_S_AVN0_OG_UCS2_FNUC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S_AVN0_OG_S1UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_UCS2_FNUC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
end





# Stage 1
# rm -rf checkpoint/G8ND_F_AVN0_OG_S1_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_F_AVN0_OG_S1_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_entity CMppProject --wandb_project CMpp


# Stage 2

# KIM
# rm -rf checkpoint/G8ND_F_AVN0_OG_S2_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_F_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_F_AVN0_OG_S2_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_F_AVN0_OG_S2_FN_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_F_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_F_AVN0_OG_S2_FN_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp

# LEE
# rm -rf checkpoint/G8ND_F_AVN0_OG_S2_UC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_F_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_F_AVN0_OG_S2_UC_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_F_AVN0_OG_S2_FNUC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_F_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_F_AVN0_OG_S2_FNUC_cos_6 --scale full --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp



:<<end

# Single Stage
# rm -rf checkpoint/G8ND_FIX1_S_AVN0_OG_6_SEV17/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX1_S_AVN0_OG_6_SEV17 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_FIX1_S_AVN0_OG_6_SEV23/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX1_S_AVN0_OG_6_SEV23 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_FN_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_FN_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_UC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_UC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_FNUC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_FNUC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


# Stage 1
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S1_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S1_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_project CMpp
# rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --use_consistency_loss --wandb --wandb_project CMpp


# Stage 2, Normal
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S2_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S2_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S2_FN_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S2_FN_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S2_UC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S2_UC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_S2_FNUC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_S2_FNUC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


# Stage 2, Normal + Consistency
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_UCS2_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_UCS2_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_UCS2_FN_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_UCS2_FN_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_UCS2_UC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_UCS2_UC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_FIX2_S_AVN0_OG_UCS2_FNUC_cos_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_FIX2_S_AVN0_OG_S1_UC_cos_6/models/last.ckpt --model CMpp_equiassem --logpath G8ND_FIX2_S_AVN0_OG_UCS2_FNUC_cos_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp




# rm -rf checkpoint/DEBUG_OVERFIT/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_OVERFIT --scale overfitting --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 --n_worker 6
rm -rf checkpoint/DEBUG_OVERFIT_FLIP1/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_OVERFIT_FLIP1 --scale overfitting --multiplicity 100 --visualize --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/DEBUG_OVERFIT_FLIP2/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_OVERFIT_FLIP2 --scale overfitting --multiplicity 100 --visualize --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 --n_worker 6 --wandb --wandb_project CMpp



rm -rf checkpoint/DEBUG_OVERFIT_FLIP_AVN/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_OVERFIT_FLIP_AVN --scale overfitting --multiplicity 100 --visualize --epochs 0 --detach_mode --use_opt_gram --flip_normal --gpus 0 --n_worker 6 --wandb --wandb_project CMpp



rm -rf checkpoint/G8ND_S_AVN0_OG_6_DB_FN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_DB_FN --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --double_backbone --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 
rm -rf checkpoint/G8ND_S_AVN0_OG_6_DB_FNUC/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_DB_FNUC --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --double_backbone --flip_normal --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 
rm -rf checkpoint/G8ND_S_AVN0_OG_6_DB_UC/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_DB_UC --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --double_backbone --use_consistency_loss --gpus 0 1 2 3 4 5 6 7 --n_worker 6 
rm -rf checkpoint/G8ND_S_AVN0_OG_6_DB/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_DB --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --double_backbone --gpus 0 1 2 3 4 5 6 7 --n_worker 6 




rm -rf checkpoint/DEBUG_OVERFIT_FLIP/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath DEBUG_OVERFIT_FLIP --scale overfitting --multiplicity 100 --epochs 0 --n_avn 0 --gpus 0 --n_worker 6 --flip_normal --wandb --wandb_project CMpp

rm -rf checkpoint/G8ND_FIX1_S_AVN0_OG_6_SEV23_V2/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_FIX1_S_AVN0_OG_6_SEV23_V2 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


rm -rf checkpoint/G8ND_S_AVN0_OG_FNOCC_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FNOCC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp


rm -rf checkpoint/G8ND_S_AVN0_OG_6_V3/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --gpus 0 --n_worker 6
rm -rf checkpoint/G8ND_S_AVN0_OG_FN_6_V3/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --flip_normal --gpus 0 --n_worker 6
rm -rf checkpoint/G8ND_S_AVN0_OG_FNOCC_6/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FNOCC_6 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --occ_mode --gpus 0 --n_worker 6
end


rm -rf checkpoint/G8ND_S_AVN0_OG_6_V3/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_FN_6_V3/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FN_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --flip_normal --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
rm -rf checkpoint/G8ND_S_AVN0_OG_FNOCC_6_V3/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_FNOCC_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --occ_mode --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp

rm -rf checkpoint/G8ND_S_AVN0_OG_OCC_6_V3/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --model CMpp_equiassem --logpath G8ND_S_AVN0_OG_OCC_6_V3 --scale small --multiplicity 1 --epochs 0 --n_avn 0 --occ_mode --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_project CMpp
