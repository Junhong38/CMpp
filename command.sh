:<<end

## TEST
rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_DET_CM --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_DET_CMbyCMpp --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test


rm -rf checkpoint/TEST_DEBUG/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DEBUG --scale full --multiplicity 1 --deterministic --gpus 0 --n_worker 6 --move_smaller
rm -rf checkpoint/TEST_DEBUG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DEBUG --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller
end


:<<end
rm -rf checkpoint/G8ND_S1_F_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8ND_S1_F_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/G8ND_S2_F_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S1_F_FV0DV0MH_6/models/last.ckpt --logpath G8ND_S2_F_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_S2_S_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath G8ND_S2_S_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale small --multiplicity 1 --epochs 1 --gpus 0 --n_worker 6

rm -rf checkpoint/G8ND_S2_F_FV1DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load_ori checkpoint/G8ND_S1_F_FV0DV0MH_6/models/last.ckpt --logpath G8ND_S2_F_FV1DV0MH_6 --model CMpp_equiassem --backbone vn_unet_deep --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_S2_S_FV1DV0MH_6/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath G8ND_S2_S_FV1DV0MH_6 --model CMpp_equiassem --backbone vn_unet_deep --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale small --multiplicity 1 --epochs 1 --gpus 0 --n_worker 6



rm -rf checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6_S_HARD/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/NDET_MT1_F_NCD_NPM_NODOCC_OG_6/models/last.ckpt --logpath NDET_MT1_F_NCD_NPM_NODOCC_OG_6_S_HARD --model CMpp_equiassem --backbone vn_unet --n_avn 0 --mlp_mode CMpp --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8ND_F_FV0DV0MH_HARD_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8ND_F_FV0DV0MH_HARD_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --wandb --wandb_entity CMppProject --wandb_project CMpp
end


:<<end
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 1 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 --n_worker 1 --hard_negative --batch_size 1
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1 python main.py --logpath IMPLE --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 1 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 --n_worker 1 --hard_negative --batch_size 1


rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 1 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 2 --wandb --wandb_entity CMppProject --wandb_project CMpp


# rm -rf checkpoint/IMPLE_TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/IMPLE/models/last.ckpt --logpath IMPLE_TEST --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale small --gpus 0 1 2 3 4 5 6 7 --n_worker 6


rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative mix --distance_type l2 --batch_size 1 --no_matching_loss
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative mix --distance_type cossim --batch_size 1 --no_matching_loss


rm -rf checkpoint/G8NDB1_F_BV0V0_MH_HN_SINK_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB1_F_BV0V0_MH_HN_SINK_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 1 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB2_F_BV0V0_MH_HN_SIG_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_HN_SIG_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 2 --matching_norm_mode sigmoid --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB2_F_BV0V0_MH_HN_SOF_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_HN_SOF_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 2 --matching_norm_mode softmax --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB2_F_BV0V0_MH_HN_SIG_NOSLACK_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_HN_SIG_NOSLACK_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 2 --matching_norm_mode sigmoid --no_slack_variable --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB2_F_BV0V0_MH_HN_SOF_NOSLACK_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_HN_SOF_NOSLACK_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 2 --matching_norm_mode softmax --no_slack_variable --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/G8NDB1_F_BV0V0_MH_HN_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1 python main.py --logpath G8NDB1_F_BV0V0_MH_HN_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1--n_worker 6 --hard_negative --batch_size 1 --no_matching_loss # --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_HN_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_HN_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative --batch_size 3 --no_matching_loss --wandb --wandb_entity CMppProject --wandb_project CMpp





rm -rf checkpoint/G8NDB3_F_BV0V0_MH_BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --balance_mode none --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode all --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N1P005_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N1P005_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.0 --balance_mode none --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode all --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.0 --pos_offset 0.00125 --neg_offset 0.45 --balance_mode none --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode all --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AD+N145P005_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AD+N145P005_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.0 --pos_offset 0.00125 --neg_offset 0.45 --balance_mode none --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp




# TEST
rm -rf checkpoint/TEST_IMPLE/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6/models/last.ckpt --logpath TEST_IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --move_smaller

"""
G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6 (G8NDB3_F_BV0V0_MH_BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6)
NAN_G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA_MCOS+MNN+NS_S1P0_6
G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N1P005_MCOS+MNN+NS_S1P0_6
G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6
G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AD+N145P005_MCOS+MNN+NS_S1P0_6
"""
rm -rf checkpoint/TEST_G8NDB3_F_BV0V0_MH_BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6/models/last.ckpt --logpath TEST_G8NDB3_F_BV0V0_MH_BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --move_smaller
rm -rf checkpoint/TEST_G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6/models/last.ckpt --logpath TEST_G8NDB3_F_BV0V0_MH_BN+HNM+NK1000+DCOS+AA+N145P005_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --move_smaller




# Origin
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/TEST_G8NDB3_F_BV0V0_MH_P010M005+N140M145+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6/models/last.ckpt --logpath TEST_G8NDB3_F_BV0V0_MH_P010M005+N140M145+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test


# Change implementation
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.10 --neg_margin 1.40 --pos_offset 0.05 --neg_offset 0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/TEST_G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/models/last.ckpt --logpath TEST_G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test


# No LR scheduler
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_NLRS_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_NLRS_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --scheduler_mode none --batch_size 3 --pos_margin 0.10 --neg_margin 1.40 --pos_offset 0.05 --neg_offset 0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/TEST_G8NDB3_F_BV0V0_MH_NLRS_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/G8NDB3_F_BV0V0_MH_NLRS_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6/models/last.ckpt --logpath TEST_G8NDB3_F_BV0V0_MH_NLRS_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNN+NS_S1P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --matching_score_mode cossim --matching_norm_mode none --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test


# Matching loss + Softmax / Matching loss + Softmax + CM-type Score
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCOS+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.10 --neg_margin 1.40 --pos_offset 0.05 --neg_offset 0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode cossim --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.10 --neg_margin 1.40 --pos_offset 0.05 --neg_offset 0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
end


# [DONE] Origin + Softmax + Distance: cossim + PO0.05PM0.10/NO1.45NM1.40 no offset + Hard Negative:mix + No balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M010+N145M140+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.10 --neg_margin 1.40 --pos_offset 0.05 --neg_offset 0.05 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Sinkhorn
# rm -rf checkpoint/G8NDB1_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSink_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB1_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSink_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 1 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode sinkhorn --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Sinkhorn + Hard Negative:mix
# rm -rf checkpoint/G8NDB1_F_BV0V0_MH_P010M005+N140M145+BH+HNM+NK0+DL2+AD_MCM+MNSink_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB1_F_BV0V0_MH_P010M005+N140M145+BH+HNM+NK0+DL2+AD_MCM+MNSink_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 1 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative mix --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode sinkhorn --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax + Distance: cossim
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (learnable temperature)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --learnable_softmax_temperature --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (learnable temperature) + P0.005/N1.45 no offset
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M005+N145M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M005+N145M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --learnable_softmax_temperature --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE(NAN -> AGAIN)]Origin + Softmax (learnable temperature) + Distance: cossim
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoftLT_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoftLT_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --learnable_softmax_temperature --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp



# [DONE] Origin + Softmax (learnable temperature, 2.0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT20_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DL2+AD_MCM+MNSoftLT20_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type l2 --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --learnable_softmax_temperature --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE]Origin + Softmax (learnable temperature, 2.0) + Distance: cossim
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoftLT20_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoftLT20_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --learnable_softmax_temperature --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp






# [RE, OK] Origin + Softmax (temperature, 1.0) + Distance: cossim
# rm -rf checkpoint/RE_G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath RE_G8NDB3_F_BV0V0_MH_P010M005+N140M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset -0.05 --neg_offset -0.05 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.1, offset 0) / negative(1.40, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M010+N140M140+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M010+N140M140+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.1 --neg_margin 1.4 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.005, offset 0) / negative(1.0, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N10M10+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N10M10+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.0 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.05, offset 0) / negative(1.45, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P005M005+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P005M005+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.05 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.10, offset 0) / negative(1.45, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P010M010+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P010M010+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.10 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK1000 with no_balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative none --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp



# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.085, offset 0) / negative(1.45, offset 0)
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P008M008+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P008M008+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.08 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.090, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK1000 with no_balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P009M009+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P009M009+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.09 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK2000 with no_balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK2000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK2000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative none --neg_topk 2000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK4000 with no_balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK4000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK4000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative none --neg_topk 4000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:MIX with all hard
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BOH+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BOH+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode only_hard --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK500 with no_balance
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK500+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BN+HNN+NK500+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative none --neg_topk 500 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK1000 with all_hard
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BAH+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BAH+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode all_hard --hard_negative none --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK2000 with all_hard
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BAH+HNN+NK2000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BAH+HNN+NK2000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode all_hard --hard_negative none --neg_topk 2000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [STOP] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:TOPK1000 with double
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNN+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative none --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:mix with double
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:mix with double + FILP_NORMAL
rm -rf checkpoint/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_FN_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_FN_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --flip_normal --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [DONE] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:mix with double + FILP_NORMAL(+ consistency loss)
rm -rf checkpoint/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_FNC_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_FNC_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --flip_normal --consistency_loss --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [ING] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:mix with double + PRED_NORMAL_MODE:cross + FILP_NORMAL(MIX)
rm -rf checkpoint/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPGFNM_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPGFNM_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --normal_pred_mode gram --flip_normal_mode mix --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp

# [ING] Origin + Softmax (temperature, 1.0) + Distance: cossim + positive(0.075, offset 0) / negative(1.45, offset 0) + Hard Negative:mix with double + PRED_NORMAL_MODE:cross + FILP_NORMAL(MIX)(+ consistency loss)
rm -rf checkpoint/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPGFNMC_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPGFNMC_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --pos_margin 0.075 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode double --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --normal_pred_mode gram --flip_normal_mode mix --consistency_loss --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --viz_train_epoch 30 --wandb --wandb_entity CMppProject --wandb_project CMpp



# Origin + Softmax + Distance: cossim + P0.005/N1.45 no offset
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNN+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative none --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# Origin + Softmax + Distance: cossim + P0.005/N1.45 no offset + Hard Negative:mix
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# Origin + Softmax + Distance: cossim + P0.005/N1.45 no offset + Hard Negative:mix&topk
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNM+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BH+HNM+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode half --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# [DONE] Origin + Softmax + Distance: cossim + P0.005/N1.45 no offset + Hard Negative:mix + No balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N105M105+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N105M105+BN+HNM+NK0+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.05 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative mix --neg_topk 0 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# Origin + Softmax + Distance: cossim + P0.005/N1.45 no offset + Hard Negative:mix&topk + No balance
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BN+HNM+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_P0005M0005+N145M145+BN+HNM+NK1000+DCOS+AD_MCM+MNSoft_S1P1_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --pos_margin 0.005 --neg_margin 1.45 --pos_offset 0.0 --neg_offset 0.0 --balance_mode none --hard_negative mix --neg_topk 1000 --distance_type cossim --anchor_mode default --matching_score_mode CM --matching_norm_mode softmax --p_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# Origin + Softmax + only matching loss (no circle loss)





