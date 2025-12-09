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
end



# rm -rf checkpoint/G8NDB1_F_BV0V0_MH_DL2HNMIX_SINK_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB1_F_BV0V0_MH_DL2HNMIX_SINK_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type l2 --batch_size 1 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_DL2HNMIX_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_DL2HNMIX_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type l2 --batch_size 3 --no_matching_loss --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/G8NDB3_F_BV0V0_MH_DCOSHNMIX_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_DCOSHNMIX_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type cossim --batch_size 3 --no_matching_loss --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB3_F_BV0V0_MH_DCOSHNTOPK_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_DCOSHNTOPK_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative topk --distance_type cossim --batch_size 3 --no_matching_loss --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/IMPLE_G8NDB3_F_BV0V0_MH_DCOSHNMIX_NOMATCH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE_G8NDB3_F_BV0V0_MH_DCOSHNMIX_NOMATCH_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative mix --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --s_loss_weight 0.0

rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative mix --distance_type cossim --batch_size 1 --matching_score_mode cossim






rm -rf checkpoint/G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_DCOSNOBAL_P0_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --no_balance --wandb --wandb_entity CMppProject --wandb_project CMpp

rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative none --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --s_loss_weight 0.0 --no_balance --wandb --wandb_entity CMppProject --wandb_project CMpp

rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative none --distance_type cossim --batch_size 1 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --p_loss_weight 0.0 --no_balance





rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative none --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --no_balance --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 --n_worker 1 --hard_negative none --distance_type cossim --batch_size 1 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable  --no_balance
rm -rf checkpoint/IMPLE/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath IMPLE --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --no_balance --wandb --wandb_entity CMppProject --wandb_project CMpp

rm -rf checkpoint/G8NDB3_F_BV0V0_MH_DCOSNOBAL_P1HN_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8NDB3_F_BV0V0_MH_DCOSNOBAL_P1HN_6 --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --hard_negative mix --distance_type cossim --batch_size 3 --matching_score_mode cossim --matching_norm_mode none --no_slack_variable --no_balance --wandb --wandb_entity CMppProject --wandb_project CMpp

