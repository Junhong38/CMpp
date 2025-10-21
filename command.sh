# FINAL EXPERIMENTS
rm -rf checkpoint/T3_S_CMorigin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath T3_S_CMorigin --scale small --multiplicity 1 --epochs 0 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_CMoriginByCM/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T3_S_CMoriginByCM --scale small --multiplicity 1 --epochs 0 --visualize --scheduler_mode CM --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_CM/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T3_S_CM --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVN/ && (CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T3_S_AVN --scale small --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp &> log_T3_S_AVN.txt)
rm -rf checkpoint/T3_S_AVNOON/ && (CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON --scale small --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --only_one_norm --wandb --wandb_project CMpp &> log_T3_S_AVNOON.txt)
rm -rf checkpoint/T3_S_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --delete_occupancy_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_EXP --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_EXP_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --delete_occupancy_loss --wandb --wandb_project CMpp



# EXP experiments
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_EXP_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --use_opt_gram --wandb --wandb_project CMpp



# VNLinearLeakyReLU experiments
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
rm -rf checkpoint/T3_S_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVN_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --wandb --wandb_project CMpp



# Circle loss experiments: No Detach, Same Opt, No Balance, CORR NO BALANCE, CORR MAX NO BALANCE
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
rm -rf checkpoint/T3_S_AVNOON_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NC_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDSO_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDSO_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --same_opt --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDNOBAL_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDNOBAL_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --no_balance --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDCORR_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDCORR_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --only_corr --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDCORRNOBAL_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDCORRNOBAL_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --only_corr --no_balance --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDCORRMAXNOBAL_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDCORRMAXNOBAL_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --only_corr --max_points --no_balance --use_opt_gram --wandb --wandb_project CMpp



# Scheduler experiments
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC_OG_SINGLE/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_NODOCC_OG_SINGLE --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --scheduler_mode none --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCD_NPM_NODOCC_OG_ONYCYCLE/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCD_NPM_NODOCC_OG_ONYCYCLE --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --scheduler_mode onecycle --wandb --wandb_project CMpp



# Log_scale experiments
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
# 24 48 60 96
rm -rf checkpoint/T3_S_AVNOON_NCDLOG36_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDLOG36_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --log_scale 36 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDLOG48_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDLOG48_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --log_scale 48 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDLOG60_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDLOG60_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --log_scale 60 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVNOON_NCDLOG96_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVNOON_NCDLOG96_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --log_scale 96 --wandb --wandb_project CMpp



# knn experiments
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
# 10 20
rm -rf checkpoint/T3_S_K10_AVNOON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_K10_AVNOON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --n_knn 10 --wandb --wandb_project CMpp



# n_avn experiments
# T3_S_AVNOON_NCD_NPM_NODOCC_OG
# 1 2 4 5
rm -rf checkpoint/T3_S_AVN1OON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVN1OON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --n_avn 1 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVN2OON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVN2OON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --n_avn 2 --wandb --wandb_project CMpp
rm -rf checkpoint/T3_S_AVN4OON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T3_S_AVN4OON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --n_avn 4 --wandb --wandb_project CMpp






rm -rf checkpoint/DEBUG_3/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath DEBUG_3 --scale overfitting --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --viz_epoch 100