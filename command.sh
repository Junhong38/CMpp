

rm -rf checkpoint/T1_S_CMorigin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath T1_S_CMorigin --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_CM/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T1_S_CM --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCDSOCORRMAXNOBAL/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCDSOCORRMAXNOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --same_opt --only_corr --max_points 128 --no_balance --wandb --wandb_project CMpp &


rm -rf checkpoint/T1_S_AVN/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T1_S_AVN --scale small --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON --scale small --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --only_one_norm --wandb --wandb_project CMpp


rm -rf checkpoint/T1_S_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --delete_occupancy_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --use_opt_gram --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --delete_occupancy_loss --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --wandb --wandb_project CMpp



rm -rf checkpoint/T1_S_AVNOON_NC_G10/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NC_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_G10/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_G10/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_G10/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --exp_scale_for_point_matching_loss --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC_G10/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --delete_occupancy_loss --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG_G10/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --use_opt_gram --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC_G10/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --delete_occupancy_loss --gradient_clip_val 10 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC_OG_G10/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC_OG_G10 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --gradient_clip_val 10 --wandb --wandb_project CMpp



rm -rf checkpoint/T1_S_AVNOON_NC_G100/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NC_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_G100/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_G100/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_G100/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_point_matching_loss --exp_scale_for_point_matching_loss --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC_G100/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --delete_occupancy_loss --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG_G100/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_EXP_NODOCC_OG_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --exp_scale_for_point_matching_loss --use_opt_gram --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC_G100/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --delete_occupancy_loss --gradient_clip_val 100 --wandb --wandb_project CMpp
rm -rf checkpoint/T1_S_AVNOON_NCD_NPM_NODOCC_OG_G100/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath T1_S_AVNOON_NCD_NPM_NODOCC_OG_G100 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --use_opt_gram --gradient_clip_val 100 --wandb --wandb_project CMpp




