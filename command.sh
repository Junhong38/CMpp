:<<end
# rm -rf checkpoint/L3_OVER_CM_origin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath L3_OVER_CM_origin --scale overfitting --multiplicity 100 --epochs 0 --wandb --wandb_project CMpp
rm -rf checkpoint/L3_OVER_CM_by_CMpp/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_by_CMpp --scale overfitting --multiplicity 100 --epochs 0 --visualize --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_OVER_CM_AVN/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN --scale overfitting --multiplicity 100 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC --scale overfitting --multiplicity 100 --epochs 0 --visualize --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM --scale overfitting --multiplicity 100 --epochs 0 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM_EXP --scale overfitting --multiplicity 100 --epochs 0 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM_EXP_NODOCC --scale overfitting --multiplicity 100 --epochs 0 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale overfitting --multiplicity 100 --epochs 0 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM_NODOCC --scale overfitting --multiplicity 100 --epochs 0 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVN_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVN_NC_NPM_NODOCC_OG --scale overfitting --multiplicity 100 --epochs 0 --visualize --use_opt_gram --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_OVER_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM_EXP --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM_EXP_NODOCC --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM_EXP_NODOCC_OG --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM_NODOCC --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NC_NPM_NODOCC_OG --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --use_opt_gram --wandb --wandb_project CMpp 



# rm -rf checkpoint/L3_TINY_CM_origin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath L3_TINY_CM_origin --scale tiny --multiplicity 33 --epochs 0 --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_by_CMpp/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_by_CMpp --scale tiny --multiplicity 33 --epochs 0 --visualize --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_TINY_CM_AVN/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN --scale tiny --multiplicity 33 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC --scale tiny --multiplicity 33 --epochs 0 --visualize --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM --scale tiny --multiplicity 33 --epochs 0 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM_EXP --scale tiny --multiplicity 33 --epochs 0 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM_EXP_NODOCC --scale tiny --multiplicity 33 --epochs 0 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale tiny --multiplicity 33 --epochs 0 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM_NODOCC --scale tiny --multiplicity 33 --epochs 0 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVN_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVN_NC_NPM_NODOCC_OG --scale tiny --multiplicity 33 --epochs 0 --visualize --use_opt_gram --wandb --wandb_project CMpp 



rm -rf checkpoint/L3_TINY_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM_EXP --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM_EXP_NODOCC --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM_EXP_NODOCC_OG --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM_NODOCC --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_TINY_CM_AVNOON_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_TINY_CM_AVNOON_NC_NPM_NODOCC_OG --scale tiny --multiplicity 33 --epochs 0 --visualize --only_one_norm --use_opt_gram --wandb --wandb_project CMpp 



# 

# rm -rf checkpoint/L3_SMALL_CM_origin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath L3_SMALL_CM_origin --scale small --multiplicity 1 --epochs 0 --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_by_CMpp/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_by_CMpp --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp


rm -rf checkpoint/L3_SMALL_CM_AVN/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN --scale small --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC --scale small --multiplicity 1 --epochs 0 --visualize --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM --scale small --multiplicity 1 --epochs 0 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP --scale small --multiplicity 1 --epochs 0 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --use_opt_gram --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_SMALL_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_NODOCC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --use_opt_gram --wandb --wandb_project CMpp 




rm -rf checkpoint/L3_OVER_CM_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NCD --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_OVER_CM_AVNOON_NCDSO/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_OVER_CM_AVNOON_NCDSO --scale overfitting --multiplicity 100 --epochs 0 --visualize --only_one_norm --detach_mode --same_opt --debugged_circle_loss  --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_SMALL_CM_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NCD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NCDSO/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NCDSO --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --same_opt --debugged_circle_loss  --wandb --wandb_project CMpp 




rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --debugged_circle_loss --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCSO/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCSO --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --same_opt --debugged_circle_loss --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCDSO/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCDSO --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --same_opt --debugged_circle_loss --wandb --wandb_project CMpp &




rm -rf checkpoint/L3_DEBUG_SMALL_CM/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp &

rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --wandb --wandb_project CMpp &

rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+CORR+NOBAL/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+CORR+NOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --only_corr --no_balance --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --wandb --wandb_project CMpp &

rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --div_mode dynamic --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD+NOBAL/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD+NOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --div_mode dynamic --no_balance --wandb --wandb_project CMpp &

# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --div_mode static --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS+NOBAL/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS+NOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --div_mode static --no_balance --wandb --wandb_project CMpp &

# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD+NOBAL_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVD+NOBAL_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --div_mode dynamic --no_balance --use_opt_gram --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS+NOBAL_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+DIVS+NOBAL_NODOCC_OG --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --detach_mode --div_mode static --no_balance --use_opt_gram --wandb --wandb_project CMpp &
end


rm -rf checkpoint/L3_DEBUG_SMALL_CM/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM --scale small --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NC --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+CORR+MAX+NOBAL/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+CORR+MAX+NOBAL --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --only_corr --max_points 128 --no_balance --wandb --wandb_project CMpp &


rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG1/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG1 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 1.0 --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG2/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG2 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 2.0 --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG4/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG4 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 4.0 --wandb --wandb_project CMpp &
rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG6/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG6 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 6.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG12/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG12 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 12.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG24/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+NOBAL+LOG24 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --no_balance --log_scale 24.0 --wandb --wandb_project CMpp &


# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG1/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG1 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 1.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG2/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG2 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 2.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG4/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG4 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 4.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG6/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG6 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 6.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG12/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG12 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 12.0 --wandb --wandb_project CMpp &
# rm -rf checkpoint/L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG24/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_DEBUG_SMALL_CM_AVNOON_NCD+LOG24 --scale small --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --detach_mode --log_scale 24.0 --wandb --wandb_project CMpp &




