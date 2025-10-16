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





# rm -rf checkpoint/L3_SMALL_CM_origin/ && CUDA_VISIBLE_DEVICES=0 python main.py --model CM_equiassem --logpath L3_SMALL_CM_origin --scale L3_SMALL --multiplicity 1 --epochs 0 --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_by_CMpp/ && CUDA_VISIBLE_DEVICES=1 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_by_CMpp --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --wandb --wandb_project CMpp


rm -rf checkpoint/L3_SMALL_CM_AVN/ && CUDA_VISIBLE_DEVICES=2 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVN_NC/ && CUDA_VISIBLE_DEVICES=3 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --debugged_circle_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_NODOCC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVN_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVN_NC_NPM_NODOCC_OG --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --use_opt_gram --wandb --wandb_project CMpp 


rm -rf checkpoint/L3_SMALL_CM_AVNOON/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_circle_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM/ && CUDA_VISIBLE_DEVICES=4 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP/ && CUDA_VISIBLE_DEVICES=5 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
# rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_EXP_NODOCC_OG --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_NODOCC/ && CUDA_VISIBLE_DEVICES=6 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_NODOCC --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --delete_occupancy_loss --wandb --wandb_project CMpp 
rm -rf checkpoint/L3_SMALL_CM_AVNOON_NC_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=7 python main.py --model CMpp_equiassem --logpath L3_SMALL_CM_AVNOON_NC_NPM_NODOCC_OG --scale L3_SMALL --multiplicity 1 --epochs 0 --visualize --only_one_norm --use_opt_gram --wandb --wandb_project CMpp 
end



