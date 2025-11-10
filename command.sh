:<<end

## TEST
rm -rf checkpoint/TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DET_T6_F_AVN_NCD_NPM_NODOCC_OG --scale full --multiplicity 1 --visualize --use_opt_gram --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CM/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_DET_CM --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test
rm -rf checkpoint/TEST_DET_CMbyCMpp/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/CM_checkpoint/CM_everyday_pa.ckpt --model CMpp_equiassem --logpath TEST_DET_CMbyCMpp --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller --wandb --wandb_project CMpp_test


rm -rf checkpoint/TEST_DEBUG/ && CUDA_VISIBLE_DEVICES=0 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DEBUG --scale full --multiplicity 1 --deterministic --gpus 0 --n_worker 6 --move_smaller
rm -rf checkpoint/TEST_DEBUG/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/DET_T6_F_AVN_NCD_NPM_NODOCC_OG/models/last.ckpt --model CMpp_equiassem --logpath TEST_DEBUG --scale full --multiplicity 1 --deterministic --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --move_smaller
end



rm -rf checkpoint/G8ND_S1_F_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --logpath G8ND_S1_F_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --only_train_normal --wandb --wandb_entity CMppProject --wandb_project CMpp


rm -rf checkpoint/G8ND_S2_F_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load checkpoint/G8ND_S1_F_FV0DV0MH_6/models/last.ckpt --logpath G8ND_S2_F_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_S2_S_FV0DV0MH_6/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath G8ND_S2_S_FV0DV0MH_6 --model CMpp_equiassem --backbone vn_unet --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale small --multiplicity 1 --epochs 1 --gpus 0 --n_worker 6

rm -rf checkpoint/G8ND_S2_F_FV1DV0MH_6/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --load_ori checkpoint/G8ND_S1_F_FV0DV0MH_6/models/last.ckpt --logpath G8ND_S2_F_FV1DV0MH_6 --model CMpp_equiassem --backbone vn_unet_deep --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale full --multiplicity 1 --epochs 0 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8ND_S2_S_FV1DV0MH_6/ && CUDA_VISIBLE_DEVICES=0 python main.py --logpath G8ND_S2_S_FV1DV0MH_6 --model CMpp_equiassem --backbone vn_unet_deep --double_bacbone vn_unet --n_avn 0 --mlp_mode half --scale small --multiplicity 1 --epochs 1 --gpus 0 --n_worker 6

