
:<<end
rm -rf checkpoint/TEST_PA_CM_OO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --model CM_equiassem --logpath TEST_PA_CM_OO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode origin &> TEST_PA_CM_OO.log
rm -rf checkpoint/TEST_PA_CM_NO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --model CM_equiassem --logpath TEST_PA_CM_NO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode origin

rm -rf checkpoint/TEST_PA_CM_ON/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --model CM_equiassem --logpath TEST_PA_CM_ON --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode new &> TEST_PA_CM_ON.log
rm -rf checkpoint/TEST_PA_CM_NN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --model CM_equiassem --logpath TEST_PA_CM_NN --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode new





rm -rf checkpoint/TEST_MPA_CM_OO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --model CM_equiassem --logpath TEST_MPA_CM_OO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode origin &> TEST_MPA_CM_OO.log
rm -rf checkpoint/TEST_MPA_CM_NO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --model CM_equiassem --logpath TEST_MPA_CM_NO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode origin

rm -rf checkpoint/TEST_MPA_CM_ON/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --model CM_equiassem --logpath TEST_MPA_CM_ON --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode new &> TEST_MPA_CM_ON.log
rm -rf checkpoint/TEST_MPA_CM_NN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --model CM_equiassem --logpath TEST_MPA_CM_NN --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode new
end





(CUDA_VISIBLE_DEVICES=0 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --logpath TEST_PA_SHONAN_origin_origin --scale full --n_worker 6 --div_mode origin --sampling_mode origin &> TEST_PA_SHONAN_origin_origin.log) &
(CUDA_VISIBLE_DEVICES=1 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --logpath TEST_PA_SHONAN_new_origin --scale full --n_worker 6 --div_mode new --sampling_mode origin &> TEST_PA_SHONAN_new_origin.log) &
(CUDA_VISIBLE_DEVICES=2 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --logpath TEST_PA_SHONAN_origin_new --scale full --n_worker 6 --div_mode origin --sampling_mode new &> TEST_PA_SHONAN_origin_new.log) &
(CUDA_VISIBLE_DEVICES=3 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_pa_everyday.ckpt --logpath TEST_PA_SHONAN_new_new --scale full --n_worker 6 --div_mode new --sampling_mode new &> TEST_PA_SHONAN_new_new.log) &




(CUDA_VISIBLE_DEVICES=4 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --logpath TEST_MPA_SHONAN_origin_origin --scale full --n_worker 6 --div_mode origin --sampling_mode origin &> TEST_MPA_SHONAN_origin_origin.log) &
(CUDA_VISIBLE_DEVICES=5 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --logpath TEST_MPA_SHONAN_new_origin --scale full --n_worker 6 --div_mode new --sampling_mode origin &> TEST_MPA_SHONAN_new_origin.log) &
(CUDA_VISIBLE_DEVICES=6 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --logpath TEST_MPA_SHONAN_origin_new --scale full --n_worker 6 --div_mode origin --sampling_mode new &> TEST_MPA_SHONAN_origin_new.log) &
(CUDA_VISIBLE_DEVICES=7 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_v2_mpa_everyday.ckpt --logpath TEST_MPA_SHONAN_new_new --scale full --n_worker 6 --div_mode new --sampling_mode new &> TEST_MPA_SHONAN_new_new.log) &




