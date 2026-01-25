
:<<end
rm -rf checkpoint/TEST_CM_OO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_CM_OO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode origin &> TEST_CM_OO.log
rm -rf checkpoint/TEST_CM_NO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_CM_NO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode origin

rm -rf checkpoint/TEST_CM_ON/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_CM_ON --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode new &> TEST_CM_ON.log
rm -rf checkpoint/TEST_CM_NN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CM_everyday_pa.ckpt --model CM_equiassem --logpath TEST_CM_NN --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode new


CUDA_VISIBLE_DEVICES=0 python test_mpa_shonan_T.py --load CM_checkpoint/CM_everyday_pa.ckpt --logpath TEST_SHONAN_origin_origin --scale full --n_worker 6 --div_mode origin --sampling_mode origin &
CUDA_VISIBLE_DEVICES=1 python test_mpa_shonan_T.py --load CM_checkpoint/CM_everyday_pa.ckpt --logpath TEST_SHONAN_new_origin --scale full --n_worker 6 --div_mode new --sampling_mode origin & 
CUDA_VISIBLE_DEVICES=2 python test_mpa_shonan_T.py --load CM_checkpoint/CM_everyday_pa.ckpt --logpath TEST_SHONAN_origin_new --scale full --n_worker 6 --div_mode origin --sampling_mode new &
CUDA_VISIBLE_DEVICES=3 python test_mpa_shonan_T.py --load CM_checkpoint/CM_everyday_pa.ckpt --logpath TEST_SHONAN_new_new --scale full --n_worker 6 --div_mode new --sampling_mode new &
end



rm -rf checkpoint/TEST_CM_OO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_pa_new.ckpt --model CM_equiassem --logpath TEST_CM_OO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode origin &> TEST_CM_OO.log
rm -rf checkpoint/TEST_CM_NO/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_pa_new.ckpt --model CM_equiassem --logpath TEST_CM_NO --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode origin

rm -rf checkpoint/TEST_CM_ON/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_pa_new.ckpt --model CM_equiassem --logpath TEST_CM_ON --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode origin --sampling_mode new &> TEST_CM_ON.log
rm -rf checkpoint/TEST_CM_NN/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load CM_checkpoint/CMNet_pa_new.ckpt --model CM_equiassem --logpath TEST_CM_NN --scale full --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --test_end_mode new --sampling_mode new


CUDA_VISIBLE_DEVICES=0 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_pa_new.ckpt --logpath TEST_SHONAN_origin_origin --scale full --n_worker 6 --div_mode origin --sampling_mode origin &
CUDA_VISIBLE_DEVICES=1 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_pa_new.ckpt --logpath TEST_SHONAN_new_origin --scale full --n_worker 6 --div_mode new --sampling_mode origin & 
CUDA_VISIBLE_DEVICES=2 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_pa_new.ckpt --logpath TEST_SHONAN_origin_new --scale full --n_worker 6 --div_mode origin --sampling_mode new &
CUDA_VISIBLE_DEVICES=3 python test_mpa_shonan_T.py --load CM_checkpoint/CMNet_pa_new.ckpt --logpath TEST_SHONAN_new_new --scale full --n_worker 6 --div_mode new --sampling_mode new &

