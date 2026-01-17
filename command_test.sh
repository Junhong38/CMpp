


rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller

rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller


rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller

rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/last.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller
rm -rf checkpoint/TEST/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint_backup/SECOND_G8NDB3_SEGATTENDICE_LOAD_G8NDB3_W10/models/model-seg_F1-epoch=086.ckpt --logpath TEST --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller



