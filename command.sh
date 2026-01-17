
# 2 stage training
:<<end
# 1st stage -> orientation training
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR1-2C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR1-2C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv1_2 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR1-3C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR1-3C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv1_3 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR2C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR2C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv2 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR3C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR3C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv3 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR4C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/FIRST_G8NDB6SN_NPCFNR5C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --only_train_normal --logpath FIRST_G8NDB6SN_NPCFNR5C1 --epochs 0 --scheduler_mode none --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode rightv5 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
end



# 2nd stage -> shape training
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR1-2C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR1-2C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR1-2C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv1_2 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR1-3C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR1-3C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR1-3C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv1_3 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR2C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR2C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR2C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv2 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR3C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR3C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR3C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv3 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR4C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv4 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_G8NDB3_NPCFNR5C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR5C1/models/last.ckpt --freeze_ori_2nd_stage --logpath SECOND_G8NDB3_NPCFNR5C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv5 --wandb --wandb_entity CMppProject --wandb_project CMpp


# 2nd stage -> shape training, but freeze frame backbone and normal prediction proj. 2nd vector proj will be not frozen
rm -rf checkpoint/SECOND_FREEZEN2ND_G8NDB3_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR4C1/models/last.ckpt --freeze_ori_2nd_stage no_2nd --logpath SECOND_FREEZEN2ND_G8NDB3_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# 2nd stage -> shape training without freezing orientation backbone network
rm -rf checkpoint/SECOND_NFREEZE_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR4C1/models/last.ckpt --logpath SECOND_NFREEZE_G8NDB2_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/SECOND_NFREEZE_G8NDB2_NPCFNR5C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR5C1/models/last.ckpt --logpath SECOND_NFREEZE_G8NDB2_NPCFNR5C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv5 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp


# 2nd stage -> shape training without freezing orientation backbone network, LR for orientation backbone network is lr * ori_backbone_lr_weight
rm -rf checkpoint/SECOND_NFREEZE01_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR4C1/models/last.ckpt --logpath SECOND_NFREEZE01_G8NDB2_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --ori_backbone_lr_weight 0.1 --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/SECOND_NFREEZE00125_G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_ori checkpoint_1stStage/FIRST_G8NDB6SN_NPCFNR4C1/models/last.ckpt --logpath SECOND_NFREEZE00125_G8NDB2_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --ori_backbone_lr_weight 0.0125 --wandb --wandb_entity CMppProject --wandb_project CMpp


# Single Stage -> shape training + Consistency training
rm -rf checkpoint/G8NDB2_NPCFNR4C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --logpath G8NDB2_NPCFNR4C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv4 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp
# rm -rf checkpoint/G8NDB2_NPCFNR5C1/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --logpath G8NDB2_NPCFNR5C1 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 2 --flip_normal_mode rightv5 --consistency_loss_weight 1.0 --wandb --wandb_entity CMppProject --wandb_project CMpp





# Single Stage -> shape training + segmentation training (mlp) + warming up (10)
rm -rf checkpoint/G8NDB3_SEGMLPDICE_W10/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --logpath G8NDB3_SEGMLPDICE_W10 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 3 --flip_normal_mode none --seg_head_mode mlp --seg_loss_mode dice --start_hard_neg_epoch 10 --wandb --wandb_entity CMppProject --wandb_project CMpp


# G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD+W10_MCM+MNSoft_S1P1_6
# Single Stage -> shape training + segmentation training (mlp) + warming up (10) after loading G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD+W10_MCM+MNSoft_S1P1_6
rm -rf checkpoint/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --scale full --load_except_seg_head checkpoint_backup/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD+W10_MCM+MNSoft_S1P1_6/models/last.ckpt --logpath SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10 --epochs 0 --scheduler_mode cos --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode none --seg_head_mode mlp --seg_loss_mode dice --wandb --wandb_entity CMppProject --wandb_project CMpp
rm -rf checkpoint/SECOND_G8NDB3_SEGATTENDICELR0001_LOAD_G8NDB3_W10/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --lr 0.001 --scale full --load_except_seg_head checkpoint_backup/G8NDB3_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD+W10_MCM+MNSoft_S1P1_6/models/last.ckpt --logpath SECOND_G8NDB3_SEGATTENDICELR0001_LOAD_G8NDB3_W10 --epochs 0 --scheduler_mode onecycle --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --batch_size 6 --flip_normal_mode none --seg_head_mode atten --seg_loss_mode dice --wandb --wandb_entity CMppProject --wandb_project CMpp




