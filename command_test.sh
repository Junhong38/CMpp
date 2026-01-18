### G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6
# 기본적인 실험. 학습때의 성능이 잘 나오는지, argument 설정이 잘 되었는지 확인하기 위한 실험.
python test.py --logpath base --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move smaller 적용
python test.py --logpath base+move_smaller --move_smaller --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + sampling_mode=same 적용
python test.py --logpath base+same_sampling --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + sampling_mode=same 적용
python test.py --logpath base+move_smaller+same_sampling --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


### using RANSAC ###
# 기본 + RANSAC 적용
python test.py --logpath RANSAC --use_RANSAC --use_predicted_normal --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move smaller 적용
python test.py --logpath RANSAC+move_smaller --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + sampling_mode=same 적용
python test.py --logpath RANSAC+same_sampling --use_RANSAC --use_predicted_normal --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + sampling_mode=same 적용
python test.py --logpath RANSAC+move_smaller+same_sampling --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


### using RANSAC + tighter normal thresholding ###
# normal threshold 180 - 45 = 135도
# strong normal threshold 180 - 20 = 160도
python test.py --logpath RANSAC_MS_SS_norm135_Snorm160 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# normal threshold 180 - 20 = 160도
# strong normal threshold 180 - 20 = 160도
python test.py --logpath RANSAC_MS_SS_norm160_Snorm160 --RANSAC_normal_threshold 160 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# normal threshold 180 - 20 = 160도
# strong normal threshold 180 - 10 = 170도
python test.py --logpath RANSAC_MS_SS_norm160_Snorm170 --RANSAC_normal_threshold 160 --RANSAC_strong_normal_threshold 170 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


### using score dependent RANSAC ###
# 기본 + score dependent RANSAC 적용
python test.py --logpath sd-RANSAC --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# score dependent RANSAC + move smaller 적용
python test.py --logpath sd-RANSAC+move_smaller --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --move_smaller --sampling_mode random --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# score dependent RANSAC + sampling_mode=same 적용
python test.py --logpath sd-RANSAC+same_sampling --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# score dependent RANSAC + move_smaller + sampling_mode=same 적용
python test.py --logpath sd-RANSAC+move_smaller+same_sampling --use_RANSAC --RANSAC_type score_dependent --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


### using score dependent RANSAC + tighter normal thresholding ###
# normal threshold 180 - 45 = 135도
# strong normal threshold 180 - 20 = 160도
python test.py --logpath sd-RANSAC_MS_SS_norm135_Snorm160 --RANSAC_type score_dependent --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# normal threshold 180 - 20 = 160도
# strong normal threshold 180 - 20 = 160도
python test.py --logpath sd-RANSAC_MS_SS_norm160_Snorm160 --RANSAC_type score_dependent --RANSAC_normal_threshold 160 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# normal threshold 180 - 20 = 160도
# strong normal threshold 180 - 10 = 170도
python test.py --logpath sd-RANSAC_MS_SS_norm160_Snorm170 --RANSAC_type score_dependent --RANSAC_normal_threshold 160 --RANSAC_strong_normal_threshold 170 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


### putative match selection ###
# 기본 + move_smaller + putative match selection (topk=128)
python test.py --logpath RANSAC_MS_top128 --infer_topk 128 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=256)
python test.py --logpath RANSAC_MS_top256 --infer_topk 256 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=64)
python test.py --logpath RANSAC_MS_top64 --infer_topk 64 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=0.001% of MN)
python test.py --logpath RANSAC_MS_top0.001% --infer_topk -0.001 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=0.01% of MN)
python test.py --logpath RANSAC_MS_top0.01% --infer_topk -0.01 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=0.05% of MN)
python test.py --logpath RANSAC_MS_top0.05% --infer_topk -0.05 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=0.1% of MN)
python test.py --logpath RANSAC_MS_top0.1% --infer_topk -0.1 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=500)
python test.py --logpath RANSAC_MS_top500 --infer_topk 500 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (topk=1000)
python test.py --logpath RANSAC_MS_top1000 --infer_topk 1000 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (mutual topk=1)
python test.py --logpath RANSAC_MS_mutual_top1 --infer_topk 1 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (mutual topk=2)
python test.py --logpath RANSAC_MS_mutual_top2 --infer_topk 2 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (mutual topk=3)
python test.py --logpath RANSAC_MS_mutual_top3 --infer_topk 3 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (soft topk=1)
python test.py --logpath RANSAC_MS_soft_top1 --infer_topk 1 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (soft topk=2)
python test.py --logpath RANSAC_MS_soft_top2 --infer_topk 2 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (soft topk=3)
python test.py --logpath RANSAC_MS_soft_top3 --infer_topk 3 --infer_match_option mutual_topk --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (unidirectional_nn_matchiing topk=1)
python test.py --logpath RANSAC_MS_unidirectional_top1 --infer_topk 1 --infer_match_option unidirectional_nn_matching --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (unidirectional_nn_matchiing topk=2)
python test.py --logpath RANSAC_MS_unidirectional_top2 --infer_topk 2 --infer_match_option unidirectional_nn_matching --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (unidirectional_nn_matchiing topk=3)
python test.py --logpath RANSAC_MS_unidirectional_top3 --infer_topk 3 --infer_match_option unidirectional_nn_matching --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (injective matching)
python test.py --logpath RANSAC_MS_injective --infer_match_option injective_matching --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 + move_smaller + putative match selection (bijective matching)
python test.py --logpath RANSAC_MS_bijective --infer_match_option bijective_matching --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

# 기본 말고 sdRANSAC이 맞는 듯...
# same은 해야 성능이 올라갈 수 있느 듯...

### normal threshold를 더 널널하게 ###
python test.py --logpath sd-RANSAC_MS_SS_norm135_Snorm135 --RANSAC_type score_dependent --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 135 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7




python test.py --logpath sd-RANSAC_MS_top500 --infer_topk 500 --RANSAC_type score_dependent --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 154 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7
python test.py --logpath sd-RANSAC_MS_SS_top0.001%_norm135_Snorm154 --infer_topk -0.001 --RANSAC_type score_dependent --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 154 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7

python test.py --logpath sd-RANSAC_MS_SS_top0.05%_norm135_Snorm154 --infer_topk -0.05 --RANSAC_type score_dependent --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 154 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


python test.py --logpath sd-RANSAC_MS_SS_top0.05%_norm90_Snorm90_dist0.01-0.008 --infer_topk -0.05 --RANSAC_type score_dependent --RANSAC_normal_threshold 90 --RANSAC_strong_normal_threshold 90 --use_RANSAC --use_predicted_normal --move_smaller --sampling_mode same --flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7



--flip_normal_mode rightv1_2 --n_worker 6 --load ./checkpoint/checkpoint_CMpp/G8NDB2_F_BV0V0_MH_P0075M0075+N145M145+BD+HNM+NK0+DCOS+AD_NPCFNR1-2_MCM+MNSoft_S1P1_6/models/last.ckpt --gpus 0 1 2 3 4 5 6 7




rm -rf checkpoint/TOP128NP5000SAMPLERAND_LOAD_G8NDB3_SEGMLPDICE_W10/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/checkpoint_CMpp/G8NDB3_SEGMLPDICE_W10/models/last.ckpt --scale full --logpath TOP128NP5000SAMPLERAND_LOAD_G8NDB3_SEGMLPDICE_W10 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test
rm -rf checkpoint/TOP128NP5000SAMPLERAND+SEG_LOAD_G8NDB3_SEGMLPDICE_W10/ && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py --load checkpoint/checkpoint_CMpp/G8NDB3_SEGMLPDICE_W10/models/last.ckpt --scale full --logpath TOP128NP5000SAMPLERAND+SEG_LOAD_G8NDB3_SEGMLPDICE_W10 --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller --wandb --wandb_entity CMppProject --wandb_project CMpp_test

python test.py --load checkpoint/checkpoint_CMpp/NO_FLIP_W10/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --scale full --logpath test --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal_mode none --seg_head_mode mlp --use_seg_result --move_smaller
python test.py --logpath M_threshold-sd-RANSAC_MS_top64_norm135_Snorm154 --seg_head_mode mlp --use_seg_result --using_seg_mode threshold --RANSAC_type score_dependent --infer_topk 64 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 154 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode none --n_worker 6 --load checkpoint/checkpoint_CMpp/NO_FLIP_W10/SECOND_G8NDB3_SEGMLPDICE_LOAD_G8NDB3_W10/models/last.ckpt --gpus 0 1 2 3 4 5 6 7


python test.py --logpath mating-sd-RANSAC_MS_top128_norm135_Snorm160 --seg_head_mode mlp --use_seg_result --RANSAC_type score_dependent --infer_topk 128 --RANSAC_normal_threshold 135 --RANSAC_strong_normal_threshold 160 --use_RANSAC --use_predicted_normal --move_smaller --flip_normal_mode none --n_worker 6 --load checkpoint/checkpoint_CMpp/G8NDB3_SEGMLPDICE_W10/models/last.ckpt --gpus 0 1 2 3 4 5 6 7