#!/usr/bin/env bash
# =============================================================================
#  Reproduce: "Learning Combinative Shape Matching with Surface Normal
#  Consistency"  (CMpp / EquiAssem, BreakingBad Everyday, 8 GPUs)
#
#  Pipeline:  3-stage training  ->  inference (2-part Table 1 & multi-part Table 3)
#  Best-model chain (flip_normal_mode rightv4 throughout):
#      STAGE 1  FIRST   : frame / normal training        (--only_train_normal)
#      STAGE 2  SECOND  : shape + matching, frame at low LR (NFREEZE, lr*0.0125)
#      STAGE 3  THIRD   : mating-surface seg head (MLP + Dice)
#
#  Checkpoints all live under checkpoint/<logpath>/models/last.ckpt and each
#  stage loads directly from the previous stage's dir (no backup copies).
#  (main.py auto-resumes from checkpoint/<logpath>/models/model-* if it exists,
#   which would silently override --load_*, hence the leading rm -rf of the
#   CURRENT stage's dir only — the previous stage's dir is left intact to load.)
#
#  The EPOCHS / USE_WANDB toggles below switch between a SMOKE TEST
#  (EPOCHS=1, wandb off -> "does the whole pipeline currently run?") and the
#  REAL run (EPOCHS=0 -> auto 90 epochs, USE_WANDB=1). Comment out finished stages.
# =============================================================================
set -eo pipefail

# ------------------------------- user config ---------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # the 8 GPUs to use
GPUS="0 1 2 3 4 5 6 7"                            # must match CUDA_VISIBLE_DEVICES
DATAPATH=/mnt/nvme2n1p1/kimsangki_datasets/breaking_bad/volume_constrained/
NWORKER=6

# --- SMOKE TEST toggles : quick "does it currently run?" check ---------------
#     real reproduce run -> set USE_WANDB=1  and  EPOCHS=0
USE_WANDB=0                                       # 0 = wandb off, 1 = log to wandb
EPOCHS=0                                          # 0 = full run (auto 90 for everyday/2-part); 1 = quick smoke run
# -----------------------------------------------------------------------------
WANDB_ENTITY=CMppProject
WANDB_PROJECT=CMpp
WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS="--wandb --wandb_entity ${WANDB_ENTITY} --wandb_project ${WANDB_PROJECT}"
fi

# logpath (= checkpoint dir name = wandb run name) for each stage
FIRST=FIRST_G8NDB6SN_NPCFNR4C1
SECOND=SECOND_NFREEZE00125_G8NDB2_NPCFNR4C1
THIRD=THIRD_SEGMLPDICE_NFREEZE00125_G8NDB2_NPCFNR4C1

# -----------------------------------------------------------------------------


# ============================== STAGE 1 : FRAME ==============================
echo "==================== STAGE 1 : frame / normal training ===================="
rm -rf checkpoint/${FIRST}/
python main.py \
    --datapath ${DATAPATH} \
    --scale full \
    --only_train_normal \
    --logpath ${FIRST} \
    --epochs ${EPOCHS} \
    --scheduler_mode none \
    --gpus ${GPUS} \
    --n_worker ${NWORKER} \
    --batch_size 6 \
    --flip_normal_mode rightv4 \
    --consistency_loss_weight 1.0 \
    ${WANDB_ARGS}



# ========================= STAGE 2 : SHAPE + MATCHING ========================
echo "==================== STAGE 2 : shape + matching training ===================="
rm -rf checkpoint/${SECOND}/
python main.py \
    --datapath ${DATAPATH} \
    --scale full \
    --load_ori checkpoint/${FIRST}/models/last.ckpt \
    --logpath ${SECOND} \
    --epochs ${EPOCHS} \
    --scheduler_mode cos \
    --gpus ${GPUS} \
    --n_worker ${NWORKER} \
    --batch_size 2 \
    --flip_normal_mode rightv4 \
    --consistency_loss_weight 1.0 \
    --ori_backbone_lr_weight 0.0125 \
    ${WANDB_ARGS}



# ========================= STAGE 3 : SEGMENTATION HEAD =======================
echo "==================== STAGE 3 : segmentation head training ===================="
rm -rf checkpoint/${THIRD}/
python main.py \
    --datapath ${DATAPATH} \
    --scale full \
    --load_except_seg_head checkpoint/${SECOND}/models/last.ckpt \
    --logpath ${THIRD} \
    --epochs ${EPOCHS} \
    --scheduler_mode cos \
    --gpus ${GPUS} \
    --n_worker ${NWORKER} \
    --batch_size 6 \
    --flip_normal_mode rightv4 \
    --seg_head_mode mlp \
    --seg_loss_mode dice \
    ${WANDB_ARGS}

# the trained model used by every inference command below
CKPT=checkpoint/${THIRD}/models/last.ckpt



# =============================================================================
#  INFERENCE  (2-part only; no training; uses CSV logger; eval dirs are scratch)
#  Maps to the paper Table 1 (2-part): woRANSAC / RANSAC / RANSAC+pen
# =============================================================================
echo "==================== INFERENCE ===================="

# --------- Table 1 : 2-part, WITHOUT RANSAC (weighted-SVD / LGR head) ---------
rm -rf checkpoint/EVAL_2part_woRANSAC/
python test.py \
    --datapath ${DATAPATH} \
    --load ${CKPT} \
    --logpath EVAL_2part_woRANSAC \
    --gpus ${GPUS} \
    --n_worker ${NWORKER} \
    --flip_normal_mode rightv4 \
    --seg_head_mode mlp \
    --move_smaller \
    --min_part 2 \
    --max_part 2 \
    --multi_part_assembly none


echo "==================== DONE: training + 2-part inference complete ===================="
