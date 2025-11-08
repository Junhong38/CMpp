import subprocess
import os
import queue
from threading import Thread

# def worker(gpu_id, task_queue):
#     """
#     Continuously get a task from the queue, set the CUDA_VISIBLE_DEVICES environment variable, and execute the task.
#     When the queue is empty, the worker will exit.
#     """
#     while True:
#         try:
#             # If the queue is empty, queue.Empty will be raised, and the worker will break the loop
#             command = task_queue.get(timeout=3)  # You can adjust the timeout as needed
#         except queue.Empty:
#             return

#         env = os.environ.copy()
#         env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

#         # Execute the command
#         process = subprocess.Popen(command, env=env, shell=True)
#         process.wait()

#         # Mark this task as done in the queue to allow another to be added if needed
#         task_queue.task_done()

def worker(task_queue):
    """
    Continuously get a task from the queue, set the CUDA_VISIBLE_DEVICES environment variable, and execute the task.
    When the queue is empty, the worker will exit.
    """
    while True:
        try:
            # If the queue is empty, queue.Empty will be raised, and the worker will break the loop
            command = task_queue.get(timeout=3)  # You can adjust the timeout as needed
        except queue.Empty:
            return

        env = os.environ.copy()
        # env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Execute the command
        process = subprocess.Popen(command, env=env, shell=True)
        process.wait()

        # Mark this task as done in the queue to allow another to be added if needed
        task_queue.task_done()

def execute_commands_on_gpus(commands, num_gpus=None):
    """
    Create a queue of commands, and have each GPU work through the queue.
    """
    # Query number of available GPUs
    if num_gpus is None:
        try:
            num_gpus = str(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
            ).decode('utf-8')).count('\n')
            assert num_gpus > 0
        except Exception as e:
            print(f"An error occurred while querying the number of GPUs using nvidia-smi: {e}")
            return

    # Create a queue for the commands
    command_queue = queue.Queue()
    for command in commands:
        command_queue.put(command)

    # Start a worker thread for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        # thread = Thread(target=worker, args=(gpu_id, command_queue))
        thread = Thread(target=worker, args=(command_queue,))
        thread.start()
        threads.append(thread)

    # Wait for all tasks in the queue to be processed
    command_queue.join()

    # The commands are all done at this point, but the worker threads are likely idle and waiting for more tasks.
    # We'll end each thread by joining them.
    for thread in threads:
        thread.join()

# def return_command(
#     option,
#     logpath
# ):
#     """Generate a command string for launching an experiment."""
#     command = f"""
#     python test.py \\
#     --scale small \\
#     --multiplicity 1 \\
#     --logpath {logpath} \\
#     --visualize \\
#     --only_one_norm \\
#     --use_opt_gram \\
#     {option}
#     """

#     return command

# # Generate command list
# commands_list = []
# option = ""
# for usage in [None, '--use_RANSAC']:
#     if usage == '--use_RANSAC':
#         for type in ['default', 'score_dependent']:
#             for match_option in ['topk', 'mutual_topk', 'soft_topk', 'unidirectional_nn_matching', 'injective_matching', 'bijective_matching']:
#                 if match_option == 'topk':
#                     for topk in ['128', '-10', '-20']:
#                         option = usage + " \\ " + '--RANSAC_type ' + type + " \\ " + "--RANSA"
#     breakpoint()
    
# for option in ['original', 'additional_VNLinearLeakyReLU', 'debugged_circle_loss', 'debugged_point_matching_loss', 'delete_occupancy_loss', 'use_opt_gram']:
#     logpath = f"tiny_{option}"
#     if option == 'original': option = None
#     commands_list.append(
#         return_command(
#             option=option,
#             logpath=logpath
#         )
#     )



# commands_list = [
#     "rm -rf checkpoint/SDR_predicted_static-topk_128/ && python test.py --logpath SDR_predicted_static-topk_128 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option topk --RANSAC_topk 128 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_dynamic-topk_10/ && python test.py --logpath SDR_predicted_dynamic-topk_10 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option topk --RANSAC_topk -10 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_dynamic-topk_20/ && python test.py --logpath SDR_predicted_dynamic-topk_20 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option topk --RANSAC_topk -20 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_mutual-topk_1/ && python test.py --logpath SDR_predicted_mutual-topk_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option mutual_topk --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_mutual-topk_2/ && python test.py --logpath SDR_predicted_mutual-topk_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option mutual_topk --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_mutual-topk_3/ && python test.py --logpath SDR_predicted_mutual-topk_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option mutual_topk --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_soft-topk_1/ && python test.py --logpath SDR_predicted_soft-topk_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option soft_topk --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_soft-topk_2/ && python test.py --logpath SDR_predicted_soft-topk_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option soft_topk --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_soft-topk_3/ && python test.py --logpath SDR_predicted_soft-topk_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option soft_topk --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_unidir_top_1/ && python test.py --logpath SDR_predicted_unidir_top_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_unidir_top_2/ && python test.py --logpath SDR_predicted_unidir_top_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_unidir_top_3/ && python test.py --logpath SDR_predicted_unidir_top_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_inject/ && python test.py --logpath SDR_predicted_inject --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option injective_matching --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/SDR_predicted_biject/ && python test.py --logpath SDR_predicted_biject --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type score_dependent --RANSAC_match_option bijective_matching --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",

#     "rm -rf checkpoint/R_predicted_static-topk_128/ && python test.py --logpath  R_predicted_static-topk_128 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option topk --RANSAC_topk 128 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_dynamic-topk_10/ && python test.py --logpath  R_predicted_dynamic-topk_10 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option topk --RANSAC_topk -10 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_dynamic-topk_20/ && python test.py --logpath  R_predicted_dynamic-topk_20 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option topk --RANSAC_topk -20 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_mutual-topk_1/ && python test.py --logpath  R_predicted_mutual-topk_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option mutual_topk --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_mutual-topk_2/ && python test.py --logpath  R_predicted_mutual-topk_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option mutual_topk --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_mutual-topk_3/ && python test.py --logpath  R_predicted_mutual-topk_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option mutual_topk --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_soft-topk_1/ && python test.py --logpath  R_predicted_soft-topk_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option soft_topk --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_soft-topk_2/ && python test.py --logpath  R_predicted_soft-topk_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option soft_topk --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_soft-topk_3/ && python test.py --logpath  R_predicted_soft-topk_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option soft_topk --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_unidir_top_1/ && python test.py --logpath  R_predicted_unidir_top_1 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 1 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_unidir_top_2/ && python test.py --logpath  R_predicted_unidir_top_2 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 2 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_unidir_top_3/ && python test.py --logpath  R_predicted_unidir_top_3 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option unidirectional_nn_matching --RANSAC_topk 3 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_inject/ && python test.py --logpath  R_predicted_inject --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option injective_matching --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",
#     "rm -rf checkpoint/R_predicted_biject/ && python test.py --logpath  R_predicted_biject --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --use_RANSAC --RANSAC_type default --RANSAC_match_option bijective_matching --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt --use_predicted_normal",

    # "rm -rf checkpoint/CMpp-topk_128/ && python test.py --logpath CMpp-topk_128 --scale full --multiplicity 1 --visualize --only_one_norm --use_opt_gram --n_avn 0 --load checkpoint/DET_T5_F_NCD_NPM_NODOCC_OG/models/model-rrmse-epoch=086.ckpt"

# ]

# commands_list = [
#     # "rm -rf checkpoint/CMbyCMpp_DS && python main.py --model CMpp_equiassem --logpath CMbyCMpp_DS --scale small --multiplicity 1 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3"
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW1.0 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW1.0 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 1.0",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.9 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.9 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.9",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.8 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.8 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.8",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.7 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.7 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.7",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.6 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.6 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.6",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.5 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.5 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.5",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.4 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.4 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.4",
#     "rm -rf checkpoint/MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.3 && python main.py --model CMpp_equiassem --logpath MR_DS_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.3 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.3",

#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW1.0 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW1.0 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 1.0 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.9 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.9 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.9 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.8 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.8 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.8 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.7 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.7 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.7 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.6 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.6 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.6 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.5 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.5 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.5 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.4 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.4 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.4 --DS_only_training",
#     "rm -rf checkpoint/MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.3 && python main.py --model CMpp_equiassem --logpath MR_DSonlyT_DET_T5_F_NCD_NPM_NODOCC_OG_PW0.3 --scale small --multiplicity 1 --epochs 0 --visualize --detach_mode --use_opt_gram --deterministic --n_avn 0 --delete_Sinkhorn --wandb --wandb_project CMpp_debugging3 --p_loss_weight 0.3 --DS_only_training",
# ]
# commands_list = commands_list[::-1]

commands_list = [
    "rm -rf checkpoint/NF_CL_add_frame_layers/ && python main.py --model CMpp_equiassem --logpath NF_CL_add_frame_layers --scale small --multiplicity 1 --epochs 0 --n_avn 0 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal --use_consistency_loss --additional_VNLLReLU_for_frame --n_afl 2 --wandb --wandb_entity triplepoint --wandb_project CMpp_new_normal",
    "rm -rf checkpoint/NF_CL_add_frame_and_2shape_layers/ && python main.py --model CMpp_equiassem --logpath NF_CL_add_frame_and_2shape_layers --scale small --multiplicity 1 --epochs 0 --n_avn 2 --detach_mode --use_opt_gram --gpus 0 1 2 3 4 5 6 7 --n_worker 6 --flip_normal --use_consistency_loss --additional_VNLLReLU_for_frame --n_afl 2 --wandb --wandb_entity triplepoint --wandb_project CMpp_new_normal",

]

# commands_list = commands_list[::-1]
# print(commands_list[0])
# Execute the commands across the GPUs
execute_commands_on_gpus(commands_list, num_gpus=1)

