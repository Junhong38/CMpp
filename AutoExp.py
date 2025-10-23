import subprocess
import os
import queue
from threading import Thread

def worker(gpu_id, task_queue):
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
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Execute the command
        process = subprocess.Popen(command, env=env, shell=True)
        process.wait()

        # Mark this task as done in the queue to allow another to be added if needed
        task_queue.task_done()

# def worker(task_queue):
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
#         # env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

#         # Execute the command
#         process = subprocess.Popen(command, env=env, shell=True)
#         process.wait()

#         # Mark this task as done in the queue to allow another to be added if needed
#         task_queue.task_done()

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
        thread = Thread(target=worker, args=(gpu_id+1, command_queue))
        # thread = Thread(target=worker, args=(command_queue,))
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
#     option_str = f"--{option} \\" if option else "" 
#     """Generate a command string for launching an experiment."""
#     command = f"""
#     python main.py \\
#     --scale tiny \\
#     --multiplicity 100 \\
#     --logpath {logpath} \\
#     --visualize \\
#     --wandb \\
#     --wandb_project CMpp_debugging \\
#     {option_str}
#     """

#     return command

# # Generate command list
# commands_list = []
# for option in ['original', 'additional_VNLinearLeakyReLU', 'debugged_circle_loss', 'debugged_point_matching_loss', 'delete_occupancy_loss', 'use_opt_gram']:
#     logpath = f"tiny_{option}"
#     if option == 'original': option = None
#     commands_list.append(
#         return_command(
#             option=option,
#             logpath=logpath
#         )
#     )

commands_list = [
    "rm -rf checkpoint/OVER_CM_origin/ && python main.py --model CM_equiassem --logpath OVER_CM_origin --scale overfitting --multiplicity 100 --visualize  --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_by_CMpp/ && python main.py --model CMpp_equiassem --logpath OVER_CM_by_CMpp --scale overfitting --multiplicity 100 --visualize --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_AVN/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN --scale overfitting --multiplicity 100 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_AVN_NC/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC --scale overfitting --multiplicity 100 --visualize --debugged_circle_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_AVN_NC_NPM/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM --scale overfitting --multiplicity 100 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp_debugging3",
    # "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_EXP/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_EXP --scale overfitting --multiplicity 100 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_EXP_NODOCC/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_EXP_NODOCC --scale overfitting --multiplicity 100 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale overfitting --multiplicity 100 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_NODOCC/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_NODOCC --scale overfitting --multiplicity 100 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_NODOCC_OG --scale overfitting --multiplicity 100 --visualize --use_opt_gram --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/OVER_CM_AVN_NC_NPM_NODOCC_OG_10KNN/ && python main.py --model CMpp_equiassem --logpath OVER_CM_AVN_NC_NPM_NODOCC_OG_10KNN --scale overfitting --multiplicity 100 --visualize --use_opt_gram --n_knn 10 --wandb --wandb_project CMpp_debugging3",


    "rm -rf checkpoint/TINY_CM_origin/ && python main.py --model CM_equiassem --logpath TINY_CM_origin --scale tiny --multiplicity 33 --visualize --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_by_CMpp/ && python main.py --model CMpp_equiassem --logpath TINY_CM_by_CMpp --scale tiny --multiplicity 33 --visualize --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_AVN/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN --scale tiny --multiplicity 33 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_AVN_NC/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC --scale tiny --multiplicity 33 --visualize --debugged_circle_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_AVN_NC_NPM/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM --scale tiny --multiplicity 33 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp_debugging3",
    # "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_EXP/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_EXP --scale tiny --multiplicity 33 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_EXP_NODOCC/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_EXP_NODOCC --scale tiny --multiplicity 33 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale tiny --multiplicity 33 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_NODOCC/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_NODOCC --scale tiny --multiplicity 33 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_NODOCC_OG --scale tiny --multiplicity 33 --visualize --use_opt_gram --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/TINY_CM_AVN_NC_NPM_NODOCC_OG_10KNN/ && python main.py --model CMpp_equiassem --logpath TINY_CM_AVN_NC_NPM_NODOCC_OG_10KNN --scale tiny --multiplicity 33 --visualize --use_opt_gram --n_knn 10 --wandb --wandb_project CMpp_debugging3",


    "rm -rf checkpoint/SMALL_CM_origin/ && python main.py --model CM_equiassem --logpath SMALL_CM_origin --scale small --multiplicity 1 --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_by_CMpp/ &&  python main.py --model CMpp_equiassem --logpath SMALL_CM_by_CMpp --scale small --multiplicity 1 --visualize --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_AVN/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN --scale small --multiplicity 1 --visualize --additional_VNLinearLeakyReLU --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_AVN_NC/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC --scale small --multiplicity 1 --visualize --debugged_circle_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM --scale small --multiplicity 1 --visualize --debugged_point_matching_loss --wandb --wandb_project CMpp_debugging3",
    # "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_EXP/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_EXP --scale small --multiplicity 1 --visualize --debugged_point_matching_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_EXP_NODOCC/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_EXP_NODOCC --scale small --multiplicity 1 --visualize --delete_occupancy_loss --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging",
    # "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_EXP_NODOCC_OG --scale small --multiplicity 1 --visualize --use_opt_gram --exp_scale_for_point_matching_loss --wandb --wandb_project CMpp_debugging"
    "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_NODOCC/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_NODOCC --scale small --multiplicity 1 --visualize --delete_occupancy_loss --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_NODOCC_OG/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_NODOCC_OG --scale small --multiplicity 1 --visualize --use_opt_gram --wandb --wandb_project CMpp_debugging3",
    "rm -rf checkpoint/SMALL_CM_AVN_NC_NPM_NODOCC_OG_10KNN/ && python main.py --model CMpp_equiassem --logpath SMALL_CM_AVN_NC_NPM_NODOCC_OG_10KNN --scale small --multiplicity 1 --visualize --use_opt_gram --n_knn 10 --wandb --wandb_project CMpp_debugging3"

]

commands_list = commands_list[::-1]
# print(commands_list[0])
# Execute the commands across the GPUs
execute_commands_on_gpus(commands_list, num_gpus=7)

