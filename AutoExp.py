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

def return_command(
    pos_margin,
    neg_margin,
    log_scale,
    logpath
):
    """Generate a command string for launching an experiment."""
    command = f"""
    python main.py \\
    --scale overfitting \\
    --reverse_normal \\
    --attention none \\
    --pos_margin {pos_margin} \\
    --neg_margin {neg_margin} \\
    --log_scale {log_scale} \\
    --logpath {logpath} \\
    --gpus 0 1 2 3 4 5 6 7
    """

    return command

# Generate command list
commands_list = []
for log_scale in ['24', '1', '15', '5']:
    for pos_margin in ['0.05', '0.1']:
        for neg_margin in ['0.4', '0.6', '0.8', '1.0', '1.2']:
            if log_scale == '24' and pos_margin == '0.05' and neg_margin in ['0.4', '0.6', '0.8']:
                continue
            logname = "NewCircleLoss_overfitting_P"+pos_margin+"_N"+neg_margin+"_LS"+log_scale
            commands_list.append(
                return_command(
                    pos_margin=pos_margin,
                    neg_margin=neg_margin,
                    log_scale=log_scale,
                    logpath=logname
                )
            )

# print(commands_list[0])
# Execute the commands across the GPUs
execute_commands_on_gpus(commands_list, num_gpus=1)

