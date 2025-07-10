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
        thread = Thread(target=worker, args=(gpu_id, command_queue))
        thread.start()
        threads.append(thread)

    # Wait for all tasks in the queue to be processed
    command_queue.join()

    # The commands are all done at this point, but the worker threads are likely idle and waiting for more tasks.
    # We'll end each thread by joining them.
    for thread in threads:
        thread.join()

def return_command(
    score_comb,
    topk,
    match_selection,
    results_folder,
    optimal_matching
):
    """Generate a command string for launching an experiment."""
    command = f"""
    python test_mpa_shonan_mating.py \\
    --min_part 2 \\
    --max_part 2 \\
    --scale full \\
    --load ./checkpoint/CM-v2-mpa-every-model-trmse-epoch=325.ckpt \\
    --gt_normal_threshold -0.7 \\
    --gt_mating_surface \\
    --score_comb {score_comb} \\
    --distance_threshold 0.02 \\
    --topk {topk} \\
    --initial_match_selection {match_selection} \\
    --results_folder {results_folder} \\
    --optimal_matching_choice {optimal_matching} \\
    """

    return command

# Generate command list
commands_list = []
for match_selection in ['topk', 'mutual', 'soft']:
    for score_comb in ['sum']: #, 'intersection'
        for optimal_matching in ['many-to-one', 'one-to-one']:
            if match_selection == 'topk':
                topk_library = ['0', '128']
            elif match_selection == 'mutual' or 'soft':
                topk_library = ['1', '2', '3']
            for topk in topk_library:
                results_folder = "test_MS_FULL_" + match_selection + "_" + topk + "_" + score_comb + "_" + optimal_matching + ".txt"
                commands_list.append(
                    return_command(
                        score_comb=score_comb,
                        topk=topk,
                        match_selection=match_selection,
                        results_folder=results_folder,
                        optimal_matching=optimal_matching
                    )
                )

# print(commands_list[0])
# Execute the commands across the GPUs
execute_commands_on_gpus(commands_list, num_gpus=8)

