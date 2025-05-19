# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit for ViM-CLIP models.
"""
import argparse
import os
import uuid
from pathlib import Path

# Import the main script for ViM-CLIP
import main_clip as classification 
import submitit

# Import the model definition to register models (if needed by get_args_parser)
import vim_clip.models_mamba_clip

def parse_args():
    # Use the argument parser from main_clip.py
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for ViM-CLIP", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request") # Default to 1 node for easier testing
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job in minutes")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit") # Adjust if needed
    parser.add_argument("--use_volta32", action='store_true', help="Set if using Volta 32GB GPUs")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    # Add specific arguments for main_clip.py if they need different defaults here
    # For example:
    # parser.set_defaults(model='vim_tiny_patch16_224_clip_prompts') 
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    # Adjust shared folder path if necessary for your environment
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments/vim_clip") # Specific subfolder
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Fallback or alternative shared path
    p = Path(f"./shared_experiments/vim_clip") 
    p.mkdir(parents=True, exist_ok=True)
    print(f"Warning: /checkpoint/ not found, using local shared folder: {p.resolve()}")
    return p
    # raise RuntimeError("No shared folder available") # Or raise error

def get_init_file():
    # Init file must not exist, but its parent dir must exist.
    shared_folder = get_shared_folder()
    os.makedirs(str(shared_folder), exist_ok=True)
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # Ensure the correct main script is called
        import main_clip as classification
        # Make sure model definitions are registered
        import vim_clip.models_mamba_clip

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        # Check for the specific checkpoint file in the output directory
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth") 
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        # Use the job ID to create a unique output directory path
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}, local_rank: {job_env.local_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        # Define the job directory using the shared folder and substituting job ID
        args.job_dir = get_shared_folder() / "%j"

    # Define the output directory based on the job directory pattern
    args.output_dir = str(args.job_dir) # Output dir will include job ID after setup

    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=str(args.job_dir), slurm_max_num_timeout=30)

    # Cluster configuration
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    # Set SLURM parameters
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node, # Adjust memory based on needs
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10, # Adjust based on dataloading/processing needs
        nodes=nodes,
        timeout_min=timeout_min,
        # Cluster-specific parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120, 
        **kwargs
    )

    # Set job name (optional)
    executor.update_parameters(name="vim_clip_train") # Changed job name

    # Set distributed training parameters
    args.dist_url = get_init_file().as_uri()
    # args.output_dir is already set to the job directory pattern

    # Create trainer instance and submit job
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id} to partition {partition}")
    print(f"Job directory: {str(args.job_dir).replace('%j', job.job_id)}")
    print(f"Output directory: {str(args.output_dir).replace('%j', job.job_id)}")

if __name__ == "__main__":
    main() 