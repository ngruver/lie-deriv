# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import copy
import argparse
import os
import uuid
from pathlib import Path, PosixPath

import exps_e2e as exp
import submitit
import wandb

def parse_args():
    training_parser = exp.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for ssl robustness", parents=[training_parser], add_help=False)
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=3000, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--mail", default="", type=str,
                        help="Email this user when the job finishes if specified")
    parser.add_argument("--sweep_id", default="", type=str,
                        help="id of wandb sweep")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/scratch/").is_dir():
        p = Path(f"/scratch/nvg7279/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import exps_e2e as exp

        self._setup_gpu_args()

        path_args = {k: str(v) for k, v in self.args.__dict__.items() if isinstance(v, PosixPath)}
        args = copy.deepcopy(self.args)
        args.__dict__.update(path_args)

        func = lambda: exp.main(args)
        wandb.agent(self.args.sweep_id, function=func, project="LieDerivEquivariance")

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
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
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main(_args):
    args = parse_args()
    args.__dict__.update(_args)

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    # executor = submitit.local.local.LocalExecutor(args.job_dir)
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    kwargs = {}

    executor.update_parameters(
        mem_gb=32,
        # gpus_per_node=num_gpus_per_node,
        tasks_per_node=max(1,num_gpus_per_node),  # one task per GPU
        cpus_per_task=4,
        nodes=nodes,
        timeout_min=60 * 48,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_gres=f"gpu:{num_gpus_per_node}", #you can choose to comment this, or change it to v100 as per your need
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="ssl_robustness")
    if args.mail:
        executor.update_parameters(
            additional_parameters={'mail-user': args.mail, 'mail-type': 'END'})

    executor.update_parameters(slurm_additional_parameters={
        'gres-flags': 'enforce-binding',
        'partition': 'rtx8000,gpu_misc_v100',
        # 'account': 'cds',
        # 'qos': 'cds'
    })

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    print(args)

    trainer = Trainer(args)
    # trainer()
    job = executor.submit(trainer)

    print(job.job_id)


if __name__ == "__main__":
    main()
