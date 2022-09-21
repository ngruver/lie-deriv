import wandb
import time
import numpy as np

import sweep_configs.e2e_configs as e2e_cfgs
import sweep_configs.layerwise_configs as layerwise_cfgs

from submitit_e2e import main as submitit_e2e
from submitit_layerwise import main as submitit_layerwise

all_e2e_cfgs = [
    e2e_cfgs.core_models,
    # e2e_cfgs.other_models,
]

all_layerwise_cfgs = [
    # layerwise_cfgs.core_models
]

def main():
    for sweep_configuration in all_e2e_cfgs:
        sweep_id = wandb.sweep(sweep_configuration, project="LieDerivEquivariance")
        vals = [d['values'] for d in sweep_configuration["parameters"].values() if "values" in d]
        num_jobs = np.prod([len(v) for v in vals])
        
        print("Submitting {} jobs!".format(num_jobs))
        for _ in range(num_jobs):
            try:
                submitit_e2e({"sweep_id": sweep_id})
                time.sleep(1)
            except Exception as e:
                print(e)

    for sweep_configuration in all_layerwise_cfgs:
        sweep_id = wandb.sweep(sweep_configuration, project="LieDerivEquivariance")
        vals = [d['values'] for d in sweep_configuration["parameters"].values() if "values" in d]
        num_jobs = np.prod([len(v) for v in vals])
        
        print("Submitting {} jobs!".format(num_jobs))
        for _ in range(num_jobs):
            try:
                submitit_layerwise({"sweep_id": sweep_id})
                time.sleep(1)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    main()
