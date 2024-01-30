import os
import trajectory.utils as utils
import pandas as pd
import numpy as np

class Parser(utils.Parser):
    config: str = 'config.offline'
    dataset: str = 'halfcheetah-medium-v2'
    folders_retnet: list = []
    folders_gpt: list = []

args = Parser().parse_args('plot', mkdir=False)

def extract_times(path, param, folders, model):
    for model_plans in os.listdir(path):
        if model in model_plans and param in model_plans:
            path = os.path.join(path, model_plans)
            break
    if not folders:
        all_dirs = [os.path.join(path, directory) for directory in os.listdir(path)]
        folders = [directory for directory in filter(os.path.isdir, all_dirs)]
    all_list = []
    for folder in folders:
        data = pd.read_csv(os.path.join(folder, "plan_times.csv"))
        all_list.append(data)
    all_array = np.array(all_list).squeeze()
    all_means = np.mean(all_array, axis=0)
    mean_step = all_means[1]
    std_step = all_means[2]
    mean_plan_time = all_means[3]
    std_plan_time = np.std(all_array, axis=0)[3]
    df = pd.DataFrame.from_dict({
        "mean_step": [mean_step],
        "std_step": [std_step],
        "mean_plan_time": [mean_plan_time],
        "std_plan_time": [std_plan_time]
    })
    df.to_csv(os.path.join(path, "times.csv"))

if __name__ == '__main__':
    params = ['H15', 'H10', 'H5']
    datasets = ['halfcheetah-medium-v2', 'walker2d-medium-v2', 'hopper-medium-v2']
    for dataset in datasets:
        for param in params:
            gpt_path = os.path.join(args.logbase, dataset, args.prefix)
            retnet_path = os.path.join(args.logbase, dataset, args.prefix)
            extract_times(gpt_path, param, args.folders_gpt, model="gpt")
            extract_times(gpt_path, param, args.folders_retnet, model="retnet")