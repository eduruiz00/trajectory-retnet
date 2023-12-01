import numpy as np
import pandas as pd
import trajectory.utils as utils
import os
import matplotlib
import matplotlib.pyplot as plt

class Parser(utils.Parser):
    config: str = 'config.offline'
    dataset: str = 'halfcheetah-medium-v2'
    folders_retnet: list = []
    folders_gpt: list = []

args = Parser().parse_args('plot', mkdir=False)

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_gpt_vs_retnet(gpt_curve, retnet_curve, title, xlabel, ylabel, savepath, verbose=False):
    fig = plt.gcf()
    ax = plt.gca()
    gpt_mean= gpt_curve[1][0]
    gpt_std = gpt_curve[1][1]
    retnet_mean = retnet_curve[1][0]
    retnet_std = retnet_curve[1][1]
    ax.plot(gpt_curve[0], gpt_mean, label="GPT")
    ax.fill_between(gpt_curve[0], gpt_mean-gpt_std, gpt_mean+gpt_std, alpha=0.3)
    ax.plot(retnet_curve[0], retnet_curve[1][0], label="RetNet")
    ax.fill_between(retnet_curve[0], retnet_mean-retnet_std, retnet_mean+retnet_std, alpha=0.3)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if verbose:
        plt.show()
    plt.savefig(savepath)
    plt.close()

def explore_folders(path, folders, model):
    all_steps = []
    all_rewards = []
    all_total_rewards = []
    all_scores = []
    for model_plans in os.listdir(path):
        if model in model_plans:
            path = os.path.join(path, model_plans)
    if not folders:
        folders = os.listdir(path)
    for folder in folders:
        data = pd.read_csv(os.path.join(path, folder, "plan_curves.csv"))
        all_rewards.append(data["reward"])
        all_total_rewards.append(data["total_reward"])
        all_scores.append(data["score"])
        all_steps.append(data["step"])
    return all_steps, all_rewards, all_total_rewards, all_scores

def extract_mean_and_std(data):
    data_array = np.stack(data, axis=0)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    return mean, std

if __name__ == '__main__':

    #################
    ## latex
    #################
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)
    #################

    gpt_path = os.path.join(args.logbase, args.dataset, args.prefix)
    retnet_path = os.path.join(args.logbase, args.dataset, args.prefix)

    gpt_steps, gpt_rewards, gpt_total_rewards, gpt_scores = explore_folders(gpt_path, args.folders_gpt, model="gpt")
    retnet_steps, retnet_rewards, retnet_total_rewards, retnet_scores = explore_folders(retnet_path, args.folders_retnet, model="retnet")

    plots_path = os.path.join(args.logbase, args.dataset, "plots")
    if not os.path.isdir(plots_path):
        os.mkdir(plots_path)
    plot_gpt_vs_retnet(
        (np.array(gpt_steps[0]), extract_mean_and_std(gpt_rewards)),
        (np.array(retnet_steps[0]), extract_mean_and_std(retnet_rewards)),
        "Reward Curves",
        "Steps in episode",
        "reward",
        os.path.join(plots_path, "reward_curves.png")
    )

    plot_gpt_vs_retnet(
        (np.array(gpt_steps[0]), extract_mean_and_std(gpt_total_rewards)),
        (np.array(retnet_steps[0]), extract_mean_and_std(retnet_total_rewards)),
        "Total Reward Curves",
        "Steps in episode",
        "accumulated reward",
        os.path.join(plots_path, "total_reward_curves.png")
    )

    plot_gpt_vs_retnet(
        (np.array(gpt_steps[0]), extract_mean_and_std(gpt_scores)),
        (np.array(retnet_steps[0]), extract_mean_and_std(retnet_scores)),
        "Normalized Total Reward Curves",
        "Steps in episode",
        "score",
        os.path.join(plots_path, "score_curves.png")
    )