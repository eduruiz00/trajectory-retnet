import numpy as np
import pandas as pd
import trajectory.utils as utils
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class Parser(utils.Parser):
    config: str = 'config.offline'
    dataset: str = 'halfcheetah-medium-v2'
    params: str = 'freq1_H15_beam32'
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

def smooth(y, window=20, poly=1):
    return savgol_filter(y, window, poly)

def plot_gpt_vs_retnet_multiple_curves(gpt_curves, retnet_curves, title, xlabel, ylabel, savepath, verbose=False):
    fig = plt.gcf()
    ax = plt.gca()
    cmap =  plt.get_cmap('tab10')
    gpt_means = [gpt_curve[1][0] for gpt_curve in gpt_curves]
    gpt_stds = [gpt_curve[1][1] for gpt_curve in gpt_curves]
    retnet_means = [retnet_curve[1][0] for retnet_curve in retnet_curves]
    retnet_stds = [retnet_curve[1][1] for retnet_curve in retnet_curves]
    for i in range(len(gpt_curves)):
        ax.plot(gpt_curves[i][0], smooth(gpt_means[i]), label="GPT", linestyle='--', linewidth=1.0, color=cmap(i+2))
        ax.fill_between(gpt_curves[i][0], smooth(gpt_means[i]-gpt_stds[i]), smooth(gpt_means[i]+gpt_stds[i]), alpha=0.3, color=cmap(i+2), linewidth=0)
        ax.plot(retnet_curves[i][0], smooth(retnet_curves[i][1][0]), label="RetNet", linestyle='-', linewidth=1.0, color=cmap(i+2))
        ax.fill_between(retnet_curves[i][0], smooth(retnet_means[i]-retnet_stds[i]), smooth(retnet_means[i]+retnet_stds[i]), alpha=0.3, color=cmap(i+2), linewidth=0)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if verbose:
        plt.show()
    plt.savefig(savepath, dpi=800)
    plt.close()
    
def plot_multiple_curves(curves, labels, title, xlabel, ylabel, savepath, verbose=False):
    fig = plt.gcf()
    ax = plt.gca()
    cmap =  plt.get_cmap('tab10')
    means = [curve[1][0] for curve in curves]
    stds = [curve[1][1] for curve in curves]
    for i in range(len(curves)):
        ax.plot(curves[i][0], smooth(means[i]), label="H="+labels[i], linewidth=1.0, color=cmap(i+2))
        ax.fill_between(curves[i][0], smooth(means[i]-stds[i]), smooth(means[i]+stds[i]), alpha=0.3, color=cmap(i+2), linewidth=0)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if verbose:
        plt.show()
    plt.savefig(savepath, dpi=800)
    plt.close()

def plot_gpt_vs_retnet(gpt_curve, retnet_curve, title, xlabel, ylabel, savepath, verbose=False):
    fig = plt.gcf()
    ax = plt.gca()
    gpt_mean= gpt_curve[1][0]
    gpt_std = gpt_curve[1][1]
    retnet_mean = retnet_curve[1][0]
    retnet_std = retnet_curve[1][1]
    ax.plot(gpt_curve[0], smooth(gpt_mean), label="GPT")
    ax.fill_between(gpt_curve[0], smooth(gpt_mean-gpt_std), smooth(gpt_mean+gpt_std), alpha=0.3)
    ax.plot(retnet_curve[0], smooth(retnet_mean), label="RetNet")
    ax.fill_between(retnet_curve[0], smooth(retnet_mean-retnet_std), smooth(retnet_mean+retnet_std), alpha=0.3)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if verbose:
        plt.show()
    plt.savefig(savepath)
    plt.close()

def explore_folders(path, params, folders, model):
    all_steps = []
    all_rewards = []
    all_total_rewards = []
    all_scores = []
    for model_plans in os.listdir(path):
        if model in model_plans and params in model_plans:
            path = os.path.join(path, model_plans)
            break
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
    mean = np.mean(data_array, axis=-2)
    std = np.std(data_array, axis=-2)
    return  np.stack([mean, std], axis=-2)

if __name__ == '__main__':

    #################
    ## latex
    #################
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    #################

    gpt_path = os.path.join(args.logbase, args.dataset, args.prefix)
    retnet_path = os.path.join(args.logbase, args.dataset, args.prefix)

    all_params = ['freq1_H15_beam32', 'freq1_H5_beam32']
    all_gpt_steps = []
    all_gpt_rewards = []
    all_gpt_total_rewards = []
    all_gpt_scores = []
    all_retnet_steps = []
    all_retnet_rewards = []
    all_retnet_total_rewards = []
    all_retnet_scores = []
    for params in all_params:
        gpt_steps, gpt_rewards, gpt_total_rewards, gpt_scores = explore_folders(gpt_path, params, args.folders_gpt, model="gpt")
        retnet_steps, retnet_rewards, retnet_total_rewards, retnet_scores = explore_folders(retnet_path, params, args.folders_retnet, model="retnet")
        all_gpt_steps.append(gpt_steps)
        all_gpt_rewards.append(gpt_rewards)
        all_gpt_total_rewards.append(gpt_total_rewards)
        all_gpt_scores.append(gpt_scores)
        all_retnet_steps.append(retnet_steps)
        all_retnet_rewards.append(retnet_rewards)
        all_retnet_total_rewards.append(retnet_total_rewards)
        all_retnet_scores.append(retnet_scores)

    plots_path = os.path.join(args.logbase, args.dataset, "plots")
    if not os.path.isdir(plots_path):
        os.mkdir(plots_path)
    
    xy_gpt_rewards = [(np.array(gpt_steps[0]), mean_std) for gpt_steps, mean_std in zip(all_gpt_steps, extract_mean_and_std(all_gpt_rewards))]
    xy_retnet_rewards = [(np.array(retnet_steps[0]), mean_std) for retnet_steps, mean_std in zip(all_retnet_steps, extract_mean_and_std(all_retnet_rewards))]
    
    xy_gpt_total_rewards = [(np.array(gpt_steps[0]), mean_std) for gpt_steps, mean_std in zip(all_gpt_steps, extract_mean_and_std(all_gpt_total_rewards))]
    xy_retnet_total_rewards = [(np.array(retnet_steps[0]), mean_std) for retnet_steps, mean_std in zip(all_retnet_steps, extract_mean_and_std(all_retnet_total_rewards))]
    
    xy_gpt_scores = [(np.array(gpt_steps[0]), mean_std) for gpt_steps, mean_std in zip(all_gpt_steps, extract_mean_and_std(all_gpt_scores))]
    xy_retnet_scores = [(np.array(retnet_steps[0]), mean_std) for retnet_steps, mean_std in zip(all_retnet_steps, extract_mean_and_std(all_retnet_scores))]

    labels = [param.split("_")[1][1:] for param in all_params]

    plot_gpt_vs_retnet(
        xy_gpt_rewards[0],
        xy_retnet_rewards[0],
        args.dataset.split("-")[0].capitalize() + " Reward Curves",
        "Steps in Episode",
        "Reward",
        os.path.join(plots_path, args.dataset + "_" + all_params[0] + "_reward_curves.png")
    )

    plot_gpt_vs_retnet(
        xy_gpt_total_rewards[0],
        xy_retnet_total_rewards[0],
        args.dataset.split("-")[0].capitalize() + " Total Reward Curves",
        "Steps in Episode",
        "Accumulated Reward",
        os.path.join(plots_path, args.dataset + "_" + all_params[0] + "_total_reward_curves.png")
    )
    
    plot_gpt_vs_retnet(
        xy_gpt_scores[0],
        xy_retnet_scores[0],
        args.dataset.split("-")[0].capitalize() + " Normalized Total Reward Curves",
        "Steps in Episode",
        "Score",
        os.path.join(plots_path, args.dataset + "_" + all_params[0] + "_score_curves.png")
    )

    plot_multiple_curves(
        xy_gpt_rewards,
        labels,
        args.dataset.split("-")[0].capitalize() + " Reward Curves GPT",
        "Steps in Episode",
        "Reward",
        os.path.join(plots_path, args.dataset + "_gpt_horizons_reward_curves.png")
    )

    plot_multiple_curves(
        xy_retnet_rewards,
        labels,
        args.dataset.split("-")[0].capitalize() + " Reward Curves RetNet",
        "Steps in Episode",
        "Reward",
        os.path.join(plots_path, args.dataset + "_retnet_horizons_reward_curves.png")
    )

    # plot_gpt_vs_retnet_multiple_curves(
    #     xy_gpt_total_rewards,
    #     xy_retnet_total_rewards,
    #     "Total Reward Curves",
    #     "Steps in Episode",
    #     "Accumulated Reward",
    #     os.path.join(plots_path, "all_total_reward_curves.png")
    # )

    # plot_gpt_vs_retnet_multiple_curves(
    #     xy_gpt_scores,
    #     xy_retnet_scores,
    #     "Normalized Total Reward Curves",
    #     "Steps in Episode",
    #     "Score",
    #     os.path.join(plots_path, "all_score_curves.png")
    # )