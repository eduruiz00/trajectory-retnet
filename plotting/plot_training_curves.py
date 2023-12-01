import numpy as np
import pandas as pd
import trajectory.utils as utils
import os
import matplotlib
import matplotlib.pyplot as plt

class Parser(utils.Parser):
    config: str = 'config.offline'
    dataset: str = 'halfcheetah-medium-v2'
    folder_retnet: str
    folder_gpt: str

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
    ax.plot(gpt_curve[0], gpt_curve[1], label="GPT")
    ax.plot(retnet_curve[0], retnet_curve[1], label="RetNet")
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if verbose:
        plt.show()
    plt.savefig(savepath)
    plt.close()


if __name__ == '__main__':

    #################
    ## latex
    #################
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # matplotlib.rc('text', usetex=True)
    #################

    gpt_path = os.path.join(args.logbase, args.dataset, args.gpt_loadpath, args.folder_gpt)
    retnet_path = os.path.join(args.logbase, args.dataset, args.retnet_loadpath, args.folder_retnet)

    train_loss_gpt = pd.read_csv(os.path.join(gpt_path, "learning_curves.csv"))[["iteration", "loss"]]
    train_loss_retnet = pd.read_csv(os.path.join(retnet_path, "learning_curves.csv"))[["iteration", "loss"]]

    plots_path = os.path.join(args.logbase, args.dataset, "plots")
    if not os.path.isdir(plots_path):
        os.mkdir(plots_path)
    plot_gpt_vs_retnet(
        (np.array(train_loss_gpt["iteration"]), np.array(train_loss_gpt["loss"])),
        (np.array(train_loss_retnet["iteration"]), np.array(train_loss_retnet["loss"])),
        "Training Curves",
        "Iterations",
        "Loss",
        os.path.join(plots_path, "learning_curves.svg")
    )

    validation_gpt = pd.read_csv(os.path.join(gpt_path, "total_reward_curves.csv"))[["epoch", "total_reward"]]
    validation_retnet = pd.read_csv(os.path.join(retnet_path, "total_reward_curves.csv"))[["epoch", "total_reward"]]
    plot_gpt_vs_retnet(
        (np.array(validation_gpt["epoch"]), np.array(validation_gpt["total_reward"])),
        (np.array(validation_retnet["epoch"]), np.array(validation_retnet["total_reward"])),
        "Validation Curves",
        "Epoch",
        "Total reward in a 300 episodes run",
        os.path.join(plots_path, "validation_curves.svg")
    )