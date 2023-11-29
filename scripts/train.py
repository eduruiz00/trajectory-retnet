import os
import numpy as np
import torch
import datetime
import pdb
import pandas as pd

import trajectory.utils as utils
from trajectory.utils.timer import Timer
import trajectory.datasets as datasets
from trajectory.models.transformers import GPT
from evaluation import evaluate

class Parser(utils.Parser):
    dataset: str = 'bullet-halfcheetah-medium-v0'
    config: str = 'config.offline'
    exp_name: str = 'gpt/pretrained'

#######################
######## setup ########
#######################

args = Parser().parse_args('train')
plan_args = Parser().parse_args('plan')

#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)

sequence_length = args.subsampled_sequence_length * args.step
args.exp_name = args.gpt_exp_name

dataset_config = utils.Config(
    datasets.DiscretizedDataset,
    savepath=(args.savepath, 'data_config.pkl'),
    env=args.dataset,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    step=args.step,
    discount=args.discount,
    discretizer=args.discretizer,
)

dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

model_config = utils.Config(
    GPT,
    savepath=(args.savepath, 'model_config.pkl'),
    ## discretization
    vocab_size=args.N, block_size=block_size,
    ## architecture
    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## loss weighting
    action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
    ## dropout probabilities
    embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
)

model = model_config()
model.to(args.device)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=0,
    device=args.device,
    dataset=args.dataset,
)

trainer = trainer_config()

#######################
###### main loop ######
#######################
## scale number of epochs to keep number of updates constant
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
save_freq = int(n_epochs // args.n_saves)

df_times = pd.DataFrame(columns=['epoch', 'time_trainer', 'time_epoch', 'acc_time'])

training_timer = Timer()

curves_file = os.path.join(args.savepath, "total_reward_curves.csv")
df_empty = pd.DataFrame(columns=["epoch", "total_reward"])
df_empty.to_csv(curves_file, mode='w')

for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name} | time: {datetime.datetime.now()}')

    losses, time = trainer.train(model, dataset, starting_epoch=epoch)

    evaluate(model, dataset, trainer.writer, plan_args, training_epoch=epoch, max_episode_steps=args.training_episode_steps, render=False, curves_file=curves_file)
    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    save_epoch = epoch // save_freq * save_freq
    statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
    print(f'Saving model to {statepath}')

    ## save state to disk
    state = model.state_dict()
    torch.save(state, statepath)

    acc_time, epoch_time = training_timer(flag=True)
    df_times = pd.concat([df_times, pd.DataFrame({
        'epoch': [epoch],
        'time_trainer (s)': [time],
        'time_epoch (s)': [epoch_time],
        'acc_time (min)': [acc_time / 60],
    })], ignore_index=True)
    df_times.to_csv(os.path.join(args.savepath, 'time_table.csv'))
