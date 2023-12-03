import json
import pdb
from os.path import join
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from trajectory.utils.timer import Timer
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)

def evaluate(
        model,
        dataset,
        writer,
        args,
        training_epoch=None,
        max_episode_steps=None,
        render=False,
        curves_file=None,
    ):
    env = datasets.load_environment(args.dataset)
    renderer = utils.make_renderer(args) if render else None
    timer = utils.timer.Timer()
    if curves_file is None:
        # plan
        curves_file = os.path.join(args.savepath, "plan_curves.csv")
        df_empty = pd.DataFrame(columns=["step", "reward", "total_reward", "score"])
        df_empty.to_csv(curves_file, mode='w')
    discretizer = dataset.discretizer
    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    value_fn = lambda x: discretizer.value_fn(x, args.percentile)
    preprocess_fn = datasets.get_preprocess_fn(env.name)

    observation = env.reset()
    total_reward = 0

    ## observations for rendering
    rollout = [observation.copy()]

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    T = max_episode_steps if max_episode_steps is not None else env.max_episode_steps
    timer = Timer()
    time_steps = []
    for t in range(T):

        observation = preprocess_fn(observation)

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                model, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
            )

        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        ## update rollout observations and context transitions
        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        if training_epoch is None:
            print(
                f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
                f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
            )

            writer.add_scalar('reward', reward, t)
            writer.add_scalar('total_reward', total_reward, t)
            writer.add_scalar('score', score, t)

            step_df = pd.DataFrame([[t,reward, total_reward, score]])
            step_df.to_csv(curves_file, mode='a', header = False)


        # visualization
        if render and (t % args.vis_freq == 0 or terminal or t == T):

            # save current plan
            renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), writer, sequence_recon, env.state_vector(), to_tensorboard_only=True, tag='plan', step=t)

            # save rollout thus far
            renderer.render_rollout(join(args.savepath, f'rollout.mp4'), writer, rollout, to_tensorboard_only=True, fps=80, tag='rollout', step=t)

        cum_time, time_step = timer(flag=True)
        time_steps.append(time_step)
        if terminal: break

        observation = next_observation
    
    if training_epoch is not None:
        writer.add_scalar('total_reward', total_reward, training_epoch)
        step_df = pd.DataFrame([[training_epoch, total_reward]])
        step_df.to_csv(curves_file, mode='a', header = False)

    return score, t, total_reward, terminal, time_steps