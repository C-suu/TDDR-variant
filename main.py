"""Train and evaluate the six TDDR variants with the legacy Gym API."""

import argparse
import os

import gym
import numpy as np
import torch

import buffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_policy(policy, env_name, seed, discount=0.99, eval_episodes=10):
    """Return mean episode reward and mean discounted rollout return."""
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.0
    state_values = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state_value = 0.0
        step = 0
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state_value += reward * discount**step
            step += 1
        state_values.append(state_value)
    avg_reward /= eval_episodes
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.6f}")
    return avg_reward



def main():
    """Run one method, environment and training seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="DADC",
        choices=["DADC", "DASC", "SASC", "DADC-R", "DASC-R", "SASC-R"],
    )
    parser.add_argument("--env", default="Ant-v2")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--qweight", default=0.3, type=float)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--max_timesteps", default=1000000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    representation = args.policy.endswith("-R")
    if args.policy == "DADC":
        from DADC import DADC as Algorithm
    elif args.policy == "DASC":
        from DASC import DASC as Algorithm
    elif args.policy == "SASC":
        from SASC import SASC as Algorithm
    elif args.policy == "DADC-R":
        from DADC_R import DADC_R as Algorithm
    elif args.policy == "DASC-R":
        from DASC_R import DASC_R as Algorithm
    else:
        from SASC_R import SASC_R as Algorithm

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        q_weight=args.qweight,
    )
    if not representation:
        kwargs.update(device=device, discount=0.99, tau=0.005)
    policy = Algorithm(**kwargs)
    replay_buffer = buffer.ReplayBuffer(state_dim, action_dim)

    file_name = f"{args.policy}_{args.env}_{args.seed}_{args.qweight}"
    os.makedirs("results", exist_ok=True)
    if representation:
        os.makedirs("reward_results", exist_ok=True)
    print(
        f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, upsilon: {args.qweight}"
    )
    avg_reward = eval_policy(policy, args.env, args.seed)
    evaluations = [avg_reward]
    estimated_values = []

    state, done = env.reset(), False
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state
        episode_reward += reward
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        if done:
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.6f}"
            )
            state, done = env.reset(), False
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            avg_reward, true_value = eval_policy(policy, args.env, args.seed)
            evaluations.append(avg_reward)
            if representation:
                estimates = [value.cpu().detach().numpy() for value in estimated_values]
                np.save(f"reward_results/{file_name}_avg_rewards.npy", evaluations)
            else:
                np.save(f"results/{file_name}.npy", evaluations)


if __name__ == "__main__":
    main()
