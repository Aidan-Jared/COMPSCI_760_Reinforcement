import argparse
import torch
from utils import make_envs, take_action, Logger, Storage, VectorEnvVisualizer
from feudalnet import FeudalNetwork, feudal_loss
from feudaltransformer import FeudalTransformer
import gymnasium as gym
import time
import os
from tetris_gymnasium.envs.tetris import Tetris

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='ALE/MsPacman-v5',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=100,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e8),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=1,
                    help='toggle to feedforward ML architecture')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=40,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=8,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=.5,
                    help='Random Gausian goal for exploration')
parser.add_argument('--decay', type=float, default=.999,
                    help='how much eps decays')
parser.add_argument('--decay-limit', type=float, default=1e-3,
                    help='how much eps decays')
parser.add_argument('--layers', type=int, default=5,
                    help='transformer layers for manager')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

def experiment(args):
    save_steps =  list(torch.arange(0, int(args.max_steps), int(args.max_steps) // 10).numpy())
    logger = Logger(args.run_name, args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    envs = make_envs(args.env_name, args.num_workers, args)
    feudalnet = FeudalTransformer(
        num_workers=args.num_workers,
        input_dim=envs.single_observation_space.shape,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=envs.single_action_space.n,
        time_horizon=args.time_horizon,
        dilation=args.layers,
        device=device,
        mlp=args.mlp,
        args=args)
    
    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr = args.lr, alpha=.99, eps=1e-5)
    goals, states, masks, actions, rewards, zs = feudalnet.init_obj()

    x, info = envs.reset(seed=args.seed)
    step = 0
    visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
    scalar = torch.amp.GradScaler(device)
    batch_idx = 0
    while step < args.max_steps:
        feudalnet.repackage_hidden()
        goals, states, zs, actions, rewards = feudalnet.detach_sequences(goals, states, zs, actions, rewards)
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                'adv_m', 'adv_w'])

        for _ in range(args.num_steps):
            action_dist, goals, states, zs, value_m, value_w = feudalnet(x, zs, actions, rewards, goals, states, masks)
            action, logp, entropy = take_action(action_dist)
            x, reward, terminated, truncated, info = envs.step(action)
            actions.pop(0)
            rewards.pop(0)
            actions.append(torch.FloatTensor(action).unsqueeze(1).to(device))
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            if step % 160 == 0:
                visualizer.capture_frame(envs, step, action, reward, terminated, truncated, info)
            logger.log_episode(info, step)
            mask = torch.FloatTensor(1 - (terminated + truncated)).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': feudalnet.intrinsic_reward(states, goals, masks),
                'v_w': value_w,
                'v_m': value_m,
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': feudalnet.state_goal_cosine(states, goals, masks),
                'm': mask
            })

            step += args.num_workers

            if step % 640 == 0 and feudalnet.manager.eps > args.decay_limit:
                feudalnet.eps_decay()

        with torch.no_grad():
            *_, next_v_m, next_v_w = feudalnet(x, zs, actions, rewards, goals, states, masks, save = False)
            next_v_m = next_v_m.detach()
            next_v_w = next_v_w.detach()
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
        scalar.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
        optimizer.step()
        if batch_idx % 20 == 0:
            torch.cuda.synchronize()
            time.sleep(2)
        batch_idx += 1
        logger.log_scalars(loss_dict, step)
        if len(save_steps) > 0 and step > save_steps[0]:
            torch.save({
                'model': feudalnet.state_dict(),
                'args': args,
                'processor_mean': feudalnet.preprocessor.rms.mean,
                'optim': optimizer.state_dict()},
                f'models/{args.env_name[4:]}_{args.run_name}_step={step}.pt')
            logger.save()
            save_steps.pop(0)

    envs.close()
    torch.save({
    'model': feudalnet.state_dict(),
    'args': args,
    'processor_mean': feudalnet.preprocessor.rms.mean,
    'optim': optimizer.state_dict()},
    f'models/{args.env_name}_{args.run_name}_steps={step}.pt')

if __name__ == "__main__":
    run_name = args.run_name
    for seed in range(2):
        args.seed = seed
        args.run_name = f"{run_name}_seed={seed}"
        experiment(args)