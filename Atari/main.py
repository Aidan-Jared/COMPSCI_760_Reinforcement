import argparse
import torch
from utils import make_envs, take_action, Logger, Storage, VectorEnvVisualizer
from feudalnet import FeudalNetwork, Qlearn, feudal_loss
from feudaltransformer import FeudalTransformer
import gymnasium as gym
from train import Train

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='ALE/MsPacman-v5',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=32,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=400,
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
parser.add_argument('--time-horizon', type=int, default=30,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=128,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.99,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.999,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=.75,
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')
parser.add_argument('--decay', type=float, default=.9999,
                    help='how much eps decays')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='feudalv2',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')
parser.add_argument('--model', type=str, choices=['feudal', 'feudalTransformer', 'qlearn'],
                    default='qlearn', help="which model to train")
parser.add_argument('--decay-limit', type=float, default=1e-3,
                    help='how much eps decays')

# QLEARN SPECIFIC PARAMETERS
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for Q-learning')
parser.add_argument('--target-update', type=int, default=10000, help='steps between target syncs')
parser.add_argument('--eps-decay-freq', type=int, default=1280, help='steps between epsilon decays')

args = parser.parse_args()

def experiment(args):
    # save_steps =  list(torch.arange(0, int(args.max_steps), int(args.max_steps) // 10).numpy())
    logger = Logger(args.run_name, args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using Cuda Device: {device}")
    args.device = device
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    envs = make_envs(args.env_name, args.num_workers, args)
    if args.model == 'feudal':
        feudalnet = FeudalNetwork(
            num_workers=args.num_workers,
            input_dim=envs.single_observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)
    elif args.model =='feudalTransformer':
        feudalnet = FeudalTransformer(
            num_workers=args.num_workers,
            input_dim=envs.single_observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=envs.single_action_space.n,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)
    else:
        feudalnet = Qlearn(
            input_dim=envs.single_observation_space.shape,
            hidden_dim= args.hidden_dim_manager,
            n_actions=envs.single_action_space.n,
            device=device,
            mlp=args.mlp,
        )
    
    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr = args.lr, alpha=.99, eps=1e-5)
    train = Train(args, feudalnet, optimizer, envs, logger)
    print(f"Using model {args.model}")
    if args.model == 'feudal':
        train.train_feudal()
    elif args.model == 'feudalTransformer':
        train.train_transformer()
    elif args.model == 'qlearn':
        train.train_qmodel()
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == "__main__":
    base = args.run_name
    for seed in range(2):
        args.seed = seed
        args.run_name = f"{base}_seed={seed}"
        experiment(args)