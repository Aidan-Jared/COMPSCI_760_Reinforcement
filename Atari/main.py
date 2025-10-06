import argparse
import torch
from utils import make_envs, take_action, Logger, Storage, VectorEnvVisualizer
from feudalnet import FeudalNetwork, Qlearn
from feudaltransformer import FeudalTransformer
import gymnasium as gym
from train import Train
from RND import RNDModel

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='ALE/MsPacman-v5',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=1000,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e8),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.02,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=0,
                    help='toggle to feedforward ML architecture')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=40,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=128,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.95,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.99,
                    help="discount factor manager"),
parser.add_argument('--alpha', type=float, default=.2,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=.9,
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')
parser.add_argument('--decay', type=float, default=.9985,
                    help='how much eps decays')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='feudalv4',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')
parser.add_argument('--model', type=str, choices=['feudal', 'feudalTransformer', 'qlearn'],
                    default='feudal', help="which model to train")
parser.add_argument('--decay-limit', type=float, default=2e-1,
                    help='how much eps decays')

# QLEARN SPECIFIC PARAMETERS
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for Q-learning')
parser.add_argument('--target-update', type=int, default=10000, help='steps between target syncs')
parser.add_argument('--eps-decay-freq', type=int, default=1280, help='steps between epsilon decays')

parser.add_argument('--maximal', type=bool, default=False, help='use maximal reward')

args = parser.parse_args()

def experiment(args):
    # save_steps =  list(torch.arange(0, int(args.max_steps), int(args.max_steps) // 10).numpy())
    logger = Logger(args.run_name, args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using Cuda Device: {device}")
    args.device = device
    torch.manual_seed(args.seed)
    rnd_model = RNDModel(device=device) # a seperate model to encourage exploration, less seen states give higher rewards, as states are more common, reduce the reward
    if torch.cuda.is_available() and args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    envs = make_envs(args.env_name, args.num_workers, args, rnd_model=rnd_model)
    n_actions = envs.single_action_space.n
    if args.model == 'feudal':
        feudalnet = FeudalNetwork(
            num_workers=args.num_workers,
            input_dim=envs.single_observation_space.shape,
            hidden_dim_manager=args.hidden_dim_manager,
            hidden_dim_worker=args.hidden_dim_worker,
            n_actions=n_actions,
            time_horizon=args.time_horizon,
            dilation=args.dilation,
            device=device,
            mlp=args.mlp,
            args=args)
        optimizer = torch.optim.RMSprop([
            {'params': feudalnet.manager.parameters(), 'lr': args.lr * .7},
            {'params': feudalnet.worker.parameters(), 'lr': args.lr},
            {'params': feudalnet.perception.parameters(), 'lr': args.lr},
        ], lr= args.lr, alpha=.99, eps=1e-5)
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
        
        optimizer = torch.optim.RMSprop([
            {'params': feudalnet.manager.parameters(), 'lr': args.lr / 2},
            {'params': feudalnet.worker.parameters(), 'lr': args.lr},
            {'params': feudalnet.perception.parameters(), 'lr': args.lr},
        ], lr= args.lr, alpha=.99, eps=1e-5)
    else:
        feudalnet = Qlearn(
            input_dim=envs.single_observation_space.shape,
            hidden_dim= args.hidden_dim_manager,
            n_actions=n_actions,
            device=device,
            mlp=args.mlp,
        )
        optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr = args.lr, alpha=.99, eps=1e-5)
    
    train = Train(args, feudalnet, optimizer, envs, logger, rnd_model)
    
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