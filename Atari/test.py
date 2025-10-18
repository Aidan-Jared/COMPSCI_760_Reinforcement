import argparse
import torch
from feudaltransformer import FeudalTransformer
from feudalnet import FeudalNetwork, Qlearn
import gymnasium as gym
from utils import make_envs, take_action, Storage, VectorEnvVisualizer
import ale_py
gym.register_envs(ale_py)
import json

parser = argparse.ArgumentParser(description='Feudal Nets')

parser.add_argument('--model', type=str, default='models/MsPacman-v5_feudalv6_seed=0_step=10000384.pt',
                    help='path to model save data')

arg = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_feudal(model, args, envs, iter):
    terminated, truncated = [False, False], [False, False]
    total_reward = 0
    goals, states, masks = model.init_obj()
    step = 0 
    x, info = envs.reset()
    # if iter == 0:
    #         hidden_m = model.hidden_m
    #         hidden_w = model.hidden_w
    #         torch.onnx.export(
    #             model,
    #             (torch.as_tensor(x), torch.stack(goals), torch.stack(states), masks[-1], False, hidden_m, hidden_w),
    #             'model.feudalModel.onnx',
    #             input_names=['current_state', 'goals', 'states', 'mask', 'save', 'hidden_m', 'hidden_w'],
    #             output_names= ['action_dist', 'goals', 'states', 'value_m', 'value_w'],
    #             dynamo=True,
    #             dynamic_shapes=[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, tuple, tuple],
    #             verify=True,
    #             optimize=True
    #         )
    while not terminated[0] and not truncated[0]:
        action_dist, goals, states, value_m, value_w = model(x, goals, states, masks[-1])
        action = torch.argmax(action_dist, dim=-1).cpu()
        # action = torch.multinomial(action_dist, num_samples=1).cpu().numpy()[0]
        # action, logp, entropy = take_action(action_dist)
        x, reward, terminated, truncated, info = envs.step(action)
        if step % 4 == 1:
            visualizer.capture_frame(envs, step, action, reward, terminated, truncated, info)
        step += 1
        total_reward = info['total_reward']

    return total_reward


def test_qlearn(model, args, envs, iter):
    terminated, truncated = False, False
    total_reward = 0
    step = 0 
    x, info = envs.reset()
    if iter == 0:
        torch.onnx.export(
            model,
            torch.as_tensor(x),
            'model.qModel.onnx',
            input_names=['current_state'],
            output_names= ['qvalues'],
            dynamo=True
        )
    while not terminated and not truncated:
        action_dist = model(x)
        action = torch.argmax(action_dist, dim=-1).cpu()
        # action = torch.multinomial(action_dist, num_samples=1).cpu().numpy()[0]
        # action, logp, entropy = take_action(action_dist)
        x, reward, terminated, truncated, info = envs.step(action)
        if step % 4 == 1:
            visualizer.capture_frame(envs, step, action, reward, terminated, truncated, info)
        step += 1
        total_reward = info['total_reward']
    return total_reward
    


if __name__ == "__main__":

    save_data = torch.load(arg.model, weights_only=False)
    args = save_data['args']
    model_weights = save_data['model']
    # args.num_workers = 2
    envs = make_envs(args.env_name, args.num_workers, args, train=False)
    visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
    if args.model == 'feudal':
        model = FeudalNetwork(
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
        model.load_state_dict(model_weights)
        model.to(device)
        model.eval()
    elif args.model =='qlearn':
        model = Qlearn(
            input_dim=envs.single_observation_space.shape,
            hidden_dim= args.hidden_dim_manager,
            n_actions=envs.single_action_space.n,
            device=device,
            mlp=args.mlp,
            #init_weights=model_weights
        )
        model.load_state_dict(model_weights)
        model.to(device)
        model.eval()
    
    scores = dict()
    
    with torch.no_grad():
        for i in range(100):
            if args.model == 'feudal':
                reward = test_feudal(model, args, envs, i)
            elif args.model =='qlearn':
                reward = test_qlearn(model, args, envs, i)

            scores[i] = reward[0]
            print(reward)
    with open(f'scores/{args.model}_{args.env_name[4:]}', 'w') as r:
        json.dump(scores, r)