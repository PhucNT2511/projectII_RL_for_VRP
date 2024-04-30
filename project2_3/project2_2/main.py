import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from Models.actor import DRL4VRP
from Models.critic import StateCritic
from Tasks.vrp import VehicleRoutingDataset, reward, render

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))

def read_input_file(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    truck_capacity = int(lines[2].split(':')[1].strip())
    num_customers = int(lines[1].split(':')[1].strip())

    location_index = lines.index("LOCATION\n") + 1
    demand_index = lines.index("DEMAND_SECTION\n") + 1

    locations = []
    demands = []
    for i in range(location_index, demand_index - 1):
        parts = lines[i].split()
        locations.append([float(parts[1]), float(parts[2])])

    for i in range(demand_index, len(lines) - 1):
        if "DEPOT_SECTION" in lines[i] or "EOF" in lines[i]:
            break
        parts = lines[i].split()
        demands.append(int(parts[1]))

    max_demand = max(demands)
    max_load = truck_capacity

    # Scale coordinates
    scaler = MinMaxScaler()
    locations_scaled = scaler.fit_transform(locations)

    # Normalize demands
    demands = [demand / float(max_load) for demand in demands]

    return num_customers, max_load, max_demand, locations_scaled, demands

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        print(x0[0])
        print(static[0])
        print(dynamic[0])
        
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
            print(tour_indices[0])

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn, render_fn, 
          batch_size, actor_lr, critic_lr, max_grad_norm, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf
    actor_checkpoint, critic_checkpoint = None, None  # Initialize checkpoint paths
    for epoch in range(5):  ##the number of epochs
        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            #print('static_train 0: ',static[0])
            #print('dynamic_train 0: ',dynamic[0])
            x0 = x0.to(device) if len(x0) > 0 else None

            tour_indices, tour_logp = actor(static, dynamic, x0)
            #print(tour_indices)
            reward = reward_fn(static, tour_indices)

            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward.float() - critic_est.float())

            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn, valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)
            actor_checkpoint = save_path  # Update actor checkpoint path

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)
            critic_checkpoint = save_path  # Update critic checkpoint path

        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))
    return actor_checkpoint, critic_checkpoint
def train_vrp(args):
    print('Starting VRP training')
    #Load some data from input.txt
    num_customers, max_load, max_demand, locations, demands = read_input_file('/content/drive/MyDrive/prjII/project2_3/project2_2/input.txt')

    train_data = VehicleRoutingDataset(args.train_size, num_customers, max_load, max_demand,args.seed)
    print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size, num_customers, max_load, max_demand,args.seed+1)

    # Assigning static_size and dynamic_size
    train_data.static_size = 2
    train_data.dynamic_size = 2

    actor = DRL4VRP(train_data.static_size, train_data.dynamic_size, args.hidden_size, 
                    train_data.update_dynamic, train_data.update_mask, args.num_layers, args.dropout).to(device)
    print('Actor: {} '.format(actor))

    critic = StateCritic(train_data.static_size, train_data.dynamic_size, args.hidden_size).to(device)
    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = reward
    kwargs['render_fn'] = render
    actor_checkpoint, critic_checkpoint = None, None

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        actor_checkpoint, critic_checkpoint = train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size, num_customers, max_load, max_demand, args.seed + 2)
    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, reward, render, test_dir, num_plot=5)
    print('Average tour length: ', out)
    return actor_checkpoint, critic_checkpoint, actor

def test_vrp(input_file, actor):
    print('Starting VRP testing')

    # Load data from input file
    num_customers, max_load, max_demand, locations, demands = read_input_file(input_file)

    # Create dynamic_data from demands và max_load
    dynamic_data = torch.tensor([[1.0, demand] for demand in demands], dtype=torch.float).unsqueeze(0).transpose(1, 2).to(device)

    # Create static_data from locations
    static_data = torch.tensor(locations, dtype=torch.float).unsqueeze(0).transpose(1, 2).to(device)

    # choose 0 to be the depot
    x0 = torch.tensor(locations[0], dtype=torch.float).unsqueeze(0).unsqueeze(2).to(device)

    print(x0)
    print(static_data)
    print(dynamic_data)

    actor.eval()    #Take model actor to evaluating state

   # Use actor model to exec tour_indices
    with torch.no_grad():
        tour_indices, _ = actor.forward(static_data, dynamic_data, x0)
    
    # Print tour indices
    print('Computed tour indices:', tour_indices.tolist()) 

def random_data(num_customers, max_load, max_demand):
    """
    Generate random data for the Vehicle Routing Problem (VRP).
    
    Args:
        num_customers (int): Number of customers.
        max_load (float): Maximum load capacity of the vehicles.
        max_demand (float, optional): Maximum demand of each customer. If None, it will be set to max_load.
    
    Returns:
        static_data (torch.Tensor): Static data representing customer locations.
        dynamic_data (torch.Tensor): Dynamic data representing customer demands and vehicle loads.
        start_loc (torch.Tensor): Start location for the vehicles.
    """    
    # Create only one data sample
    dataset = VehicleRoutingDataset(num_samples=1, input_size=num_customers, max_load=max_load, max_demand= max_demand)

    # Take data from dataset
    static_data, dynamic_data, start_loc = dataset[0]
    # Add a dimension to normalize data (because of having only 1 sample)
    static_data = static_data.unsqueeze(0)
    dynamic_data = dynamic_data.unsqueeze(0)
    start_loc= start_loc.unsqueeze(0)
    return static_data, dynamic_data, start_loc, dataset

def test_vrp_with_random_data(actor_checkpoint, critic_checkpoint, num_customers, max_load, max_demand):
    print('Starting VRP testing with random data')

    # Implement random data
    static_data, dynamic_data, start_loc, dataset = random_data(num_customers, max_load, max_demand)
    static_data = static_data.to(device)
    dynamic_data = dynamic_data.to(device)
    start_loc = start_loc.to(device)
    print(static_data)
    print(dynamic_data)
    print(start_loc)

    #Establish some features of dataset 
    dataset.static_size = 2
    dataset.dynamic_size = 2

    # Define and load model actor
    actor = DRL4VRP(dataset.static_size, dataset.dynamic_size, 128, 
                    dataset.update_dynamic, dataset.update_mask, num_layers=1, dropout=0.1).to(device)
    actor.load_state_dict(torch.load(actor_checkpoint, map_location=device))
    actor.eval()

    # Use actor model to exec tour_indices
    with torch.no_grad():
        tour_indices, _ = actor.forward(static_data, dynamic_data, start_loc)
    
    # Print tour indices
    print('Computed tour indices:', tour_indices.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--num_nodes', default=10, type=int, help='Number of nodes (customers)')
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--train_size',default=1000000, type=int)
    parser.add_argument('--valid_size', default=1000, type=int)
    args = parser.parse_args()

    actor_checkpoint, critic_checkpoint, actor = train_vrp(args)
    print(actor_checkpoint)
    #Exec with input from input file after training model
    test_vrp('/content/drive/MyDrive/prjII/project2_3/project2_2/input.txt', actor)
    #exec with random input
    test_vrp_with_random_data(actor_checkpoint, critic_checkpoint, 10, 250, 106)