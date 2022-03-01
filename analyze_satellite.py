import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, copy, time, pickle
from urllib.request import urlretrieve
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import torch
import torch.nn as nn
import torch.nn.functional as F

from dissipative_hnns.models import MLP, DHNN, HNN, DHNN_sat, HNN_sat, MLP_sat
from dissipative_hnns.train import train, get_args
from dissipative_hnns.experiment_satellite.data import get_dataset, hamiltonian_fn
from dissipative_hnns.experiment_satellite.data import potential_energy_over_M_sat, kinetic_energy_over_M_sat, total_energy_over_M_sat

import random
import time

def print_stats(results):
    k = 3
    stats_last = lambda v: (np.mean(v[-k:]), np.std(v[-k:]))
    
    metrics = ['train_loss', 'test_loss']
    print("\t" + " & ".join(metrics), end='\n\t')
    for metric in metrics:
        print("{:.2e} \pm {:.2e}".format(*stats_last(results[metric])), end=' & ')
    
    # print("LaTeX format", end='\n\t')
    # for metric in metrics:
    #     print("{:.2e}".format(np.mean((results[metric])[-k:])), end=' & ')

def integrate_model(model, t_span, y0, **kwargs):
    
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,6)
        t = torch.zeros_like(x[...,:1])
        dx = model(x, t=t).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

# def hamiltonian_fn(coords):
#     k = 2.4  # this coefficient must be fit to the data
#     q, p = np.split(coords,2)
#     H = k*(1-np.cos(q)) + p**2 # pendulum hamiltonian
#     return H

name = 'satellite'
exp_dir = './experiment_satellite'
satellite_problem = True
data_percentage_usage = 1
mode = 'DHNN'
train_all = False

args = get_args()
# args.batch_size = 150
args.total_steps = 20000
#args.learning_rate = 5e-5
args.test_every = 5000

args.verbose = True
args.gpu_enable = True
args.gpu_select = 0
args.device = torch.device('cuda:' + str(args.gpu_select) if args.gpu_enable else 'cpu')

[f(args.seed) for f in [np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all]]
data = get_dataset(name, exp_dir, satellite_problem, data_percentage_usage)

# parameters to test
# hidden_dim_list = [250, 256, 260, 275, 300]
# learn_rate_list = [1e-5, 5e-5, 1e-4]
# batch_size_list = [128, 256, 512]

hidden_dim_list = [256]
learn_rate_list = [5e-5]
batch_size_list = [128]

for hidden_dim in hidden_dim_list:
    for learn_rate in learn_rate_list:
        for batch_size in batch_size_list:

            tic = time.time()

            if args.verbose:
                if mode == 'MLP':
                    print("\nTraining baseline model:")
                elif mode == 'HNN':
                    print("\nTraining HNN model:")
                else:
                    print("\nTraining D-HNN model:")
                print("hidden_dim: " + str(hidden_dim) + ", learn_rate: " + str(learn_rate) + ", batch_size: " + str(batch_size) + '\n')

            if train_all == False:
                if mode == 'MLP':
                    model = MLP(args.input_dim, 6, hidden_dim)
                    model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model
                elif mode == 'HNN':
                    model = HNN_sat(args.input_dim, hidden_dim)
                    model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model
                else:
                    model = DHNN_sat(args.input_dim, hidden_dim, assume_canonical_coords=True, device = args.device)
                    model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model
            else:
                model = MLP_sat(args.input_dim, 6, hidden_dim)
                model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model

                model = HNN_sat(args.input_dim, hidden_dim)
                model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model

                model = DHNN_sat(args.input_dim, hidden_dim, assume_canonical_coords=True, device = args.device)
                model_results = train(model, args, learn_rate, batch_size, data[0])  # training the model

                # print('D-HNN') ; print_stats(dhnn_results)
                # print('\nHNN') ; print_stats(hnn_results)
                # print('\nMLP') ; print_stats(mlp_results)

            print('Elapsed (s): ', time.time() - tic)

x = data[1]['x_test'] 
# Each line of 'x' is in the form (qx1, qx2, qy1, qy2, px1, px2, py1, py2) in original 2-body experiment and (qx1, qx2, qy1, qy2, qz1, qz2, px1, px2, py1, py2, pz1, pz2) in the satellite-problem experiment
TE = data[1]['energy_test']
KE = data[1]['ke_test']
PE = data[1]['pe_test']
lengths = data[1]['lengths_test']

lengths_num = len(lengths)
index = random.randint(0, lengths_num-1)
n = 0
trajectory_start = 1
while n < index:
    # print(n)
    trajectory_start = trajectory_start + lengths[n]
    n = n + 1
trajectory_end = trajectory_start + lengths[index] - 1
trajectory_states = x[trajectory_start:trajectory_end, :]
TE_trajectory = TE[trajectory_start:trajectory_end]
KE_trajectory = KE[trajectory_start:trajectory_end]
PE_trajectory = PE[trajectory_start:trajectory_end]

# draw trajectories

fig = plt.figure(figsize=[20,8], dpi=100)
ax = plt.subplot(1,2,1, projection='3d')

ax.plot3D(trajectory_states[:, 0], trajectory_states[:, 1], trajectory_states[:, 2], 'blue', label='Satellite trajectory')
ax.scatter3D(0, 0, 0, 'green', label='Point Earth')
ax.set_title('Ground Truth Trajectory', fontsize=20)
ax.set_xlabel('X (Normalised)', fontsize=15)
ax.set_ylabel('Y (Normalised)', fontsize=15)
ax.set_zlabel('Z (Normalised)', fontsize=15)
ax.legend(fontsize=7)

ax = fig.add_subplot(1, 2, 2)

ax.plot(TE_trajectory, 'green', label='Total Energy')
ax.plot(KE_trajectory, 'blue', label='Potential Energy')
ax.plot(PE_trajectory, 'orange', label='Kinetic Energy')
ax.set_title('Ground Truth Energy', fontsize=20)
ax.set_xlabel('Timestep', fontsize=15)
ax.set_ylabel('J (Normalised)', fontsize=15)
ax.legend(fontsize=7)

plt.show()

# get trajectory of true test data
# t_eval = np.squeeze( data['time_test'] - data['time_test'].min() )
t_span = [0,5]
x0 = x[trajectory_start]

trajectory_states_GT = trajectory_states
true_x = trajectory_states_GT[:, :3]

t_eval = np.linspace(t_span[0], t_span[1], trajectory_end - trajectory_start)

# # integrate along baseline vector field
# mlp_path = integrate_model(mlp_model, t_span, x0, t_eval=t_eval)
# mlp_x = mlp_path['y'].T

# # integrate along HNN vector field
# hnn_path = integrate_model(hnn_model, t_span, x0, t_eval=t_eval)
# hnn_x = hnn_path['y'].T

# # integrate along D-HNN vector field
# dhnn_path = integrate_model(dhnn_model, t_span, x0, t_eval=t_eval)
# dhnn_x = dhnn_path['y'].T

# integrate along D-HNN vector field
path = integrate_model(model, t_span, x0, t_eval=t_eval)
model_x = path['y'].T

# plotting
tpad = 7

fig = plt.figure(figsize=[20,8], dpi=100)

ax = plt.subplot(1,3,1, projection='3d')
ax.plot3D(true_x[:, 0], true_x[:, 1], true_x[:, 2], 'k', label='Ground truth trajectory')
# ax.plot3D(mlp_x[:, 0], mlp_x[:, 1], mlp_x[:, 2], 'r', label='Predicted (baseline) trajectory')
# ax.plot3D(hnn_x[:, 0], hnn_x[:, 1], hnn_x[:, 2], 'g', label='Predicted (HNN) trajectory')
# ax.plot3D(dhnn_x[:, 0], dhnn_x[:, 1], dhnn_x[:, 2], 'b', label='Predicted (D-HNN) trajectory')
label = 'Predicted (' + mode + ') trajectory' 
ax.plot3D(model_x[:, 0], model_x[:, 1], model_x[:, 2], 'b', label=label)
ax.scatter3D(0, 0, 0, 'green', label='Point Earth')
ax.set_title('Trajectories', fontsize=20)
ax.set_xlabel('X (Normalised)', fontsize=15)
ax.set_ylabel('Y (Normalised)', fontsize=15)
ax.set_zlabel('Z (Normalised)', fontsize=15)
ax.legend(fontsize=7)

# plt.title("Predictions", pad=tpad) ; plt.xlabel('$q$') ; plt.ylabel('$p$')
# plt.plot(true_x[:,0], true_x[:,1], 'k-', label='Ground truth', linewidth=2)
# plt.plot(mlp_x[:,0], mlp_x[:,1], 'r-', label='MLP', linewidth=2)
# plt.plot(hnn_x[:,0], hnn_x[:,1], 'g-', label='HNN', linewidth=2)
# plt.plot(dhnn_x[:,0], dhnn_x[:,1], 'b-', label='D-HNN (ours)', linewidth=2)
# plt.xlim(-1.2,2) ; plt.ylim(-1.2,2)
# plt.legend(fontsize=7)

ax = fig.add_subplot(1, 3, 2)
ax.plot(TE_trajectory, 'green', label='Total Energy (GT)')
ax.plot(KE_trajectory, 'blue', label='Potential Energy (GT)')
ax.plot(PE_trajectory, 'orange', label='Kinetic Energy (GT)')


TE_cal_list = []
PE_cal_list = []
KE_cal_list = []

for point in model_x:
    twobody_points = np.stack((point, np.array([0, 0, 0, 0, 0, 0])))
    TE_cal_list.append(total_energy_over_M_sat(twobody_points)[0])
    PE_cal_list.append(potential_energy_over_M_sat(twobody_points))
    KE_cal_list.append(kinetic_energy_over_M_sat(twobody_points))

label = 'Total Energy (' + mode + ')'
ax.plot(TE_cal_list, color='green', linestyle='dashed', label=label)
label = 'Potential Energy (' + mode + ')'
ax.plot(PE_cal_list, color='blue', linestyle='dashed', label=label)
label = 'Kinetic Energy (' + mode + ')'
ax.plot(KE_cal_list, color='orange', linestyle='dashed', label=label)
ax.set_title('Energy', fontsize=20)
ax.set_xlabel('Timestep', fontsize=15)
ax.set_ylabel('J (Normalised)', fontsize=15)
ax.legend(fontsize=7)

# ax = fig.add_subplot(1, 3, 3)
# ax.plot(settings['t_eval'], potential_energy_over_M_sat(base_orbit[:, 1:]), 'blue', label='Potential Energy')
# ax.plot(settings['t_eval'], kinetic_energy_over_M_sat(base_orbit[:, 1:]), 'orange', label='Kinetic Energy')
# ax.plot(settings['t_eval'], total_energy_over_M_sat(base_orbit[:, 1:])[0], 'green', label='Total Energy')
# ax.set_title('Neural Network (baseline) energy', fontsize=20)
# ax.set_xlabel('Timestep', fontsize=15)
# ax.set_ylabel('J (Normalised)', fontsize=15)
# ax.legend(fontsize=15)

ax = fig.add_subplot(1, 3, 3)
ax.set_title("MSE between coordinates", pad=tpad) ; ax.set_xlabel('Time step')
# ax.plot(t_eval, ((true_x[:, :3]-mlp_x[:, :3])**2).mean(-1), 'r-', label='MLP', linewidth=2)
# ax.plot(t_eval, ((true_x[:, :3]-hnn_x[:, :3])**2).mean(-1), 'g-', label='HNN', linewidth=2)
# ax.plot(t_eval, ((true_x[:, :3]-dhnn_x[:, :3])**2).mean(-1), 'b-', label='D-HNN', linewidth=2)
ax.plot(t_eval, ((true_x[:, :3]-model_x[:, :3])**2).mean(-1), 'b-', label=mode, linewidth=2)
ax.legend(fontsize=7)

# plt.subplot(1,3,3)
# plt.title("Total energy", pad=tpad)
# plt.xlabel('Time step')
# true_e = np.stack([hamiltonian_fn(c) for c in true_x])
# mlp_e = np.stack([hamiltonian_fn(c) for c in mlp_x])
# hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
# dhnn_e = np.stack([hamiltonian_fn(c) for c in dhnn_x])
# plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=2)
# plt.plot(t_eval, mlp_e, 'r-', label='MLP', linewidth=2)
# plt.plot(t_eval, hnn_e, 'g-', label='HNN', linewidth=2)
# plt.plot(t_eval, dhnn_e, 'b-', label='D-HNN (ours)', linewidth=2)
# plt.legend(fontsize=7)

plt.tight_layout() ; plt.show()
fig.savefig('./static/satellite_results.pdf')