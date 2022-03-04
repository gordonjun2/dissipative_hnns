# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus

import torch
import numpy as np
import os, copy, time


# simplifies accessing the hyperparameters.
class ObjectView(object):
    def __init__(self, d): self.__dict__ = d
    
def get_args(as_dict=False):
  arg_dict = {'input_dim': 2*6,                         
              'hidden_dim': 256, # capacity
              'output_dim': 2,                        
              'learning_rate': 1e-2, 
              'test_every': 100,
              'print_every': 100000,
              'batch_size': 128,
              'train_split': 0.80,  # train/test dataset percentage
              'total_steps': 50000,  # because we have a synthetic dataset
              'device': 'cuda', # {"cpu", "cuda"} for using GPUs
              'seed': 42,
              'as_separate': False,
              'decay': 0,
              'verbose' : True}
  return arg_dict if as_dict else ObjectView(arg_dict)


def get_batch(v, step, args, batch_size):  # helper function for moving batches of data to/from GPU
  dataset_size, num_features = v.shape
  bix = (step*batch_size) % dataset_size
  v_batch = v[bix:bix + batch_size, :]  # select next batch
  return torch.tensor(v_batch, requires_grad=True,  dtype=torch.float32, device=args.device)


def train(model, args, learn_rate, batch_size, data):
  """ General training function"""
  model = model.to(args.device)  # put the model on the GPU
  optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=args.decay)  # setting the Optimizer
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000000, gamma=0.5, last_epoch=- 1, verbose=False)

  model.train()     # doesn't make a difference for now
  t0 = time.time()  # logging the time
  results = {'train_loss':[], 'test_loss':[], 'test_acc':[], 'global_step':0}  # Logging the results

  # initialise infinity loss first
  best_test_loss = float('inf')

  for step in range(args.total_steps):  # training loop 

    x, t, dx = [get_batch(data[k], step, args, batch_size) for k in ['x', 't', 'dx']]
    dx_hat = model(x, t=t)  # feeding forward
    loss = (dx-dx_hat).pow(2).mean()  # L2 loss function
    loss.backward(retain_graph=False); optimizer.step(); optimizer.zero_grad()  # backpropogation
    scheduler.step()

    results['train_loss'].append(loss.item())  # logging the training loss

    # Testing our data with desired frequency (test_every)
    if step % args.test_every == 0:

      x_test, t_test, dx_test = [get_batch(data[k], step=0, args=args, batch_size = batch_size)
                          for k in ['x_test', 't_test', 'dx_test']]
      dx_hat_test = model(x_test, t=t_test)  # testing our model.
      test_loss = (dx_test-dx_hat_test).pow(2).mean().item() # L2 loss

      results['test_loss'].append(test_loss)
    if step % args.print_every == 0 or step == args.total_steps-1:
      print('Current Learning Rate: ', scheduler.get_last_lr())
      print('step {}, dt {:.3f}, train_loss {:.2e}, test_loss {:.2e}'
            .format(step, time.time()-t0, loss.item(), test_loss)) #.item() is just the integer of PyTorch scalar. 
      t0 = time.time()

    if args.satellite_case and args.save_weights and step % args.print_every == 0:
      if test_loss < best_test_loss:
        path = '{}/{}-satellite-orbits-{}-step={}.tar'.format(args.save_dir, args.name, args.label, step)
        torch.save(model.state_dict(), path)
        best_test_loss = test_loss

  model = model.cpu()
  return results
