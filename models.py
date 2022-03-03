# Dissipative Hamiltonian Neural Networks
# Andrew Sosanya, Sam Greydanus | 2020

import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim):
      super(MLP, self).__init__()
      self.lin_1 = nn.Linear(input_dim, hidden_dim)
      self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
      self.lin_3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t=None):
      inputs = torch.cat([x, t], axis=-1) if t is not None else x
      h = self.lin_1(inputs).tanh() 
      h = h + self.lin_2(h).tanh()
      y_hat = self.lin_3(h)
      return y_hat

class MLP_sat(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim):
      super(MLP_sat, self).__init__()
      self.lin_1 = nn.Linear(input_dim, hidden_dim)
      self.lin_2 = nn.Linear(hidden_dim, hidden_dim)
      self.lin_3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, t=None):
      inputs = x
      h = self.lin_1(inputs).tanh() 
      h = h + self.lin_2(h).tanh()
      y_hat = self.lin_3(h)
      return y_hat

# class DHNN(nn.Module):
#   def __init__(self, input_dim, hidden_dim):
#     super(DHNN, self).__init__()  # Inherit the methods of the Module constructor
#     self.mlp = MLP(input_dim, 2, hidden_dim)  # Instantiate an MLP for learning the conservative component
    
#   def forward(self, x, t=None, as_separate=False): 
#     inputs = torch.cat([x, t], axis=-1) if t is not None else x
#     output = self.mlp(inputs)  # Bx2 Get the scalars from our baseline model
#     D,H = output[...,0], output[...,1]  # Separate out the Dissipative (D) and Hamiltonian (H) functions

#     irr_component = torch.autograd.grad(D.sum(), x, create_graph=True)[0]  # Take their gradients
#     rot_component = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

#     # For H, we need the symplectic gradient, and therefore
#     #   we split our tensor into 2 and swap the chunks.
#     dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1]//2, dim=1)
#     q_dot_hat, p_dot_hat = dHdp, -dHdq
#     rot_component = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
#     if as_separate:
#         return irr_component, rot_component  # Return the two fields seperately, or return the composite field. 

#     return irr_component + rot_component  # return decomposition if as_separate else sum of fields


class DHNN(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(DHNN, self).__init__()  # Inherit the methods of the Module constructor
    self.mlp_h = MLP(input_dim, 1, hidden_dim)  # Instantiate an MLP for learning the conservative component
    self.mlp_d = MLP(input_dim, 1, hidden_dim)  # Instantiate an MLP for learning the dissipative component
    
  def forward(self, x, t=None, as_separate=False): 
    inputs = torch.cat([x, t], axis=-1) if t is not None else x
    D = self.mlp_d(inputs)
    H = self.mlp_h(inputs)

    irr_component = torch.autograd.grad(D.sum(), x, create_graph=True)[0]  # Take their gradients
    rot_component = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

    # For H, we need the symplectic gradient, and therefore
    #   we split our tensor into 2 and swap the chunks.
    dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1]//2, dim=1)
    q_dot_hat, p_dot_hat = dHdp, -dHdq
    rot_component = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
    if as_separate:
        return irr_component, rot_component  # Return the two fields seperately, or return the composite field. 

    return irr_component + rot_component  # return decomposition if as_separate else sum of fields

class DHNN_sat(nn.Module):
  def __init__(self, input_dim, hidden_dim, assume_canonical_coords=True, device = None):
    super(DHNN_sat, self).__init__()  # Inherit the methods of the Module constructor
    self.mlp_h = MLP_sat(input_dim, 1, hidden_dim)  # Instantiate an MLP for learning the conservative component
    self.mlp_d = MLP_sat(input_dim, 1, hidden_dim)  # Instantiate an MLP for learning the dissipative component
    self.assume_canonical_coords = assume_canonical_coords
    self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor

  def forward(self, x, t=None, as_separate=False): 
    inputs = x
    D = self.mlp_d(inputs)
    H = self.mlp_h(inputs)

    irr_component = torch.autograd.grad(D.sum(), x, create_graph=True)[0]  # Take their gradients
    # irr_component = irr_component @ self.M.t()
    rot_component = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
    # rot_component = rot_component @ self.M.t()

    # For H, we need the symplectic gradient, and therefore
    #   we split our tensor into 2 and swap the chunks.
    dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1]//2, dim=1)
    q_dot_hat, p_dot_hat = dHdp, -dHdq
    rot_component = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
    if as_separate:
        return irr_component, rot_component  # Return the two fields seperately, or return the composite field. 

    return irr_component + rot_component  # return decomposition if as_separate else sum of fields

  def permutation_tensor(self,n):
      M = None
      if self.assume_canonical_coords:
          M = torch.eye(n)
          M = torch.cat([M[n//2:], -M[:n//2]])
      else:
          '''Constructs the Levi-Civita permutation tensor'''
          M = torch.ones(n,n) # matrix of ones
          M *= 1 - torch.eye(n) # clear diagonals
          M[::2] *= -1 # pattern of signs
          M[:,::2] *= -1
  
          for i in range(n): # make asymmetric
              for j in range(i+1, n):
                  M[i,j] *= -1
      return M


class HNN(nn.Module): 
  def __init__(self, input_dim, hidden_dim):
    super(HNN, self).__init__()  # Inherit the methods of the Module constructor
    self.mlp = MLP(input_dim, 1, hidden_dim)  # Instantiate an instance of our baseline model.
    
  def forward(self, x, t=None):
    inputs = torch.cat([x, t], axis=-1) if t is not None else x
    output = self.mlp(inputs)  # Bx2 Get the scalars from our baseline model

    H = output[...,0]  # Separate out the Dissipative (D) and Hamiltonian (H) functions
    H_grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

    # For H, we need the symplectic gradient, and therefore
    #   we split our tensor into 2 and swap the chunks.
    dHdq, dHdp = torch.split(H_grads, H_grads.shape[-1]//2, dim=1)
    q_dot_hat, p_dot_hat = dHdp, -dHdq
    H_hat = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
    return H_hat

class HNN_sat(nn.Module): 
  def __init__(self, input_dim, hidden_dim):
    super(HNN_sat, self).__init__()  # Inherit the methods of the Module constructor
    self.mlp = MLP_sat(input_dim, 1, hidden_dim)  # Instantiate an instance of our baseline model.
    
  def forward(self, x, t=None):
    inputs = x
    output = self.mlp(inputs)  # Bx2 Get the scalars from our baseline model

    H = output[...,0]  # Separate out the Dissipative (D) and Hamiltonian (H) functions
    H_grads = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

    # For H, we need the symplectic gradient, and therefore
    #   we split our tensor into 2 and swap the chunks.
    dHdq, dHdp = torch.split(H_grads, H_grads.shape[-1]//2, dim=1)
    q_dot_hat, p_dot_hat = dHdp, -dHdq
    H_hat = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
    return H_hat
