import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import math
import numpy as np
from scipy.integrate import odeint
import random
import pickle

# Tell it to use GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    print('Using GPU')
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)
    print('No GPU found, using cpu')

### Derivative

# Code to take the derivative with respect to the input.
def diff(u, t, order=1):
    # code adapted from neurodiffeq library
    # https://github.com/NeuroDiffGym/neurodiffeq/blob/master/neurodiffeq/neurodiffeq.py
    """The derivative of a variable with respect to another.
    """
    # ones = torch.ones_like(u)

    der = torch.cat([torch.autograd.grad(u[:, i].sum(), t, create_graph=True)[0] for i in range(u.shape[1])], 1)
    if der is None:
        print('derivative is None')
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):

        der = torch.cat([torch.autograd.grad(der[:, i].sum(), t, create_graph=True)[0] for i in range(der.shape[1])], 1)
        # print()
        if der is None:
            print('derivative is None')
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    return der

### NN

class MyNetwork_Ray_Tracing(nn.Module):
    """
    function to learn the hidden states derivatives hdot
    """
    def __init__(self, number_dims=100, number_dims_heads=100, depth_body=6,  N=1):
        """ number_dims is the number of nodes within each layer
        N is the number of heads
        """
        super(MyNetwork_Ray_Tracing, self).__init__()
        self.N=N
        self.depth_body= depth_body
        self.number_dims = number_dims
        # Tanh activation function
        self.nl = nn.Tanh()
        self.lin1 = nn.Linear(1,number_dims)
        self.lin2 = nn.ModuleList([nn.Linear(number_dims, number_dims)])
        self.lin2.extend([nn.Linear(number_dims, number_dims) for i in range(depth_body-1)])
        self.lina = nn.ModuleList([nn.Linear(number_dims, number_dims_heads)])
        self.lina.extend([nn.Linear(number_dims, number_dims_heads) for i in range(N-1)])
        # 4 outputs for x,y, p_x, p_y
        self.lout1= nn.ModuleList([nn.Linear(number_dims_heads, 4, bias=True)])
        self.lout1.extend([nn.Linear(number_dims_heads, 4, bias=True) for i in range(N-1)])
    def forward(self, t):
        x = self.lin1(t)
        x = self.nl(x)
        for m in range(self.depth_body):
          x = self.lin2[m](x)
          x = self.nl(x)
        d={}
        for n in range(self.N):
          xa= self.lina[n](x)
          d[n]= self.lout1[n](xa)
        return d




### Numerical solver

# Font sizes
lineW = 3
lineBoxW=2
font = {'size'   : 24}

plt.rc('font', **font)
# plt.rcParams['text.usetex'] = True


# Use below in the Scipy Solver   
# CHANGE
# A_ is .1 and sig is 0.03
def f_general(u, t, means_Gaussians, lam=1, sig=0.03, A_=.1):
    # unpack current values of u
    x, y, px, py = u  

    V=0
    Vx=0
    Vy=0

    # CHANGE
    A=A_

    for i in means_Gaussians:
      muX1=i[0]
      muY1=i[1]
      V+=  - A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2)
      Vx+= A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) * (x-muX1)/sig**2 
      Vy+= A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) * (y-muY1)/sig**2 
    
    # derivatives of x, y, px, py
    derivs = [px, py, -Vx, -Vy] 
    
    return derivs

# Scipy Solver   
# CHANGE
# A_ is .1 and sig is 0.03
def rayTracing_general(t, x0, y0, px0, py0, means_Gaussians, lam=1, sig=0.03, A_=.1):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f_general, u0, t, args=(means_Gaussians, lam, sig, A_,))
    xP = solPend[:,0];    yP  = solPend[:,1];
    pxP = solPend[:,2];   pyP = solPend[:,3]
    return xP, yP, pxP, pyP


# Code for training
# CHANGE
# sig=0.03, A_=.1
def N_heads_run_Gaussiann(sig=0.03, A_=.1, initial_x=0, final_t=1, means=[[7.51, 4.6], [8.78, 6.16]], alpha_=1, width_=40, width_heads=8, \
  epochs_=20000, grid_size=200, number_of_heads=1, PATH="models", print_legend=False, loadWeights=False, energy_conservation=True,\
   norm_clipping=True):
  '''
  means should be of the forms [[mu_x1,mu_y1],..., [mu_xn,mu_yn]]
  initial_x: is the (common) starting x value for our rays
  final_y: is the final time
  width_ is the width of the base
  width_heads is the width of each head
  epochs_ is the number of epochs we train the NN for 
  number_of_heads is the number of heads
  '''
  # We will time the process
  # Access the current time
  t0=time.time()

  # Set out tensor of times
  t=torch.linspace(0,final_t,grid_size,requires_grad=True).reshape(-1,1)

  # Number of epochs
  num_epochs = epochs_

  # We keep a log of the loss as a fct of the number of epochs
  loss_log=np.zeros(num_epochs)

  # For comparaison
  temp_loss=np.inf

  # Set up the network
  network = MyNetwork_Ray_Tracing(number_dims=width_, number_dims_heads=width_heads, N=number_of_heads)
  # Make a deep copy
  network2 = copy.deepcopy(network)

  # Specify optimizer and learning rate
  # TO DO: ADD A SCHEDULER ON THE LEARNING RATE
  # EVERY 100 epochs multiply the learning rate by 0.999
  optimizer = optim.Adam(network.parameters(),lr=1e-3)
  # Dictionary for the initial conditions
  ic={}
  # Dictionary for the initial energy for each initial conditions
  H0_init={}

  # Random create initial conditions (and add the opposite)
  for j in range(number_of_heads):
    # Initial conditions
    # CHANGE
    # Now the initial condition is in [0,1]
    initial_condition=random.randint(0,100)/100
    print('The initial condition (for y) is {}'.format(initial_condition))
    ic[j]=initial_condition

  # Keep track of the number of epochs
  total_epochs=0

  ## LOADING WEIGHTS PART if PATH file exists and loadWeights=True
  if loadWeights==True:
      print("We loaded the previous model")
      checkpoint=torch.load(PATH)
      device = torch.device("cuda")
      network.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      network.to(device)
      total_epochs+=checkpoint['total_epochs']
      print("We previously trained for {} epochs".format(total_epochs))
      print('The loss was:', checkpoint['loss'], 'achieved at epoch', checkpoint['epoch'])

  # Dictionary keeping track of the loss for each head
  losses_part={}
  for k in range(number_of_heads):
    losses_part[k]=np.zeros(num_epochs)

  # For every epoch...
  for ne in range(num_epochs):
      optimizer.zero_grad()
      # Random sampling
      t=torch.rand(grid_size,requires_grad=True)*final_t
      t, ind=torch.sort(t)
      t[0]=0
      t=t.reshape(-1,1)
      # Forward pass through the network
      d = network(t)
      # loss
      loss=0
      # for saving the best loss (of individual heads) 
      losses_part_current={}

      # For each head...
      for l in range(number_of_heads):
        # Get the current head
        head=d[l]
        # Get the corresponding initial condition
        initial_y=ic[l]
        
        # Outputs
        x=head[:,0]
        y=head[:,1]
        px=head[:,2]
        py=head[:,3]
        x=x.reshape((-1,1))
        y=y.reshape((-1,1))
        px=px.reshape((-1,1))
        py=py.reshape((-1,1))
        # Derivatives
        x_dot=diff(x,t,1)
        y_dot=diff(y,t,1)
        px_dot=diff(px,t,1)
        py_dot=diff(py,t,1)

        # Loss
        L1=((x_dot-px)**2).mean()
        L2=((y_dot-py)**2).mean()

        # For the other components of the loss, we need the potential V
        # and its derivatives
        ## Partial derivatives of the potential (updated below)
        partial_x=0
        partial_y=0

        ## Energy at the initial time (updated below)
        H_0=1/2
        H_curr=(px**2+py**2)/2
        for i in range(len(means)):
          # Get the current means
          mu_x=means[i][0]
          mu_y=means[i][1]

          # Building the potential and updating the partial derivatives
          # CHANGE
          potential=-A_*torch.exp(-(1/(2*sig**2))*((x-mu_x)**2+(y-mu_y)**2))
          # Partial wrt to x
          partial_x+=-potential*(x-mu_x)/sig**2 
          # Partial wrt to y
          partial_y+=-potential*(y-mu_y)/sig**2 

          # Updating the energy
          # CHANGE
          H_0+=-A_*math.exp(-(1/(2*sig**2))*((initial_x-mu_x)**2+(initial_y-mu_y)**2))
          H_curr+=-A_*torch.exp(-(1/(2**sig**2))*((x-mu_x)**2+(y-mu_y)**2))

        ## We can finally set the energy for head l
        H0_init[l]=H_0

        # Other components of the loss
        L3=((px_dot+partial_x)**2).mean()
        L4=((py_dot+partial_y)**2).mean()

        # Nota Bene: L1,L2,L3 and L4 are Hamilton's equations

        # Initial conditions taken into consideration into the loss
        ## Position
        L5=((x[0,0]-initial_x)**2)
        L6=((y[0,0]-initial_y)**2)
        ## Velocity
        L7=(px[0,0]-1)**2
        L8=(py[0,0]-0)**2

        # Could add the penalty that H is constant L9
        L9=((H_0-H_curr)**2).mean()
        if not energy_conservation:
          # total loss
          loss+=L1+L2+L3+L4+L5+L6+L7+L8
          # loss for current head
          lossl_val=L1+L2+L3+L4+L5+L6+L7+L8
        if energy_conservation:
          # total loss
          loss+=L1+L2+L3+L4+L5+L6+L7+L8+L9
          # loss for current head
          lossl_val=L1+L2+L3+L4+L5+L6+L7+L9

        # the loss for head l at epoch ne is stored
        losses_part[l][ne]=lossl_val

        # the loss for head l
        losses_part_current[l]=lossl_val

      # Backward
      loss.backward()

      # Here we perform clipping 
      # (source: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch)
      if norm_clipping:
        # Check that this is correct
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=500)

      optimizer.step()

      # the loss at epoch ne is stored
      loss_log[ne]=loss.item()

      # If it is the best loss so far, we update the best loss and saved the model
      if loss.item()<temp_loss:
        epoch_mini=ne+total_epochs
        network2=copy.deepcopy(network)
        temp_loss=loss.item()
        individual_losses_saved=losses_part_current

        
  try:
    print('The best loss we achieved was:', temp_loss, 'at epoch', epoch_mini)
  except UnboundLocalError:
    print("Increase number of epochs")

  maxi_indi=0
  for g in range(number_of_heads):
    if individual_losses_saved[g]>maxi_indi:
      maxi_indi=individual_losses_saved[g]  
  print('The maximum of the individual losses was {}'.format(maxi_indi))
  total_epochs+=num_epochs

  ### Save network2 here (to train again in the next cell) ######################
  torch.save({'model_state_dict': network2.state_dict(), 'loss':temp_loss,  
              'epoch':epoch_mini, 'optimizer_state_dict':optimizer.state_dict(),
              'total_epochs': total_epochs},PATH)
  ###############################################################################


  ########## Saving to a file  #####################################
  # Saving the network
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Network_state'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(network2.state_dict(),f)
  f.close()
  #################################################################

  # Forward pass (network2 is the best network now)
  d2=network2(t)

  ########## Saving to a file  #####################################
  # Saving the loss
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'loss'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(loss_log,f)
  f.close()

  # Saving the individual losses
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'losses_part'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(losses_part,f)
  f.close()
  #################################################################

  # Now plot the individual trajectories and the individual losses
  for m in range(number_of_heads):
    # Get head m
    uf=d2[m]
    # The x trajectory
    x_traj=uf[0]
    # The y trjaectory
    y_traj=uf[1]
    # The loss
    loss_=losses_part[m]
    
    ########## Saving to a file  #####################################
    # Saving the trajectories
    filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
    str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
    'energyconservation_'+str(energy_conservation)+\
    '_normclipping_'+str(norm_clipping)+'_'\
    'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
    'Trajectory_NN_x'+'.p'
    #os.mkdir(filename)
    f=open(filename,"wb")
    pickle.dump(uf.cpu().detach()[:,0],f)
    f.close()
    # Saving the trajectories
    filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
    str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
    'energyconservation_'+str(energy_conservation)+\
    '_normclipping_'+str(norm_clipping)+'_'\
    'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
    'Trajectory_NN_y'+'.p'
    #os.mkdir(filename)
    f=open(filename,"wb")
    pickle.dump(uf.cpu().detach()[:,1],f)
    f.close()
    # Saving the trajectories
    filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
    str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
    'energyconservation_'+str(energy_conservation)+\
    '_normclipping_'+str(norm_clipping)+'_'\
    'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+\
    'Trajectory_NN_px'+'.p'
    #os.mkdir(filename)
    f=open(filename,"wb")
    pickle.dump(uf.cpu().detach()[:,2],f)
    f.close()
    # Saving the trajectories
    filename = 'Head_'+str(m)+'Initial_x_'+str(initial_x)+'final_t_'+\
    str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
    'energyconservation_'+str(energy_conservation)+\
    '_normclipping_'+str(norm_clipping)+'_'\
    'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Trajectory_NN_py'+'.p'
    #os.mkdir(filename)
    f=open(filename,"wb")
    pickle.dump(uf.cpu().detach()[:,3],f)
    f.close()
    ################################################################# 

  ######


  # Initial conditions for y
  Max=max(ic)
  Min=min(ic)

  ########## Saving ###########
  # Saving the initial conditions
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+\
  str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Initial_conditions'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(ic,f)
  f.close()
  ############################# 

  # define the time
  Nt=500
  t = np.linspace(0,final_t,Nt)

  # For the comparaison between the NN solution and the numerical solution,
  # we need to have the points at the same time
  # Set our tensor of times
  t_comparaison=torch.linspace(0,final_t,Nt,requires_grad=True).reshape(-1,1)
  d_comparaison=network2(t_comparaison)

  ########## Saving ###########
  # Saving the initial conditions
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+\
  str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'d_comparaison'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(d_comparaison,f)
  f.close()
  ############################# 


  # Initial posiiton and velocity
  x0, px0, py0 =  0, 1, 0.; 
  # Initial y position
  Y0 = ic

  Min=0
  Max=1

  # Maximum and mim=nimum x at final time
  maximum_x=initial_x
  maximum_y=0
  minimum_y=0
  min_final=np.inf

  for i in range(number_of_heads):
      print('The initial condition used is', Y0[i])
      x, y, px, py = rayTracing_general(t, x0, Y0[i], px0, py0, means_cell)
      if x[-1]>maximum_x:
        maximum_x=x[-1]
      if x[-1]<min_final:
        min_final=x[-1]
      if min(y)<minimum_y:
        minimum_y=min(y)
      if max(y)>maximum_y:
        maximum_y=max(y)

      ########## Saving ###########
      # Saving the (numerical trajectories)
      filename = 'Head_'+str(i)+'Initial_x_'+str(initial_x)+\
      'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
      'energyconservation_'+str(energy_conservation)+\
      '_normclipping_'+str(norm_clipping)+'_'\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'TrajectoriesNumerical_x'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(x,f)
      f.close()
      # Saving the (numerical trajectories)
      filename = 'Head_'+str(i)+'Initial_x_'+str(initial_x)+\
      'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
      'energyconservation_'+str(energy_conservation)+\
      '_normclipping_'+str(norm_clipping)+'_'\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'TrajectoriesNumerical_y'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(y,f)
      f.close()
      # Saving the (numerical trajectories)
      filename = 'Head_'+str(i)+'Initial_x_'+str(initial_x)+\
      'final_t_'+str(final_t)+'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
      'energyconservation_'+str(energy_conservation)+\
      '_normclipping_'+str(norm_clipping)+'_'\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'TrajectoriesNumerical_px'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(px,f)
      f.close()
      # Saving the (numerical trajectories)
      filename = 'Head_'+str(i)+'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
      'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
      'energyconservation_'+str(energy_conservation)+\
      '_normclipping_'+str(norm_clipping)+'_'\
      'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'TrajectoriesNumerical_py'+'.p'
      #os.mkdir(filename)
      f=open(filename,"wb")
      pickle.dump(py,f)
      f.close()
      #############################



  y1=np.linspace(minimum_y-.1,maximum_y+.1,500); x1= np.linspace(-.1,maximum_x,500)
  x, y = np.meshgrid(x1, y1)
  
  ########## Saving ###########
  # Saving the means passed in
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Means'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(means_cell,f)
  f.close()

  # Saving the mesh grid - x1
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid-x1'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(x1,f)
  f.close()

  # Saving the mesh grid - y1
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid-y1'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(y1,f)
  f.close()

  # Saving the mesh grid
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+\
  'alpha_'+ str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(x,f)
  pickle.dump(y,f)
  f.close()
  #############################

  V=0
  Vx=0
  Vy=0

  # CHANGE
  sig=0.03
  A_=.1

  for i in means_cell:
    muX1=i[0]
    muY1=i[1]
    V+=  - A_*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) 

  ########## Saving ###########
  # Saving the values of V on the grid
  filename = 'Initial_x_'+str(initial_x)+'final_t_'+str(final_t)+'alpha_'+\
   str(alpha_)+'width_'+str(width_)+'width_heads_'+str(width_heads)+\
  'energyconservation_'+str(energy_conservation)+\
  '_normclipping_'+str(norm_clipping)+'_'\
  'epochs_'+str(epochs_)+'grid_size_'+str(grid_size)+'Grid_potential_values'+'.p'
  #os.mkdir(filename)
  f=open(filename,"wb")
  pickle.dump(V,f)
  f.close()
  ############################# 

# I took the potential from Marios
# std =  0.07531199161310201 (this was calculated using np.std(V) though, was it the same before?)

if __name__=='__main__':
  means_cell=[[0.69328886, 0.62302152],
       [0.62972374, 0.68102272],
       [0.35136855, 0.86476039],
       [0.2457959 , 0.30488395],
       [0.43302775, 0.3037174 ],
       [0.2920545 , 0.58190847],
       [0.25480357, 0.2826869 ],
       [0.72043632, 0.97845852],
       [0.39262171, 0.780991  ],
       [0.69264611, 0.43098572],
       [0.53137197, 0.36109684],
       [0.39207023, 0.23181951],
       [0.67612807, 0.24105851],
       [0.27432858, 0.83761906],
       [0.2446997 , 0.03918383],
       [0.26430048, 0.20931155],
       [0.48313039, 0.30458524],
       [0.93054253, 0.22026465],
       [0.6534496 , 0.86990403],
       [0.22873698, 0.51596415],
       [0.8242501 , 0.31343169],
       [0.07013505, 0.01035017],
       [0.39194914, 0.12206969],
       [0.50188023, 0.9421176 ],
       [0.51851362, 0.41315729],
       [0.23069509, 0.66829742],
       [0.42707137, 0.52838229],
       [0.99268802, 0.4732617 ],
       [0.50827165, 0.25481151],
       [0.78231963, 0.63622204],
       [0.98584809, 0.15917754],
       [0.686844  , 0.73084527],
       [0.71270836, 0.60876685],
       [0.07069536, 0.20174712],
       [0.47055071, 0.74085157],
       [0.43598188, 0.20926862],
       [0.77685756, 0.83513752],
       [0.03934816, 0.3780164 ],
       [0.19721233, 0.82084329],
       [0.88300091, 0.86994626],
       [0.36565405, 0.91898782],
       [0.34477498, 0.61927163],
       [0.26112272, 0.56559439],
       [0.08066489, 0.63508465],
       [0.90375391, 0.90630822],
       [0.78162643, 0.78715488],
       [0.14675199, 0.17238519],
       [0.20230789, 0.20954889],
       [0.89263846, 0.09397119],
       [0.41300506, 0.29932929],
       [0.00652997, 0.36290346],
       [0.02955892, 0.31081316],
       [0.98478028, 0.36599197],
       [0.53045133, 0.4577757 ],
       [0.11265293, 0.90468273],
       [0.86243902, 0.82198721],
       [0.15541893, 0.35910267],
       [0.38395148, 0.82486313],
       [0.12626633, 0.74048552],
       [0.67723934, 0.47309336],
       [0.45998951, 0.0456377 ],
       [0.08414303, 0.51762913],
       [0.66969847, 0.40136521],
       [0.95722118, 0.20554883],
       [0.98212535, 0.99472204],
       [0.18858644, 0.71447836],
       [0.63407734, 0.21679316],
       [0.04553423, 0.76775082],
       [0.90108713, 0.65962794],
       [0.23724257, 0.19465689],
       [0.71113517, 0.9324531 ],
       [0.5397087 , 0.6675822 ],
       [0.51964917, 0.69593061],
       [0.93059511, 0.16501089],
       [0.18466471, 0.09990244],
       [0.59433978, 0.23121713],
       [0.08075388, 0.54965616],
       [0.33432759, 0.23345882],
       [0.18919413, 0.55432353],
       [0.81927138, 0.74639199],
       [0.94236385, 0.46802752],
       [0.9229066 , 0.47704559],
       [0.39218096, 0.86850983],
       [0.28495168, 0.45401228],
       [0.46925987, 0.45978391],
       [0.72848875, 0.63783759],
       [0.99439799, 0.80080784],
       [0.03081596, 0.30919357],
       [0.33705227, 0.95739791],
       [0.62507195, 0.44324632],
       [0.53384847, 0.05518808],
       [0.5130672 , 0.16601456],
       [0.25736049, 0.12527952],
       [0.81627604, 0.52687942],
       [0.38151881, 0.99245342],
       [0.36206498, 0.90817642],
       [0.36651755, 0.55728959],
       [0.53637457, 0.50082012],
       [0.92767367, 0.16358248],
       [0.82517191, 0.65982193],
       [0.17250297, 0.93257064],
       [0.07460555, 0.41295947],
       [0.87941881, 0.14902439],
       [0.51908387, 0.8318594 ],
       [0.8420881 , 0.04181111],
       [0.4431039 , 0.25719247],
       [0.66335985, 0.57408672],
       [0.46680594, 0.36087934],
       [0.47635924, 0.82876929],
       [0.19347517, 0.85150253],
       [0.59654293, 0.95250603],
       [0.49344813, 0.07385078],
       [0.02953921, 0.94803508],
       [0.0211277 , 0.96541423],
       [0.65454881, 0.53177392],
       [0.08062432, 0.29065971],
       [0.03302955, 0.42727707],
       [0.60471609, 0.23992   ],
       [0.30986327, 0.30486802],
       [0.70142952, 0.13009902],
       [0.58170402, 0.9352433 ],
       [0.14748494, 0.76010235],
       [0.24764802, 0.85584304],
       [0.52664166, 0.72288131],
       [0.3898681 , 0.48533184],
       [0.1088854 , 0.48297433],
       [0.73853363, 0.82968724],
       [0.12628585, 0.80053709],
       [0.48002587, 0.03470227],
       [0.76990073, 0.64709848],
       [0.41529376, 0.74118477],
       [0.88123604, 0.03229834],
       [0.30646548, 0.51364109],
       [0.03644331, 0.16025551],
       [0.72099122, 0.28907357],
       [0.36652211, 0.39922883],
       [0.4275408 , 0.15222184],
       [0.2412875 , 0.33606908],
       [0.41278203, 0.06022276],
       [0.62387162, 0.25624721],
       [0.91234033, 0.50119869],
       [0.83081854, 0.19540723],
       [0.47652942, 0.73757264],
       [0.53914831, 0.73776151],
       [0.09597372, 0.79662722],
       [0.78934317, 0.28251807],
       [0.04925713, 0.83197026],
       [0.16078311, 0.24088763],
       [0.72631218, 0.05849439],
       [0.14155282, 0.05322373],
       [0.12172829, 0.54418637],
       [0.28729071, 0.49367869],
       [0.20625798, 0.49526827],
       [0.40217293, 0.2251016 ],
       [0.25519847, 0.928009  ],
       [0.15074866, 0.16201404],
       [0.1609175 , 0.33213512],
       [0.12165613, 0.62829024],
       [0.22793273, 0.54506243],
       [0.72072985, 0.90504972],
       [0.06709414, 0.96255601],
       [0.49715694, 0.90808823],
       [0.53402368, 0.30992067],
       [0.64035945, 0.04600602],
       [0.81685181, 0.06650662],
       [0.73824526, 0.77159705],
       [0.90811529, 0.07021155],
       [0.92633352, 0.64112242],
       [0.27957405, 0.77158541],
       [0.13307637, 0.04214366],
       [0.39080086, 0.22419985],
       [0.79487371, 0.28805123],
       [0.60068759, 0.57325972],
       [0.13159972, 0.6202648 ],
       [0.8429458 , 0.53668504],
       [0.63194593, 0.73033927],
       [0.85142267, 0.68603946],
       [0.41701097, 0.70811919],
       [0.58924298, 0.47957525],
       [0.39532785, 0.25239882],
       [0.80301044, 0.69618195],
       [0.03805653, 0.36240816],
       [0.21402138, 0.12856325],
       [0.24486784, 0.35431111],
       [0.82752941, 0.21565475],
       [0.16717406, 0.20547143],
       [0.44420708, 0.75381892],
       [0.08437253, 0.01474421],
       [0.54574273, 0.34686528],
       [0.36650591, 0.31214747],
       [0.38493034, 0.21607199],
       [0.69266925, 0.73034133],
       [0.03648276, 0.41798161],
       [0.94219908, 0.66376044],
       [0.52614605, 0.26407267],
       [0.30959919, 0.81611857],
       [0.09140571, 0.23799709],
       [0.02884327, 0.95817383],
       [0.20797279, 0.77761005],
       [0.45146932, 0.06693628]]
  N_heads_run_Gaussiann(grid_size=30000, means=means_cell,final_t=1, width_=64, width_heads=32, epochs_=30000)
