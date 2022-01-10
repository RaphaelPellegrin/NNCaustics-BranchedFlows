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
def f_general(u, t, means_Gaussians, lam=1, sig=2, A_=1):
    # unpack current values of u
    x, y, px, py = u  

    V=0
    Vx=0
    Vy=0

    A=A_/(2*np.pi*sig**2)

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
def rayTracing_general(t, x0, y0, px0, py0, means_Gaussians, lam=1, sig=2, A_=1):
    u0 = [x0, y0, px0, py0]
    # Call the ODE solver
    solPend = odeint(f_general, u0, t, args=(means_Gaussians, lam, sig, A_,))
    xP = solPend[:,0];    yP  = solPend[:,1];
    pxP = solPend[:,2];   pyP = solPend[:,3]
    return xP, yP, pxP, pyP


# Code for training

def N_heads_run_Gaussiann(initial_x=0, final_t=50, means=[[7.51, 4.6], [8.78, 6.16]], alpha_=1, width_=40, width_heads=8, \
  epochs_=20000, grid_size=200, number_of_heads=20, PATH="models", print_legend=False, loadWeights=False, energy_conservation=True,\
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
    initial_condition=random.randint(0,2000)/100
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

          sig=2
          # Building the potential and updating the partial derivatives
          potential=-(1/(2*math.pi*sig**2))*torch.exp(-(1/(2*sig**2))*((x-mu_x)**2+(y-mu_y)**2))
          # Partial wrt to x
          partial_x+=-potential*(x-mu_x)*(1/sig*2)
          # Partial wrt to y
          partial_y+=-potential*(y-mu_y)*(1/sig*2)

          # Updating the energy
          H_0+=-(1/(2*math.pi*sig**2))*math.exp(-(1/2*sig**2)*((initial_x-mu_x)**2+(initial_y-mu_y)**2))
          H_curr+=-(1/(2*math.pi*sig**2))*torch.exp(-(1/2*sig**2)*((x-mu_x)**2+(y-mu_y)**2))

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
  Max=20

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



  y1=np.linspace(minimum_y-1,maximum_y+1,500); x1= np.linspace(-1,maximum_x,500)
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

  sig=2
  A_=1
  A=A_/(2*np.pi*sig**2)

  for i in means_cell:
    muX1=i[0]
    muY1=i[1]
    V+=  - A*np.exp(- (( (x-muX1)**2 + (y-muY1)**2) / sig**2)/2) 

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

if __name__=='__main__':
  means_cell=[[7.51, 4.6],
  [8.78, 6.16],
  [6.78, 7.39],
  [5.04, 0.36],
  [4.6, 9.45],
  [7.13, 0.13], [3,3], [5,5],[6,6],[5,6]]
  N_heads_run_Gaussiann(means=means_cell,final_t=14, width_=16, width_heads=32, epochs_=60000)
