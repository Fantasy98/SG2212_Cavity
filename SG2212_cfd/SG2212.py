import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation
import math 
import scipy.sparse as sp
import scipy.linalg as scl
from scipy.sparse.linalg import splu
params = {'legend.fontsize': 12,
          'legend.loc':'best',
          'figure.figsize': (8,5),
          'lines.markerfacecolor':'none',
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize':12,
          'ytick.labelsize':12,
          'grid.alpha':0.6}
pylab.rcParams.update(params)

def avg(A,axis=0):
    """
    Averaging function to go from cell centres (pressure nodes)
    to cell corners (velocity nodes) and vice versa.
    avg acts on index idim; default is idim=1.
    """
    if (axis==0):
        B = (A[:-1]+ A[1:])/2.
    elif (axis==1):
        B = (A[:,:-1] + A[:,1:])/2.
    else:
        raise ValueError('Wrong value for axis')
    return B           

def DD(n,h):
    """
    One-dimensional finite-difference derivative matrix 
    of size n times n for second derivative:
    h^2 * f''(x_j) = -f(x_j-1) + 2*f(x_j) - f(x_j+1)

    Homogeneous Neumann boundary conditions on the boundaries 
    are imposed, i.e.
    f(x_0) = f(x_1) 
    if the wall lies between x_0 and x_1. This gives then
    h^2 * f''(x_j) = + f(x_0) - 2*f(x_1) + f(x_2)
                   = + f(x_1) - 2*f(x_1) + f(x_2)
                   =              f(x_1) + f(x_2)

    For n=5 and h=1 the following result is obtained:
 
    A =
        -1     1     0     0     0
         1    -2     1     0     0
         0     1    -2     1     0
         0     0     1    -2     1
         0     0     0     1    -1
    """
    data = np.concatenate( (np.array([-1]), np.ones( (n-2,1) ) @ np.array([-2]),np.array([-1])))
    
    diags = np.array([0])
    A = sp.spdiags(data.T, diags, n,n) / h**2

    ones = np.vstack([np.ones(n),np.ones(n)])
    diags_one = np.array([-1,1])
    A_one = sp.spdiags(ones,diags_one,n,n)/h**2

    A+=A_one
    return A
    
DD(5,1).toarray()

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

Pr = 0.71
Re = 50
# Ri = 0. 
dt = 1e-3
Tf = 20
Lx = 1.
Ly = 1.
Nx =30
Ny = 30
namp = 0.
ig = 20

# number of iteratins
Nit = 1000
# edge coordinates
x = np.linspace(0,1,Nx+1)
y = np.linspace(0,1,Ny+1)
# grid spacing
hx = Lx/Nx
hy = Ly/Ny

# boundary conditions
Utop = 1.; Ttop = 1.; Tbottom = 0.;
uN = x*0 + Utop;  uN = uN[:,np.newaxis];    vN = avg(x,0)*0;    vN = vN[:,np.newaxis];
uS = x*0 + 0;  uS = uS[:,np.newaxis];         vS = avg(x,0)*0;  vS = vS[:,np.newaxis];
uW = avg(y,0)*0;  uW = uW[np.newaxis,:];       vW = y*0;   vW = vW[np.newaxis,:];
uE = avg(y,0)*0;  uE = uE[np.newaxis,:];       vE = y*0;    vE = vE[np.newaxis,:];

tN = x*0 ; tS = y*0

# Compute system matrices for pressure 
# Laplace operator on cell centres: Fxx + Fyy
# First set homogeneous Neumann condition all around
Lp = np.kron( sp.eye(Ny).toarray(),DD(Nx,hx).toarray()) + np.kron( DD(Ny,hy).toarray(),sp.eye(Nx).toarray());
# Set one Dirichlet value to fix pressure in that point
Lp[:,0] = 0; Lp[0,:] =0; Lp[0,0] = 1;
Lp_lu, Lp_piv = scl.lu_factor(Lp)
Lps = sp.csc_matrix(Lp)
Lps_lu = splu(Lps)

U = np.zeros((Nx-1,Ny))
V = np.zeros((Nx,Ny-1))
T = np.zeros((Nx,Ny))+ \
    namp*(np.random.rand(Nx,Ny)-0.5); 


Ue = np.vstack((uW, U, uE)); Ue = np.hstack( (2*uS-Ue[:,0,np.newaxis], Ue, 2*uN-Ue[:,-1,np.newaxis]));

Ua = avg(Ue,1)
Ue.shape,Ua.shape



if (ig>0):
    metadata = dict(title='Lid-driven cavity', artist='SG2212')
    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=metadata)
    matplotlib.use("Agg")
    fig=plt.figure()
    writer.setup(fig,"/home/yuning/KTH/yuning_Project/cavity.mp4",dpi=200)

# progress bar
print('[         |         |         |         |         ]')
tic()
for k in range(Nit):
    print("Iteration k=%i time=%.2e" % (k,k*dt))

    # include all boundary points for u and v (linear extrapolation
    # for ghost cells) into extended array (Ue,Ve)
    Ue = np.vstack((uW, U, uE)); Ue = np.hstack( (2*uS-Ue[:,0,np.newaxis], Ue, 2*uN-Ue[:,-1,np.newaxis]));
    Ve = np.hstack( [vS,V,vN]  ); Ve = np.hstack(  [2*vW.T-Ve[0,:,np.newaxis], Ve.T, 2*vE.T-Ve[-1,:,np.newaxis]]  ).T

    # averaged (Ua,Va) of u and v on corners
    Ua = avg(Ue,1)
    Va = avg(Ve,0)

    #  construct individual parts of nonlinear terms
    dUVdx = np.diff(Ua[:,1:-1]*Va[:,1:-1],axis=0,n=1)/hx
    dUVdy = np.diff(Ua[1:-1,:]*Va[1:-1,:],axis =1,n=1)/hy
    Ub    = avg( Ue*Ue,0);   
    Vb    = avg( Ve*Ve,1)
    dU2dx = np.diff( Ub[:,1:-1],n=1,axis=0)/hx;
    dV2dy = np.diff( Vb[1:-1,:],n=1,axis=1)/hy;

    # treat viscosity explicitly
    viscu = np.diff( Ue[:,1:-1],axis=0,n=2 )/hx**2 + np.diff(Ue[1:-1,:],axis=1,n=2)/hy**2  ;
    viscv = np.diff( Ve[:,1:-1],axis=0,n=2 )/hx**2 + np.diff(Ve[1:-1,:],axis=1,n=2)/hy**2 ;

    # compose final nonlinear term + explicit viscous terms
    U = U + (dt/Re)*viscu - dt*(dU2dx + dUVdy)
    V = V + (dt/Re)*viscv - dt*(dV2dy + dUVdx)

    # pressure correction, Dirichlet P=0 at (1,1)
    rhs = (np.diff( np.vstack([uW,U,uE]),n=1,axis=0)/hx + np.diff( np.hstack([vS,V,vN]),n=1 ,axis=1)/hy )/dt;
    rhs = np.reshape(rhs,(Nx*Ny,1));
    rhs[0] = 0;

    # different ways of solving the pressure-Poisson equation:
    P = Lps_lu.solve(rhs)

    P = np.reshape(P,(Nx,Ny))

    # apply pressure correction
    U = U - dt*np.diff(P,n=1,axis=0)/hx;
    V = V - dt*np.diff(P,n=1,axis=1)/hy; 

    # Temperature equation
    

    # do postprocessing to file
    if (ig>0 and np.floor(k/ig)==k/ig):
        plt.clf()
        # Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
        # Va = np.vstack((vW,avg(np.hstack((vS,V,
        #                           vN)),0),vE));

        # plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20,cmap="jet")
        # plt.quiver(x,y,Ua.T,Va.T)
        plt.contourf(avg(x),avg(y),T.T,levels=np.arange(0,1.05,0.05),cmap = "jet")
        plt.gca().set_aspect(1.)
        plt.colorbar()
        plt.title(f'Velocity at t={k*dt:.2f}')
        writer.grab_frame()

    # update progress bar
    if np.floor(51*k/Nit)>np.floor(51*(k-1)/Nit): 
        print('.',end='')

# finalise progress bar
print(' done. Iterations k=%i time=%.2f' % (k,k*dt))
toc()

if (ig>0):
    writer.finish()


Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
Va = np.vstack((vW,avg(np.hstack((vS,V,
                                  vN)),0),vE));
plt.figure(-1)
plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20)
plt.quiver(x,y,Ua.T,Va.T)
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Velocity at t={k*dt:.2f}')
plt.savefig("velocity_field")

div = (np.diff( np.vstack( (uW,U, uE)),axis=0)/hx + np.diff( np.hstack(( vS, V, vN)),axis=1)/hy)
plt.figure()
plt.pcolor(avg(x),avg(y),div.T,shading='nearest')
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Divergence at t={k*dt:.2f}')
plt.savefig("Divergence")
