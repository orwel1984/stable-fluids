import numpy as np                                
import math

import scipy.sparse as sparse
from scipy.sparse import csc_matrix, linalg as sla
from scipy.sparse import spdiags, eye

class Solver:
    nu =   0.0000001      # rate of density-diffusion
    visc = 0.00001        # rate of velocity-diffusion

    def __init__(self, m, n):
        self.m = m
        self.n = n
        nu = self.nu
        visc = self.visc
        
        self.M = m-1        # index of last element in x-axis (zero based array-indexing)
        self.N = n-1 
        #######################
        # Domain Ω=[0,1]×[0,1]
        #######################
        self.L = 1
        x = np.linspace(0, self.L, n)   
        y = np.linspace(0, self.L, m)
        xx, yy = np.meshgrid(x,y)
        ######################
        # Δ𝑡, Δ𝑥, Δ𝑦
        ######################
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dt = 0.1
        ######################
        # CFL = Δ𝑡/(Δ𝑥Δ𝑦)
        ######################
        self.c = self.dt/(self.dx*self.dy);        
        
        # denisty fields
        self.D  =  np.full( (m, n),    0, dtype=float)
        self.D0 =  np.full( (m, n),    0, dtype=float)
        
        # velocity fields
        self.V  =  np.full( (m, n, 2), 0, dtype=float)
        self.V0 =  np.full( (m, n, 2), 0, dtype=float)     
        
        # implict solver matrices A, for diffusion
        self.A_d = self.implicit_mat(nu);
        self.A_v = self.implicit_mat(visc);
        self.A_p = self.poisson_mat();
        
    def reset(self):
        self.D0[:,:] = 0
        self.V0[:,:] = [0,0]
        self.D[:,:] = 0
        self.V[:,:] = [0,0]  
    
    def step(self):
        self.velocity_step()            
        self.desnity_step()        

    def desnity_step(self):        
        self.D = self.diffuse(self.A_d, self.D0)
        self.boundary(self.D, 0)        
        self.D = self.advect(self.D0, self.D, self.V)
        self.boundary(self.D, 0)        
        self.D0 = self.D
        
    def velocity_step(self):        
        u0 = self.V0[:, :, 0];
        v0 = self.V0[:, :, 1];        
        
        # diffuse velocity fields
        u = self.diffuse(self.A_v, u0)
        v = self.diffuse(self.A_v, v0)
        self.boundary(u, 1)
        self.boundary(v, 2)
        
        self.V[:, :, 0] = u;
        self.V[:, :, 1] = v;

        # project
        self.project(u0, v0, u, v)   # u0,v0 are updated
        
        # advect velocity fields
        self.advect(u, u0, self.V0)  # output = u
        self.advect(v, v0, self.V0)  # output = v
        self.boundary(u, 1)
        self.boundary(v, 2)        
        
        # project
        self.project(u, v, u0, v0)  # u,v are updated
        
        # Save
        self.V[:, :, 0] = u;
        self.V[:, :, 1] = v;
        self.V0[:,:,:] = self.V[:,:,:]
                
    def advect(self, d, d0, V):
        dt0 = self.dt/self.dx
        dt1 = self.dt/self.dy        
        for j in range(1, self.N-1):
            for i in range(1, self.M-1):

                u = V[i, j, 0]
                v = V[i, j, 1]

                x = i - u*dt0;      # cells travelled in x-driection during dt        
                y = j - v*dt1;      # cells travelled in y-driection during dt
                
                s0,s1,t0,t1,i0i,i1i,j0i,j1i = self.clamp(x,y)

                d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                          s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                
        return d;

                
    def project(self, u, v, p, div):
        
        div[1:-1, 1:-1] = -0.5* self.dx * \
                                ( u[2:, 1:-1] - u[:-2, 1:-1] +
                                  v[1:-1, 2:] - v[1:-1, :-2])

        p[:, :] = 0

        self.boundary(div, 0)
        self.boundary(p, 0)
        
        #p = self.diffuse(self.A_p, div)
        self.lin_solve(p, div, 1, 4)        
                               
        u[1:-1, 1:-1] -= 0.5 * (1/self.dx) * (p[ 2:  ,  1:-1] - p[  :-2, 1:-1])
        v[1:-1, 1:-1] -= 0.5 * (1/self.dy) * (p[ 1:-1,  2:  ] - p[ 1:-1,  :-2])

        self.boundary(u, 1)
        self.boundary(v, 2)
        

    def lin_solve(self, x, x0, a, c):
        for iteration in range(0,5):
            x[1:-1, 1:-1] =  (1/c)* ( x0[1:-1, 1:-1] + \
                                      a * (x[2:  , 1:-1] + x[ :-2, 1:-1] + \
                                           x[1:-1, 2:  ] + x[1:-1,  :-2]))
            
    def poisson_mat(self):
        D = self.diffusion_mat();
        A = (1/(self.dx*self.dy))*D
        A_inv = sla.splu(csc_matrix(A));
        return A_inv;
            
    def diffuse(self, A, Q_n):
        b = Q_n.reshape(self.m*self.n, 1);
        q = A.solve(b);    
        return q.reshape(self.m,self.n);    
    
    def implicit_mat(self, diff):
        MN = self.m*self.n;
        I = eye(MN,MN);        
        D = self.diffusion_mat();
        A = (I - (self.c*diff) * D);
        A_inv = sla.splu(csc_matrix(A));
        return A_inv;
    
    def diffusion_mat(self):        
        e = np.ones(self.m)
        T_x = spdiags( [e, -2*e, e],            # diagonal data  - stencil 1,-2,1
                       [-1, 0, 1],              # diagonal indices
                       self.m, self.m).toarray()
        
        e = np.ones(self.n)
        T_y = spdiags( [e, -2*e, e],            # diagonal data - stencil 1,-2,1
                       [-1, 0, 1],              # diagonal indices
                       self.n, self.n).toarray()
        
        I_m = eye(self.m);
        I_n = eye(self.n);
        
        # Boundary Conditions 
        T_x[0,0] = -1;
        T_x[self.M,self.M] = -1;
        T_y[0,0] = -1;
        T_y[self.N,self.N] = -1;
        
        D = sparse.kron(I_m, T_y) + sparse.kron(T_x, I_n);
        return D;
    
    def boundary(self, G, cond):
        # Note: remeber B.C.s already applied in diffusion solver's A 
        # Boundarie are on rows: 0, M and columns: 0,N   
        M = self.M
        N = self.N
        
        slipX = 1;
        slipY = 1;
        if cond == 1:
            slipY = -1
        if cond == 2:
            slipX = -1
        
        G[0,:] = G[1,:]    * slipY  # Top: row 0        
        G[M,:] = G[M-1,:]  * slipY  # Bottom: row M        
        G[:,0] = G[:,1]    * slipX  # Left: column 0
        G[:,N] = G[:,N-1]  * slipX  # Right: column N
                
        # corners
        G[0,0] = 0.5 * (G[0,1]   + G[1,1])
        G[M,N] = 0.5 * (G[M-1,N] + G[M,N-1])
        G[0,N] = 0.5 * (G[0,N-1] + G[1,N])
        G[M,0] = 0.5 * (G[M-1,0] + G[M,1])
        
    def clamp(self, x,y):        
        if x < 0.5: x = 0.5
        if x > (self.M-1) + 0.5: x = (self.M-1) + 0.5
        i0 = int(math.floor(x))
        i1 = int(i0 + 1.0)

        if y < 0.5: y = 0.5
        if y > (self.N-1) + 0.5: y = (self.N-1) + 0.5
        j0 = int(math.floor(y))
        j1 = int(j0 + 1.0)

        s1 = x - i0        # fraction of amount that x > nearest int i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        return s0,s1, t0,t1, i0,i1,j0,j1