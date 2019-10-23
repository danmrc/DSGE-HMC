include("galis_example.jl")

GAMMA_0 = [bet    0     0  0;
           1      sig   0  0;
           0      0     0  0;
           0      0     0  1]

GAMMA_1 = [1      -kappa  0  0;
           0       sig    1  0;
           -phi_pi  -phi_y  1 -1;
           0       0      0  rho_v]

PSI = [0; 0; 0; 1]

PI = [bet  0;
      1    sig;
      0    0;
      0    0]
