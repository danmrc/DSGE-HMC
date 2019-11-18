using QuantEcon

function log_like_dsge(par,data)
    #order to par
    #alfa
    #beta
    #epsilon
    #theta
    #sigma this is the std dev of the innovation
    #phi_pi
    #phi_y
    #phi_v
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
    sol = gensys(GAMMA_0,GAMMA_1,PSI,PI)
    if sum(sol.eu) != 2
        return(-9999999999999)
    end
    A = sol.Theta1[4,4]
    G = sol.Theta1[1,4]
    R = 0
    Q = sigma^2
    kalman_res = Kalman(A,G,Q,R)
