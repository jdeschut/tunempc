#
#    This file is part of TuneMPC.
#
#    TuneMPC -- A Tool for Economic Tuning of Tracking (N)MPC Problems.
#    Copyright (C) 2020 Jochem De Schutter, Mario Zanon, Moritz Diehl (ALU Freiburg).
#
#    TuneMPC is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    TuneMPC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with TuneMPC; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#!/usr/bin/python3
"""Implementation of the LQR example as proposed in 

Zanon et al.
Indefinite linear MPC and approximated economic MPC for nonlinearsystems, (section 6). 
Journal of Process Control 2014

:author: Jochem De Schutter

"""

import control
import tunempc.convexifier as convexifier
import numpy as np
import matplotlib.pyplot as plt


# system data
A = np.matrix('-0.3319  0.7595 1.5399; -0.3393 0.1250 0.4245; -0.5090 0.9388 0.8864')
B = np.matrix('0.1060; -1.3835; -0.1496')

# (indefinite) cost matrices
Q = np.matrix('-1.0029 -0.0896 1.1050; -0.0896 1.6790 -0.5762; 1.1050 -0.5762 -0.4381')
N = np.matrix('-0.0420; 0.2112; -0.2832')
R = np.matrix('0.6192')

# convexify cost matrices
dHc, dQc, dRc, dNc = convexifier.convexify(A,B,Q,R,N)

# stabilizing LQR with tuned weighting matrices
Pc, Ec, Kc = control.mateqn.dare(A, B, Q + dQc[0], R + dRc[0], N + dNc[0], np.eye(A.shape[0]))

# stabilizing LQR
P, E, K = control.mateqn.dare(A, B, Q, R, N, np.eye(A.shape[0]))

# check equivalence
assert np.linalg.norm(K - Kc) < 1e-5, 'Feedback laws should be identical.'

# define closed-loop systems
sys0 = control.ss(A- np.matmul(B,K), B, np.eye(A.shape[0]), np.zeros((A.shape[0],B.shape[1])),1)
sysc = control.ss(A-np.matmul(B,Kc), B, np.eye(A.shape[0]), np.zeros((A.shape[0],B.shape[1])),1)

# check equivalence visually
T, Yout = control.initial_response(sys0,T=range(30),X0=np.matrix('1;0;0'))
T, Youtc = control.initial_response(sysc,T=range(30),X0=np.matrix('1;0;0'))
plt.plot(T, Yout[0])
plt.plot(T, Youtc[0])
plt.show()
