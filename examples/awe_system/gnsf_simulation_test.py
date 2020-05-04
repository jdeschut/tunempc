
import tunempc.pmpc as pmpc
import tunempc.preprocessing as preprocessing
import tunempc.mtools as mtools
import tunempc.closed_loop_tools as clt
import casadi as ca
import casadi.tools as ct
import numpy as np
import pickle
import collections
import copy
import matplotlib.pyplot as plt

# load user input
with open('user_input.pkl','rb') as f:
    user_input = pickle.load(f)

# load convexified reference
with open('convex_reference.pkl','rb') as f:
    sol = pickle.load(f)

# add variables to sys again
vars = collections.OrderedDict()
for var in ['x','u','us']:
    vars[var] = ca.MX.sym(var, sol['sys']['vars'][var])
sol['sys']['vars'] = vars
nx = sol['sys']['vars']['x'].shape[0]
nu = sol['sys']['vars']['u'].shape[0]
ns = sol['sys']['vars']['us'].shape[0]

# set-up open-loop scenario
Nmpc  = 20

# prepare tracking cost and initialization
tracking_cost = mtools.tracking_cost(nx+nu+ns)
lam_g0 = copy.deepcopy(sol['lam_g'])

# tuned tracking MPC
tuning = {'H': sol['S']['Hc'], 'q': sol['S']['q']}
TUNEMPC = pmpc.Pmpc(
    N = Nmpc,
    sys = sol['sys'],
    cost = tracking_cost,
    wref = sol['wsol'],
    tuning = tuning,
    lam_g_ref = lam_g0,
)

# get system dae
alg = user_input['dyn']

# solver options
opts = {}
opts['integrator_type'] = 'GNSF'
opts['sim_method_num_steps'] = 50
opts['tf'] = Nmpc*user_input['ts']

_, GNSF_integrator = TUNEMPC.generate(
    alg, opts = opts, name = 'awe_system'
    )

# set up simulation
x0 = np.squeeze(sol['wsol']['x',0].full())

# GNSF step
x_sim_gnsf = []
for i in range(user_input['p']):
    GNSF_integrator.set('x', x0)
    GNSF_integrator.set('u', np.squeeze(sol['wsol']['u',i].full()))
    gnsf_status = GNSF_integrator.solve()
    x0 = GNSF_integrator.get('x')
    x_sim_gnsf.append(x0)

opts['integrator_type'] = 'IRK'
_, IRK_integrator = TUNEMPC.generate(
    alg, opts = opts, name = 'irk'
    )

# IRK step
x0 = np.squeeze(sol['wsol']['x',0].full())
x_sim_irk = []
for i in range(user_input['p']):
    IRK_integrator.set('x', x0)
    IRK_integrator.set('u', np.squeeze(sol['wsol']['u',i].full()))
    gnsf_status = IRK_integrator.solve()
    x0 = IRK_integrator.get('x')
    x_sim_irk.append(x0)


# plot x-position
plt.figure()
plt.plot([x_sim_gnsf[k][0] for k in range(user_input['p'])])
plt.plot([x_sim_irk[k][0] for k in range(user_input['p'])])
plt.plot([sol['wsol']['x',0][0] for k in range(user_input['p'])], linestyle = '--', color = 'black')
plt.show()
import ipdb; ipdb.set_trace()