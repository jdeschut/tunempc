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
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
"""
SQP method

:author: Jochem De Schutter (2019), based on implementation by Mario Zanon (2016).
"""


import casadi.tools as ct
import casadi as ca
import numpy as np
from scipy.linalg import eig
from scipy.linalg import null_space
import logging

class Sqp(object):

    def __init__(self, problem, options = {}):

        """ Constructor
        """

        # problem data
        self.__w = problem['x']
        self.__f = problem['f']
        self.__g = problem['g']
        self.__p = problem['p']
        self.__lbg = problem['lbg']
        self.__ubg = problem['ubg']

        # default settings
        self.__options = {
            'regularization': 'reduced',
            'regularization_tol': 1e-6,
            'tol': 1e-6,
            'lam_tresh': 1e-8,
            'max_ls_iter': 300,
            'ls_step_factor': 0.8,
            'hessian_approximation': 'exact',
            'max_iter': 2000
        }

        # overwrite default
        for opt in options:
            self.__options[opt] = options[opt]

        self.__construct_sensitivities()
        self.__construct_qp_solver()

        return None

    def __construct_sensitivities(self):

        """ Construct NLP sensitivities
        """

        # convenience
        w = self.__w
        p = self.__p

        # cost function
        self.__f_fun = ca.Function('f_fun',[w,p], [self.__f])
        self.__jacf_fun = ca.Function('jacf_fun',[w,p], [ca.jacobian(self.__f,self.__w)])
        
        # constraints
        self.__g_fun = ca.Function('g_fun',[w,p],[self.__g])
        self.__jacg_fun = ca.Function('jacg_fun',[w,p], [ca.jacobian(self.__g,self.__w)])
        self.__gzeros = np.zeros((self.__g.shape[0],1))

        # exact hessian
        lam_g = ca.MX.sym('lam_g',self.__g.shape)
        lag = self.__f + ct.mtimes(lam_g.T, self.__g)
        self.__jlag_fun = ca.Function('jLag',[w,p,lam_g],[ca.jacobian(lag,w)])

        if self.__options['hessian_approximation'] == 'exact':
            self.__H_fun = ca.Function('H_fun',[w,p,lam_g],[ca.hessian(lag,w)[0]])
        else:
            self.__H_fun = self.__options['hessian_approximation']

    def __construct_qp_solver(self):

        Hlag = self.__H_fun(self.__w, self.__p, ca.MX.sym('lam_g',self.__g.shape))
        jacg = self.__jacg_fun(self.__w, self.__p)

        # qp cost function and constraints
        qp = {
            'h': Hlag.sparsity(),
            'a': jacg.sparsity()
        }

        # qp options
        opts = {
            'enableEqualities':True,
            'printLevel':'none',
            'sparse': True,
            'enableInertiaCorrection':True,
            'enableCholeskyRefactorisation':1,
            # 'enableRegularisation': True,
            # 'epsRegularisation': 1e-6
            # 'enableFlippingBounds':True
            # 'enableFarBounds': False,
            # 'enableFlippingBounds': True,
            # 'epsFlipping': 1e-8,
            # 'initialStatusBounds': 'inactive',
        }

        self.__solver = ca.conic(
            'qpsol',
            'qpoases',
            qp,
            opts
        )

        return None

    def solve(self, w0, p0, lam_g_ip):

        """ Vanilla SQP-method, no line-search.
        """

        # Pre-filter multipliers from interior-point method
        lam_g0 = self.__prefilter_lam_g(lam_g_ip)

        # Perform SQP iterations
        converged = self.__check_convergence(w0,p0,lam_g0,0.0, 0)
        converged = False

        k = 0
        while not converged:

            # evaluate constraints residuals
            g0 = self.__g_fun(w0,p0)

            # regularize hessian
            H = self.__regularize_hessian(w0, p0, lam_g0)

            # qp matrices
            qp_p = {
                'h': H,
                'g': self.__jacf_fun(w0,p0),
                'a': self.__jacg_fun(w0,p0),
                'lba': self.__lbg - g0,
                'uba': self.__ubg - g0,
                'lam_a0': lam_g0
            }

            # solve qp
            res = self.__solver(**qp_p)

            # perform line-search
            dw = self.__linesearch(w0, p0, res['x'].full())

            # check convergence
            w0 = w0 + dw
            lam_g0 = res['lam_a']

            k += 1
            converged = self.__check_convergence(w0,p0,lam_g0,dw, k)

        # Solution sanity check
        S = self.__postprocessing(w0,p0,lam_g0, k)

        return {'x':w0, 'lam_g': lam_g0, 'S': S}

    def __postprocessing(self, w0, p0, lam_g0, iter):

        """ Perform sanity checks on solution and compute sensitivities
        """

        # compute sensitivities
        H = self.__H_fun(w0,p0,lam_g0)

        # check if reduced hessian is PD
        jacg_active, as_idx = self.__active_constraints_jacobian(w0,p0,lam_g0)
        Z = null_space(jacg_active)
        Hred = ct.mtimes(ct.mtimes(Z.T, H),Z)

        # check eigenvalues of reduced hessian
        if Hred.shape[0] > 0:
            min_eigval = np.min(np.linalg.eigvals(Hred))
            assert min_eigval > self.__options['regularization_tol'], 'Reduced Hessian is not positive definite!'

        # retrieve active set changes w.r.t initial guess
        nAC = len([k for k in self.__as_idx_init if k not in as_idx])
        nAC += len([k for k in as_idx if k not in self.__as_idx_init])

        # save stats
        info = self.__solver.stats()
        self.__stats = {
            'x': w0,
            'lam_g': lam_g0,
            'p0': p0,
            'iter_count': iter,
            'f': self.__f_fun(w0,p0),
            't_wall_total': info['t_wall_solver'],
            'return_status': info['return_status'],
            'nAC': nAC,
            'nAS': len(as_idx)
        }

        return {'H': H}

    def __prefilter_lam_g(self, lam_g0):

        """ Prefilter multipliers obtained from the interior-point solver
            for use in the active-set SQP-method. Due to limited accuracy of
            the IP-solver, multipliers of passive constraints are not exactly zero.
        """

        import copy
        lam_g0 = copy.deepcopy(lam_g0)

        # filter multipliers with treshold value
        for i in range(lam_g0.shape[0]):
            if abs(lam_g0[i]) < self.__options['lam_tresh']:
                lam_g0[i] = 0.0
        
        return lam_g0

    def __check_convergence(self, w0, p0, lam_g0, dw, k):

        """ Check convergence of SQP algorithm and print stats.
        """

        # dual infeasibility
        dual_infeas = np.linalg.norm(self.__jlag_fun(w0,p0,lam_g0).full(),np.inf)

        if k == 0:
            f0 = self.__f_fun(w0,p0)
            g0 = self.__g_fun(w0,p0)
            lbg = self.__lbg.cat.full()
            ubg = self.__ubg.cat.full()
            infeas = np.linalg.norm(
                abs(np.array([*map(min,np.array(g0) - lbg,self.__gzeros)]) +
                    np.array([*map(max,np.array(g0) - ubg,self.__gzeros)])),
                    np.inf)

            self.__ls_filter = np.array([[ f0, infeas ]])
            self.__alpha = 0.0
            self.__reg = 0.0
            _, self.__as_idx_init = self.__active_constraints_jacobian(w0,p0,lam_g0)

        # print stats
        if k%10 == 0:
            logging.info('iter\tf\t\tstep\t\tinf_du\t\tinf_pr\t\talpha\t\treg')
        logging.info('{:3d}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.2e}\t{:.2e}'.format(
            k,
            self.__ls_filter[-1,0],
            np.linalg.norm(dw),
            dual_infeas,
            self.__ls_filter[-1,1],
            self.__alpha,
            self.__reg))

        # check convergence
        if (self.__ls_filter[-1,1] < self.__options['tol'] and
            dual_infeas < self.__options['tol']):
            converged = True
            if k != 0:
                logging.info('Optimal solution found.')
        elif k == self.__options['max_iter']:
            converged = True
            logging.info('Maximum number of iterations exceeded.')
        else:
            converged = False

        return converged

    def __linesearch(self, w0, p0, dw):

        alpha = 1.0
        wnew = w0 + alpha*dw

        # filter method
        f0 = self.__f_fun(wnew,p0)
        g0 = self.__g_fun(wnew,p0)
        lbg = self.__lbg.cat.full()
        ubg = self.__ubg.cat.full()
        infeas = np.linalg.norm(
            abs(np.array([*map(min,np.array(g0) - lbg,self.__gzeros)]) +
                np.array([*map(max,np.array(g0) - ubg,self.__gzeros)])),
                np.inf)

        for ls_k in range(self.__options['max_ls_iter']):
            check = np.array(list(map(int,float(f0) > self.__ls_filter[:,0]))) \
                    + np.array(list(map(int, infeas > self.__ls_filter[:,1] )))
            check = check.tolist()
            check = [check[i] > 1.5 for i in range(len(check))]
            if sum(check) > 1:   # Allow to worsen the filter for max 0 iterations
                alpha *= self.__options['ls_step_factor']
                # iter_btls += 1
                wnew = w0 + alpha*dw
                f0 = self.__f_fun(wnew,p0)
                g0 = self.__g_fun(wnew,p0)
                infeas = np.linalg.norm(
                    abs(np.array([*map(min,np.array(g0) - lbg,self.__gzeros)]) +
                        np.array([*map(max,np.array(g0) - ubg,self.__gzeros)])),
                        np.inf)
            else:
                break

        self.__alpha = alpha
        self.__ls_filter = np.append( self.__ls_filter, np.array([[ f0, infeas ]]), axis=0 )

        return alpha*dw

    def __regularize_hessian(self, w0, p0, lam_g):

        # evaluate hessian
        H = self.__H_fun(w0,p0,lam_g).full()

        # eigenvalue tolerance
        tol = self.__options['regularization_tol']

        if self.__options['regularization']  == 'reduced':

            # active constraints jacobian
            jacg_active, _ = self.__active_constraints_jacobian(w0,p0,lam_g)

            # compute reduced hessian
            Z   = null_space(jacg_active)
            Hr  = np.dot(np.dot(Z.T, H), Z)

            # compute eigenvalues
            if Hr.shape[0] != 0:
                eva, evec = eig(Hr)
                regularize = (min(eva.real) < tol)
            else:
                regularize = False

            # check if regularization is needed
            if regularize:

                # detect non-positive eigenvalues
                evmod = eva*0
                for i in range(len(eva)): 
                    if eva[i] < tol:
                        evmod[i] = tol
                    else:
                        evmod[i] = eva[i]

                # eigenvalue regularization vector
                deva = evmod - eva
                self._reg = max(deva.real)
                dHr = np.dot(np.dot(evec,np.diag(deva)),np.linalg.inv(evec))
                H = H + np.dot(np.dot( Z, dHr), Z.T )

                # make sure that the Hessian is symmetric
                H = (H.real + H.real.T)/2.0

                # check regularized reduced hessian to be sure
                rH = np.dot(np.dot( Z.T, H ), Z)
                eva_r, evec = eig(rH)
                for e in eva_r:
                    if e < tol/1e2:
                        print('Regularization of reduced Hessian failed. Eigenvalue: {}'.format(e))


        elif self.__options['regularization'] == 'full':

            # comput eigenvaluescc
            eva, evec = eig(H)

            # detect non-positive eigenvalues
            evmod = eva*0
            for i in range(len(eva)): 
                if eva[i] < tol:
                    evmod[i] = tol
                else:
                    evmod[i] = eva[i]

            # eigenvalue regularization vector
            deva = evmod - eva
            self.__reg = max(deva.real)
            H = H + np.dot(np.dot(evec,np.diag(deva)),np.linalg.inv(evec))

            # make sure that the Hessian is symmetric
            H = (H.real + H.real.T)/2.0

        else:
            self.__reg = 0.0

        return H

    def __active_constraints_jacobian(self, w0, p0, lam_g):

        # evaluate constraints jacobian
        jacg = self.__jacg_fun(w0,p0).full()

        # constraint bounds
        bounds = (self.__ubg.cat - self.__lbg.cat).full()

        # equality constraints
        eq_idx = [i for i, e in enumerate(bounds) if e == 0]
        jacg_active = jacg[eq_idx,:]

        # inequality constraints
        ineq_idx = [i for i, e in enumerate(bounds) if e != 0]
        as_idx = []
        for i in ineq_idx:
            if lam_g[i] != 0:
                jacg_active = np.append(jacg_active,[jacg[i,:]], axis = 0)
                as_idx.append(i)

        return jacg_active, as_idx

    @property
    def ls_filter(self):
        return self.__ls_filter

    @property
    def stats(self):
        return self.__stats