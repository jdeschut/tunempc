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
:author: Jochem De Schutter
"""

import numpy as np
import matplotlib.pyplot as plt
from tunempc.logger import Logger

def check_equivalence(controllers, cost, h, x0, dx, alpha, flag = 'tunempc'):

    """ Check local equivalence of different controllers.
    """

    Logger.logger.info(60*'=')
    Logger.logger.info(15*' '+'Compare feedback policies...')
    Logger.logger.info(60*'=')
    Logger.logger.info('')

    log = []

    # compute feedback law in direction dx for different alpha values
    for alph in alpha:

        # determine initial condition
        print('alpha: {}'.format(alph))
        x_init = x0 + alph*dx

        log.append(initialize_log(controllers, x_init))

        for name in list(controllers.keys()):

            # compute feedback law and store results
            print('Compute MPC feedback for controller {}'.format(name))
            if flag == 'tunempc':
                u0 = controllers[name].step(x_init)
                wsol = controllers[name].w_sol
            elif flag == 'acados':
                u0 = controllers[name].step_acados(x_init)
                wsol = controllers[name].w_sol_acados
            log[-1]['u'][name] = wsol['u',:]
            log[-1]['x'][name] = wsol['x',:]
            log[-1]['l'][name] = [cost(wsol['x',k],wsol['u',k]).full()[0][0] for k in range(len(wsol['u',:]))]
            log[-1]['h'][name] = [h(wsol['x',k],wsol['u',k]).full()[0][0] for k in range(len(wsol['u',:]))]

            # reset controller
            controllers[name].reset()

    return log

def closed_loop_sim(controllers, cost, h, F, x0, N, flag = 'tunempc'):

    """ Perform closed-loop simulations for different controllers starting from x0
    """

    Logger.logger.info(60*'=')
    Logger.logger.info(10*' '+'Compute closed-loop responses...')
    Logger.logger.info(60*'=')
    Logger.logger.info('')

    log = initialize_log(controllers, x0)

    for i in range(N):

        Logger.logger.info('======================================')
        Logger.logger.info('Closed-loop simulation step {} of {}'.format(i, N))
        Logger.logger.info('======================================')

        for name in list(controllers.keys()):

            # compute feedback law and store results
            print('Compute MPC feedback for controller {}'.format(name))
            if flag == 'tunempc':
                log['u'][name].append(controllers[name].step(log['x'][name][-1]))
            elif flag == 'acados':
                log['u'][name].append(controllers[name].step_acados(log['x'][name][-1]))
            log['l'][name].append(cost(log['x'][name][-1], log['u'][name][-1]).full()[0][0])
            log['h'][name].append(h(log['x'][name][-1], log['u'][name][-1]).full())

            # simulate
            log['x'][name].append(F(x0 = log['x'][name][-1], p = log['u'][name][-1])['xf'])

    return log

def initialize_log(controllers, x0=None):
    
    # initialize log
    log = {'u':{},'l':{},'h':{}}
    
    if x0 is not None:
        log['x'] = {}

    for name in list(controllers.keys()):
        for log_key in log.keys():
            log[log_key][name] = []

        if x0 is not None:
            log['x'][name] = [x0]
    
    return log