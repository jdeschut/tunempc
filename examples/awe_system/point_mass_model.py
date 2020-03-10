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
#!/usr/bin/python3
""" TBD

:author: Jochem De Schutter

"""

import numpy as np
from casadi.tools import vertcat

def data_dict():

    data_dict = {}
    data_dict['name'] = 'point_mass'

    data_dict['geometry'] = geometry() # kite geometry
    data_dict['aero_deriv'] = aero_deriv() # stability derivatives


    # (optional: on-board battery model)
    coeff_min = np.array([0, -80*np.pi/180.0])
    coeff_max = np.array([2, 80*np.pi/180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

    return data_dict

def geometry():

    geometry = {}
    # 'aerodynamic parameter identification for an airborne wind energy pumping system', licitra, williams, gillis, ghandchi, sierbling, ruiterkamp, diehl, 2017
    # 'numerical optimal trajectory for system in pumping mode described by differential algebraic equation (focus on ap2)' licitra, 2014
    geometry['s_ref'] = 500.0  # [m^2]
    geometry['m_k'] = 20*geometry['s_ref']  # [kg]
    geometry['ar'] = 10.0
    geometry['c_ref'] = np.sqrt(geometry['s_ref']/geometry['ar'])  # [m]
    geometry['b_ref'] = np.sqrt(geometry['s_ref']*geometry['ar'])  # [m]

    # dirty fix to get correct CI
    geometry['ar'] = 50.0

    geometry['j'] = np.array([[25., 0.0, 0.47],
                              [0.0, 32., 0.0],
                              [0.47, 0.0, 56.]])

    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    geometry['delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    geometry['ddelta_max'] = np.array([2., 2., 2.])

    geometry['c_root'] = 1.4 * geometry['c_ref']
    geometry['c_tip'] = 2. * geometry['c_ref'] - geometry['c_root']

    geometry['fuselage'] = True
    geometry['wing'] = True
    geometry['tail'] = True
    geometry['wing_profile'] = None

    # tether attachment point
    geometry['r_tether'] = np.zeros((3,1))

    return geometry

def aero_deriv():
    # 'numerical optimal trajectory for system in pumping mode described by differential algebraic equation (focus on ap2)' licitra, 2014

    aero_deriv = {}
    
    aero_deriv['CD0'] = 0.02

    return aero_deriv

def set_options(options):

    options['params']['tether']['sigma_max'] = 3.9e9
    options['params']['tether']['f_sigma'] = 5.0
    options['params']['tether']['rho'] = 1450.0

    options['user_options']['wind']['u_ref'] = 10.0

    options['model']['aero']['three_dof']['coeff_max'] = [1.0, 80.0 * np.pi / 180.]
    options['model']['aero']['three_dof']['coeff_min'] = [0.0, -80.0 * np.pi / 180.]

    options['model']['model_bounds']['dcoeff_max'] = [5., 5.0]
    options['model']['model_bounds']['dcoeff_min'] = [-5., -5.0]
    options['model']['system_bounds']['u']['dkappa'] = [-1000.0, 1000.0]


    return options

def battery_model_parameters(coeff_max, coeff_min):

    battery_model = {}

    # guessed values for battery model
    battery_model['flap_length'] = 0.2
    battery_model['flap_width'] = 0.1
    battery_model['max_flap_defl'] = 20.*(np.pi/180.)
    battery_model['min_flap_defl'] = -20.*(np.pi/180.)
    battery_model['c_dl'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[0] - coeff_max[0])
    battery_model['c_dphi'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[1] - coeff_max[1])
    battery_model['defl_lift_0'] = battery_model['min_flap_defl'] - battery_model['c_dl']*coeff_max[0]
    battery_model['defl_roll_0'] = battery_model['min_flap_defl'] - battery_model['c_dphi']*coeff_max[1]
    battery_model['voltage'] = 3.7
    battery_model['mAh'] = 5000.
    battery_model['charge'] = battery_model['mAh']*3600.*1e-3
    battery_model['number_of_cells'] = 15.
    battery_model['conversion_efficiency'] = 0.7
    battery_model['power_controller'] = 50.
    battery_model['power_electronics'] = 10.
    battery_model['charge_fraction'] = 1.

    return battery_model