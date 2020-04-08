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

from setuptools import setup, find_packages

import sys
print(sys.version_info)

if sys.version_info < (3,5):
    sys.exit('Python version 3.5 or later required. Exiting.')

setup(name='tunempc',
   version='0.1',
   python_requires='>=3.5, <3.8',
   description='A tool for economic tuning of tracking (N)MPC problems',
   url='https://github.com/jdeschut/tunempc',
   author='Jochem De Schutter',
   license='LGPLv3.0',
   packages = find_packages(),
  #  include_package_data = True,
  #  setup_requires=['setuptools_scm'],
  #  use_scm_version={
  #    "fallback_version": "0.1-local",
  #    "root": "../..",
  #    "relative_to": __file__
  #  },
   install_requires=[
      'numpy',
      'scipy',
      'casadi==3.5.1',
      'matplotlib',
      'picos'
   ],
  #  package_data={'': [
  #      'c_templates_tera/main.in.c',
  #      'c_templates_tera/Makefile.in',
  #      'c_templates_tera/model.in.h',
  #      'c_templates_tera/main.in.h',
  #      'c_templates_tera/acados_solver.in.c',
  #      'c_templates_tera/acados_solver.in.h',
  #      'c_templates_tera/acados_sim_solver.in.c',
  #      'c_templates_tera/acados_sim_solver.in.h',
  #      'c_templates_tera/acados_solver_sfun.in.c',
  #      'c_templates_tera/p_constraint.in.h',
  #      'c_templates_tera/h_constraint.in.h',
  #      'c_templates_tera/h_e_constraint.in.h',
  #      'c_templates_tera/phi_e_constraint.in.h',
  #      'c_templates_tera/phi_constraint.in.h',
  #      'c_templates_tera/make_sfun.in.m',
  #      'c_templates_tera/cost_y_fun.in.h',
  #      'c_templates_tera/cost_y_e_fun.in.h',
  #      'c_templates_tera/external_cost.in.h',
  #      'c_templates_tera/external_cost_e.in.h',
  #       'acados_layout.json',
  #      'acados_sim_layout.json',
  #      ]},
  #  zip_safe=False
)
