%
%    This file is part of TuneMPC.
%
%    TuneMPC -- A Tool for Economic Tuning of Tracking (N)MPC Problems.
%    Copyright (C) 2020 Jochem De Schutter, Mario Zanon, Moritz Diehl (ALU Freiburg).
%
%    TuneMPC is free software; you can redistribute it and/or
%    modify it under the terms of the GNU Lesser General Public
%    License as published by the Free Software Foundation; either
%    version 3 of the License, or (at your option) any later version.
%
%    TuneMPC is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%    Lesser General Public License for more details.
%
%    You should have received a copy of the GNU Lesser General Public
%    License along with TuneMPC; if not, write to the Free Software Foundation,
%    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
%

% function that converts general DAE model to GNSF-structured model
% @author: Jonathan Frey
function convert(json_filename)
    dae_json_to_gnsf_json( json_filename );
end