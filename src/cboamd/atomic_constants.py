# Copyright (C) 2018 Johannes Flick
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# constants are from octopus
# src/basic/global.F90
# src/basic/unit_system.F90

P_a_B =  0.52917720859
P_Ang =  1.0 / P_a_B
P_Ry  =  13.60569193
P_eV  =  1.0 / P_Ry
P_Kb  =  (8.617343e-5)/(2.0*P_Ry)  # Boltzmann constant in Ha/K
P_c   =  137.035999679
P_g   =  2.0023193043768
P_pe =  1/(0.0005485799110) #'1/12 of the mass of C^12'


# my definitions

P_Har  =  2.0*P_Ry

# 1 Hartree expressed in kcal/mol (unit-system definition; used when
# converting kcal/mol data from classical force fields)
P_Har_kcalmol = 627.5094740631

# octopus
# share/pseudopotentials/elements.dat
dicct_atomic_mass = {\
'H': 1.00784,\
'D': 2.01410178,\
'T': 3.0160492,\
'He': 4.0026022,\
'Li': 6.938,\
'Be': 9.01218315,\
'B': 10.806,\
'C': 12.0096,\
'N': 14.00643,\
'O': 15.99903,\
'F': 18.9984031636,\
'Ne': 20.17976,\
'Na': 22.989769282,\
'Mg': 24.304,\
'Al': 26.98153857,\
'Si': 28.084, \
'Ce': 140.116} 

#unit_invcm%factor = M_ONE/CNST(219474.63)
#src/basic/unit_system.F90:    unit_invcm%abbrev = 'cm^-1'
#src/basic/unit_system.F90:    unit_invcm%name   = 'h times c over centimeters'


# not from octopus
P_fs = 0.6582119514
P_cm1 = 219474.63068	
