import numpy as np
import os

if not os.path.isdir("deepmd"):
    os.system('mkdir deepmd')

if not os.path.isdir("deepmd/training_data/set.000"):
    os.system('mkdir deepmd/training_data/set.000')

if not os.path.isdir("deepmd/validation_data/set.000"):
    os.system('mkdir deepmd/validation_data/set.000')


#def to_npy(filename, which, training):
#    if which == 'energy':
#        dim = 1
#        load = 'energy.dat'
#        save = 'energy.npy'
#    obs = np.loadtxt(load)
#    array = np.zeros((len(obs), dim))
#    array[:,0] = energy[:,2]
#    np.save('deepmd/' + save, array)
#    obs = np.load('deepmd/' + save)
#    print('shape', obs.shape)
#    print(obs)
#    return

import ase.io as aio
atoms = aio.read('../../geometries/co2-jb-augccpvdz-disp.xyz')
symbols = atoms.get_chemical_symbols()
type_map_raw = np.unique(symbols)
with open("deepmd/type_map.raw", "w") as f:
    for key in type_map_raw:
        f.write(key + '\n')

type_raw = []

for ii in range(0, len(symbols)):
    count = 0
    for jj in range(0, len(type_map_raw)):     
        if symbols[ii] == type_map_raw[jj]:
            type_raw.append(count)
        count += 1

with open("deepmd/type.raw", "w") as f:
    for key in type_raw:
        f.write(str(key) + '\n')


training = 1900 #rest will go to validation

energy = np.loadtxt('energy.dat')
array = np.zeros((len(energy), 1))

ranorder = np.arange(0, len(energy))
print(ranorder)
import random
random.shuffle(ranorder)
print(ranorder)

array[:,0] = energy[ranorder,2]
np.save('deepmd/energy.npy', array)
energy = np.load('deepmd/energy.npy')
print('energy shape', energy.shape)
print(energy)
np.save('deepmd/training_data/set.000/energy.npy', energy[:training])
np.save('deepmd/validation_data/set.000/energy.npy', energy[training:])

force = np.loadtxt('force.dat')
array = np.zeros((len(force), 9))
array[:,:] = force[ranorder,2:]
np.save('deepmd/force.npy', array)
force = np.load('deepmd/force.npy')
print('force shape', force.shape)
print(force)
np.save('deepmd/training_data/set.000/force.npy', force[:training])
np.save('deepmd/validation_data/set.000/force.npy', force[training:])

position = np.loadtxt('position.dat')
array = np.zeros((len(position), 9))
array[:,:] = position[ranorder,2:11]
np.save('deepmd/coord.npy', array)
coord = np.load('deepmd/coord.npy')
print('coord shape', coord.shape)
print(coord)
np.save('deepmd/training_data/set.000/coord.npy', coord[:training])
np.save('deepmd/validation_data/set.000/coord.npy', coord[training:])

dipole = np.loadtxt('dipole.dat')
array = np.zeros((len(position), 3))
array[:,:] = dipole[ranorder,2:5]
np.save('deepmd/dipole.npy', array)
dipole = np.load('deepmd/dipole.npy')
print('dipole shape', dipole.shape)
print(dipole)
np.save('deepmd/training_data/set.000/dipole.npy', dipole[:training])
np.save('deepmd/validation_data/set.000/dipole.npy', dipole[training:])

polar = np.loadtxt('polarizability.dat')
array = np.zeros((len(position), 9))
array[:,:] = polar[ranorder,2:11]
np.save('deepmd/polarizability.npy', array)
polar = np.load('deepmd/polarizability.npy')
print('polar shape', polar.shape)
print(polar)
np.save('deepmd/training_data/set.000/polarizability.npy', polar[:training])
np.save('deepmd/validation_data/set.000/polarizability.npy', polar[training:])


array = np.zeros((len(position), 9))
array[:,[0,4,8]] = 10
np.save('deepmd/box.npy', array)
coord = np.load('deepmd/box.npy')
print('box shape', coord.shape)
print(coord)
np.save('deepmd/training_data/set.000/box.npy', coord[:training])
np.save('deepmd/validation_data/set.000/box.npy', coord[training:])

