import numpy as np
from cboamd.atomic_constants import P_cm1, P_Ang, P_Har
import argparse

def fourier_transform(signal, dt, sigma=0, ws = 0):
  if sigma != 0:
    damp = np.exp(-(dt*np.arange(len(signal)))/float(sigma))
    signal *= damp
  NN=len(signal)
  ws=2*np.pi/NN
  wnorm = np.arange(-np.pi,np.pi,ws)
  wnorm=wnorm[0:NN]
  w=wnorm/dt
  signal=np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))*dt
  return w, signal

def main():
    parser = argparse.ArgumentParser(description="Parse mode name.")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="ase",
        help="Name of the mode (default: ase)"
    )

    args = parser.parse_args()
    mode = args.mode

    if mode == 'octopus':
        data = np.loadtxt('td.general/multipoles')
        dt = (data[1,1] - data[0,1])*10*P_Har #/27.21#z) #10/0.6582119514)#*
        signal = (data[:,3])/P_Ang
    elif mode == 'ase':
        dipole = np.loadtxt('dipole.dat')
        dt = dipole[1,1] - dipole[0,1]
        signal = dipole[:,2]
    elif mode == 'auto':

        dipole = np.loadtxt('dipole.dat')
        dt = dipole[1,1] - dipole[0,1]
        from ase.io.trajectory import Trajectory
        traj = Trajectory('md.traj')

        signal = []
        icount = 0
        for atoms in traj:
            if icount == 0:
                velo0 = atoms.get_velocities()
            else:
                velo = atoms.get_velocities()
                auto = 0
                for ii in range(0, len(velo)):
                    auto += (velo[ii] @ velo0[ii])
                signal.append(auto)
            icount += 1

    print('dt', dt)
    wx, signalx = fourier_transform(signal, dt)


    np.savetxt(f'spectrum_{mode}.dat', np.c_[wx*P_cm1, abs(signalx)])

if __name__ == "__main__":
    main()