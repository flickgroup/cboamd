import numpy as np
import os

data_sys = ["training", "validation"]
nep_sys = ["train", "test"]

coord_train = None
coord_test = None

Eshift = -5093.82

def data_to_nep(data):
    out_str = ""
    header = '\nLattice="ABC" Properties=species:S:1:pos:R:3:force:R:3 energy=ENER dipole="DIP" pol="POL" pbc="F F F"\n'
    for iframe in range(data["nframe"]):
        out_str += str(data["natom"])
        print(data["box"][iframe].shape)
        box_str = np.array2string(data["box"][iframe], separator=' ', max_line_width=np.inf, formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
        print(box_str)
        dip_str = np.array2string(data["dipole"][iframe], separator=' ', max_line_width=np.inf, formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
        pol_str = np.array2string(data["polarizability"][iframe], separator=' ', max_line_width=np.inf, formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
        header_ = header.replace("ABC", box_str)
        header_ = header_.replace("ENER", f"{data['energy'][iframe][0]:.6e}")
        header_ = header_.replace("DIP", dip_str)
        header_ = header_.replace("POL", pol_str)
        out_str += header_
        # out_str += "\n"
        for iatom in range(data["natom"]):
            coord_str = np.array2string(data['coord'][iframe, np.arange(iatom*3, iatom*3+3)], separator=' ', formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
            force_str = np.array2string(data['force'][iframe, np.arange(iatom*3, iatom*3+3)], separator=' ', formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
            # apol_str = np.array2string(data['atomic_polarizability'][iframe, np.arange(iatom*9, iatom*9+9)], separator=' ', formatter={'float_kind': lambda x: f"{x:.6e}"})[1:-1]
            # print(iatom, apol_str)
            out_str += f"{data['type_map'][data['atype'][iatom]]} {coord_str} {force_str}\n"
        # out_str += "\n"
    return out_str

for idx_sys, data_dir in enumerate(data_sys):
    print(f"idx_sys: {idx_sys}, data_dir: {data_dir}")
    dir_ = os.path.join("deepmd", data_dir+"_data", "set.000")
    dir_atomic = dir_
    # dir_global = dir_.replace("ABC", "global")
    # print(dir_atomic)
    # print(dir_global)

    data = {}
    data["coord"] = np.load(os.path.join(dir_atomic, "coord.npy"))
    data["energy"] = np.load(os.path.join(dir_atomic, "energy.npy")) - Eshift
    data["force"] = np.load(os.path.join(dir_atomic, "force.npy"))
    data["box"] = np.load(os.path.join(dir_atomic, "box.npy"))
    data["polarizability"] = np.load(os.path.join(dir_atomic, "polarizability.npy"))
    data["dipole"] = np.load(os.path.join(dir_atomic, "dipole.npy"))
    data["atype"] = np.loadtxt(os.path.join(os.path.dirname(dir_atomic), "type.raw"), dtype=int)
    data["type_map"] = np.loadtxt(os.path.join(os.path.dirname(dir_atomic), "type_map.raw"), dtype=str)

    nframe = data["coord"].shape[0]
    natom = int(data["coord"].shape[1] / 3)

    data["nframe"] = nframe
    data["natom"] = natom

    assert nframe == data["coord"].shape[0]
    assert nframe == data["box"].shape[0] 
    assert nframe == data["polarizability"].shape[0] 
    assert natom == data["coord"].shape[1] / 3 
    print(data["coord"].shape)
    print(data["box"].shape)
    print(data["polarizability"].shape)

    nep_data = data_to_nep(data)
    with open(os.path.join("nep_data", nep_sys[idx_sys]+".xyz"), "w") as f:
        f.write(nep_data)

#     if idx_sys == 0:
#         coord_train = data["coord"]
#     else:
#         coord_test = data["coord"]

# print(coord_train.shape)
# print(coord_test.shape)
# print(coord_train - coord_test)


    
