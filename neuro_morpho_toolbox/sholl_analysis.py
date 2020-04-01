import numpy as np
import pandas as pd
import argparse
import os

# Define classes and functions

class neuron:
    def __init__(self, file, zyx=False):
        self.file = file
        self.name = file.split("/")[-1].split(".")[0]

        n_skip = 0
        with open(self.file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        f.close()
        if zyx:
            names = ["##n", "type", "z", "y", "x", "r", "parent"]
        else:
            names = ["##n", "type", "x", "y", "z", "r", "parent"]
        swc = pd.read_csv(self.file, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          names=names
                          )
        self.swc = swc
        _ = self.get_soma()
        return

    def get_soma(self):
        soma = self.swc[((self.swc.type==1) & (self.swc.parent==-1))]
        if len(soma)!=1:
            print(("Invalid number of soma found: %d" % len(soma)))
            self.soma = pd.DataFrame({"x": np.nan,
                                      "y": np.nan,
                                      "z": np.nan}, index=[self.name])
        else:
            soma = pd.DataFrame({"x": soma.x.iloc[0],
                                 "y": soma.y.iloc[0],
                                 "z": soma.z.iloc[0]}, index=[self.name])
            self.soma = soma
        return self.soma

    def get_segments(self, custom_soma=None):
        # lab = [i for i,name in enumerate(self.swc.index.tolist()) if self.swc.loc[name, "parent"]!=(-1)]
        child = self.swc[self.swc.parent != (-1)]
        parent = self.swc.loc[child.parent]
        soma = np.array(self.soma[['x', 'y', 'z']])
        if custom_soma is not None:
            soma = custom_soma
        # Distance to soma
        cr = np.sqrt(np.sum(np.square(child[['x', 'y', 'z']] - soma), axis=1))
        pr = np.sqrt(np.sum(np.square(parent[['x', 'y', 'z']] - soma), axis=1))
        res = pd.DataFrame({"cx":child.x.tolist(),
                            "cy":child.y.tolist(),
                            "cz":child.z.tolist(),
                            "cr":cr.tolist(),
                            "c_name":child.index.tolist(),
                            "px":parent.x.tolist(),
                            "py":parent.y.tolist(),
                            "pz":parent.z.tolist(),
                            "pr":pr.tolist(),
                            "p_name":parent.index.tolist(),
                            })
        return res

    def flip(self, axis, axis_max):
        self.swc[axis] = axis_max - self.swc[axis]
        return

    def scale(self, xyz_scales, inplace=True):
        if inplace:
            self.swc["x"] = self.swc["x"] * xyz_scales[0]
            self.swc["y"] = self.swc["y"] * xyz_scales[1]
            self.swc["z"] = self.swc["z"] * xyz_scales[2]
            return self.swc
        else:
            res = self.swc.copy()
            res["x"] = self.swc["x"] * xyz_scales[0]
            res["y"] = self.swc["y"] * xyz_scales[1]
            res["z"] = self.swc["z"] * xyz_scales[2]
            return res

    def shift(self, xyz_shift, inplace=True):
        if inplace:
            self.swc["x"] = self.swc["x"] + xyz_shift[0]
            self.swc["y"] = self.swc["y"] + xyz_shift[1]
            self.swc["z"] = self.swc["z"] + xyz_shift[2]
            return self.swc
        else:
            res = self.swc.copy()
            res["x"] = res["x"] + xyz_shift[0]
            res["y"] = res["y"] + xyz_shift[1]
            res["z"] = res["z"] + xyz_shift[2]
            return res

    def save(self, file_name):
        self.swc.x = self.swc.x.round(2)
        self.swc.y = self.swc.y.round(2)
        self.swc.z = self.swc.z.round(2)
        self.swc.to_csv(file_name, sep=" ")
        return

def get_crossing(cur_neuron, step, r_max, custom_soma=None):
    df = cur_neuron.get_segments(custom_soma)
    df['is_cross'] = False
    df['at_cross'] = -1
    r_list = []
    ct_list = []
    cur_r = 0
    while cur_r < r_max:
        lab = ((df.pr <= cur_r) & (df.cr > cur_r)) | ((df.pr > cur_r) & (df.cr <= cur_r))
        df.loc[lab, 'is_cross'] = True
        df.loc[lab, 'at_cross'] = cur_r
        r_list.append(cur_r)
        ct_list.append(int(sum(lab)))
        cur_r = cur_r + step
    ct_df = pd.DataFrame({'r': r_list, 'count': ct_list})
#     # For testing purpose
#     cur_neuron.swc.loc[df[df.is_cross].p_name.tolist(), 'type'] = (df[df.is_cross].at_cross // step).tolist()
#     cur_neuron.swc.loc[df[df.is_cross].c_name.tolist(), 'type'] = (df[df.is_cross].at_cross // step).tolist()
#     cur_neuron.swc.type = cur_neuron.swc['type'].astype('int')
#     cur_neuron.save('sholl_test.swc')
    return cur_neuron, df, ct_df

if __name__=='__main__':
    # Load input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_swc', type=str, default=None, help='input swc file')
    parser.add_argument('-d', '--input_dir', type=str, default=None,
                        help='input swc directory, will overwrie -i option when specified')
    parser.add_argument('-s', '--step', type=float, default=5, help='step size for increasing shell radius')
    parser.add_argument('-r', '--r_max', type=float, default=300, help='maximum shell radius')
    parser.add_argument('-o', '--output_file', type=str, default='sholl.csv', help='name of output file')
    parser.add_argument('-X', '--soma_x', type=float, default=None, help='specify soma coordinate: x-axis, use -X "-**" for negative values')
    parser.add_argument('-Y', '--soma_y', type=float, default=None, help='specify soma coordinate: y-axis, use -Y "-**" for negative values')
    parser.add_argument('-Z', '--soma_z', type=float, default=None, help='specify soma coordinate: z-axis, use -Z "-**" for negative values')

    args = parser.parse_args()
    input_swc = args.input_swc  # necessary argument for single file mode
    input_dir = args.input_dir  # necessary argument for batch mode
    step = args.step
    r_max = args.r_max
    output_csv = args.output_file
    custom_soma = np.array([[args.soma_x,
                             args.soma_y,
                             args.soma_z]])
    if ((args.soma_x is None) | (args.soma_y is None) | (args.soma_z is None)):
        custom_soma = None

    # Sholl analysis
    assert ((input_swc is not None) | (input_dir is not None)), 'please specify -i or -d option'
    # 1. Batch mode
    if input_dir is not None:
        swc_list = [i for i in os.listdir(input_dir) if i.endswith('swc')]
    # 2. Single file mode
    elif input_swc is not None:
        input_dir = "/".join(input_swc.split("/")[:-1])+"/"
        swc_list = [input_swc.split("/")[-1]]

    ct_df = None
    for cur_swc in swc_list:
        cur_neuron = neuron(input_dir + cur_swc)
        _, _, tp = get_crossing(cur_neuron,
                                step=step,
                                r_max=r_max,
                                custom_soma=custom_soma)
        if ct_df is None:
            ct_df = pd.DataFrame(index=tp.index)
            ct_df['r'] = tp['r'].tolist()
        ct_df[cur_swc.split('.')[0]] = tp['count'].tolist()

    ct_df.rename(columns={'r':'Distance_to_soma'}, inplace=True)
    ct_df.to_csv(output_csv)

    # # For testing purposes
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # for cur_swc in swc_list:
    #     cur_name = cur_swc.split('.')[0]
    #     ax.plot(ct_df['Distance_to_soma'], ct_df[cur_name], label=cur_name)
    # ax.legend(loc='upper right')
    # ax.set_xlabel('Distance to soma (um)')
    # ax.set_ylabel('Number of crossings')
    # plt.show()

