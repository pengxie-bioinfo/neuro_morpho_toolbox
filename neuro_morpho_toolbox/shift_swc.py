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
        # _ = self.get_soma()
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
        # self.swc.x = self.swc.x.round(2)
        # self.swc.y = self.swc.y.round(2)
        # self.swc.z = self.swc.z.round(2)
        self.swc.to_csv(file_name, sep=" ")
        return


if __name__=='__main__':
    # Load input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_swc', type=str, default=None, help='input swc file')
    parser.add_argument('-I', '--input_dir', type=str, default=None,
                        help='input swc directory, will overwrie -i option when specified')
    parser.add_argument('-o', '--output_swc', type=str, default=None, help='name of output file, default is [input_swc.scalingfactor.swc]')
    parser.add_argument('-O', '--output_dir', type=str, default=None,
                        help='output swc directory, will overwrie -o option when specified')
    parser.add_argument('-x', '--x_scale', type=float, default=None, help='specify scaling factor for x, x_output = x_input * x_scale')
    parser.add_argument('-y', '--y_scale', type=float, default=None, help='specify scaling factor for y, y_output = y_input * y_scale, will be set equal to x_scale, if not specified')
    parser.add_argument('-z', '--z_scale', type=float, default=None, help='specify scaling factor for z, z_output = z_input * z_scale, will be set equal to x_scale, if not specified')

    args = parser.parse_args()
    input_swc = args.input_swc  # necessary argument for single file mode
    input_dir = args.input_dir  # necessary argument for batch mode
    output_swc = args.output_swc
    output_dir = args.output_dir

    sx = args.x_scale
    sy = args.y_scale
    sz = args.z_scale
    assert sx is not None, "Error: please sepcify the scaling factor"
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    # Output tag
    if ((sx==sy) & (sx==sz)):
        scale_tag = "s"+str(sx)
    else:
        scale_tag = "x"+str(sx)+"y"+str(sy)+"z"+str(sz)
    assert ((input_swc is not None) | (input_dir is not None)), "Error: please specify the input file using -i or -I"

    # Scaling
    cwd = os.getcwd()+"/"
    hmd = os.getenv("HOME") + "/"
    if input_dir is None: # single file mode
        assert ((input_swc.lower().endswith(".swc")) | (input_swc.lower().endswith(".eswc"))), "Error: invalid suffix of input file, must ends with .swc/.eswc"
        if input_swc.startswith("~/"):
            input_swc = hmd + input_swc.replace("~/", "")
        if os.path.isfile(input_swc):
            input_swc = input_swc
        elif os.path.isfile(cwd+input_swc):
            input_swc = cwd + input_swc
        else:
            input_swc = ""
            print("Error: file specified by -i cannot be found")
        if output_swc is None:
            if input_swc.lower().endswith(".swc"):
                output_swc = input_swc.lower().replace(".swc", "."+scale_tag+".swc")
            elif input_swc.lower.endswith(".eswc"):
                output_swc = input_swc.lower().replace(".eswc", "."+scale_tag + ".eswc")
        cur_neuron = neuron(input_swc)
        cur_neuron.scale([sx, sy, sz])
        cur_neuron.save(output_swc)
    else:                # batch mode
        if input_dir.startswith("~/"):
            input_dir = hmd + input_dir.replace("~/", "")
        if input_dir.startswith("./"):
            input_dir = cwd + input_dir.split(".")[-1]
        if not input_dir.endswith("/"):
            input_dir = input_dir + "/"
        if os.path.isdir(input_dir):
            input_dir = input_dir
        elif os.path.isdir(cwd+input_dir):
            input_dir = cwd+input_dir
        else:
            input_dir = ""
            print("Error: path specified by -I cannot be found")
        if output_dir is None:
            output_dir = input_dir + scale_tag + "/"
        elif not output_dir.endswith("/"):
            output_dir = output_dir+"/"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        input_swc_list = [i for i in os.listdir(input_dir) if i.lower().endswith('swc')]
        for input_swc in input_swc_list:
            output_swc = input_swc
            # if input_swc.lower().endswith(".swc"):
            #     output_swc = input_swc.lower().replace(".swc", "."+scale_tag+".swc")
            # elif input_swc.lower().endswith(".eswc"):
            #     output_swc = input_swc.lower().replace(".eswc", "."+scale_tag + ".eswc")
            cur_neuron = neuron(input_dir+input_swc)
            # cur_neuron.scale([sx, sy, sz])
            cur_neuron.shift([-sx, -sy, -sz])
            cur_neuron.save(output_dir + output_swc)

