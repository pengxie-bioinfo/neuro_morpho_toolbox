
def load_swc(swc_file):
    n_skip = 0
    with open(swc_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                n_skip += 1
            else:
                break
    f.close()
    swc = pd.read_csv(swc_file, index_col=0, skiprows=n_skip, sep=" ",
                      usecols=[0,1,2,3,4,5,6],
                      names=["", "type", "x", "y", "z", "r", "parent"]
                     )
    return swc

class swc:
    def __init__(self, file):
        self.file = file
        self.name = file.

        n_skip = 0
        with open(swc_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("#"):
                    n_skip += 1
                else:
                    break
        f.close()
        swc = pd.read_csv(swc_file, index_col=0, skiprows=n_skip, sep=" ",
                          usecols=[0, 1, 2, 3, 4, 5, 6],
                          names=["", "type", "x", "y", "z", "r", "parent"]
                          )
