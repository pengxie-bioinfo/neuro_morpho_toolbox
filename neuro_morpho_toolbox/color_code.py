import random
import colorlover as cl
import matplotlib

bupu = cl.scales['9']['seq']['BuPu']
greens = cl.scales['9']['seq']['Greens']
set2 = cl.scales['7']['qual']['Set2']
spectral = cl.scales['9']['div']['Spectral']
paired = cl.scales['10']['qual']['Paired']
mpl_colors = []
for i in range(9):
    tp = []
    for j in list(matplotlib.colors.to_rgb("C"+str(i))):
        tp.append(str(int(j*255)))
    tp = ", ".join(tp)
    tp = "rgb(" + tp + ")"
    mpl_colors.append(tp)


# Assign random color to single cells
def single_cell_colors(cell_list):
    random.shuffle(cell_list)
    single_cell_color_dict = dict(zip(cell_list, cl.to_rgb(cl.interp( paired, len(cell_list) ))))
    return single_cell_color_dict

