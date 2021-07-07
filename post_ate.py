import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
sys.path.append("/home/cvv5110/Packages/jenny/")
from python.jnPostproc import postproc

def plot_fields(cfd_field, gpr_field, ana_field, **args):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sign = args['sign']
    index = args['index']
    for _i in range(len(args['times'])):
        time = args['times'][_i]
        ax.plot(cfd.xp, sign*cfd_field[:,index,time], linestyle='-', color=colors[_i])
        ax.plot(gpr.coor[0:100,0], sign*gpr_field[index,time,0:100], linestyle='--', color=colors[_i])
        #ax.plot(ana.xp, sign*ana_field[:,index,time],linestyle='-.')
    ax.set_xlabel('x, meters', size=14)
    ax.set_ylabel(args['title'], size=14)

cfd_dir = "/home/cvv5110/Packages/simg/test/hypate/ate_res/"
gpr_dir = "/home/cvv5110/Packages/interface/res/"
ana_dir = "/home/cvv5110/Packages/simg/test/hypate/l1_res/"

cfd = postproc(fileName=cfd_dir+"ate2d_l3_F.hdf")
gpr = postproc(fileName=gpr_dir+"ht2d_l22_F.hdf")
ana = postproc(fileName=ana_dir+"ate2d_l1_F.hdf")
colors = ['k', 'coral', 'orange', 'green', 'blue', 'orchid', 'r', 'gold', 'pink', 'brown']

ana.probeAt(xlim=[-0.5,0.5], ylim=0.01)
cfd.probeAt(xlim=[-0.5,0.5], ylim=0.01)
ana_data = ana.prbData
cfd_data = cfd.prbData
gpr_data = np.array(gpr.rawDatVec)

plot_fields(cfd_data, gpr_data, ana_data, index=2, times=[100,200,300,400,500,600,700,800,900], sign=1, title='Displacement')
plt.show()

plot_fields(cfd_data, gpr_data, ana_data, index=0, times=[100, 200, 300, 400, 500, 600], sign=-1, title='Force')
plot_fields(cfd_data, gpr_data, ana_data, index=1, times=[100, 200, 300, 400, 500, 600], sign=1, title='Heat Flux')
plt.show()


print()
