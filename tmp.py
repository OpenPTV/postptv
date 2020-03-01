


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from flowtracks.io import trajectories_ptvis

inName = './data/particles/xuap.%d'
trajects = trajectories_ptvis(inName, xuap = True) # the directory with the input files

############
# PLOTTING #
############

fig = pyplot.figure(figsize = (12, 10))
# fig.canvas.manager.window.move(900, 300)
ax = Axes3D(fig)
ax.view_init(35, -160) # view angle: (rotation of z-axis, rotation around z-axis)
mpl.rcParams['legend.fontsize'] = 10

positions = [tr.pos() for tr in trajects]
for pos in positions:
	trajX = [pos[k][0] for k in range(len(pos))]
	trajY = [pos[k][1] for k in range(len(pos))]
	trajZ = [pos[k][2] for k in range(len(pos))]
	ax.plot(trajX, trajY, trajZ, '.')
#ax.plot([], [], [], '.', label = 'Trajectories')

ax.legend()
pyplot.show()

