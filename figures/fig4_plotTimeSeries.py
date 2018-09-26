"""

Plot a time series of Sx and Sy (at a single grid point) using 
the predictions from the neural networks from three different 
regions, and compare with the truth.

Also plot time series of the spatially-averaged Sx and Sy, which
tells you how much of a source/sink of momentum each neural network
is contributing as a function of time.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath
import scipy.io as sio

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )

##### Load Data #####

# load data
predsX1 = np.load('../data/Predictions/Sx_Pred_R1_str2.npz')
predsX2 = np.load('../data/Predictions/Sx_Pred_R2_str2.npz')
predsX3 = np.load('../data/Predictions/Sx_Pred_R3_str2.npz')

predsY1 = np.load('../data/Predictions/Sy_Pred_R1_str2.npz')
predsY2 = np.load('../data/Predictions/Sy_Pred_R2_str2.npz')
predsY3 = np.load('../data/Predictions/Sy_Pred_R3_str2.npz')

truth = np.load('../data/Predictions/SxSy_True_str2.npz')

SxTrue, SyTrue = truth['SxT'], truth['SyT']

SxPred1, SyPred1 = predsX1['SP'], predsY1['SP']
SxPred2, SyPred2 = predsX2['SP'], predsY2['SP']
SxPred3, SyPred3 = predsX3['SP'], predsY3['SP']

# choose a point in space to plot (in km from origin)
x0, y0 = 1920, 2400

# convert x0 and y0 to grid points
x0, y0 = int( x0/7.5 ), int( y0/7.5 )

##### Plotting: Time-Series #####

t = np.linspace(1,350,350)

fig, (ax0,ax1,ax2,ax3) = plt.subplots( 4, 1, sharex=True )

bgColor = 'white'
gridColor = 'lightgray'

C = [ 'black', 'red', 'blue', 'gray' ]


# Sx
ax0.plot( t, np.squeeze( SxTrue[:,y0,x0] )*1e6, label='Truth', color=C[0], linewidth=2.5, ls='--' )
ax0.plot( t, np.squeeze( SxPred1[:,y0,x0] )*1e6, label='CNN$_x$1', color=C[1], linewidth=1.5 )
ax0.plot( t, np.squeeze( SxPred2[:,y0,x0] )*1e6, label='CNN$_x$2', color=C[2], linewidth=1.5 )
ax0.plot( t, np.squeeze( SxPred3[:,y0,x0] )*1e6, label='CNN$_x$3', color=C[3], linewidth=1.5)

ax0.xaxis.grid( color=gridColor, lw=1 )
ax0.set_title(r'$S_x$ and $\tilde{S}_x$ Time-Series (at $x=1920$km, $y=2400$km)', fontsize=12 )
ax0.set_xlim( (0,350) )
ax0.set_facecolor( bgColor )
ax0.set_ylabel(r'$S_x$ or $\tilde{S}_x$'+'\n'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax0.legend( shadow=True, loc=1, bbox_to_anchor=(1.15,1), facecolor=bgColor  )
ax0.text(3,1.32,'a.',fontsize=12)


# Sy
ax1.plot( t, np.squeeze( SyTrue[:,y0,x0] )*1e6, label='Truth', color=C[0], linewidth=2.5, ls='--' )
ax1.plot( t, np.squeeze( SyPred1[:,y0,x0] )*1e6, label='CNN$_y$1', color=C[1], linewidth=1.5 )
ax1.plot( t, np.squeeze( SyPred2[:,y0,x0] )*1e6, label='CNN$_y$2', color=C[2], linewidth=1.5 )
ax1.plot( t, np.squeeze( SyPred3[:,y0,x0] )*1e6, label='CNN$_y$3', color=C[3], linewidth=1.5 )

ax1.xaxis.grid( color=gridColor, lw=1 )
ax1.set_title(r'$S_y$ and $\tilde{S}_y$ Time-Series (at $x=1920$km, $y=2400$km)', fontsize=12 )
ax1.set_xlim( (0,350) )
ax1.set_facecolor( bgColor )
ax1.set_ylabel(r'$S_y$ or $\tilde{S}_y$'+'\n'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax1.legend( shadow=True, loc=1, bbox_to_anchor=(1.15,1), facecolor=bgColor  )
ax1.text(3,0.8,'b.',fontsize=12)


yLim = 0.035

# U Momentum input
ax2.plot( t, np.mean( SxTrue, axis=(1,2) )*1e6, label='Truth', color=C[0], linewidth=2.5, ls='--' )
ax2.plot( t, np.mean( SxPred1, axis=(1,2) )*1e6, label='$f_x(\overline{\psi},\mathbf{w}_1)$', color=C[1], linewidth=1.5 )
ax2.plot( t, np.mean( SxPred2, axis=(1,2) )*1e6, label='$f_x(\overline{\psi},\mathbf{w}_2)$', color=C[2], linewidth=1.5 )
ax2.plot( t, np.mean( SxPred3, axis=(1,2) )*1e6, label='$f_x(\overline{\psi},\mathbf{w}_3)$', color=C[3], linewidth=1.5 )

ax2.xaxis.grid( color=gridColor, lw=1 )
#ax0.yaxis.grid( color=gridColor, lw=1 )
ax2.set_title(r'Spatially-Averaged $S_x$ and $\tilde{S}_x$', fontsize=12 )
ax2.set_xlim( (0,350) )
ax2.set_ylim( (-yLim,yLim) )
ax2.set_facecolor( bgColor )
ax2.set_ylabel(r'Spatial-Mean $S_x$ or $\tilde{S}_x$'+'\n'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax2.legend( shadow=True, loc=1, bbox_to_anchor=(1.15,1), facecolor=bgColor  )
ax2.text(3,0.037,'c.',fontsize=12)



# V Momentum input
ax3.plot( t, np.mean( SyTrue, axis=(1,2) )*1e6, label='Truth', color=C[0], linewidth=2.5, ls='--' )
ax3.plot( t, np.mean( SyPred1, axis=(1,2) )*1e6, label='$f_y(\overline{\psi},\mathbf{w}_1)$', color=C[1], linewidth=1.5 )
ax3.plot( t, np.mean( SyPred2, axis=(1,2) )*1e6, label='$f_y(\overline{\psi},\mathbf{w}_2)$', color=C[2], linewidth=1.5 )
ax3.plot( t, np.mean( SyPred3, axis=(1,2) )*1e6, label='$f_y(\overline{\psi},\mathbf{w}_3)$', color=C[3], linewidth=1.5 )

ax3.xaxis.grid( color=gridColor, lw=1 )
ax3.set_title(r'Spatially-Averaged $S_y$ and $\tilde{S}_y$', fontsize=12 )
ax3.set_xlim( (0,350) )
ax3.set_ylim( (-yLim,yLim) )
ax3.set_facecolor( bgColor )
ax3.xaxis.set_label_coords(0.5, -0.09)
ax3.set_xlabel('Day')
ax3.set_ylabel(r'Spatial-Mean $S_y$ or $\tilde{S}_y$'+'\n'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax3.legend( shadow=True, loc=1, bbox_to_anchor=(1.15,1), facecolor=bgColor  )
ax3.text(3,0.037,'d.',fontsize=12)


fig.set_size_inches( 11, 9, forward=True )
plt.subplots_adjust( left=0.15, top=0.95, right=0.85, hspace=0.4 )

# before saving figure, move bottom two subplots down a bit and
# add a line separating top from bottom
pos1 = ax1.get_position()
pos2 = ax2.get_position()
ax2.set_position( [ pos2.x0, pos2.y0-0.02, pos2.width, pos2.height ] )
pos3 = ax3.get_position()
ax3.set_position( [ pos3.x0, pos3.y0-0.02, pos3.width, pos3.height ] )

ax2.plot( [0.045,0.96], [ 0.53, 0.53 ], color='black', clip_on=False, transform=fig.transFigure, solid_capstyle='round' )


plt.savefig('timeSeries.png', format='png', dpi=300 )


plt.show()
