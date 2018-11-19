"""
Plot the time-series and std dev of the predictions generated 
by the momentum conversing approaches A, B, and C.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import matplotlib.patheffects as mpath

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )

##### Make Predictions #####

# load data
predsXA = np.load('../data/Predictions/Sx_Pred_R1_str2_MOM_A.npz')
predsXB = np.load('../data/Predictions/Sx_Pred_R1_str2_MOM_B.npz')
predsXC = np.load('../data/Predictions/Sx_Pred_R1_str2.npz')

predsYA = np.load('../data/Predictions/Sy_Pred_R1_str2_MOM_A.npz')
predsYB = np.load('../data/Predictions/Sy_Pred_R1_str2_MOM_B.npz')
predsYC = np.load('../data/Predictions/Sy_Pred_R1_str2.npz')

truth = np.load('../data/Predictions/SxSy_True_str2.npz')

SxTrue, SyTrue = truth['SxT'], truth['SyT']

SxPredA, SyPredA = predsXA['SP'], predsYA['SP']
SxPredB, SyPredB = predsXB['SP'], predsYB['SP']
SxPredC, SyPredC = predsXC['SP'], predsYC['SP']

# post-processing of approach C
SxPredC = SxPredC - np.mean( SxPredC, axis=(1,2), keepdims=True )
SyPredC = SyPredC - np.mean( SyPredC, axis=(1,2), keepdims=True )


print "Mean zonal bias of approach A: ", np.mean( np.mean( SxPredA, axis=(1,2) ) )*1e6
print "Mean zonal bias of approach B: ", np.mean( np.mean( SxPredB, axis=(1,2) ) )*1e6
print "Mean zonal bias of approach C: ", np.mean( np.mean( SxPredC, axis=(1,2) ) )*1e6

print "Mean meridional bias of approach A: ", np.mean( np.mean( SyPredA, axis=(1,2) ) )*1e6
print "Mean meridional bias of approach B: ", np.mean( np.mean( SyPredB, axis=(1,2) ) )*1e6
print "Mean meridional bias of approach C: ", np.mean( np.mean( SyPredC, axis=(1,2) ) )*1e6

##### Plotting #####

t = np.linspace(1,350,350)

ax0 = plt.subplot2grid( (4,3), (0,0), colspan=1 )
ax1 = plt.subplot2grid( (4,3), (0,1), colspan=1 )
ax2 = plt.subplot2grid( (4,3), (0,2), colspan=1 )
ax345 = plt.subplot2grid( (4,3), (1,0), colspan=3 )

ax6 = plt.subplot2grid( (4,3), (2,0), colspan=1 )
ax7 = plt.subplot2grid( (4,3), (2,1), colspan=1 )
ax8 = plt.subplot2grid( (4,3), (2,2), colspan=1 )
ax91011 = plt.subplot2grid( (4,3), (3,0), colspan=3 )

bgColor = 'white'
gridColor = 'lightgray'

##### GLOBAL MOMENTUM INPUT #####

lineColors = [ 'blue', 'darkred', 'green' ]

yLim = 0.035

# U Momentum input
ax345.plot( t, np.mean( SxTrue, axis=(1,2) )*1e6, label='Truth', color='black', linewidth=3, linestyle='--', zorder=4 )
ax345.plot( t, np.mean( SxPredA, axis=(1,2) )*1e6, label='A: $f_x(\overline{\psi},\mathbf{w}_1^A)$', color=lineColors[0], linewidth=1.5, zorder=3 )
ax345.plot( t, np.mean( SxPredB, axis=(1,2) )*1e6, label='B: $f_x(\overline{\psi},\mathbf{w}_1^B)$', color=lineColors[1], linewidth=1.5, zorder=2 )
ax345.plot( t, np.mean( SxPredC, axis=(1,2) )*1e6, label='C: $f_x(\overline{\psi},\mathbf{w}_1^C)$', color=lineColors[2], linewidth=2, zorder=1 )


ax345.xaxis.grid( color=gridColor, lw=1 )
ax345.set_title(r'Spatially-Averaged $S_x$ and $\tilde{S}_x$', fontsize=9 )
ax345.set_xlim( (0,350) )
ax345.set_ylim( (-yLim,yLim) )
ax345.set_facecolor( bgColor )
ax345.set_xticklabels( [] )
ax345.xaxis.set_label_coords(0.5, -0.07)
ax345.set_ylabel(r'Spatial-Mean $S_x$ or $\tilde{S}_x$ (10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax345.legend( shadow=True, loc=1, bbox_to_anchor=(1.2,1), facecolor=bgColor, fontsize=8  )
ax345.text( 6, 0.025, 'd.', fontsize=12 )


# V Momentum input
ax91011.plot( t, np.mean( SyTrue, axis=(1,2) )*1e6, label='Truth', color='black', linewidth=3, linestyle='--', zorder=4 )
ax91011.plot( t, np.mean( SyPredA, axis=(1,2) )*1e6, label='$A: f_y(\overline{\psi},\mathbf{w}_1^A)$', color=lineColors[0], linewidth=1.5, zorder=3 )
ax91011.plot( t, np.mean( SyPredB, axis=(1,2) )*1e6, label='$B: f_y(\overline{\psi},\mathbf{w}_1^B)$', color=lineColors[1], linewidth=1.5, zorder=2 )
ax91011.plot( t, np.mean( SyPredC, axis=(1,2) )*1e6, label='$C: f_y(\overline{\psi},\mathbf{w}_1^C)$', color=lineColors[2], linewidth=2, zorder=1 )


ax91011.xaxis.grid( color=gridColor, lw=1 )
ax91011.set_title(r'Spatially-Averaged $S_y$ and $\tilde{S}_y$', fontsize=9 )
ax91011.set_xlim( (0,350) )
ax91011.set_ylim( (-yLim,yLim) )
ax91011.set_facecolor( bgColor )
ax91011.xaxis.set_label_coords(0.5, -0.07)
ax91011.set_xlabel('Day')
ax91011.set_ylabel(r'Spatial-Mean $S_y$ or $\tilde{S}_y$ (10$^{-6}$ms$^{-2}$)', fontsize=9 )
ax91011.legend( shadow=True, loc=1, bbox_to_anchor=(1.205,1), facecolor=bgColor, fontsize=8  )
ax91011.text( 6, 0.025, 'h.', fontsize=12 )

##### STANDARD DEVIATIONS OF MOMENTUM-CONSERVING APPROACHES #####

x0, y0 = 13, 450
cLim = 1.4
cMap = 'magma_r'

dx = 0.04

# Std Predicted Sx (training region 1)
im0 = ax0.imshow( np.squeeze( np.std( SxPredA, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax0.set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_1^A)$', fontsize=9)
ax0.set_yticks( (0,255,511) )
ax0.set_yticklabels( ('0km','1920km','3840km'), fontsize=9 )
ax0.set_xticks( (0,255,511) )
ax0.set_xticklabels( [] )
ax0.text( x0, y0, 'a.', fontsize=12 )
ax0.text( 13, 15, 'Approach A', fontsize=12 )

# Std Predicted Sx (training region 2)
im1 = ax1.imshow( np.squeeze( np.std( SxPredB, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax1.set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_1^B)$', fontsize=9 )
ax1.set_yticks( (0,255,511) )
ax1.set_yticklabels( [] )
ax1.set_xticks( (0,255,511) )
ax1.set_xticklabels( [] )
ax1.text( x0, y0, 'b.', fontsize=12 )
ax1.text( 13, 15, 'Approach B', fontsize=12 )

# Std Predicted Sx (training region 3)
im2 = ax2.imshow( np.squeeze( np.std( SxPredC, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax2.set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_1^C)$', fontsize=9 )
ax2.set_yticks( (0,255,511) )
ax2.set_yticklabels( [] )
ax2.set_xticks( (0,255,511) )
ax2.set_xticklabels( [] )
ax2.text( x0, y0, 'c.', fontsize=12 )
ax2.text( 13, 15, 'Approach C', fontsize=12 )


fig = plt.gcf()
fig.set_size_inches( 8, 9, forward=True )
plt.subplots_adjust( right=0.85, hspace=0.4, top=0.93, left=0.12 )


# adjust position and size of subplots
pos = ax0.get_position()
pos = [ pos.x0, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax0.set_position( pos )

pos = ax1.get_position()
pos = [ pos.x0-0.5*dx, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax1.set_position( pos )

pos = ax2.get_position()
pos = [ pos.x0-dx, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax2.set_position( pos )

# colorbar row 0
pos = ax2.get_position()
cax = fig.add_axes( [ pos.x0+pos.width - 0.06, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im2, ax=cax )
cBar.ax.set_ylabel(r'Std Dev $\tilde{S}_x$'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )

## STD DEV Sy

x0, y0 = 13, 460
cLim = 1
cMap = 'magma_r'

dx = 0.04

# Std Predicted Sy (training region 1)
im6 = ax6.imshow( np.squeeze( np.std( SyPredA, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax6.set_title(r'Std Dev $f_y(\overline{\psi},\mathbf{w}_1^A)$', fontsize=9)
ax6.set_yticks( (0,255,511) )
ax6.set_yticklabels( ('0km','1920km','3840km'), fontsize=9 )
ax6.set_xticks( (0,255,511) )
ax6.set_xticklabels( [] )
ax6.text( x0, y0, 'e.', fontsize=12 )
ax6.text( 13, 15, 'Approach A', fontsize=12 )

# Std Predicted Sy (training region 2)
im7 = ax7.imshow( np.squeeze( np.std( SyPredB, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax7.set_title(r'Std Dev $f_y(\overline{\psi},\mathbf{w}_1^B)$', fontsize=9 )
ax7.set_yticks( (0,255,511) )
ax7.set_yticklabels( [] )
ax7.set_xticks( (0,255,511) )
ax7.set_xticklabels( [] )
ax7.text( x0, y0, 'f.', fontsize=12 )
ax7.text( 13, 15, 'Approach B', fontsize=12 )

# Std Predicted Sy (training region 3)
im8 = ax8.imshow( np.squeeze( np.std( SyPredC, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
ax8.set_title(r'Std Dev $f_y(\overline{\psi},\mathbf{w}_1^C)$', fontsize=9 )
ax8.set_yticks( (0,255,511) )
ax8.set_yticklabels( [] )
ax8.set_xticks( (0,255,511) )
ax8.set_xticklabels( [] )
ax8.text( x0, y0, 'g.', fontsize=12 )
ax8.text( 13, 15, 'Approach C', fontsize=12 )


# adjust position and size of subplots
pos = ax6.get_position()
pos = [ pos.x0, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax6.set_position( pos )

pos = ax7.get_position()
pos = [ pos.x0-0.5*dx, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax7.set_position( pos )

pos = ax8.get_position()
pos = [ pos.x0-dx, pos.y0-0.5*dx, pos.width+dx, pos.height+dx ]
ax8.set_position( pos )

# colorbar row 2
pos = ax8.get_position()
cax = fig.add_axes( [ pos.x0+pos.width - 0.06, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im8, ax=cax )
cBar.ax.set_ylabel(r'Std Dev $\tilde{S}_x$'+'(10$^{-6}$ms$^{-2}$)', fontsize=9 )

plt.savefig('momABC.png', format='png', dpi=300 )

plt.show()
