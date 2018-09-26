"""

Compare Sx from the truth and the predictions of the three neural
networks by plotting the following diagnoistics:

- Snapshots.
- Time-means.
- Standard-deviations.
- Correlation between truth and predictions.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath
import scipy.io as sio

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )


##### Load data #####

# load data
preds1 = np.load('../data/Predictions/Sx_Pred_R1_str2.npz')
preds2 = np.load('../data/Predictions/Sx_Pred_R2_str2.npz')
preds3 = np.load('../data/Predictions/Sx_Pred_R3_str2.npz')

truth = np.load('../data/Predictions/SxSy_True_str2.npz')

SxTrue, SyTrue = truth['SxT'], truth['SyT']

SxPred1 = preds1['SP']
SxPred2 = preds2['SP']
SxPred3 = preds3['SP']

# calculate correlations
r = np.zeros( (3,512,512) )

for i in range(512) :  # loop through x

    if i % 10 == 0 : print i, " / 512"

    for j in range(512) :  # loop throughy y

        # correlation between true Sx and predictions
        r[0,j,i] = np.corrcoef( np.squeeze( SxTrue[:,j,i] ), np.squeeze( SxPred1[:,j,i] ), rowvar=False )[0,1]
        r[1,j,i] = np.corrcoef( np.squeeze( SxTrue[:,j,i] ), np.squeeze( SxPred2[:,j,i] ), rowvar=False )[0,1]
        r[2,j,i] = np.corrcoef( np.squeeze( SxTrue[:,j,i] ), np.squeeze( SxPred3[:,j,i] ), rowvar=False )[0,1]



##### Plotting: Sx Predictions #####

t0 = 200 # snapshot time

fig, axArr = plt.subplots( 4, 4, sharey=True, figsize=(7.9,9) )


## SNAPSHOTS ##

cLim = 2.4

x0, y0 = 13, 450

# True Sx
im0 = axArr[0,0].imshow( np.squeeze( SxTrue[t0,:,:] )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[0,0].set_title('True $S_x$ Snapshot', fontsize=9 )
axArr[0,0].set_yticks( (0,255,511) )
axArr[0,0].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[0,0].set_xticks( (0,255,511) )
axArr[0,0].set_xticklabels( ('','','') )
axArr[0,0].text( x0, y0, 'a.', fontsize=9 )

# Predicted Sx (training region 1)
im1 = axArr[0,1].imshow( np.squeeze( SxPred1[t0,:,:] )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[0,1].set_title(r'Snapshot $f_x(\overline{\psi},\mathbf{w}_1)$', fontsize=9 )
axArr[0,1].set_yticks( (0,255,511) )
axArr[0,1].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[0,1].set_xticks( (0,255,511) )
axArr[0,1].set_xticklabels( ('','','') )
axArr[0,1].text( x0, y0, 'b.', fontsize=9 )

# Predicted Sx (training region 2)
im2 = axArr[0,2].imshow( np.squeeze( SxPred2[t0,:,:] )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[0,2].set_title(r'Snapshot $f_x(\overline{\psi},\mathbf{w}_2)$', fontsize=9 )
axArr[0,2].set_yticks( (0,255,511) )
axArr[0,2].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[0,2].set_xticks( (0,255,511) )
axArr[0,2].set_xticklabels( ('','','') )
axArr[0,2].text( x0, y0, 'c.', fontsize=9 )

# Predicted Sx (training region 3)
im3 = axArr[0,3].imshow( np.squeeze( SxPred3[t0,:,:] )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[0,3].set_title(r'Snapshot $f_x(\overline{\psi},\mathbf{w}_3)$', fontsize=9 )
axArr[0,3].set_yticks( (0,255,511) )
axArr[0,3].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[0,3].set_xticks( (0,255,511) )
axArr[0,3].set_xticklabels( ('','','') )
axArr[0,3].text( x0, y0, 'd.', fontsize=9 )

## MEANS ##

cLim = 1.5

# Mean True Sx
im4 = axArr[1,0].imshow( np.squeeze( np.mean( SxTrue, axis=0 ) )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[1,0].set_title('Mean of True $S_x$', fontsize=9 )
axArr[1,0].set_yticks( (0,255,511) )
axArr[1,0].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[1,0].set_xticks( (0,255,511) )
axArr[1,0].set_xticklabels( ('','','') )
axArr[1,0].text( x0, y0, 'e.', fontsize=9 )

# Mean Predicted Sx (training region 1)
im5 = axArr[1,1].imshow( np.squeeze( np.mean( SxPred1, axis=0 ) )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[1,1].set_title(r'Mean $f_x(\overline{\psi},\mathbf{w}_1)$', fontsize=9 )
axArr[1,1].set_yticks( (0,255,511) )
axArr[1,1].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[1,1].set_xticks( (0,255,511) )
axArr[1,1].set_xticklabels( ('','','') )
axArr[1,1].text( x0, y0, 'f.', fontsize=9 )

# Mean Predicted Sx (training region 2)
im6 = axArr[1,2].imshow( np.squeeze( np.mean( SxPred2, axis=0 ) )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[1,2].set_title(r'Mean $f_x(\overline{\psi},\mathbf{w}_2)$', fontsize=9 )
axArr[1,2].set_yticks( (0,255,511) )
axArr[1,2].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[1,2].set_xticks( (0,255,511) )
axArr[1,2].set_xticklabels( ('','','') )
axArr[1,2].text( x0, y0, 'g.', fontsize=9 )

# Mean Predicted Sx (training region 3)
im7 = axArr[1,3].imshow( np.squeeze( np.mean( SxPred3, axis=0 ) )*1e6, origin='lower', cmap='seismic', vmin=-cLim, vmax=cLim )
axArr[1,3].set_title(r'Mean $f_x(\overline{\psi},\mathbf{w}_3)$)', fontsize=9 )
axArr[1,3].set_yticks( (0,255,511) )
axArr[1,3].set_yticklabels( ('0km','1920km','3840km'), fontsize=9 )
axArr[1,3].set_xticks( (0,255,511) )
axArr[1,3].set_xticklabels( ('','','') )
axArr[1,3].text( x0, y0, 'h.', fontsize=9 )

## STD ##

cLim = 1.5
cMap = 'magma_r'

# Std True Sx
im8 = axArr[2,0].imshow( np.squeeze( np.std( SxTrue, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
axArr[2,0].set_title('Std Dev of True $S_x$', fontsize=9 )
axArr[2,0].set_yticks( (0,255,511) )
axArr[2,0].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[2,0].set_xticks( (0,255,511) )
axArr[2,0].set_xticklabels( ('','','') )
axArr[2,0].text( x0, y0, 'i.', fontsize=9 )

# Std Predicted Sx (training region 1)
im9 = axArr[2,1].imshow( np.squeeze( np.std( SxPred1, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
axArr[2,1].set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_1)$', fontsize=9 )
axArr[2,1].set_yticks( (0,255,511) )
axArr[2,1].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[2,1].set_xticks( (0,255,511) )
axArr[2,1].set_xticklabels( ('','','') )
axArr[2,1].text( x0, y0, 'j.', fontsize=9 )

# Std Predicted Sx (training region 2)
im10 = axArr[2,2].imshow( np.squeeze( np.std( SxPred2, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
axArr[2,2].set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_2)$', fontsize=9 )
axArr[2,2].set_yticks( (0,255,511) )
axArr[2,2].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[2,2].set_xticks( (0,255,511) )
axArr[2,2].set_xticklabels( ('','','') )
axArr[2,2].text( x0, y0, 'k.', fontsize=9 )

# Std Predicted Sx (training region 3)
im11 = axArr[2,3].imshow( np.squeeze( np.std( SxPred3, axis=0 ) )*1e6, origin='lower', cmap=cMap, vmin=0, vmax=cLim )
axArr[2,3].set_title(r'Std Dev $f_x(\overline{\psi},\mathbf{w}_3)$', fontsize=9 )
axArr[2,3].set_yticks( (0,255,511) )
axArr[2,3].set_yticklabels( ('0km','1920km','3840km'), fontsize=9 )
axArr[2,3].set_xticks( (0,255,511) )
axArr[2,3].set_xticklabels( ('','','') )
axArr[2,3].text( x0, y0, 'l.', fontsize=9 )

## CORRELATION ##

cMap = 'seismic'
axArr[3,0].axis('off')

# Corr Predicted Sx (training region 1)
im13 = axArr[3,1].imshow( np.squeeze( r[0,:,:] ), origin='lower', cmap=cMap, vmin=-1, vmax=1 )
axArr[3,1].set_title(r'Corr($S_x,\tilde{S}_x$): $f_x(\overline{\psi},\mathbf{w}_1)$', fontsize=9 )
axArr[3,1].set_yticks( (0,255,511) )
axArr[3,1].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[3,1].set_xticks( (0,255,511) )
axArr[3,1].set_xticklabels( ('0km','1920km','3840km'), rotation=-90, fontsize=9  )
axArr[3,1].text( x0, y0, 'm.', fontsize=9, color='white' )

# Corr Predicted Sx (training region 2)
im14 = axArr[3,2].imshow( np.squeeze( r[1,:,:] ), origin='lower', cmap=cMap, vmin=-1, vmax=1 )
axArr[3,2].set_title(r'Corr($S_x,\tilde{S}_x$): $f_x(\overline{\psi},\mathbf{w}_2)$', fontsize=9 )
axArr[3,2].set_yticks( (0,255,511) )
axArr[3,2].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[3,2].set_xticks( (0,255,511) )
axArr[3,2].set_xticklabels( ('0km','1920km','3840km'), rotation=-90, fontsize=9  )
axArr[3,2].text( x0, y0, 'n.', fontsize=9, color='white' )

# Corr Predicted Sx (training region 3)
im15 = axArr[3,3].imshow( np.squeeze( r[2,:,:] ), origin='lower', cmap=cMap, vmin=-1, vmax=1 )
axArr[3,3].set_title(r'Corr($S_x,\tilde{S}_x$): $f_x(\overline{\psi},\mathbf{w}_3)$', fontsize=9 )
axArr[3,3].set_yticks( (0,255,511) )
axArr[3,3].set_yticklabels( ('0km','1920km','3840km'), fontsize=9  )
axArr[3,3].set_xticks( (0,255,511) )
axArr[3,3].set_xticklabels( ('0km','1920km','3840km'), rotation=-90, fontsize=9  )
axArr[3,3].text( x0, y0, 'o.', fontsize=9, color='white' )

## COLORBARS ##

plt.subplots_adjust( wspace=0.06, hspace=0.3, right=0.85, top=0.85 )

dx=-0.05

# colorbar row 0
pos = axArr[0,3].get_position()
cax = fig.add_axes( [ pos.x0+pos.width+dx, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im0, ax=cax )
cBar.ax.set_ylabel(r'$\tilde{S}_x$ (10$^{-6}$ms$^{-2}$)', fontsize=9 )
cBar.ax.tick_params( labelsize=9 )

# colorbar row 1
pos = axArr[1,3].get_position()
cax = fig.add_axes( [ pos.x0+pos.width+dx, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im4, ax=cax )
cBar.ax.set_ylabel(r'Time-Mean $\tilde{S}_x$ (10$^{-6}$ms$^{-2}$)', fontsize=9 )
cBar.ax.tick_params( labelsize=9 )

# colorbar row 2
pos = axArr[2,3].get_position()
cax = fig.add_axes( [ pos.x0+pos.width+dx, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im9, ax=cax )
cBar.ax.set_ylabel(r'Std Dev $\tilde{S}_x$ (10$^{-6}$ms$^{-2}$)', fontsize=9  )
cBar.ax.tick_params( labelsize=9 )

# colorbar row 3
pos = axArr[3,3].get_position()
cax = fig.add_axes( [ pos.x0+pos.width+dx, pos.y0, 0.07, pos.height] )
cax.axis('off')
cBar = plt.colorbar( im13, ax=cax )
cBar.ax.set_ylabel(r'Pearson Correlation', fontsize=9 )
cBar.set_ticks( [-1,-0.5,0,0.5,1] )
cBar.ax.set_yticklabels( ['-1','-0.5','0','0.5','1'] )
cBar.ax.tick_params( labelsize=9 )

# move the first column slightly to the left
dx = -0.03

for row in range(3) :
    pos = axArr[row,0].get_position()
    pos = [ pos.x0 + dx, pos.y0, pos.width, pos.height ]
    axArr[row,0].set_position( pos )

# add a bit of annotation
posX0, posX1, posX2, posX3 = axArr[0,0].get_position(), axArr[0,1].get_position(), axArr[0,2].get_position(), axArr[0,3].get_position()
x_line = ( posX0.x0 + posX0.width + posX1.x0 ) * 0.5
axArr[0,0].plot( [x_line,x_line], [0.03,0.98], color='black', clip_on=False, transform=fig.transFigure )
plt.text( posX0.x0 + 0.5*posX0.width, 0.91, r'\textbf{Truth} $S_x$', ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure )
plt.text( posX2.x0 + 0.5*posX2.width, 0.95, r'\textbf{Convolutional Neural Networks} $\tilde{S}_x = f_x(\overline{\psi},\mathbf{w}_R)$', ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure )

plt.text( posX1.x0 + 0.5*posX1.width, 0.92, r'\textbf{Region 1:}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )
plt.text( posX2.x0 + 0.5*posX2.width, 0.92, r'\textbf{Region 2:}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )
plt.text( posX3.x0 + 0.5*posX3.width, 0.92, r'\textbf{Region 3:}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )

plt.text( posX1.x0 + 0.5*posX1.width, 0.90, r'\textbf{Western Boundary}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )
plt.text( posX2.x0 + 0.5*posX2.width, 0.90, r'\textbf{Eastern Boundary}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )
plt.text( posX3.x0 + 0.5*posX3.width, 0.90, r'\textbf{Southern Gyre}', ha='center', fontsize=9, fontweight='bold', transform=fig.transFigure )


plt.savefig('SxPredictions.png',format='png',dpi=300 )

plt.show()
