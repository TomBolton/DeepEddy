"""

A neural network has been trained to predict the middle-layer 
streamfunction psi_2 using the upper-layer streamfunction psi_1.
The neural network was trained using data from region 1, in the 
same way as the neural networks trained to predict Sx.

The middle-layer streamfunction is then predicted and compared with
the truth. Also, the predicted middle-layer streamfunction is fed back
into the neural network as an input, in an attempt to predict the 
bottom-layer streamfunction psi_3. The predicted bottom-layer
streamfunction is also compared with the truth.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )

##### Load data #####

# load data
dataMid = np.load('data/overlappingMidPsiPred1_str2.npz')
dataBot = np.load('data/overlappingBotPsiPred1_str2.npz')

psiTop = dataMid['top']
psiMid = dataMid['mid']
psiMidPred = dataMid['midPred']

psiBot = dataBot['bot']
psiBotPred = dataBot['botPred']

# move time dimension to the left
psiBot = np.moveaxis( psiBot, 2, 0)

print psiBot.shape
print psiBotPred.shape

# calculate correlations
r = np.zeros( (2,512,512) )

for i in range(512) :  # loop through x

    if i % 10 == 0 : print i, " / 512"

    for j in range(512) :  # loop throughy y

        # correlation between true Sx and predictions
        r[0,j,i] = np.corrcoef( np.squeeze( psiMid[:,j,i] ), np.squeeze( psiMidPred[:,j,i] ), rowvar=False )[0,1]
        r[1,j,i] = np.corrcoef( np.squeeze( psiBot[:,j,i] ), np.squeeze( psiBotPred[:,j,i] ), rowvar=False )[0,1]

##### Plotting #####

fig, axArr = plt.subplots( 4, 3, sharex=True, sharey=True, figsize=(6.7,9) )

# middle layer

im00 = axArr[0,1].imshow( np.mean( psiMid, axis=0 )*1e-5, origin='lower', cmap = 'seismic', vmin=-0.35, vmax=0.35 )
im10 = axArr[1,1].imshow( np.mean( psiMidPred, axis=0 )*1e-5, origin='lower', cmap='seismic', vmin=-0.35, vmax=0.35 )
im01 = axArr[0,2].imshow( np.std( psiMid, axis=0 )*1e-5, origin='lower', cmap='magma_r', vmin=0, vmax=0.35 )
im11 = axArr[1,2].imshow( np.std( psiMidPred, axis=0 )*1e-5, origin = 'lower', cmap='magma_r', vmin=0, vmax=0.35 )
im12 = axArr[1,0].imshow( r[0,:,:], origin='lower', cmap='seismic', vmin=-1, vmax=1 )

axArr[0,1].set_title('a. True Mean $\psi_2$', fontsize=10 )
axArr[0,2].set_title('b. True Std Dev $\psi_2$', fontsize=10 )
axArr[1,1].set_title(r'd. Mean $\tilde{\psi}_2$', fontsize=10 )
axArr[1,2].set_title(r'e. Std Dev $\tilde{\psi}_2$', fontsize=10 )
axArr[1,0].set_title(r'c. Corr($\psi_2$,$\tilde{\psi}_2$)', fontsize=10 )

for i in range(3) :
    for j in range(2) :
        axArr[j,i].set_xticks( [0,256,512] )
        axArr[j,i].set_yticks( [0,256,512] )
        axArr[j,i].set_xticklabels( [] )
        axArr[j,i].set_yticklabels( ['0km', '1920km', '3840km' ], fontsize=8 )

fig.delaxes( axArr[0,0] )

# bottom layer
cLim1, cLim2 = 0.25, 0.25


im20 = axArr[2,1].imshow( np.mean( psiBot, axis=0 )*1e-5, origin='lower', cmap = 'seismic', vmin=-cLim1, vmax=cLim1 )
im30 = axArr[3,1].imshow( np.mean( psiBotPred, axis=0 )*1e-5, origin='lower', cmap='seismic', vmin=-cLim1, vmax=cLim1 )
im21 = axArr[2,2].imshow( np.std( psiBot, axis=0 )*1e-5, origin='lower', cmap='magma_r', vmin=0, vmax=cLim2 )
im31 = axArr[3,2].imshow( np.std( psiBotPred, axis=0 )*1e-5, origin = 'lower', cmap='magma_r', vmin=0, vmax=cLim2 )
im32 = axArr[3,0].imshow( r[1,:,:], origin='lower', cmap='seismic', vmin=-1, vmax=1 )

axArr[2,1].set_title('f. True Mean $\psi_3$', fontsize=10 )
axArr[2,2].set_title('g. True Std Dev $\psi_3$', fontsize=10 )
axArr[3,1].set_title(r'i. Mean $\tilde{\psi}_3$', fontsize=10 )
axArr[3,2].set_title(r'j. Std Dev $\tilde{\psi}_3$', fontsize=10 )
axArr[3,0].set_title(r'h. Corr($\psi_3$,$\tilde{\psi}_3$)', fontsize=10 )

for i in range(3) :
    for j in range(2) :
        axArr[j+2,i].set_xticks( [0,256,512] )
        axArr[j+2,i].set_yticks( [0,256,512] )
        axArr[j+2,i].set_xticklabels( [] )
        axArr[j+2,i].set_yticklabels( ['0km', '1920km', '3840km' ], fontsize=8 )

fig.delaxes( axArr[2,0] )

# add text above each correlation plot
pos = axArr[1,0].get_position()
axArr[1,0].text( pos.x0 + 0.5*pos.width, 0.91, 'Predicting the\nMiddle-Layer\nStreamfunction: ',
                 ha='center', va='center', fontsize=12, transform=fig.transFigure )
axArr[1,0].text( pos.x0 + 0.5*pos.width, 0.85, r'$\tilde{\psi}_2 = f(\overline{\psi}_1,\mathbf{W})$',
                 ha='center', va='center', fontsize=12, transform=fig.transFigure )

pos = axArr[3,0].get_position()
axArr[3,0].text( pos.x0 + 0.5*pos.width, 0.40, 'Predicting the\nBottom-Layer\nStreamfunction: ',
                 ha='center', va='center', fontsize=12, transform=fig.transFigure )
axArr[3,0].text( pos.x0 + 0.5*pos.width, 0.34, r'$\tilde{\psi}_3 = f(\overline{\tilde{\psi}}_2,\mathbf{W})$',
                 ha='center', va='center', fontsize=12, transform=fig.transFigure )


# before making the colorbars, move all the subplots in the top half of
# the figure up a bit, and all subplots in bottom half down a bit
for i in range(3) :
    for j in range(2) :

        # top-half
        pos = axArr[j,i].get_position()
        pos = [ pos.x0, pos.y0 + 0.08, pos.width, pos.height]
        axArr[j,i].set_position( pos )

        # bottom-half
        pos = axArr[j+2,i].get_position()
        pos = [ pos.x0, pos.y0  - 0.02, pos.width, pos.height ]
        axArr[j+2,i].set_position( pos )

# Colorbars: bottom layer
dy = 0.02; dx = 0.01

pos = axArr[3,0].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width - 2*dx, 0.01 ]

cax32 = fig.add_axes( pos )
cBar = plt.colorbar( im32, cax=cax32, orientation='horizontal' )
cBar.ax.set_xlabel('Pearson Correlation', fontsize=8 )
cBar.set_ticks( (-1,-0.5,0,0.5,1) )
cBar.set_ticklabels( ('-1','-0.5','0','0.5','1') )
cBar.ax.tick_params( labelsize=8 )

pos = axArr[3,2].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width - 2*dx, 0.01 ]

cax31 = fig.add_axes( pos )
cBar = plt.colorbar( im31, cax=cax31, orientation='horizontal' )
cBar.ax.set_xlabel('Std Dev $\psi$\n(10$^5$m$^2$s$^{-2}$)', fontsize=8 )
cBar.set_ticks( (0,cLim2*0.5,cLim2) )
cBar.set_ticklabels( ['0.00','0.125','0.25'])
cBar.ax.tick_params( labelsize=8 )

pos = axArr[3,1].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width-2*dx, 0.01 ]

cax30 = fig.add_axes( pos )
cBar = plt.colorbar( im30, cax=cax30, orientation='horizontal' )
cBar.ax.set_xlabel('Time-Mean $\psi$\n(10$^5$m$^2$s$^{-2}$)', fontsize=8 )
cBar.set_ticks( (-cLim1,0,cLim1) )
cBar.ax.tick_params( labelsize=8 )


# Colorbars: middle layer

pos = axArr[1,0].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width - 2*dx, 0.01 ]

cax12 = fig.add_axes( pos )
cBar = plt.colorbar( im12, cax=cax12, orientation='horizontal' )
cBar.ax.set_xlabel('Pearson Correlation', fontsize=8 )
cBar.set_ticks( (-1,-0.5,0,0.5,1) )
cBar.set_ticklabels( ('-1','-0.5','0','0.5','1') )
cBar.ax.tick_params( labelsize=8 )

pos = axArr[1,2].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width- 2*dx, 0.01 ]

cax11 = fig.add_axes( pos )
cBar = plt.colorbar( im11, cax=cax11, orientation='horizontal' )
cBar.ax.set_xlabel('Std Dev $\psi$\n(10$^5$m$^2$s$^{-2}$)', fontsize=8 )
cBar.set_ticks( (0,0.17,0.34) )
cBar.ax.tick_params( labelsize=8 )

pos = axArr[1,1].get_position()
pos = [ pos.x0 + dx, pos.y0 - dy, pos.width - 2*dx, 0.01 ]

cax10 = fig.add_axes( pos )
cBar = plt.colorbar( im10, cax=cax10, orientation='horizontal' )
cBar.ax.set_xlabel('Time-Mean $\psi$\n(10$^5$m$^2$s$^{-2}$)', fontsize=8 )
cBar.set_ticks( (-0.34,0,0.34) )
cBar.ax.tick_params( labelsize=8 )

# add separation line
axArr[1,0].plot( [0.03,0.93], [ 0.5, 0.5 ], color='black', clip_on=False, transform=fig.transFigure, solid_capstyle='round' )

plt.savefig('subSurfPsi.png', format='png', dpi=300 )

plt.show()

