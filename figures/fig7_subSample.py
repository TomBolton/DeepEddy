"""

Plot results of neural networks trained on under-sampled training
data. Show the RMSE of the CNN trained on dense training data (i.e.
the original neural network), and compare to RMSE of the neural net
trained with 50% or 20% sub-sampled data.

Underneath, plot the RMSE as a function of the percentage of points
sampled in the training data.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )

##### Calculate RMSE #####

predsFull = np.load('../data/Predictions/Sx_Pred_R1_str2.npz')
SxFull = predsFull['SP']
truth = np.load('../data/Predictions/SxSy_True_str2.npz')
SxTrue = truth['SxT']

N = [ 1500, 1350, 1200, 1050, 900, 750, 600, 450, 300, 150, 75, 30 ]

labels = [ str(n) for n in N ]

RMSE = np.zeros( (512,512,1+len(labels)) )
RMSE[:,:,0] = np.sqrt( np.mean( np.square( SxFull - SxTrue ), axis=0 ) )

# calculate RMSE
mu, sigma = np.mean( SxFull ), np.std( SxFull )

for n, label in enumerate( labels ) :

	data = np.load('../data/Predictions/Sx_Pred_R1_str2_'+label+'ss.npz')

	SxPred = data['SP']
        SxPred = ( SxPred - np.mean( SxPred ) ) / np.std( SxPred )
	SxPred = SxPred * sigma + mu

	RMSE[:,:,n+1] = np.sqrt( np.mean( np.square( SxPred - SxTrue ), axis=0 ) )
	


##### Plotting #####
fig, axes = plt.subplots( 2, 2, figsize=(8,8) )

#plt.subplots_adjust( hspace=0.25, right=0.87, top=0.9, left = 0.3 )

# plot RMSE of densely trained CNNS
im0 = axes[0,0].imshow( RMSE[:,:,0]*1e6, origin='lower', cmap='magma_r', vmin=0, vmax=1.4 )

axes[0,0].autoscale(False)
axes[0,0].set_yticks( (0,255,511) )
axes[0,0].set_yticklabels( [ '0km','1920km','3840km'], fontsize=8 )
axes[0,0].set_xticks( (0,255,511) )
axes[0,0].set_xticklabels(['0km', '1920km', '3840km'], fontsize=8)
axes[0,0].set_title('RMSE of Neural Network\nTrained with Dense (100$\%$) Data' )
axes[0,0].text( 15, 465, 'a.', fontsize=12, color='black' )

# plot RMSE of under-sampled CNN
im1 = axes[0,1].imshow( RMSE[:,:,9]*1e6, origin='lwoer', cmap='magma_r', vmin=0, vmax=1.4 )

axes[0,1].autoscale(False)
axes[0,1].set_yticks( (0,255,511) )
axes[0,1].set_yticklabels( [] )
axes[0,1].set_xticks( (0,255,511) )
axes[0,1].set_xticklabels( ['0km','1920km','3840km'], fontsize=8 )
axes[0,1].set_title( 'RMSE of Neural Network\nTrained with Sub-Sampled (18.75$\%$) Data' )
axes[0,1].text( 15, 465, 'b.', fontsize=12, color='black' )

#  colorbar
dx = 0.05; dy = 0.0; s = "5%"; p = 0.0
pos = axes[0,1].get_position()
cax = fig.add_axes( [ pos.x0 +pos.width+0.01, pos.y0 + 0.01, 0.01, pos.height - 0.02 ] )
vMin, vMax = im0.get_clim()
cBar = plt.colorbar(im0, cax=cax )
cBar.ax.set_ylabel(r'RMSE (10$^{-6}$ms$^{-2}$)', fontsize=8 )
cBar.set_ticks( [ 0, 0.7, 1.4 ] )
cBar.set_ticklabels( [ '0', '0.7', '1.4' ] )
cBar.ax.tick_params(labelsize=8)

# remove bottom-right axis, and extend bottom-left axis
# from left to right 
pos11 = axes[1,1].get_position()
fig.delaxes( axes[1,1] )
pos10 = axes[1,0].get_position()
pos10 = [ pos10.x0, pos10.y0, pos11.x0 + pos11.width - pos10.x0, pos10.height ]
axes[1,0].set_position( pos10 )

# plot RMSE as function of percentage of spatial points sampled
axes[1,0].scatter( 100 * np.array( [1600] + N ) / 1600.0, np.mean( RMSE, axis=(0,1) )*1e6, s=50, 
	facecolor='black', edgecolor='darkred', linewidth=2 )
#axes[1,0].plot( np.linspace(0,100,100), np.ones( (100,) )*np.mean( RMSE[:,:,0] )*1e6, 'k--', linewidth=0.5 )

axes[1,0].set_xlabel('Percentage of spatial points\nsampled in training data', fontsize=8 )
axes[1,0].set_ylabel('RMSE (10$^{-6}$ms$^{-2}$)', fontsize=8 )
axes[1,0].set_title('RMSE vs Percentage of Spatial Points Sampled in Training Data')
axes[1,0].set_ylim( [0,0.275] )
axes[1,0].set_xlim([0,100])
axes[1,0].set_xticks( [0,20,40,60,80,100] )
axes[1,0].set_xticklabels( [ '0$\%$', '20$\%$', '40$\%$', '60$\%$', '80$\%$', '100$\%$' ] )
axes[1,0].invert_xaxis()
axes[1,0].yaxis.grid(True)
axes[1,0].text( 99, 0.255, 'c.', fontsize=12 )
axes[1,0].set_axisbelow(True)

plt.savefig('underSampling.png',format='png',dpi=300)

