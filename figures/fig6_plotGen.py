"""

Take the neural network trained on region 1 to predict Sx, and 
make predictions of Sx in new QG ocean models with different 
wind forcings and viscosity coefficients.

Plot the standard deviation of the true Sx, the standard deviation
of the predicted Sx, and the correlation between them.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rc('text', usetex=True )
plt.rc('font', family='serif' )

##### Load Data #####

preds3 = np.load('../data/Predictions/Sx_Pred_R1_str2_tau3.npz')
preds6 = np.load('../data/Predictions/Sx_Pred_R1_str2_tau6.npz')
preds8 = np.load('../data/Predictions/Sx_Pred_R1_str2.npz')
preds9 = np.load('../data/Predictions/Sx_Pred_R1_str2_tau9.npz')
preds200 = np.load('../data/Predictions/Sx_Pred_R1_str2_200.npz')

truth8 = np.load('../data/Predictions/SxSy_True_str2.npz')

Sx3 = preds3['SP']
Sx6 = preds6['SP']
Sx8 = preds8['SP']
Sx9 = preds9['SP']
Sx200 = preds200['SP']

SxTrue3 = preds3['ST']
SxTrue6 = preds6['ST']
SxTrue8 = truth8['SxT']
SxTrue9 = preds9['ST']
SxTrue200 = preds200['ST']

##### Calcualte STD DEV #####

stdPreds = np.zeros( (5,512,512) )

stdPreds[0,:,:] = np.std( Sx200, axis=0 )
stdPreds[1,:,:] = np.std( Sx3, axis=0 )
stdPreds[2,:,:] = np.std( Sx6, axis=0 )
stdPreds[3,:,:] = np.std( Sx8, axis=0 )
stdPreds[4,:,:] = np.std( Sx9, axis=0 )

stdTruth = np.zeros( (5,512,512) )

stdTruth[0,:,:] = np.std( SxTrue200, axis=0 )
stdTruth[1,:,:] = np.std( SxTrue3, axis=0 )
stdTruth[2,:,:] = np.std( SxTrue6, axis=0 )
stdTruth[3,:,:] = np.std( SxTrue8, axis=0 )
stdTruth[4,:,:] = np.std( SxTrue9, axis=0 )


##### Calculate Pearson Correlation #####

r = np.zeros( (5,512,512) )

for i in range(512) :  # loop through x

    if i % 10 == 0 : print i, " / 512"

    for j in range(512) :  # loop throughy y

        # correlation between true Sx and predictions
        r[0,j,i] = np.corrcoef( np.squeeze( SxTrue200[:, j, i]), np.squeeze( Sx200[:,j,i] ), rowvar=False )[0,1]
        r[1,j,i] = np.corrcoef( np.squeeze( SxTrue3[:,j,i] ), np.squeeze( Sx3[:,j,i] ), rowvar=False )[0,1]
        r[2,j,i] = np.corrcoef( np.squeeze( SxTrue6[:,j,i] ), np.squeeze( Sx6[:,j,i] ), rowvar=False )[0,1]
        r[3,j,i] = np.corrcoef( np.squeeze( SxTrue8[:,j,i] ), np.squeeze( Sx8[:,j,i] ), rowvar=False )[0,1]
        r[4,j,i] = np.corrcoef( np.squeeze( SxTrue9[:,j,i] ), np.squeeze( Sx9[:,j,i] ), rowvar=False )[0,1]

##### Plotting #####

fig, axes = plt.subplots( 5, 3, sharex=True, sharey=True, figsize=(7,8.9) )
plt.subplots_adjust( hspace=0.25, right=0.87, top=0.9, left = 0.3 )

titleStr1 = 'CNN$_x$'
titleStr2 = r'Corr$(S_x,\tilde{S}_x)$: '
layers = [ r'$\nu$ =' + '\n200 m$^2$s$^{-2}$',  r'$\tau_0$ ='+'\n0.3 Nm$^{-2}$',
           r'$\tau_0$ ='+'\n0.6 Nm$^{-2}$', r'$\tau_0$ ='+'\n0.8 Nm$^{-2}$', r'$\tau_0$ ='+'\n0.9 Nm$^{-2}$' ]
labels1 = [ ['a.','d.','g.','j.','m.'], ['b.','e.','h.','k','n'], ['c.','f.','i.','l.','o.'] ]
limits = [ 1, 0.4, 1, 1.4, 1.4 ]



# plot std dev
for region in range(5) : 

    # plot std dev truth
    im = axes[region,0].imshow( stdTruth[region,:,:]*1e6, origin='lower', cmap='magma_r', vmin=0, vmax=limits[region] )

    # tidying up
    axes[region,0].autoscale(False)
    if region == 0 : axes[region,0].set_title( 'Std Dev of $S_x$\n(Truth)', fontsize=10 )
    axes[region,0].set_yticks( (0,255,511) )
    axes[region,0].set_yticklabels( [] )
    axes[region,0].set_xticks( (0,255,511) )
    axes[region,0].set_xticklabels(['0km', '1920km', '3840km'], rotation=-90, fontsize=8)
    axes[region,0].text( 15, 445, labels1[0][region], fontsize=10, color='black' )

    #  colorbar
    dx = 0.05; dy = 0.0; s = "5%"; p = 0.0

    # annotate
    axes[region,0].text(-600, 256, layers[region], va='center', ha='center', fontsize=12 )


    pos = axes[region,0].get_position()
    cax = fig.add_axes( [ pos.x0 - 0.03, pos.y0 + 0.01, 0.01, pos.height - 0.02 ] )
    #cax.axis('off')
    vMin, vMax = im.get_clim()
    cBar = plt.colorbar(im, cax=cax )
    cBar.ax.set_ylabel(r'Std Dev (10$^{-6}$ms$^{-2}$)', fontsize=8 )
    cBar.set_ticks( [ 0, 0.5*vMax, vMax ] )
    cBar.set_ticklabels( [ '0', str(0.5*vMax), str(vMax) ] )
    cBar.ax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # plot std dev predictions
    imB = axes[region,1].imshow(stdPreds[region, :, :] * 1e6, origin='lower', cmap='magma_r', vmin=0, vmax=limits[region] )

    # tidying up
    axes[region,1].autoscale(False)
    if region == 0 : axes[region,1].set_title( r'Std Dev $\tilde{S}_x = f_x(\overline{\psi},\mathbf{w}_1)$'+'\n(Predictions)', fontsize=10)
    axes[region,1].set_yticks((0, 255, 511))
    axes[region,1].set_yticklabels( [] )
    axes[region,1].set_xticks((0, 255, 511))
    axes[region,1].set_xticklabels(['0km', '1920km', '3840km'], rotation=-90, fontsize=8)

    axes[region,1].text(15, 445, labels1[1][region], fontsize=10, color='black')



# plot correlation maps
for region in range(5) :  # regions 1, 2, and 3

    # main plotting function
    im = axes[region,2].imshow( np.squeeze( r[region,:,:] ), origin='lower', vmin=-1, vmax=1, cmap='seismic' )

    # tidying up
    axes[region,2].autoscale(False)
    axes[region,2].set_xticks( [0,256,511] )
    axes[region,2].set_yticks( [0,256,511] )
    if region == 0 : axes[region,2].set_title( r'Corr($S_x$,$\tilde{S}_x$)', fontsize=10 )
    axes[region,2].set_xticklabels( ['0km','1920km','3840km'], rotation=-90, fontsize=8 )
    axes[region,2].set_yticklabels( [] )
    axes[region,2].text( 15, 445, labels1[2][region], fontsize=12, color='white', fontweight='bold' )


    # corr colorbar
    dx = 0.05; dy = 0.0; s = "5%"; p = 0.0

    pos = axes[region,2].get_position()
    cax = fig.add_axes( [ pos.x0 + pos.width + 0.02, pos.y0 + 0.01, 0.01, pos.height - 0.02 ] )
    cBar = plt.colorbar( im, cax=cax )
    cBar.set_ticks( (-1,-0.5,0,0.5,1) )
    cBar.set_ticklabels( ('-1','-0.5','0','0.5','1') )
    cBar.ax.tick_params( labelsize=8 )
    cBar.ax.set_ylabel(r'Pearson Correlation', fontsize=8 )



plt.savefig('generalisationMaps.png',format='png',dpi=300)

plt.show()
