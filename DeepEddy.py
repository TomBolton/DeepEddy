"""

Tom Bolton
tom.bolton@physics.ox.ac.uk
Atmospheric, Oceanic, and Planetary Physics, University of Oxford
23/09/2018

This script contains various functions for the preparation,
training, and testing of the convolutional neural networks (CNN)
used in the paper "Applications of Deep Learning to Ocean Data 
Inference and Sub-Grid Parameterisation".

The functions are split into the following made catergories:

- Data preparation.
- Training and predict.
- Plotting/Misc.

"""

import time
import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from keras import backend as K
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Convolution2D, Dense, Flatten, MaxPooling2D, Lambda
from keras.optimizers import Adam


########################################################################
#
# Data Preparation Functions
#
########################################################################

def calcEddyForcing( psiEddy, psiBar, l ) :
    """

    Given the filtered-streamfunction 'psiBar' and the sub-filter
    streamfunction psiEddy = psi - psiBar, calculate the components
    of the sub-filter eddy momentum forcing Sx and Sy

    (Sx,Sy) = (U.grad)U - filter( (u.grad)u ),

    where U is the velocity from the filtered-streamfunction, and
    u is the full velocity (filtered + sub-filter).    The calculation
    requires more spatial-filtering, which is why the length-scale
    of the filter 'l' (in km) is required as an input variable.

    """

    # spatial-resolution of QG model (7.5km)
    dx = 7.5e3 

    # streamfunction for calculating u and v
    psi = psiBar + psiEddy

    # calculate gradients
    [ psi_t, psi_y, psi_x] = np.gradient( psi, dx )
    [ psiBar_t, psiBar_y, psiBar_x] = np.gradient( psiBar, dx )

    u, v = -psi_y, psi_x
    U, V = -psiBar_y, psiBar_x
    
    # Calcluate filtered-advection term
    [ U_t, U_y, U_x ] = np.gradient( U, dx );  del U_t
    [ V_t, V_y, V_x ] = np.gradient( V, dx );  del V_t

    # ( Ud/dx + Vd/dy )U and ( Ud/dx + Vd/dy )V
    adv1_x = U * U_x + V * U_y
    adv1_y = U * V_x + V * V_y

    del U_x, U_y, V_x, V_y

    # Calculate sub-filter advection term
    [u_t, u_y, u_x] = np.gradient(u, dx); del u_t
    [v_t, v_y, v_x] = np.gradient(v, dx); del v_t

    # ( ud/dx + vd/dy )u + ( ud/dx + vd/dy )v
    adv2_x = u * u_x + v * u_y
    adv2_y = u * v_x + v * v_y

    del u_x, u_y, v_x, v_y

    for t in range( adv2_x.shape[0] ) :
        adv2_x[t,:,:] = gaussian_filter( adv2_x[t,:,:], (l*1e3)/dx )
        adv2_y[t,:,:] = gaussian_filter( adv2_y[t,:,:], (l*1e3)/dx )

    # Calculate the eddy momentum forcing components
    Sx = adv1_x - adv2_x
    Sy = adv1_y - adv2_y

    return Sx, Sy

def loadAndNormDataS( l, axis, region, MOMENTUM_B=False ):
    """

    This function loads the filtered-streamfunction and
    streamfunction anomaly from a .mat file, and calculates
    the corresponding sub-filter eddy momentum forcing. Both the eddy
    momentum    forcing and the filtered-streamfunctions are normalised
    to zero    mean and unit variance, and split into training and test data.

    l = spatial-scale (in km) of low-pass Gaussian filter
    axis = 'x' or 'y',  specifying if Sx or Sy is calculated
    region = 1, 2, or 3, the training regions being considered

    For example, for l=30km, axis='y', and region=2, then this function
    will form training and test datasets for the filtered-streamfunction
    (input) and the meridional eddy momentum forcing Sy (output), using
    data only from region=2, which corresponds to the eastern boundary.

    """

    # construct file and variable names
    fileName = 'data/Training/psiTrain' + str(region) + '_' + str(l) + 'km.mat'
    varName1 = 'psi' + str(region) + 'Anom'                 # sub-filter streamfunction
    varName2 = 'psi' + str(region) + '_' + str(l) + 'km'    # filtered streamfunction

    # load data
    data = sio.loadmat( fileName )

    # extract the filtered \bar(psi) and sub-filter psi' for this region
    psiAnom = data[ varName1 ]
    psiFilt = data[ varName2 ]

    # move time dimension to the left
    psiAnom = np.moveaxis( psiAnom, 2, 0 )
    psiFilt = np.moveaxis( psiFilt, 2, 0 )

    # calculate eddy (sub-filter) momentum forcing
    Sx, Sy = calcEddyForcing( psiAnom, psiFilt, l )

    # choose x or y axis depending on 'axis' variable
    # Sx = zonal eddy momentum forcing
    # Sy = meridional eddy momentum forcing
    S = Sx if axis == 'x' else Sy

    # standarize the variables to zero mean and unit variance,
    # this is important in particular for neural networks
    mu1, sigma1 = np.mean( psiFilt ), np.std( psiFilt )
    mu2, sigma2 = np.mean( S ), np.std( S )

    psiFilt = (psiFilt - mu1) / sigma1
    S = (S - mu2) / sigma2

    scalings = [mu1, sigma1, mu2, sigma2]

    # split into training and test data (9 years for training, 
    # 1 year for testing)
    xTrain, xTest = psiFilt[:3300, :, :], psiFilt[3300:, :, :]
    yTrain, yTest = S[:3300, :, :], S[3300:, :, :]

    # split the original 160x160 region into 16 sub-regions
    # of size 40x40. Then combine into a single data set (which
    # will have sixteen times as many training samples).
    xTrain = np.reshape(np.array(np.split(np.array(np.split(xTrain, 4, axis=2)), 4, axis=2)), (4 * 4 * 3300, 40, 40))
    yTrain = np.reshape(np.array(np.split(np.array(np.split(yTrain, 4, axis=2)), 4, axis=2)), (4 * 4 * 3300, 40, 40))

    xTest = np.reshape(np.array(np.split(np.array(np.split(xTest, 4, axis=2)), 4, axis=2)), (4 * 4 * 350, 40, 40))
    yTest = np.reshape(np.array(np.split(np.array(np.split(yTest, 4, axis=2)), 4, axis=2)), (4 * 4 * 350, 40, 40))

    if MOMENTUM_B :
        # remove the spatial-mean of S, at each time-slice, in order to make
        # the net input of momentum zero in the training data (approach B to
        # conserving momentum by pre-processing training data).
        yTrain = yTrain - np.mean( yTrain, axis=(1,2), keepdims=True )
        yTest = yTest - np.mean( yTest, axis=(1,2), keepdims=True )

    # add singleton dimension for input variables (this is for Keras)
    xTrain = np.reshape(xTrain, (-1, 40, 40, 1))
    xTest = np.reshape(xTest, (-1, 40, 40, 1))

    # reshape outputs from 2D (40x40) to 1D vector 1600
    yTrain = np.reshape(yTrain, (-1, 40 * 40))
    yTest = np.reshape(yTest, (-1, 40 * 40)) 

    return xTrain, yTrain, xTest, yTest, scalings



########################################################################
#
# Training and Predicting Functions
#
########################################################################

def trainCNN( l, axis, region, MOMENTUM_A=False, MOMENTUM_B=False ) :
    
    """
    
    This function loads data from a particular region of the QG model, 
    and then trains a convolutional neural network (CNN) to predict
    either the zonal (Sx) or meridional (Sy) component of the eddy momentum
    forcing. The CNN is trained for 200 epochs, from which the model is
    saved, including the validation loss during training as a function
    of the number of epochs.
    
    """
    
    # load data 
    xTrain, yTrain, xTest, yTest, scalings = loadAndNormDataS( l, axis, region, MOMENTUM_B )

    # number of training and validation samples
    nTrain = xTrain.shape[0]
    nTest = xTest.shape[0]

    print "Number of training samples: ", nTrain
    print "Number of validation samples: ", nTest

    ########## Construct Layers ##########

    input_layer = Input( shape=( 40, 40, 1 ) )

    # Convolution layers
    conv_1 = Convolution2D( 16, (8,8), strides=(2,2), padding='valid', activation='selu')( input_layer )
    conv_2 = Convolution2D( 8, (4,4), padding='valid', activation='selu')( conv_1 )
    conv_3 = Convolution2D( 8, (4,4), padding='valid', activation='selu')( conv_2 )

    # Max Pooling
    pool_1 = MaxPooling2D( pool_size=(2,2) )( conv_3 )
    flat = Flatten()(pool_1)

    if MOMENTUM_A :
        # dense (fully-connected) layer
        dense_1 = Dense( units=40*40, activation='linear' )( flat )
        
        # Lambda layer to remove spatial-mean from time-slice. This is
        # approach A to conserving momentum, i.e., altered-architecture.
        output_layer = Lambda( lambda x: x - K.mean( x, axis=1, keepdims=True ) )( dense_1 )
    else :
        # dense (fully-connected) layer
        output_layer = Dense( units=40*40, activation='linear' )( flat )
        
    ########## Train CNN ###########

    myModel = Model( inputs=input_layer, outputs=output_layer )
    myOpt = Adam( lr=0.001 )
    myModel.compile( loss='mean_squared_error', optimizer=myOpt )

    # show the architecture and the parameters
    print myModel.summary()

    # train the model
    History = myModel.fit( xTrain, yTrain, batch_size=16, epochs=200, verbose=2, validation_data=( xTest, yTest )  )

    # make file name
    fileName = 'cnn30km_'+str(region)+'_S'+axis+'_200e'
    if MOMENTUM_A : 
        fileName = fileName + '_MOM_A'
    elif MOMENTUM_B :    
        fileName = fileName + '_MOM_B'

    # save model
    myModel.save( fileName + '.h5' )

    # save training loss history
    with open('history_' + fileName, 'wb') as file_pi :
        pickle.dump( History.history, file_pi )


def makeOverlapPreds( l, axis, region, MOMENTUM_A=False, MOMENTUM_B=False, MOMENTUM_C=False ) :
    
    """
    
    This function loads a trained neural network and makes predictions 
    for the final year of validation data, over the full region. 

    As each neural network makes predictions for a 40x40 grid point area,
    multiple overlapping predictions have to be made over the full 
    domain (512x512) and then averaged at each grid point.
    
    region = 1, 2, or 3, the region on which the models are trained 
    l = length-scale (in km) of the spatial-filtering
    model = the CNN trained to predict Sx or Sy
    
    The predictions (and truth) and then saved in zipped numpy files.
    
    """
    modelName = 'cnn30km_'+str(region)+'_S'+axis+'_200e'
    if MOMENTUM_A :
        modelName = modelName + '_MOM_A'
    elif MOMENTUM_B :
        modelName = modelName + '_MOM_B'

    model = load_model('models/'+modelName+'.h5')
    
    # get scalings for psi, and either Sx or Sy
    _, _, _, _, scalings = loadAndNormDataS(l, axis, region )

    muPsi, sigmaPsi, muS, sigmaS = scalings[0], scalings[1], scalings[2], scalings[3]

    # load filtered-streamfunction from full region (final year) to 
    # use as the input variable to make preditions with
    data = sio.loadmat('data/Validation/psiPred_30km.mat')

    # extract filtered streamfunction and anomaly
    psiAnom = data['psiPredAnom']
    psiFilt = data['psiPred_30km']

    # move time dimension to the left
    psiAnom = np.moveaxis(psiAnom, 2, 0)
    psiFilt = np.moveaxis(psiFilt, 2, 0)

    # calculate the TRUE eddy source term
    SxTrue, SyTrue = calcEddyForcing(psiAnom, psiFilt, l)

    # standarize the input variables, i.e. the smoothed streamfunction,
    # by removing the mean and dividing by the standard deviation
    psiFilt = (psiFilt - muPsi) / sigmaPsi

    # We now want to make the predictions for the entire region, at every
    # time step. We move the neural network one grid point at a time over
    # the entire region, making predictions as it moves along. The predictions
    # at each grid point are then averaged.
    SPred = np.zeros( (350,512,512) )

    mask = np.zeros( (512,512) )

    stride = 2

    for i in range( 0, 512-40+1, stride):   # loop through points in x

        print i  # progress update

        t0 = time.time()

        for j in range( 0, 512-40+1, stride ):   # loop through points in y

            # make predictions at this point
            SPred[:,j:j+40,i:i+40] += model.predict( np.reshape( psiFilt[:,j:j+40,i:i+40], (-1,40,40,1) ) ).reshape( (-1,40,40) )

            # update number of predictions made at each grid point
            mask[j:j+40,i:i+40] += 1

        t1 = time.time()
        print t1-t0

    # average the predictions
    SPred = np.divide( SPred, mask )

    # rescale psi, either Sx or Sy
    psiFilt = psiFilt * sigmaPsi + muPsi
    SPred = SPred * sigmaS + muS

    # save as zipped numpy files
    fileName = 'S'+axis+'_Pred_R'+str(region)+'_str2'
    if MOMENTUM_A :
        fileName = fileName + '_MOM_A'
        SPred -= muS 
    elif MOMENTUM_B :
        fileName = fileName + '_MOM_B'
        SPred -= muS
    
    
    np.savez(fileName+'.npz', SP=SPred )   # predictions
    np.savez('SxSy_True_str2.npz', SxT=SxTrue, SyT=SyTrue )      # truth
    
def predictNewModels() :
    
    """
    
    This function loads a trained neural network and makes predictions 
    for the final year of validation data, over the full region - the 
    same as the function above 'makeOverlapPreds', but this time using
    data from new QG models with different forcings.
    
    Each new model has either a different wind forcing or viscosity.
    For each model, predictions are made for one year of data, using the
    filtered-streamfunction as the input as before. The various models
    have the following parameters:
    
    nu = 200 m2s-1
    tau = 0.3 Nm-2
    tau = 0.6 Nm-2
    tau = 0.9 Nm-2 

    We use the neural network trained on region 1, namely 
    'cnn30km_1_Sx_200e.h5', and predict the zonal component Sx
    in these new models. The original model that cnn30km_1_Sx_200e.h5
    was trained on had a wind forcing of tau = 0.8 Nm-2 and 75 m2s-1.
    
    """
    
    # load models
    model = load_model('models/cnn30km_1_Sx_200e.h5')
    
    # get scalings for psi and Sx
    _, _, _, _, scalings = loadAndNormDataS( 30, 'x', 1 )

    muPsi, sigmaPsi, muS, sigmaS = scalings[0], scalings[1], scalings[2], scalings[3]

    # loop through each of the new models, and make overlapping 
    # predictions of Sx for one year of data
    labels = [ '200', 'tau3', 'tau6', 'tau9' ]
    
    for string in labels :
		
		print "Currently making predictions for model: ", string
		
		data = sio.loadmat('data/Validation/psiPred_30km_'+string+'.mat')

		# extract filtered streamfunction and anomaly
		psiAnom = data['psiPredAnom']
		psiFilt = data['psiPred_30km']

		# move time dimension to the left
		psiAnom = np.moveaxis(psiAnom, 2, 0)
		psiFilt = np.moveaxis(psiFilt, 2, 0)

		# calculate the TRUE eddy source term
		SxTrue, SyTrue = calcEddyForcing(psiAnom, psiFilt, 30)

		# standarize the input variables, i.e. the smoothed streamfunction,
		# by removing the mean and dividing by the standard deviation
		psiFilt = (psiFilt - muPsi) / sigmaPsi

		# We now want to make the predictions for the entire region, at every
		# time step. We move the neural network one grid point at a time over
		# the entire region, making predictions as it moves along. The predictions
		# at each grid point are then averaged.
		SPred = np.zeros( (350,512,512) )

		mask = np.zeros( (512,512) )

		stride = 2

		for i in range( 0, 512-40+1, stride):   # loop through points in x

			if i % 100 == 0 : print i  # progress update

			t0 = time.time()

			for j in range( 0, 512-40+1, stride ):   # loop through points in y

				# make predictions at this point
				SPred[:,j:j+40,i:i+40] += model.predict( np.reshape( psiFilt[:,j:j+40,i:i+40], (-1,40,40,1) ) ).reshape( (-1,40,40) )

				# update number of predictions made at each grid point
				mask[j:j+40,i:i+40] += 1

			t1 = time.time()
			print t1-t0

		# average the predictions
		SPred = np.divide( SPred, mask )

		# rescale psi, either Sx or Sy
		psiFilt = psiFilt * sigmaPsi + muPsi
		SPred = SPred * sigmaS + muS

		# save as zipped numpy files
		np.savez('Sx_Pred_R1_str2_'+string+'.npz', SP=SPred, ST=SxTrue )   # predictions


########################################################################
#
# Plotting and Misc Functions
#
########################################################################

def getActivations( X, model ) :

    # add a singelton dimension onto X
    X = np.reshape( X, (1,40,40,1) )

    # Get activations of layer 1
    input_layer = Input(shape=(40, 40, 1))
    conv_1 = Convolution2D( 16, (8, 8), strides=(2,2), padding='valid', activation='selu', weights=model.layers[1].get_weights() )(input_layer)

    newModel1 = Model( inputs=input_layer, outputs=conv_1 )
    act1 = np.squeeze( newModel1.predict(X) )

    # Get activations of layer 2
    conv_2 = Convolution2D( 8, (4, 4), padding='valid', activation='selu', weights=model.layers[2].get_weights() )(conv_1)

    newModel2 = Model(inputs=input_layer, outputs=conv_2 )
    act2 = np.squeeze( newModel2.predict(X) )

    # Get activations of layer 3
    conv_3 = Convolution2D( 8, (4, 4), padding='valid', activation='selu', weights=model.layers[3].get_weights() )(conv_2)

    newModel3 = Model( inputs=input_layer, outputs=conv_3 )
    act3 = np.squeeze( newModel3.predict(X) )

    return act1, act2, act3
    
def plotSyntheticActivationMaps() :
	
	"""
	
	This function examines the activation maps at each
	stage of a trained convolutional neural network, by
	generating a 'fake' streamfunction as the input.
	
	A radially symmetric Gaussian is used as the fake input
	to represent an eddy. The resulting activations maps from
	each convolution layer are then plotted.
	
    """

	# load model
	myModelSx = load_model('models/cnn30km_1_Sx_200e.h5')
	print myModelSx.summary()

	# make 'fake' streamfunction
	x, y = np.meshgrid(np.linspace(-20,20,40), np.linspace(-20,20,40))
	d = np.sqrt(x*x+y*y)
	sigma = 8.0
	psi = np.exp(-( d**2 / ( 2.0 * sigma**2 ) ) )

	##### Calculate activations ######

	# make prediction of Sx
	Sx = myModelSx.predict( psi.reshape( (1,40,40,1) ) ).reshape( (40,40) )

	# calculate the activation layers for two regions
	act1, act2, act3 = getActivations( psi, myModelSx )

	##### Create subplots #####

	# create subplot for input streamfunction
	col0Axis = plt.subplot2grid( (8,16), (2,0), colspan=4, rowspan=4 )
	col0Axis.axis('off')

	# create subplots for each layer
	col1Axes = []; col2Axes = []
	col3Axes = []; col4Axes = []

	for row in range(8) :
		# create axes
		ax1 = plt.subplot2grid( (8,16), (row,5) )  # 1st layer
		ax2 = plt.subplot2grid( (8,16), (row,6) )  # 1st layer
		ax3 = plt.subplot2grid( (8,16), (row,8) )  # 2nd layer
		ax4 = plt.subplot2grid( (8,16), (row,10) )  # 3rd layer

		# add to lists
		col1Axes.append( ax1 ); col2Axes.append( ax2 )
		col3Axes.append( ax3 ); col4Axes.append( ax4 )


	# create subplot for Sx prediction
	col5Axis = plt.subplot2grid( (8,16), (2,12), colspan=4, rowspan=4 )
	col5Axis.axis('off')

	# set size of figure
	fig = plt.gcf()
	fig.set_size_inches( (16,9) )

	##### Plot activations ######

	# plot input streamfunction
	col0Axis.imshow( psi, cmap='seismic', origin='lower' )
	col0Axis.set_title(r'Synthetically-Generated Input ($\overline{\psi}$)')
	#col0Axis.text( 1, 36, 'a.', color='white', fontsize=15, fontweight='bold' )

	# plot activation functions of all three layers
	for row in range(8) :

		# turn of the axes of each subplot
		col1Axes[row].axis('off'); col2Axes[row].axis('off')
		col3Axes[row].axis('off'); col4Axes[row].axis('off')

		# Activation Layer I
		col1Axes[row].imshow( act1[:,:,row], cmap='seismic', origin='lower' )
		col2Axes[row].imshow( act1[:,:,row+8], cmap='seismic', origin='lower' )

		# Activation Layer II
		col3Axes[row].imshow( act2[:,:,row], cmap='seismic', origin='lower' )

		# Activation Layer III
		col4Axes[row].imshow( act3[:,:,row], cmap='seismic', origin='lower' )

	# plot predicted Sx
	col5Axis.imshow( Sx, cmap='seismic', origin='lower' )
	col5Axis.set_title(r'Output ($\tilde{S}_x$)')
	col5Axis.yaxis.tick_right()
	#col5Axis.text( 1, 36, 'e.', color='black', fontsize=15, fontweight='bold' )

	# add horizontal arrows between each layer
	plt.text( 0.32, 0.5, r'$\longrightarrow$', transform=fig.transFigure, fontsize=25 )
	plt.text( 0.475, 0.5, r'$\rightarrow$', transform=fig.transFigure, fontsize=25 )
	plt.text( 0.575, 0.5, r'$\rightarrow$', transform=fig.transFigure, fontsize=25 )
	plt.text( 0.665, 0.5, r'$\longrightarrow$', transform=fig.transFigure, fontsize=25 )

	# label each layer
	posX1, posX2, posX3, posX4 = col1Axes[0].get_position(), col2Axes[0].get_position(), col3Axes[0].get_position(), col4Axes[0].get_position()

	plt.text( 0.5*( posX1.x0 + posX1.width + posX2.x0 ), 0.9, 'b.\nConvolution\nLayer 1', transform=fig.transFigure, fontweight='bold', fontsize=12, ha='center' )
	plt.text( posX3.x0 + 0.5 * posX3.width, 0.9, 'c.\nConvolution\nLayer 2', transform=fig.transFigure, fontweight='bold', fontsize=12, ha='center' )
	plt.text( posX4.x0 + 0.5 * posX4.width, 0.9, 'd.\nConvolution\nLayer 3', transform=fig.transFigure, fontweight='bold', fontsize=12, ha='center' )

	# add number of filters to each layer
	plt.text( 0.5*( posX1.x0 + posX1.width + posX2.x0 ), 0.065, '16 feature maps\n(16 filters)', transform=fig.transFigure, fontweight='bold', ha='center', fontsize=12 )
	plt.text( posX3.x0 + 0.5 * posX3.width, 0.065, '8 feature maps\n(16x8 filters)', transform=fig.transFigure, fontweight='bold', ha='center', fontsize=12 )
	plt.text( posX4.x0 + 0.5 * posX4.width, 0.065, '8 feature maps\n(8x8 filters)', transform=fig.transFigure, fontweight='bold', ha='center', fontsize=12 )

	# add equation above input
	plt.text( 0.22, 0.78, 'a.\nInput: Gaussian\nstreamfunction', fontsize=15, transform=fig.transFigure, ha='center', fontweight='bold' )
	plt.text( 0.22, 0.73, r'$\overline{\psi} = e^{-r^2/2\sigma^2}$', fontsize=20, transform=fig.transFigure, ha='center', fontweight='bold' )
	plt.text( 0.22, 0.25, r'( $\sigma =$' + str(sigma*7.5) + 'km )', fontsize=15, transform=fig.transFigure, ha='center' )

	# add text above output
	posX5 = col5Axis.get_position()
	plt.text( posX5.x0 + 0.5 * posX5.width, 0.75, 'e.\nOutput: Prediction from\ntrained neural network\nCNN$_x$1', fontsize=15, ha='center', transform=fig.transFigure, fontweight='bold' )

	plt.show()
