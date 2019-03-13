# --- MODEL.PY ---
# This is the file where you are expected to define your model (as a `PricingModel` class, see below).
# It also has a command-line interface that trains your model using your data, and outputs a `model.pickle` file,
#  containing the trained model.



import numpy as np
import pandas as pd
import sklearn
import pickle
import sys
import logging

# add imports here






class PricingModel:
	'''Your pricing model. This class encapsulates all the logic for 
	   (1) training a model on data, and (2) predicting the price for
	   new datapoints.

	   You will be asked to submit (along with your code) an object of
	   this class that you have trained on your data. This object should
	   be trained using _only your_ data, and should _never_ try and access
	   anything on disk. It should have a `train` and `predict_price` method,
	   as currently described. Besides these restrictions, you are free to do
	   whatever you want within the class.

	   Objects from this class should be `pickle.dump`able (i.e. be 
	   storable in a pickle file. (this should usually be OK though)'''



	def __init__(self):
		'''Initialisation method -- this should not take any compulsory argument.
		   Put here any code that you need to initialise the model (parameters etc.)'''

		# YOUR CODE HERE


	def train(self, X_train, y_train):
		'''Train your model with data. This should not return anything, but should learn
		   internal parameters of the model.

		   INPUT:
		    - `X_train`: a Pandas dataframe, where rows are individual records;
		    - `y_train`: a Numpy array, containing the total value of claims of each individual.'''

		# YOUR CODE HERE



	def predict_price(self, X):
		'''Predict the price that for the records described by X would pay. This function
		   should _not_ attempt to retrain the model with X, and should always return the
		   same result when presented with the same X.

		   INPUT:
		    - `X`: a Pandas dataframe, where every row is an individual record.

		   OUTPUT:
		    - a Numpy array of same length as `X.index` of strictly positive floats.'''

		# YOUR CODE HERE







# COMMAND LINE INTERFACE

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

	# parameters of the training process
	output_filename = 'model.pickle' # pickled model filename
	claims_column   = 'claim_amount' # column of the claims in the CSV


	# get the command-line arguments 
	cli_args = sys.argv[1:]

	# check that the data path is specific
	if len(cli_args) < 1:
		logging.error('Please specify a CSV file to use (as first argument to this module).')
		sys.exit(1)

	data_filename = cli_args[0]


	try:
		# open the dataframe using pandas
		data = pd.read_csv( data_filename )

		# extract the attributes (X) and claims (y)
		y = data[claims_column]                # claims column
		X = data.drop(columns=[claims_column]) # everything except this column

	except Exception as err:
		logging.error('Could not load the data. Did you specify the correct file? (Error: "%s")' % err)
		sys.exit(1)

	logging.info('Initialising and training your model...')
	# train the model
	model = PricingModel()
	model.train(X, y)

	logging.info('Success! Writing to file...')


	# save the model 
	with open(output_filename, 'wb') as ff:
		pickle.dump(model, ff)

	logging.info('Done. Your model is in file "%s".' % output_filename)