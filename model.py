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
from sklearn.linear_model import LogisticRegression, LinearRegression



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

		# Our super simple model uses two linear models.
		self.clf = LogisticRegression(class_weight='balanced')
		self.reg = LinearRegression()
		self.claims = 0


	def preprocess_X(self, X):
		'''Preprocess the dataset: this method prepares the dataset for the learning of
		   the algorithm/application of the algorithm to data. This does all the feature
		   extraction/selection that is necessary.

		   INPUT:
		    - `X`: a Pandas dataframe, where rows are individual records (the original data)

		   OUTPUT:
		    - a Pandas dataframe with as many records as the input (the processed data)'''

		# YOUR CODE HERE

		# For the sake of example, we do very very basic feature selection, by arbitrarily
		#  selecting 5 columns. You should put all the tricks you use here related to 
		#  feature engineering (including one-hot encoding).
		# WARNING: this is very basic and SHOULD be changed!
		good_cols = ['drv_age1', 'drv_age_lic1', 'vh_age', 'vh_value', 'vh_speed']
		X_clean = X[good_cols]
		X_clean.fillna(0, inplace=True)
		return X_clean


	def train(self, X_train, y_train):
		'''Train your model with data. This should not return anything, but should learn
		   internal parameters of the model.

		   INPUT:
		    - `X_train`: a Pandas dataframe, where rows are individual records;
		    - `y_train`: a Numpy array, containing the total value of claims of each individual.'''

		# YOUR CODE HERE

		# Training of the simple model we give you: linear + linear.
		# 1) process the dataset.
		X_clean = self.preprocess_X(X_train)
		X_clean = np.array(X_clean)
		y_train = np.array(y_train)
		self.claims = sum(y_train)

		# 2) the binary attribute: whether there was a claim.
		y_clf = y_train > 0

		# 3) Fit the logistic regression to predict whether there will be a claim.
		self.clf.fit(X_clean, y_clf)

		# 4) Restrict X to where there was a claim.
		nnz = np.where(y_train != 0)[0]
		x_reg = X_clean[nnz]
		y_reg = y_train[nnz]

		# 5) Fit the linear regression.
		self.reg.fit(x_reg, y_reg)


	def predict_claim_probability(self, X_clean):
		'''Predict the probability that the records described by X will result in a nonzero claim.

		   INPUT:
		    - `X_clean`: a preprocessed Pandas dataframe, where every row is an individual record.

		   OUTPUT:
		    - a Numpy array of the same length as `X.index` of positive floats less than or equal to 1.'''

		# YOUR CODE HERE

		# In our example, we just apply the logistic classifier.
		return self.clf.predict_proba(X_clean)[:,1]


	def predict_claim_amount(self, X_clean):
		'''Predict the claim amount that the records described by X will cost.

		   INPUT:
		    - `X_clean`: a preprocessed Pandas dataframe, where every row is an individual record.

		   OUTPUT:
		    - a Numpy array of the same length as `X.index` of positive floats.'''

		# YOUR CODE HERE

		# In our example, we just apply the linear regressor.
		predictions = self.reg.predict(X_clean)

		# Important: prices can not be negative!
		predictions[predictions < 0] = 0
		return predictions


	def profit_strategy(self, X_clean, probability, claim_amount):
		'''Sets out the profit making strategy for the pricing model.

		   INPUT:
		    - `X_clean`: a preprocessed Pandas dataframe, where every row is an individual record.
		    - `probability`: a Numpy array with the estimated probability of making a claim for each record.
		    - `claim_amount`: a Numpy array with the estimated claim amount for each record.

		   OUTPUT:
		    - a Numpy array of the same length as `X.index` of floats representing prices.'''

		# YOUR CODE HERE

		# We give you a simple strategy to illustrate what you can do here (constant * E[claim]).
		pure_prices = probability * claim_amount
		scale = self.claims / sum(pure_prices)
		scaled_prices = scale * pure_prices

		profit_margin = 0.1
		return (1 + profit_margin) * scaled_prices


	def predict_price(self, X):
		'''Predict the price that for the records described by X would pay. This function
		   should _not_ attempt to retrain the model with X, and should always return the
		   same result when presented with the same X.

		   INPUT:
		    - `X`: a Pandas dataframe, where every row is an individual record.

		   OUTPUT:
		    - a Numpy array of same length as `X.index` of strictly positive floats.'''

		# feel free to modify this bit; but if you use our "blocks" from above, this code
		#  should work no matter what:
		X_clean = self.preprocess_X(X)
		probability  = self.predict_claim_probability(X_clean)
		claim_amount = self.predict_claim_amount(X_clean)
		final_price  = self.profit_strategy(X_clean, probability, claim_amount)
		return final_price






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
