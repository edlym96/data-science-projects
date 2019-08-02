# --- TEST_YOUR_MODEL.PY ---
# This file allows you to test that your model passes the requirements.
# just run `python3 test_your_model.py dataset.csv`
# You don't have to read it :-)

# PLEASE DO NOT MODIFY THIS FILE: YOUR CODE _SHOULD_ BE ABLE TO RUN HERE WITHOUT ANY ISSUE.


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import sys

# this imports the main class from your model
from model import PricingModel



# some configuration utilities
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# definition of the tests

def test_profitable_on_market(model, X, y, dataname='training data', require_10percent=True):
    '''This test computes the profit that the model gets on the full market,
       in a monopolistic situation (gets all the contracts).
       It requires the model to have a profit of at least 10% sum(claims)'''
    prices = model.predict_price(X)
    if (prices < 0).any():
        logging.error('Some of your prices are negative.')
        return False
    try:
        gains = prices.sum()
    except Exception as err:
        logging.error(
            'Your prices are not in the correct format (expected numpy array). Error: "%s"' % (err))
        return False
    claims = y.sum()
    profit = gains - claims
    logging.info('Your profit on %s is %.3f.' % (dataname, profit))
    if require_10percent:
        if profit < 0.1*claims:
            logging.error('Your model is not at least 10%% profitable on %s :-(' % dataname)
            ratio = (1.100001*claims/gains)
            logging.error('Hint: multiply your prices by %.6f ;-)' % ratio) # (1.1*claims/prices) * prices - claims = 0.1 claims
            return False
        else:
            logging.info('Your model is 10% profitable on training data: well done!')
    else:
        if profit < 0:
            logging.error('Your model is not profitable on %s :-(' % dataname)
            return False
    return True


def test_profitable_on_random_subset(model, X, y, n_splits=10):
    '''This test uses the previous one on subsets of the training data.'''
    # use the powerful KFold tool for this
    success_count = 0
    split = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (_, index) in enumerate(split.split(X)):
        Xsub = X.loc[index]
        ysub = y[index]
        # apply the test
        success = test_profitable_on_market(
            model, Xsub, ysub, 'fold %d of the training data' % (i+1), require_10percent=False)
        success_count += success
    logging.info('Your model was profitable on %d/%d folds of the training data.' %
                 (success_count, n_splits))
    return success_count, n_splits


# actually apply the tests
# important parameters
claims_column = 'claim_amount'
model_filename = 'model.pickle'


# try and load the data
cli_args = sys.argv[1:]
if not cli_args:
    logging.critical(
        'Missing data filename (to pass to the script as first argument)!')
    sys.exit(1)
datapath = cli_args[0]

try:
    data = pd.read_csv(datapath)
    y = data[claims_column]
    X = data.drop(columns=[claims_column])
except Exception as err:
    logging.critical('There was an error loading your data: "%s".' % err)

logging.info('The data loaded successfully.')


# try and load the model.
try:
    with open(model_filename, 'rb') as ff:
        model = pickle.load(ff)
except Exception as err:
    logging.critical('There was an error loading your model: "%s".' % err)
    sys.exit(1)

logging.info('Your model loaded successfully.')



# run the tests on the model
logging.info('Starting the tests for your model ("%s")' % model_filename)


try:
    test1 = test_profitable_on_market(model, X, y)
    print()
    test2 = test_profitable_on_random_subset(model, X, y)
except Exception as err:
    logging.critical('There was an error when using your model: "%s".' % err)
    sys.exit(1)
