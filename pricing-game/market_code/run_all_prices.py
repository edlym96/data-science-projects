# This script runs the (pickled) models found in a folder's subfolder.
# All folders are assumed to contain (1) model.pickle (the trained model)
#  and (2) model.py, the code describing the class.
#
# It takes two compulsory arguments as part of its command line interface:
#  - the test dataset to use;
#  - the folder in which to find the models.
#  [- (optional) output_file = prices.pickle]
#
# The code explores every subfolder, and when said folder contains both a
#  "model.py" file and a "model.pickle" file, it imports the module, unpickles
#  the model, and runs the code on the test set. The resulting price is then
#  stored in a Pandas dataframe, with columns for every price (and a new column
#  for the claims made by each costumer).
#
# This code assumes that the name of each subfolder is the name of the corresponding
#  team, and thus uses that name as identifier for prices.


import os
import sys
import pickle
import imp

import numpy as np
import pandas as pd


# Since we import submodules in each folder (called model.py), but we
#  need a _global_ reference to the `PricingModel` class when pickling,
#  we create an empty "placeholder" variable. This variable will be overwritten
#  when importing the module (through the use of `global PricingModel`),
#  allowing for Pickle to do its job properly.
PricingModel = None


def run(test_dataset_path, source_master_folder, claims_column='claim_amount', verbose=True):
    '''Get the prices for all models located in subfolders of `source_master_folder`,
        evaluated on all contracts in `test_dataset_path` (a CSV file).
        `claims_columns` represents the column in the dataset where the claims are.
       This returns a dictionary subfolder_name -> price (np.array).'''

    # see above on why this is necessary.
    global PricingModel

    # open the data
    df = pd.read_csv(test_dataset_path)
    y = df[claims_column]
    X = df.drop(columns=[claims_column])

    # dictionary {folder name --> price vector}
    prices = {}

    for subfolder_name in os.listdir(source_master_folder):
        # current folder (referenced as cfolder)
        cfolder = os.path.join(source_master_folder, subfolder_name)
        # check that it is a directory
        if os.path.isdir(cfolder):
            # check that both necessary files are available
            if os.path.exists(os.path.join(cfolder, 'model.py')) and os.path.exists(os.path.join(cfolder, 'model.pickle')):
                if verbose:
                    print('Importing and running folder %s...' %
                          subfolder_name)
                # import the module
                sys.path.append(cfolder)
                import model
                imp.reload(model)  # reload
                PricingModel = model.PricingModel
                # load the model from the pickle
                with open(os.path.join(cfolder, 'model.pickle'), 'rb') as ff:
                    saved_model = pickle.load(ff)
                # get the prices predicted by the model
                prices[subfolder_name] = saved_model.predict_price(X)
                # (clean up) remove the folder from the sys path (for next iteration)
                sys.path = sys.path[:-1]
                del saved_model
            # missing either the .py or .pickle file
            else:
                if verbose:
                    print(
                        'Ignoring folder "%s": either "model.py" or "model.pickle" was not found.' % cfolder)
        else:
            if verbose:
                print('Ignoring "%s": not a folder.' % cfolder)

    # convert prices to Pandas dataframe
    columns = [claims_column] + [k for k in prices.keys()]
    data = list(zip(y, *[prices[k] for k in columns[1:]]))
    print(len(data), data[0], columns)
    df = pd.DataFrame(list(data), columns=columns)
    return df


# script part
if __name__ == '__main__':

    # require three arguments
    if len(sys.argv) < 3:
        print(
            'Missing arguments. Please use interface "python %s test_dataset.csv source_master_folder"' % sys.argv[0])
        sys.exit(1)

    # parse the arguments
    test_dataset_path = sys.argv[1]
    source_master_folder = sys.argv[2]

    output_path = 'prices.csv'
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]

    # get the prices
    print('Running price computation...')
    prices_df = run(test_dataset_path, source_master_folder, verbose=True)
    print('Done! Here is what it looks like:')
    print(prices_df.head())

    # save the prices
    prices_df.to_csv(output_path, index=False)
