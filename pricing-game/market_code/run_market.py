# This script runs the market, given a `prices.csv` file (which contains
#  a "claims" column with the claims corresponding to the price).

import os
import sys
import numpy as np
import pandas as pd



def run_market(prices, claims_column='claim_amount'):
    '''Run the market, using a prices DataFrame.'''

    # parse the columns to find all the competitors
    columns = list(prices.columns)
    if claims_column not in columns:
        raise Exception(
            'The claims_column you specified was not found: "%s".' % claims_column)
    columns.remove(claims_column)

    # Assign each client to its best bet (modify prices df)
    prices = prices.copy()
    #winner = []
    winner = prices[columns].idxmin(axis=1)
    # for person in tqdm.tqdm(prices.index):
    #     values = prices.iloc[person]
    #     winname = values[columns].idxmin()
    #     winner.append(winname)
    prices['winner'] = winner

    # create a dataframe with only the names to hold data
    market = pd.DataFrame(columns, columns=['name'])

    # add columns one after the other
    market['average_price_offered'] = [prices[c].mean() for c in columns]

    # compute these columns from the dataset
    market_share = []
    avg_pr_won = []
    tot_claims = []
    for player in columns:
        # get the fraction of the dataset where this player won
        df = prices[prices['winner'] == player]
        market_share.append(len(df.index) / len(prices))
        avg_pr_won.append(df[player].mean())
        tot_claims.append(df[claims_column].sum())
    # add them to the df
    market['market_share'] = market_share
    market['average_price_won'] = avg_pr_won
    market['total_loss'] = tot_claims
    # add information about monopoly profit
    total_losses = market['total_loss'].sum()
    market['monopoly_profit'] = [prices[c].sum() - total_losses for c in columns]

    # extract information from these columns
    market['revenue'] = market['market_share'] * market['average_price_won']
    market['profit'] = market['revenue'] - market['total_loss']

    market.fillna(0, inplace=True)
    
    return market


if __name__ == '__main__':

    # parse the prices file name from CLI
    if len(sys.argv) < 2:
        print(
            'Missing argument. Please use "python %s input.csv output.csv".' % sys.argv[0])
        sys.exit(1)
    prices_data = sys.argv[1]

    # parse the output file (market data)
    output_filename = 'market.csv'
    if len(sys.argv) >= 3:
        output_filename = sys.argv[2]

    # load the data
    prices = pd.read_csv(prices_data)

    # run the market
    market = run_market(prices)

    # save the market
    market.to_csv(output_filename, index=False)

    # why not?
    print(market.head())
