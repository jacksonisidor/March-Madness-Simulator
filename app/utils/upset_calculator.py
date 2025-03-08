'''
This script calculates historical upset statistics
'''

import pandas as pd
import numpy as np

matchups = pd.read_parquet('data/all_matchup_stats.parquet', 
                           columns = ['year', 'current_round', 'seed_1', 'seed_2', 'winner', 'type'])

# loop through each year
upset_data = []
for year in matchups.year.unique():
    if year not in [2020, 2021]:
        tournament_data = matchups[(matchups.year == year) & (matchups.type == 'T')]

        # loop through rounds in this year
        formatted_round = 0
        for tourney_round in tournament_data.current_round.unique():

            formatted_round+=1
            round_data = tournament_data[tournament_data.current_round == tourney_round]

            # identify upsets
            upsets = round_data[
                ((round_data.seed_1 > round_data.seed_2) & (round_data.winner == 1)) | 
                ((round_data.seed_1 < round_data.seed_2) & (round_data.winner == 0))
            ]
            upset_count = len(upsets)

            # add the info to the dataframe
            upset_data.append({'year': year, 'round': formatted_round, 'upset_count': upset_count})

# write to file
upset_counts = pd.DataFrame(upset_data)
upset_counts.to_csv('data/upset_counts.csv')
