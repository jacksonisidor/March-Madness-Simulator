'''
This script simulates a given number of brackets for each year available based on historical seed-based odds.
Structure of simulation will be very similar to bracket_model.py, but adapted to make odds-based random prediction.
'''

import pandas as pd
import numpy as np
import time


def score_bracket(predicted, actual):

        score = 0
        for (pred_index, pred_matchup), (act_index, act_matchup) in zip(predicted.iterrows(), actual.iterrows()):
            
            if (pred_matchup["team_1"] == act_matchup["team_1"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 1):
                score += (64 / pred_matchup["current_round"]) * 10
                
            elif (pred_matchup["team_2"] == act_matchup["team_2"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 0): 
                score += (64 / pred_matchup["current_round"]) * 10
                
        return int(score)

def get_winner_info(matchups):

    # identify winners
    winner_mask = matchups["prediction"].to_numpy() 

    # select columns based on the winner
    winner_data_1 = matchups.filter(regex='_1$')
    winner_data_2 = matchups.filter(regex='_2$')

    # get the winning data based on who won
    winning_data = np.where(winner_mask[:, None], winner_data_1.values, winner_data_2.values)

    # create df
    next_round_teams = pd.DataFrame(winning_data, columns=[col[:-2] for col in winner_data_1.columns])

    # add year and current_round`
    next_round_teams["year"] = matchups["year"].values
    next_round_teams["current_round"] = matchups["current_round"].values / 2

    return next_round_teams

def next_sim_matchups(winning_teams):

    # select alternating rows for team1 and team2
    team1 = winning_teams.iloc[::2].reset_index(drop=True)
    team2 = winning_teams.iloc[1::2].reset_index(drop=True)

    # create matchups DF with all non-predictors
    matchups = pd.DataFrame({
        'year': team1['year'],
        'team_1': team1['team'],
        'seed_1': team1['seed'],
        'current_round': team1['current_round'],
        'team_2': team2['team'],
        'seed_2': team2['seed']
    })

    return matchups

def predict_games(matchups):

    # generate random win probabilities for each matchup
    random_probs = np.random.rand(len(matchups))

    # compare to historical odds to make prediction
    matchups["prediction"] = (random_probs < matchups["odds"]).astype(int)

    return matchups


def sim(current_matchups, odds):

    # merge odds with matchups each round (account of order of seed1,seed2)
    current_matchups = current_matchups.merge(
        odds, 
        how="left",
        left_on=["seed_1", "seed_2"], 
        right_on=["seed_1", "seed_2"]
    ).merge(
        odds,
        how="left",
        left_on=["seed_2", "seed_1"],
        right_on=["seed_1", "seed_2"],
        suffixes=("", "_alt")
    )
    current_matchups["odds"] = current_matchups["odds"].fillna(current_matchups["odds_alt"])
    current_matchups.drop(columns=["odds_alt"], inplace=True)

    predictions = predict_games(current_matchups)

    # Base case: Assign `predicted_bracket` and stop recursion
    if predictions["current_round"].iloc[0] == 2:
       return predictions

    next_round_teams = get_winner_info(predictions)
    next_round_matchups = next_sim_matchups(next_round_teams)

    # Recursively simulate remaining rounds
    return pd.concat([predictions, sim(next_round_matchups, odds)], ignore_index=True)


matchups = pd.read_parquet('data/all_matchup_stats.parquet', 
                           columns = ['year', 'current_round', 'team_1', 'seed_1', 'team_2', 'seed_2', 'winner'])
odds = pd.read_parquet('data/seed_odds.parquet')

# simulate given number of brackets for each year and store in dictionary
num_sims = 10000
sim_results = [] 
print("STARTING SIMULATION")
for year in matchups.year.unique():

    if year not in [2020, 2021]:

        print(f"SIMULATING {year}")
        start_time = time.perf_counter()  # start timer
        
        current_data = matchups[matchups.year == year]  # Get all games for the year
        current_r64 = current_data[current_data.current_round == 64]  # Round of 64 matchups

        for i in range(1, num_sims+1):

            # Simulate and score the bracket
            bracket = sim(current_r64, odds)
            score = score_bracket(bracket, current_data)
            sim_results.append((year, i, score))

        end_time = time.perf_counter()  # End timer
        total_time = end_time - start_time
        print(f"DONE WITH {year} IN {total_time:.2f} SECONDS")

# Convert to DataFrame
scores_df = pd.DataFrame(sim_results, columns=["year", "sim_number", "score"])

# Save as Parquet
scores_df.to_parquet('data/odds_sim_scores.parquet', index=False)