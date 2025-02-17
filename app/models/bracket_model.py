# IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


## SIMULATION
class BracketSimulator: 

    def __init__(self, data, year, picked_winner=None, playstyle="Balanced", boldness="Normal"):
        self.data = data
        self.year = year
        self.boldness = boldness
        self.picked_winner = picked_winner
        self.playstyle = playstyle
        self.predicted_bracket = None

    
    def score_bracket(self):

        predicted = self.predicted_bracket[['team_1', 'team_2', 'prediction', 'current_round']]
        actual = self.data[(self.data["year"] == self.year) & (self.data["type"] == "T")][['team_1', 'team_2', 'winner', 'current_round']]

        score = 0
        for (pred_index, pred_matchup), (act_index, act_matchup) in zip(predicted.iterrows(), actual.iterrows()):
            
            if (pred_matchup["team_1"] == act_matchup["team_1"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 1):
                score += (64 / pred_matchup["current_round"]) * 10
                
            elif (pred_matchup["team_2"] == act_matchup["team_2"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 0): 
                score += (64 / pred_matchup["current_round"]) * 10
                
        return int(score)

        
    def sim_bracket(self, current_matchups=None, model=None, predictors=None):

        # get round of 64 at the start
        if current_matchups is None:

            # get the data we need to predict for the first round
            current_matchups = self.data[
                (self.data["year"] == self.year) & 
                (self.data["type"] == "T") & 
                (self.data["current_round"] == 64)
            ].copy()


        # Only train model if it hasn't been trained yet
        if model is None:

            # get all data that was not in this years tournament
            training_data = self.data[
                (self.data["year"] != self.year) | 
                ((self.data["year"] == self.year) & (self.data["type"] != "T"))
            ]
            model, predictors = self.train_model(training_data)

        predictions = self.predict_games(model, current_matchups, predictors)

        # Base case: Reached championship, no more rounds
        if predictions["current_round"].iloc[0] == 2:
            self.predicted_bracket = predictions
            return  

        next_round_teams = self.get_winner_info(predictions)
        next_round_matchups = self.next_sim_matchups(next_round_teams)

        # Recursively simulate remaining rounds
        self.sim_bracket(next_round_matchups, model, predictors)

        # After recursion, assign full bracket
        self.predicted_bracket = pd.concat([predictions, self.predicted_bracket], ignore_index=True)

    

    def train_model(self, training_data):
        
        # set predictors based on the the user's selected playstyle
        if self.playstyle == "Offensive-Minded":
            predictors = [
                        'badj_o_diff', 'efg_diff', 'ft_rate_diff', 'tov_percent_diff', 
                        'adj_tempo_diff', '3p_percent_diff', '3p_rate_diff', '2p_percent_diff', 
                        'elite_sos_diff'
                        ]
        
        elif self.playstyle == "Defense Wins":
            predictors = [
                        'badj_d_diff', 'efg_d_diff', 'ft_rate_d_diff', 'tov_percent_d_diff', 
                        'adj_tempo_diff', '3p_percent_d_diff', '2p_percent_d_diff', 
                        'elite_sos_diff'
                        ]
        
        else:
            predictors = [
                            'badj_em_diff', 'badj_o_diff', 'badj_d_diff', 'wab_diff', 'barthag_diff',
                            'efg_diff', 'efg_d_diff', 'ft_rate_diff', 'ft_rate_d_diff', 
                            'tov_percent_diff', 'tov_percent_d_diff', 'adj_tempo_diff', 
                            '3p_percent_diff', '3p_rate_diff', '2p_percent_diff', 'exp_diff', 
                            'eff_hgt_diff', 'talent_diff', 'elite_sos_diff', 'win_percent_diff'
                            ]

        model = XGBClassifier(n_estimators=100,
                                    max_depth=5,
                                    learning_rate=0.2,
                                    subsample=0.9,
                                    colsample_bytree=1,
                                    gamma=5,
                                    random_state=44
                                )

        training_data["weight"] = training_data["type"].map({"T": 1, "RS": 1})
        model.fit(training_data[predictors], training_data["winner"], sample_weight=training_data["weight"])

        return model, predictors
    

    def predict_games(self, model, matchups, predictors):

        matchups = matchups.copy()
        matchups[predictors] = matchups[predictors].apply(pd.to_numeric, errors='coerce')

        # get win probabilities (value represents probability of team_1 winning)
        probs = model.predict_proba(matchups[predictors])
        matchups.loc[:, "win probability"] = probs[:, 1]

        # add a little normally distributed randomness for fun :)
        #randomness = np.random.normal(0, 0.025)
        #matchups["win probability"] = np.clip(matchups["win probability"] + randomness, 0, 1)


        # set different thresholds based on boldness and if the team 1 is higher/lower seed
        if self.boldness == "Go Big or Go Home":
            threshold_higher_seed = 0.63
            threshold_lower_seed = 0.37
        elif self.boldness == "Bold":
            threshold_higher_seed = 0.57
            threshold_lower_seed = 0.43
        elif self.boldness == "Normal":
            threshold_higher_seed = 0.5
            threshold_lower_seed = 0.5
        elif self.boldness == "Safe":
            threshold_higher_seed = 0.43
            threshold_lower_seed = 0.57
        elif self.boldness == "So Safe":
            threshold_higher_seed = 0.37
            threshold_lower_seed = 0.63

        # if seed 1 is the higher seed (underdog)
        matchups.loc[matchups["seed_1"] > matchups["seed_2"], "prediction"] = (matchups["win probability"] > threshold_lower_seed)*1
        # if seed 1 is the lower seed (favorite)
        matchups.loc[matchups["seed_1"] < matchups["seed_2"], "prediction"] = (matchups["win probability"] > threshold_higher_seed)*1
        # if seed 1 and seed 2 are the same seed
        matchups.loc[matchups["seed_1"] == matchups["seed_2"], "prediction"] = (matchups["win probability"] > 0.5)*1

        # apply close call strategy
        matchups.loc[matchups["close_call_1"] & ~matchups["close_call_2"], "prediction"] = 0  
        matchups.loc[matchups["close_call_2"] & ~matchups["close_call_1"], "prediction"] = 1 

        # note close calls for the next round
        close_thresh = 0.1
        matchups.loc[:, "close_call_1"] = (matchups["win probability"] >= 0.5 - close_thresh) & (matchups["win probability"] <= 0.5 + close_thresh)  
        matchups.loc[:, "close_call_2"] = (1 - matchups["win probability"] >= 0.5 - close_thresh) & (1 - matchups["win probability"] <= 0.5 + close_thresh)


        # force the user-picked winner to advance (1 if they are team_1, 0 if they are team_2 to match input data)
        matchups.loc[matchups["team_1"] == self.picked_winner, "prediction"] = 1
        matchups.loc[matchups["team_2"] == self.picked_winner, "prediction"] = 0

        return matchups


    def get_winner_info(self, matchups):

        # identify winners
        winner_mask = matchups["prediction"].to_numpy() 

        # select columns based on the winner
        winner_data_1 = matchups.filter(regex='_1$')
        winner_data_2 = matchups.filter(regex='_2$')

        # get the winning data based on who won
        winning_data = np.where(winner_mask[:, None], winner_data_1.values, winner_data_2.values)

        # create df
        next_round_teams = pd.DataFrame(winning_data, columns=[col[:-2] for col in winner_data_1.columns])

        # add year and current_round
        next_round_teams["year"] = matchups["year"].values
        next_round_teams["current_round"] = matchups["current_round"].values / 2

        return next_round_teams


    def next_sim_matchups(self, winning_teams):

        # select alternating rows for team1 and team2
        team1 = winning_teams.iloc[::2].reset_index(drop=True)
        team2 = winning_teams.iloc[1::2].reset_index(drop=True)

        # create matchups DF with all non-predictors
        matchups = pd.DataFrame({
            'year': team1['year'],
            'team_1': team1['team'],
            'seed_1': team1['seed'],
            'round_1': team1['round'],
            'current_round': team1['current_round'],
            'team_2': team2['team'],
            'seed_2': team2['seed'],
            'round_2': team2['round'],
            'close_call_1': team1['close_call'],
            'close_call_2': team2['close_call']
        })

        # add stat columns (team 1, team 2, and difference)
        stat_variables = [
            'badj_em', 'badj_o', 'badj_d', 'wab', 'barthag', 'efg', 'efg_d',
            'ft_rate', 'ft_rate_d', 'tov_percent', 'tov_percent_d', 'adj_tempo',
            '3p_percent', '3p_rate', '2p_percent', '3p_percent_d', '2p_percent_d',
            'exp', 'eff_hgt', 'talent', 'elite_sos', 'win_percent'
        ]
        for var in stat_variables:
            matchups[f'{var}_1'] = team1[var].values
            matchups[f'{var}_2'] = team2[var].values
            matchups[f'{var}_diff'] = team1[var].values - team2[var].values  # Vectorized subtraction

        return matchups


# FOR TESTING

'''data = pd.read_parquet("data/all_matchup_stats.parquet")
simulator = BracketSimulator(data, 2024)
simulator.sim_bracket()
print(simulator.score_bracket())
'''