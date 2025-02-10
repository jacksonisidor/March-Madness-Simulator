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
        actual = self.data[['team_1', 'team_2', 'winner', 'current_round']]

        score = 0
        for (pred_index, pred_matchup), (act_index, act_matchup) in zip(predicted.iterrows(), actual.iterrows()):
            
            if (pred_matchup["team_1"] == act_matchup["team_1"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 1):
                score += 64 / pred_matchup["current_round"]
                
            elif (pred_matchup["team_2"] == act_matchup["team_2"]) and (pred_matchup["prediction"] == act_matchup["winner"] == 0): 
                score += 64 / pred_matchup["current_round"]
                
        return int(score)

        
    def sim_bracket(self, current_matchups=None):
        if current_matchups is None:
            current_matchups = self.data[
                (self.data["year"] == self.year) & 
                (self.data["type"] == "T") & 
                (self.data["current_round"] == 64)
            ]

        training_data = self.data[
            (self.data["year"] != self.year) | 
            ((self.data["year"] == self.year) & (self.data["type"] != "T"))
        ]
        model, predictors = self.train_model(training_data)

        predictions = self.predict_games(model, current_matchups, predictors)

        # Base case: Assign `self.predicted_bracket` and stop recursion
        if predictions["current_round"].iloc[0] == 2:
            self.predicted_bracket = predictions
            return  

        next_round_teams = self.get_winner_info(predictions)
        next_round_matchups = self.next_sim_matchups(next_round_teams)

        # Recursively simulate remaining rounds
        self.sim_bracket(next_round_matchups)

        # After recursion, assign full bracket to `self.predicted_bracket`
        self.predicted_bracket = pd.concat([predictions, self.predicted_bracket], ignore_index=True)

    

    def train_model(self, training_data):
        
        # set predictors based on the the user's selected playstyle
        if self.playstyle == "I Love Offense":
            predictors = [
                        'badj_o_diff', 'efg_diff', 'ft_rate_diff', 'tov_percent_diff', 
                        'adj_tempo_diff', '3p_percent_diff', '3p_rate_diff', '2p_percent_diff', 
                        'elite_sos_diff'
                        ]
        
        elif self.playstyle == "Defense Wins Championships":
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

        xgb_pipeline = make_pipeline(StandardScaler(), 
                                    XGBClassifier(n_estimators=300,
                                    max_depth=7,
                                    learning_rate=0.2,
                                    subsample=0.8,
                                    colsample_bytree=1.0,
                                    gamma=0
                                    ))

        xgb_pipeline.fit(training_data[predictors], training_data["winner"])

        return xgb_pipeline, predictors
    

    def predict_games(self, model, matchups, predictors):

        # get win probabilities (value represents probability of team_1 winning)
        probs = model.predict_proba(matchups[predictors])
        matchups["win probability"] = probs[:, 1]

        # add a little normally distributed randomness for fun :)
        randomness = np.random.normal(0, 0.025)
        matchups["win probability"] = np.clip(matchups["win probability"] + randomness, 0, 1)


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


        # force the user-picked winner to advance (1 if they are team_1, 0 if they are team_2 to match input data)
        matchups.loc[matchups["team_1"] == self.picked_winner, "prediction"] = 1
        matchups.loc[matchups["team_2"] == self.picked_winner, "prediction"] = 0

        return matchups


    def get_winner_info(self, matchups):
        next_round_teams_list = []
        
        for index, matchup in matchups.iterrows():
            # if team_1 wins, get all info that ends in "_1"
            if matchup["prediction"] == 1:
                winning_team_info = matchup.filter(regex='_1$').rename(lambda x: x[:-2], axis=0)
            # if team_2 wins, get all info that ends in "_2" 
            else:
                winning_team_info = matchup.filter(regex='_2$').rename(lambda x: x[:-2], axis=0)
            
            winning_team_info["year"] = matchup["year"]
            winning_team_info["current_round"] = matchup["current_round"] / 2
            
            next_round_teams_list.append(pd.DataFrame(winning_team_info).T)
        
        next_round_teams = pd.concat(next_round_teams_list, ignore_index=True)
            
        return next_round_teams

    def next_sim_matchups(self, winning_teams):
        matchups = pd.DataFrame(columns=['year', 'team_1', 'seed_1', 'round_1', 'current_round', 'team_2', 'seed_2', 'round_2'])

        matchup_info_list = []
        # iterate through data frame and jump 2 each iteration
        for i in range(0, len(winning_teams)-1, 2):
            team1_info = winning_teams.iloc[i]
            team2_info = winning_teams.iloc[i+1]

            matchup_info = {
                    'year': team1_info['year'],
                    'team_1': team1_info['team'],
                    'seed_1': team1_info['seed'],
                    'round_1': team1_info['round'],
                    'current_round': team1_info['current_round'],
                    'team_2': team2_info['team'],
                    'seed_2': team2_info['seed'],
                    'round_2': team2_info['round'],
                    'badj_em_1': team1_info['badj_em'],
                    'badj_o_1': team1_info['badj_o'],
                    'badj_d_1': team1_info['badj_d'],
                    'wab_1': team1_info['wab'],
                    'barthag_1': team1_info['barthag'],
                    'efg_1': team1_info['efg'],
                    'efg_d_1': team1_info['efg_d'],
                    'ft_rate_1': team1_info['ft_rate'],
                    'ft_rate_d_1': team1_info['ft_rate_d'],
                    'tov_percent_1': team1_info['tov_percent'],
                    'tov_percent_d_1': team1_info['tov_percent_d'],
                    'adj_tempo_1': team1_info['adj_tempo'],
                    '3p_percent_1': team1_info['3p_percent'],
                    '3p_rate_1': team1_info['3p_rate'],
                    '2p_percent_1': team1_info['2p_percent'],
                    '3p_percent_d_1': team1_info['2p_percent_d'],
                    '2p_percent_d_1': team1_info['2p_percent_d'],
                    'exp_1': team1_info['exp'],
                    'eff_hgt_1': team1_info['eff_hgt'],
                    'talent_1' : team1_info['talent'],
                    'elite_sos_1': team1_info['elite_sos'],
                    'win_percent_1': team1_info['win_percent'],
                    'badj_em_2': team2_info['badj_em'],
                    'badj_o_2': team2_info['badj_o'],
                    'badj_d_2': team2_info['badj_d'],
                    'wab_2': team2_info['wab'],
                    'barthag_2': team2_info['barthag'],
                    'efg_2': team2_info['efg'],
                    'efg_d_2': team2_info['efg_d'],
                    'ft_rate_2': team2_info['ft_rate'],
                    'ft_rate_d_2': team2_info['ft_rate_d'],
                    'tov_percent_2': team2_info['tov_percent'],
                    'tov_percent_d_2': team2_info['tov_percent_d'],
                    'adj_tempo_2': team2_info['adj_tempo'],
                    '3p_percent_2': team2_info['3p_percent'],
                    '3p_rate_2': team2_info['3p_rate'],
                    '2p_percent_2': team2_info['2p_percent'],
                    '3p_percent_d_2': team2_info['3p_percent_d'],
                    '2p_percent_d_2': team2_info['2p_percent_d'],
                    'exp_2': team2_info['exp'],
                    'eff_hgt_2': team2_info['eff_hgt'],
                    'talent_2' : team2_info['talent'],
                    'elite_sos_2': team2_info['elite_sos'],
                    'win_percent_2': team2_info['win_percent']
                    }
    
            matchup_info_list.append(matchup_info)

        matchups = pd.concat([matchups, pd.DataFrame(matchup_info_list)])
            
        # get the stat differences same as before
        stat_variables = [
                        'badj_em', 'badj_o', 'badj_d', 'wab', 'barthag', 'efg', 'efg_d', 
                        'ft_rate', 'ft_rate_d', 'tov_percent', 'tov_percent_d', 'adj_tempo', 
                        '3p_percent', '3p_rate', '2p_percent', '3p_percent_d', '2p_percent_d',
                        'exp', 'eff_hgt', 'talent', 
                        'elite_sos', 'win_percent'
                        ]
        for variable in stat_variables:
            matchups[f'{variable}_diff'] = matchups[f'{variable}_1'] - matchups[f'{variable}_2']
            
        return matchups

data = pd.read_parquet("data/all_matchup_stats.parquet")
sim = BracketSimulator(data, 2024)
sim.sim_bracket()
print(sim.score_bracket())
