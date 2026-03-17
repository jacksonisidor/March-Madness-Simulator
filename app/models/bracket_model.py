# IMPORTS
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import math
import warnings
import gc
import psutil, os
import sys
warnings.filterwarnings("ignore")
import random
import platform

import xgboost
print("[DEBUG] xgboost version:", xgboost.__version__)

SEED = 44
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

print("[DEBUG] platform:", platform.platform())
print("[DEBUG] machine:", platform.machine())
print("[DEBUG] python:", sys.version)
print("[DEBUG] numpy:", np.__version__)
print("[DEBUG] pandas:", pd.__version__)

# function to log current memory usage
def log_memory_usage(tag):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[MEMORY] {tag}: {mem:.2f} MB")
    sys.stdout.flush()  # force immediate flushing

## SIMULATION
class BracketSimulator: 

    def __init__(self, data, year, picked_winner=None, playstyle="Balanced", boldness="Normal"):
        self.data = data
        self.year = year
        self.boldness = boldness
        self.picked_winner = picked_winner
        self.playstyle = playstyle
        self.predicted_bracket = None

        print("[DEBUG] BracketSimulator.__init__")
        print("[DEBUG] year:", self.year)
        print("[DEBUG] picked_winner:", self.picked_winner)
        print("[DEBUG] playstyle:", self.playstyle)
        print("[DEBUG] boldness:", self.boldness)
        print("[DEBUG] full data shape:", self.data.shape)
        sys.stdout.flush()

    
    def score_bracket(self):
            predicted = self.predicted_bracket[['team_1', 'team_2', 'prediction', 'current_round']].reset_index(drop=True)
            actual = self.data[(self.data["year"] == self.year) & (self.data["type"] == "T")][['team_1', 'team_2', 'winner', 'current_round']].reset_index(drop=True)

            mask1 = (predicted["team_1"] == actual["team_1"]) & (predicted["prediction"] == 1) & (actual["winner"] == 1)
            mask2 = (predicted["team_2"] == actual["team_2"]) & (predicted["prediction"] == 0) & (actual["winner"] == 0)

            score = (((64 / predicted["current_round"]) * 10)[mask1].sum() + 
                    ((64 / predicted["current_round"]) * 10)[mask2].sum())

            return int(score)
        
    def sim_bracket(self, current_matchups=None, model=None, predictors=None):

        print("\n[DEBUG] sim_bracket ENTER")
        print("[DEBUG] current_matchups is None:", current_matchups is None)
        print("[DEBUG] model is None:", model is None)
        sys.stdout.flush()

        log_memory_usage("sim_bracket start")

        # get round of 64 at the start
        if current_matchups is None:

            # get the data we need to predict for the first round
            current_matchups = self.data[
                (self.data["year"] == self.year) & 
                (self.data["type"] == "T") & 
                (self.data["current_round"] == 64)
            ].copy()
            current_matchups['team1_path_odds'] = 1
            current_matchups['team2_path_odds'] = 1

            log_memory_usage("After copying round 64 data")
        
        print("[DEBUG] initial round64 shape:", current_matchups.shape)
        print("[DEBUG] initial round64 matchups:")
        print(current_matchups[["team_1", "team_2", "seed_1", "seed_2", "current_round"]].to_string())
        sys.stdout.flush()


        # only train model if it hasn't been trained yet
        if model is None:

            # get all data that was not in this years tournament (and within recent years)
            training_data = self.data[
                ((self.data["year"] < self.year) & (self.year - self.data["year"] <= 5)) | 
                ((self.data["year"] == self.year) & (self.data["type"] != "T"))
            ]

            print("[DEBUG] training_data shape before train_model:", training_data.shape)
            print("[DEBUG] training_data year counts:", training_data["year"].value_counts().sort_index().tail(10).to_dict())
            print("[DEBUG] training_data type counts:", training_data["type"].value_counts(dropna=False).to_dict())
            print("[DEBUG] training_data round counts sample:", training_data["current_round"].value_counts().sort_index().to_dict())
            print("[DEBUG] training_data first 10 rows:")
            print(training_data[["year", "type", "team_1", "team_2", "current_round", "winner"]].head(10).to_string())
            sys.stdout.flush()

            model, predictors = self.train_model(training_data)
            log_memory_usage("After training model")

        predictions = self.predict_games(model, current_matchups, predictors)
        log_memory_usage("After predicting current round")

        # base case: reached championship, no more rounds
        if predictions["current_round"].iloc[0] == 2:
            self.predicted_bracket = predictions
            log_memory_usage("Reached championship; ending recursion")
            return  

        # save a copy of current round predictions for later concatenation
        temp_predictions = predictions.copy()
        log_memory_usage("After copying predictions for later")

        next_round_teams = self.get_winner_info(predictions)
        next_round_matchups = self.next_sim_matchups(next_round_teams)
        log_memory_usage("After preparing next round matchups")

        # free intermediate objects from current round
        del predictions, next_round_teams
        gc.collect()
        log_memory_usage("After cleaning up current round objects")

        # recursively simulate remaining rounds
        self.sim_bracket(next_round_matchups, model, predictors)

        # after recursion, assign full bracket
        self.predicted_bracket = pd.concat([temp_predictions, self.predicted_bracket], ignore_index=True)
        
        log_memory_usage("After concatenating predictions")
        del temp_predictions
        gc.collect()
        log_memory_usage("End of sim_bracket")
    
    def train_model(self, training_data):

        print("\n[DEBUG] train_model ENTER")
        print("[DEBUG] incoming training_data shape:", training_data.shape)
        sys.stdout.flush()

        # define feature importance multipliers for different playstyles
        weight_multiplier = 1  
        feature_weights = {
            "Offensive-Minded": {
                'badj_o_diff': weight_multiplier, 'efg_diff': weight_multiplier, 'ft_rate_diff': weight_multiplier, 
                '3p_percent_diff': weight_multiplier, '3p_rate_diff': weight_multiplier, '2p_percent_diff': weight_multiplier
            },
            "Defense Wins": {
                'badj_d_diff': weight_multiplier, 'efg_d_diff': weight_multiplier, 'ft_rate_d_diff': weight_multiplier, 
                'tov_percent_d_diff': weight_multiplier, '3p_percent_d_diff': weight_multiplier, '2p_percent_d_diff': weight_multiplier
            },
            "Balanced": {'2p_percent_d_diff': 0, '3p_percent_d_diff': 0}  
        }

        # define all predictors
        predictors = [
            'badj_em_diff', 'badj_o_diff', 'badj_d_diff', 'wab_diff', 'barthag_diff',
            'efg_diff', 'efg_d_diff', 'ft_rate_diff', 'ft_rate_d_diff', 
            'tov_percent_diff', 'tov_percent_d_diff', 'adj_tempo_diff', 
            '3p_percent_diff', '3p_rate_diff', '2p_percent_diff', 'exp_diff', 
            'eff_hgt_diff', 'talent_diff', 'elite_sos_diff', 'win_percent_diff',
            '3p_percent_d_diff', '2p_percent_d_diff', 'recent_wab_diff'
        ]

        print("[DEBUG] predictors:", predictors)
        print("[DEBUG] number of predictors:", len(predictors))
        sys.stdout.flush()

        # apply feature weights
        training_data = training_data.copy()
        for feature, weight in feature_weights.get(self.playstyle, {}).items():
            if feature in training_data.columns:
                training_data[feature] = training_data[feature] * weight 
        
        print("[DEBUG] post-copy training_data shape:", training_data.shape)
        print("[DEBUG] null counts in predictors:")
        print(training_data[predictors].isnull().sum().to_string())
        sys.stdout.flush()

        # initialize sample weights to 1 (base weight)
        training_data["weight"] = 1.0  

        # apply dynamic sample weights based on playstyle
        
        # weight games where the offensive gap is bigger than the defensive gap
        if self.playstyle == "Offensive-Minded":
            training_data["weight"] *= (1 + (abs(training_data["badj_o_diff"]) - abs(training_data["badj_d_diff"])).clip(lower=0))

        # weight games where the defensive gap is bigger than the offensive gap
        elif self.playstyle == "Defense Wins":
            training_data["weight"] *= (1 + (abs(training_data["badj_d_diff"]) - abs(training_data["badj_o_diff"])).clip(lower=0))

        # apply further weight to tournament games
        tourney_weight = 5
        training_data["weight"] *= training_data["type"].map({"T": tourney_weight, "RS": 1}).fillna(1)

        # ensure weights are always positive and non-zero
        training_data["weight"] = training_data["weight"].clip(lower=0.01).fillna(0.01)

        print("[DEBUG] weight summary:")
        print(training_data["weight"].describe().to_string())
        print("[DEBUG] weight by type:")
        print(training_data.groupby("type")["weight"].describe().to_string())
        sys.stdout.flush()

        print("[DEBUG] X_train shape:", training_data[predictors].shape)
        print("[DEBUG] y_train shape:", training_data["winner"].shape)
        print("[DEBUG] y_train distribution:", training_data["winner"].value_counts(dropna=False).to_dict())
        print("[DEBUG] first 8 X rows:")
        print(training_data[predictors].head(8).to_string())
        print("[DEBUG] first 8 y values:", training_data["winner"].head(8).tolist())
        sys.stdout.flush()

        # train the model
        model = XGBClassifier(
            n_estimators=125,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=1,
            gamma=4,
            random_state=44,
            n_jobs=1,
            tree_method="hist",
            device='cpu',
            validate_parameters=True
        )

        # fit the model
        model.fit(training_data[predictors], 
                training_data["winner"], 
                sample_weight=training_data["weight"])
        
        print("[DEBUG] model fit complete")
        sys.stdout.flush()

        
        del training_data
        gc.collect()

        return model, predictors

    def predict_games(self, model, matchups, predictors):

        round_label = int(matchups["current_round"].iloc[0]) if len(matchups) else None
        print(f"\n[DEBUG] ===== ROUND {round_label} =====")

        print("\n[DEBUG] predict_games ENTER")
        print("[DEBUG] incoming matchups shape:", matchups.shape)
        print("[DEBUG] incoming matchups round:", matchups["current_round"].iloc[0] if len(matchups) else None)
        print("[DEBUG] incoming matchup teams:")
        print(matchups[["team_1", "team_2", "seed_1", "seed_2", "current_round"]].to_string())
        sys.stdout.flush()

        matchups = matchups.copy()
        matchups[predictors] = matchups[predictors].apply(pd.to_numeric, errors='coerce')

        # get win probabilities (value represents probability of team_1 winning)
        probs = model.predict_proba(matchups[predictors])
        matchups.loc[:, "win probability"] = probs[:, 1]
        p = matchups["win probability"]

        print("[DEBUG] raw predict_proba first rows:")
        print(pd.DataFrame({
            "team_1": matchups["team_1"],
            "team_2": matchups["team_2"],
            "prob_team1_wins": probs[:, 1]
        }).to_string())
        sys.stdout.flush()

        # factor in path likelihoods
        alpha = 1 # weighting of path odds vs win prob
        adjusted_p = (p * (matchups["team1_path_odds"] ** alpha)) / (
            (p * (matchups["team1_path_odds"] ** alpha)) + ((1 - p) * (matchups["team2_path_odds"] ** alpha))
        )
        matchups["adj win probability"] = adjusted_p

        print("[DEBUG] adjusted probabilities:")
        print(matchups[["team_1", "team_2", "win probability", "team1_path_odds", "team2_path_odds", "adj win probability"]].to_string())
        sys.stdout.flush()

        # update path likelihoods for next round
        matchups["team1_path_odds"] *= matchups["win probability"]
        matchups["team2_path_odds"] *= (1 - matchups["win probability"])

        # add a little normally distributed randomness for fun :)
        randomness = np.random.normal(0, 0.05, size=matchups.shape[0])
        randomness = np.clip(randomness, -0.1, 0.1)  # so it doesn't get too out of hand
        matchups["adj win probability"] = np.clip(matchups["adj win probability"] + randomness, 0.001, 0.999)


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
        
        # apply boldness. upset only applies to differences of >= 1 seeds (maybe 2 is better?)
        mask_upset_underdog = (matchups["seed_1"] > matchups["seed_2"]) & ((matchups["seed_1"] - matchups["seed_2"]) >= 1)
        mask_upset_favorite = (matchups["seed_1"] < matchups["seed_2"]) & ((matchups["seed_2"] - matchups["seed_1"]) >= 1)
        mask_similar = ~(mask_upset_underdog | mask_upset_favorite)
        matchups.loc[mask_upset_underdog, "prediction"] = (matchups.loc[mask_upset_underdog, "adj win probability"] > threshold_lower_seed).astype(int)
        matchups.loc[mask_upset_favorite, "prediction"] = (matchups.loc[mask_upset_favorite, "adj win probability"] > threshold_higher_seed).astype(int)
        matchups.loc[mask_similar, "prediction"] = (matchups.loc[mask_similar, "adj win probability"] > 0.5).astype(int)

        # force the user-picked winner to advance (1 if they are team_1, 0 if they are team_2 to match input data)
        matchups.loc[matchups["team_1"] == self.picked_winner, "prediction"] = 1
        matchups.loc[matchups["team_2"] == self.picked_winner, "prediction"] = 0

        print("[DEBUG] final predictions this round:")
        print(matchups[["team_1", "team_2", "seed_1", "seed_2", "adj win probability", "prediction"]].to_string())
        sys.stdout.flush()

        return matchups


    def get_winner_info(self, matchups):

        print("\n[DEBUG] get_winner_info ENTER")
        print("[DEBUG] winners this round:")
        debug_winners = matchups[["team_1", "team_2", "prediction", "current_round"]].copy()
        debug_winners["picked_team"] = np.where(debug_winners["prediction"] == 1, debug_winners["team_1"], debug_winners["team_2"])
        print(debug_winners.to_string())
        sys.stdout.flush()

        # identify winners
        winner_mask = matchups["prediction"].to_numpy() 

        # select columns based on the winner
        winner_data_1 = matchups.filter(regex='_1$')
        winner_data_2 = matchups.filter(regex='_2$')

        # get the winning data based on who won
        winning_data = np.where(winner_mask[:, None], winner_data_1.values, winner_data_2.values)
        winning_path_odds = np.where(winner_mask, matchups["team1_path_odds"], matchups["team2_path_odds"])

        # create df
        next_round_teams = pd.DataFrame(winning_data, columns=[col[:-2] for col in winner_data_1.columns])

        # add year and current_round
        next_round_teams["year"] = matchups["year"].values
        next_round_teams["current_round"] = matchups["current_round"].values / 2
        next_round_teams["path_odds"] = winning_path_odds

        print("[DEBUG] next_round_teams:")
        print(next_round_teams[["team", "seed", "round", "current_round", "path_odds"]].to_string())
        sys.stdout.flush()

        return next_round_teams


    def next_sim_matchups(self, winning_teams):

        print("\n[DEBUG] next_sim_matchups ENTER")
        print("[DEBUG] winning_teams shape:", winning_teams.shape)
        print("[DEBUG] winning_teams rows:")
        print(winning_teams[["team", "seed", "round", "current_round", "path_odds"]].to_string())
        sys.stdout.flush()

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
            'round_2': team2['round']
        })
        
        # separate path odds
        matchups["team1_path_odds"] = team1["path_odds"]
        matchups["team2_path_odds"] = team2["path_odds"]

        # add stat columns (team 1, team 2, and difference)
        stat_variables = [
            'badj_em', 'badj_o', 'badj_d', 'wab', 'recent_wab', 'barthag', 'efg', 'efg_d',
            'ft_rate', 'ft_rate_d', 'tov_percent', 'tov_percent_d', 'adj_tempo',
            '3p_percent', '3p_rate', '2p_percent', '3p_percent_d', '2p_percent_d',
            'exp', 'eff_hgt', 'talent', 'elite_sos', 'win_percent'
        ]
        for var in stat_variables:
            matchups[f'{var}_1'] = team1[var].values
            matchups[f'{var}_2'] = team2[var].values
            matchups[f'{var}_diff'] = team1[var].values - team2[var].values  # Vectorized subtraction

        print("[DEBUG] next round matchups created:")
        print(matchups[["team_1", "team_2", "seed_1", "seed_2", "current_round", "team1_path_odds", "team2_path_odds"]].to_string())
        sys.stdout.flush()

        return matchups

#FOR TESTING
#data = pd.read_parquet("data/all_matchup_stats.parquet")
#simulator = BracketSimulator(data, 2026)
#simulator.sim_bracket()
#print(simulator.predicted_bracket[['team_1', 'team_2', 'current_round', 'win probability', 'adj win probability', 'prediction']]) 