from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from models.bracket_model import BracketSimulator

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'BBallSim'

# Read in data
data = pd.read_parquet("data/all_matchup_stats.parquet")
odds_sim_scores = pd.read_parquet("data/odds_sim_scores.parquet")

# function to format the bracket as nested list
def format_bracket(results):

    num_rounds = 7  # rounds from 64 teams to a single champion
    bracket_structure = [[] for _ in range(num_rounds)]

    for _, row in results.iterrows():
        round_index = int(round((np.log2(64 / row['current_round']))))  # convert to index

        # determine the actual winner
        winner = 0 if row["prediction"] == 1 else 1  # 0 means team_1 won, 1 means team_2 won

        # store the matchup along with the winner index
        matchup = [row['team_1'], row['team_2'], winner]
        bracket_structure[round_index].append(matchup)

    # extract the final winner from the last matchup
    final_game = results[results['current_round'] == 2].iloc[0]  
    final_matchup = bracket_structure[-2][0]  # get last game stored in bracket
    final_winner = final_matchup[0] if final_game['prediction'] == 1 else final_matchup[1]

    # add the final winner as the last round
    bracket_structure[-1].append([final_winner, "", 0])  # no opponent, default winner

    return bracket_structure


def convert_bracket_format(simulation_output):
    formatted_bracket = {"rounds": []}
    num_rounds = len(simulation_output)

    for round_index, matchups in enumerate(simulation_output):
        round_data = {"round": round_index + 1, "left": [], "right": []}

        # special handling for round 6 (index 5) when there's a single matchup:
        if round_index == 5 and len(matchups) == 1:
            team1, team2, _ = matchups[0]

            # instead of one match in one side, split the teams:
            round_data["left"].append({
                "team1": team1,
                "team2": "",    # no opponent here
                "winner": ""    # no winner yet (or leave as an empty string)
            })
            round_data["right"].append({
                "team1": "",    # no team on the left in this slot
                "team2": team2,
                "winner": ""
            })
        else:
            mid_point = len(matchups) // 2  # split into left & right by default

            for i, match in enumerate(matchups):
                team1, team2, winner_index = match
                winner = team1 if winner_index == 0 else team2

                if i < mid_point:
                    round_data["left"].append({
                        "team1": team1,
                        "team2": team2,
                        "winner": winner
                    })
                else:
                    round_data["right"].append({
                        "team1": team1,
                        "team2": team2,
                        "winner": winner
                    })

        formatted_bracket["rounds"].append(round_data)
        
    return formatted_bracket




# function to get correct suffix of percentile
def ordinal(n):
    if 10 <= n % 100 <= 20:  # special case for 11th-19th
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# Route for the home page
@app.route('/')
def home():
    max_year = data["year"].max()
    valid_years = sorted(data["year"].unique()) 
    valid_years = [year for year in valid_years if year not in [2020, 2021]] 
    return render_template('home.html', max_year=max_year, valid_years=valid_years)

# Route to team list for a given year
@app.route('/get_teams/<int:year>')
def get_teams(year):
    tournament_teams = data[(data["year"] == year) & (data["type"] == "T")]
    unique_teams = sorted(set(tournament_teams["team_1"].tolist() + tournament_teams["team_2"].tolist()))
    
    return jsonify(["None"] + unique_teams)  # Send JSON response


# Roate for simulation
@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # get the json input from the request
        input_data = request.json

        # extract the selected parameters for the simulation (with default values)
        year = input_data.get('year', 2024)
        picked_winner = input_data.get('picked_winner')
        playstyle = input_data.get('playstyle', 'Balanced')
        boldness = input_data.get('boldness', 'Normal')

        # initialize and run the simulator
        simulator = BracketSimulator(data, year, picked_winner, playstyle, boldness)
        simulator.sim_bracket()
        predictions = format_bracket(simulator.predicted_bracket)
        score = simulator.score_bracket()

        # handle scoring if results exist
        try:
            score = simulator.score_bracket()  
        except Exception as e:
            score = None  # set score to None if scoring fails

        # handle percentile calculation
        percentile = None
        if score is not None:
            year_sim_scores = odds_sim_scores[odds_sim_scores["year"] == year]["score"]

            # check if there are valid historical scores
            if len(year_sim_scores) > 0:
                percentile = ordinal(int(percentileofscore(year_sim_scores, score, kind='rank')))


        # store results in session
        session['selected_params'] = input_data
        session['simulation_results'] = predictions
        session['score'] = score
        session['percentile'] = percentile

        # redirect to results page after processing
        return jsonify({'redirect_url': url_for('results')})
    
    except Exception as e:
        print("Error", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# route for the results page
@app.route('/results')
def results():
    selected_params = session.get('selected_params')
    raw_results = session.get('simulation_results')
    score = session.get('score')
    percentile = session.get('percentile')

    # format the bracket for jquery
    formatted_bracket = convert_bracket_format(raw_results)

    return render_template('results.html', selected_params=selected_params, 
                           results=raw_results, formatted_bracket=formatted_bracket,
                           score=score, percentile=percentile)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
