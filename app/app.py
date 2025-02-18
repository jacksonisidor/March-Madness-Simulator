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

    # Initialize a list of empty lists for each round
    bracket_structure = [[] for _ in range(num_rounds)]

    for _, row in results.iterrows():
        round_index = int(round((np.log2(64 / row['current_round']))))  # Convert to 0-5 index

        # Determine the actual winner
        winner = 0 if row["prediction"] == 1 else 1  # 0 means team_1 won, 1 means team_2 won

        # Store the matchup along with the winner index
        matchup = [row['team_1'], row['team_2'], winner]
        bracket_structure[round_index].append(matchup)

    # Extract the final winner from the last matchup
    final_game = results[results['current_round'] == 2].iloc[0]  
    final_matchup = bracket_structure[-2][0]  # Get last game stored in bracket
    final_winner = final_matchup[0] if final_game['prediction'] == 1 else final_matchup[1]

    # Add the final winner as the last round
    bracket_structure[-1].append([final_winner, "", 0])  # No opponent, default winner

    print(bracket_structure)  # Debugging output

    return bracket_structure


# function to format for jquery
def format_bracket_for_jquery_bracket(results):
    """
    Converts the bracket structure into jQuery Bracket's expected format.
    Splits into top and bottom halves, Final Four, and the Championship.
    """
    num_matchups = len(results[0])  # Total matchups in the first round (32)
    mid_point = num_matchups // 2   # Split point (16 per side)

    # ✅ Separate Top and Bottom Brackets
    top_teams = [[match[0], match[1]] for match in results[0][:mid_point]]
    bottom_teams = [[match[0], match[1]] for match in results[0][mid_point:]]

    # ✅ Store results separately
    top_results = []
    bottom_results = []

    ### **🔷 Process the Top Half**
    advancing_top_teams = top_teams[:]  # Copy initial teams for tracking progression

    for round_idx, round_matches in enumerate(results[:-2]):  # Exclude last 2 rounds
        top_round = []
        next_advancing_top = []  # Track advancing teams for next round

        for match in round_matches:
            if len(match) == 3:  # Ensure match contains winner flag
                winner = match[2]
                top_round.append([1, 0] if winner == 0 else [0, 1])
                next_advancing_top.append(match[winner])  # Store advancing team

        advancing_top_teams = [next_advancing_top[i:i+2] for i in range(0, len(next_advancing_top), 2)]
        top_results.append(top_round)

    ### **🔷 Process the Bottom Half (Fixed)**
    advancing_bottom_teams = bottom_teams[:]  # Copy initial teams for tracking progression

    for round_idx, round_matches in enumerate(results[:-2]):  # Exclude last 2 rounds
        bottom_round = []
        next_advancing_bottom = []  # Track advancing teams for next round

        for match in round_matches:
            if len(match) == 3:  # Ensure match contains winner flag
                team_1, team_2, winner = match
                if [team_1, team_2] in advancing_bottom_teams or [team_2, team_1] in advancing_bottom_teams:
                    bottom_round.append([1, 0] if winner == 0 else [0, 1])
                    next_advancing_bottom.append(match[winner])  # Store advancing team

        advancing_bottom_teams = [next_advancing_bottom[i:i+2] for i in range(0, len(next_advancing_bottom), 2)]
        bottom_results.append(bottom_round)

    # ✅ Ensure Final Four round exist
    final_four = results[-2] if len(results) > 1 else []  # Second to last round

    # ✅ Handle Final Four
    final_four_results = []
    final_four_teams = []

    if len(final_four) > 0:
        final_four_results.append([1, 0] if final_four[0][2] == 0 else [0, 1])
        final_four_teams.append([final_four[0][0], final_four[0][1]])

    if len(final_four) > 1:
        final_four_results.append([1, 0] if final_four[1][2] == 0 else [0, 1])
        final_four_teams.append([final_four[1][0], final_four[1][1]])

    # ✅ Handle Championship
    final_result = []

    # ✅ Print for Debugging
    print("Top Bracket Teams:", top_teams)
    print("Bottom Bracket Teams:", bottom_teams)
    print("Top Results:", top_results)
    print("Bottom Results:", bottom_results)  # ✅ Should now advance properly
    print("Final Four Teams:", final_four_teams)

    return {
        "top_bracket": {"teams": top_teams, "results": [top_results]},
        "bottom_bracket": {"teams": bottom_teams, "results": [bottom_results]},
        "final_four_bracket": {"teams": final_four_teams, "results": [[final_four_results]]}
    }




# function to get correct suffix of percentile
def ordinal(n):
    if 10 <= n % 100 <= 20:  # Special case for 11th-19th
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

        # Handle scoring if results exist
        try:
            score = simulator.score_bracket()  # Try to score the bracket
        except Exception as e:
            print(f"Scoring Error: {e}")
            score = None  # Set score to None if scoring fails

        # Handle percentile calculation
        percentile = None
        if score is not None:
            year_sim_scores = odds_sim_scores[odds_sim_scores["year"] == year]["score"]

            # Check if there are valid historical scores
            if len(year_sim_scores) > 0:
                percentile = ordinal(int(percentileofscore(year_sim_scores, score, kind='rank')))


        # Store results in session
        session['selected_params'] = input_data
        session['simulation_results'] = predictions
        session['score'] = score
        session['percentile'] = percentile

        # Redirect to results page after processing
        return jsonify({'redirect_url': url_for('results')})
    
    except Exception as e:
        print("Error", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route for the results page
@app.route('/results')
def results():
    selected_params = session.get('selected_params')
    raw_results = session.get('simulation_results')
    score = session.get('score')
    percentile = session.get('percentile')

    # Format the bracket for jQuery Bracket
    formatted_bracket = format_bracket_for_jquery_bracket(raw_results)

    return render_template('results.html', selected_params=selected_params, 
                           results=raw_results, formatted_bracket=formatted_bracket,
                           score=score, percentile=percentile)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
