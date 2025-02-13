from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
from scipy.stats import percentileofscore
from models.bracket_model import BracketSimulator

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'BBallSim'

# Read in data
data = pd.read_parquet("data/all_matchup_stats.parquet")
odds_sim_scores = pd.read_parquet("data/odds_sim_scores.parquet")

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
        predictions = simulator.predicted_bracket
        score = simulator.score_bracket()

        year_scores = odds_sim_scores[odds_sim_scores["year"] == year]["score"]
        percentile = ordinal(int(percentileofscore(year_scores, score, kind='rank')))

        # Store results in session
        session['selected_params'] = input_data
        session['simulation_results'] = predictions[['team_1', 'team_2', 'winner', 'prediction']].to_dict(orient='records')
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
    results = session.get('simulation_results')
    score = session.get('score')
    percentile = session.get('percentile')
    return render_template('results.html', selected_params=selected_params, 
                           results=results, score=score, percentile=percentile)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
