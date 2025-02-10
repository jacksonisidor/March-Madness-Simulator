from flask import Flask, render_template, request, jsonify
import pandas as pd
from models.bracket_model import BracketSimulator

# Initialize the Flask app
app = Flask(__name__)

# Read in data
data = pd.read_parquet("data/all_matchup_stats.parquet")

print('RETRIEVED GAME DATA')

# Route for the home page
@app.route('/')
def home():
    max_year = data["year"].max()
    return render_template('home.html', max_year=max_year)

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

        print("RETRIEVED INPUT DATA")

        # extract the selected parameters for the simulation (with default values)
        year = input_data.get('year', 2024)
        picked_winner = input_data.get('picked_winner')
        playstyle = input_data.get('playstyle', 'Balanced')
        boldness = input_data.get('boldness', 'Normal')

        print("EXTRACTED PARAMETERS")

        # initialize and run the simulator
        simulator = BracketSimulator(data, year, picked_winner, playstyle, boldness)
        simulated_bracket = simulator.sim_bracket()

        print("SIMULATED BRACKET")

        # return a placeholder response for now
        return jsonify({
            'status': 'success',
            'message': 'Simulation is running...',
            'year': year,
            'picked_winner': picked_winner,
            'playstyle': playstyle,
            'boldness': boldness,
            'results': simulated_bracket[['team_1', 'team_2', 'winner', 'prediction']].head().to_dict(orient='records')
        })
    except Exception as e:
        print("Error", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
           

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
