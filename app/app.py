from flask import Flask, render_template, request, jsonify
import pandas as pd
from models.bracket_model import BracketSimulator

# Initialize the Flask app
app = Flask(__name__)

# Read in data
data = pd.read_parquet("data/all_matchup_stats.parquet")


# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for simulation
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
        simulated_bracket = simulator.sim_bracket()

        # return a placeholder response for now
        return jsonify({
            'status': 'success',
            'message': 'Simulation is running...',
            'year': year,
            'picked_winner': picked_winner,
            'playstyle': playstyle,
            'boldness': boldness,
            'results': simulated_bracket.head().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
           

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
