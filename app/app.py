from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
matplotlib.use('Agg')
from io import BytesIO
import base64
from models.bracket_model import BracketSimulator

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'BBallSim'

# Read in data
data = pd.read_parquet("data/all_matchup_stats.parquet")
odds_sim_scores = pd.read_parquet("data/odds_sim_scores.parquet")
upset_counts = pd.read_csv("data/upset_counts.csv")
public_scores = pd.read_csv("data/public_bracket_scores.csv")

# produce a bar chart of confidence per round
def generate_confidence_bar_plot(bracket):
    rounds = []
    avg_confidences = []
    
    # loop through rounds
    for i, round_matchups in enumerate(bracket, start=1):
        if i == 7:
            continue  # skip round 7 (not a real round, just the champion)
        total_conf = 0.0
        count = 0

        # loop through matchups
        for matchup in round_matchups:
            conf_str = matchup[7]
            if conf_str and conf_str.endswith("%"):
                try:
                    conf_val = float(conf_str.rstrip("%"))
                    total_conf += conf_val
                    count += 1
                except ValueError:
                    continue

        # compute average confidence for the round
        avg_conf = total_conf / count if count > 0 else 0
        rounds.append(f"Round {i}")
        avg_confidences.append(avg_conf)
    
    # create a bar plot for average confidence per round
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_title("Average Confidence per Round", fontsize=12, pad=10)
    bars = ax.bar(range(len(rounds)), avg_confidences, color="#1F4DA3")
    
    # Remove x and y ticks and spines.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Annotate each bar with the round number inside it.
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"Round\n{rounds[idx][-1]}",
            ha='center',
            va='center',
            color='white',
            fontsize=9
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,  # adjust the offset as needed
            f"{avg_confidences[idx]:.1f}%",
            ha='center',
            va='bottom',
            color='#1F4DA3',
            fontsize=9,
            fontweight='bold'
        )
    
    # Remove axis labels and frame.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)
    
    # Convert the plot to a base64-encoded PNG image.
    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    
    return plot_url

# retrieve most and least confident predictions
def get_confidence_stuff(bracket):
    games = []
    upsets = []
    round_no = 0

    for round_matchups in bracket:
        round_no += 1
        for matchup in round_matchups:
            conf_str = matchup[7]
            if conf_str and conf_str.endswith("%"):
                try:
                    conf_val = float(conf_str.rstrip("%"))
                except ValueError:
                    continue

                # determine winner and loser based on the winner index
                if matchup[4] == 0:
                    winner_team = matchup[0]
                    winner_seed = round(matchup[5])
                    loser_team = matchup[1]
                    loser_seed = round(matchup[6])
                else:
                    winner_team = matchup[1]
                    winner_seed = round(matchup[6])
                    loser_team = matchup[0]
                    loser_seed = round(matchup[5])

                game_text = f"{winner_seed}. {winner_team} beats {loser_seed}. {loser_team} (round: {round_no}): {conf_val:.1f}%"
                games.append((conf_val, game_text))

                # identify if it was an upset and store as a tuple as well
                ## including 'unnaturally' picked upsets (forced by user)
                if matchup[5] < matchup[6] and matchup[4] == 1:
                    if matchup[3] > 0.5: # genuine prediction from model
                        upsets.append((conf_val, game_text))
                    else: # forced by user
                        game_text = f"{winner_seed}. {winner_team} beats {loser_seed}. {loser_team} (round: {round_no}): {100-conf_val:.1f}%"
                        upsets.append((100 - conf_val, game_text))
                elif matchup[5] > matchup[6] and matchup[4] == 0:
                    if matchup[2] > 0.5: # genuine prediction from model
                        upsets.append((conf_val, game_text))
                    else: # forced by user
                        game_text = f"{winner_seed}. {winner_team} beats {loser_seed}. {loser_team} (round: {round_no}): {100-conf_val:.1f}%"
                        upsets.append((100 - conf_val, game_text))
    
    # Get top 3 most confident games (highest confidence first)
    games.sort(key=lambda x: x[0], reverse=True)
    most_confident_games = [game[1] for game in games[:3]]
    
    # Get top 3 least confident games (lowest confidence first)
    games.sort(key=lambda x: x[0])
    least_confident_games = [game[1] for game in games[:3]]
    
    # Get top 3 most confident upsets (sorted descending)
    upsets.sort(key=lambda x: x[0], reverse=True)
    most_confident_upsets = [upset[1] for upset in upsets[:3]]
    
    return most_confident_games, least_confident_games, most_confident_upsets


# generate a color based on confidence level
def get_confidence_color(confidence_str):
    try:
        conf_float = float(confidence_str.strip('%')) / 100.0
    except Exception:
        conf_float = 0.5  # default value if conversion fails
    hue = conf_float * 120
    lightness = 50 + (1 - abs(conf_float - 0.5) * 2) * 30
    return f"hsla({hue}, 100%, {lightness}%, 0.5)"

# function to calculate the confidence level of a bracket
def calculate_average_confidence(bracket):
    total_conf = 0.0
    count = 0

    # iterate over each matchup within each round
    for round_matchups in bracket:
        for matchup in round_matchups:
            conf_str = matchup[7]

            # skip if confidence is empty (like in the championship round)
            if conf_str and conf_str.endswith("%"):
                try:
                    conf_val = float(conf_str.rstrip("%"))
                    total_conf += conf_val
                    count += 1
                except Exception:
                    continue

    if count > 0:
        avg_conf = total_conf / count
        return avg_conf
    
    return None

# function to plot simulated scores
def generate_score_distribution(user_score, sim_scores, public_user_avg=None):

    # set up plot
    bg_color = "#f0f0f0"
    fig = plt.figure(figsize=(7, 4), facecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_color)

    # create the KDE plot
    sns.kdeplot(sim_scores, color="#1F4DA3", fill=True, linewidth=2, ax=ax)
    ax_line = sns.kdeplot(sim_scores, color="#1F4DA3", fill=False, linewidth=2, ax=ax)
    x_kde, y_kde = ax_line.get_lines()[0].get_data()

    # calculate percentiles
    percentiles = [25, 50, 75, 90]
    percentile_values = np.percentile(sim_scores, percentiles)

    # place percentage labels at the curve height
    for perc, x_val in zip(percentiles, percentile_values):
        y_val = np.interp(x_val, x_kde, y_kde)
        ax.text(x_val, y_val + 0.0001, f"{perc}%", ha='left', va='center', fontsize=10, color='gray')

    # mark user point
    if user_score is not None:
        # Draw the vertical dashed line for user score (stops at 80% of the axes height)
        ax.axvline(user_score, color='red', linestyle='dashed', linewidth=2, 
                   label='Your Score', ymin=0, ymax=0.8)
        y_user = np.interp(user_score, x_kde, y_kde)
        ax.text(user_score + 10, y_user + 0.0001, "You", ha='left', va='center',
                fontsize=10, color='red')

    # mark public user point
    if public_user_avg is not None:
        ax.axvline(public_user_avg, color='blue', linestyle='dotted', linewidth=2,
                   label='Avg User Score', ymin=0, ymax=0.8)
        y_avg = np.interp(public_user_avg, x_kde, y_kde)
        ax.text(public_user_avg + 10, y_avg + 0.0001, "User Avg", ha='left', va='center',
                fontsize=10, color='blue')

    # clean up the y-axis and add labels
    ax.set_yticks([])
    ax.set_ylabel("")
    max_y = max(y_kde)
    ax.set_ylim(0, max_y * 1.1)
    ax.set_xlabel("Score")
    ax.set_title("Compared to Simulated Brackets")

    # note text with additional info
    note = "*10,000 brackets simulated with historical seed-based odds"
    wrapped_note = textwrap.fill(note, width=35)
    ax.text(0.98, 0.98, wrapped_note, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="gray", wrap=True)

    # convert for html rendering
    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    plt.close()
    
    return plot_url

# function to format the bracket as nested list (primarily going to be used for analysis)
def format_bracket(results):
    num_rounds = 7  # rounds from 64 teams to a single champion
    bracket_structure = [[] for _ in range(num_rounds)]

    for _, row in results.iterrows():
        round_index = int(round(np.log2(64 / row['current_round'])))  # convert round number to index

        # determine the winner index (0 for team_1, 1 for team_2)
        winner = 0 if row["prediction"] == 1 else 1

        # extract win probabilities and compute confidence score
        p1 = row['win probability']
        p2 = 1 - p1  
        confidence = f'{round(max(p1, p2), 2)*100}%'

        # extract seed numbers
        seed_1 = row['seed_1']
        seed_2 = row['seed_2']

        # create matchup tuple with all necessary fields
        matchup = [row['team_1'], row['team_2'], p1, p2, winner, seed_1, seed_2, confidence]
        bracket_structure[round_index].append(matchup)

    # handle final championship round
    final_game = results[results['current_round'] == 2].iloc[0]
    final_matchup = bracket_structure[-2][0]  # Last game before final
    final_winner = final_matchup[0] if final_game['prediction'] == 1 else final_matchup[1]

    # add final winner to championship round
    bracket_structure[-1].append([final_winner, "", "", "", 0, "", "", ""])

    return bracket_structure


# function to convert bracket to a dictionary (for rendering the bracket template)
def convert_bracket_format(simulation_output):
    formatted_bracket = {"rounds": []}
    num_rounds = len(simulation_output)

    for round_index, matchups in enumerate(simulation_output):
        round_data = {"round": round_index + 1, "left": [], "right": []}

        if round_index == 5 and len(matchups) == 1:  # handle final four
            team1, team2, p1, p2, winner_index, seed_1, seed_2, confidence = matchups[0]
            round_data["left"].append({"team1": team1, "team2": "", "p1": p1, "p2": p2, "winner": ""})
            round_data["right"].append({"team1": "", "team2": team2, "p1": p1, "p2": p2, "winner": ""})
        else:
            mid_point = len(matchups) // 2
            for i, match in enumerate(matchups):
                # unpack only what we need for visual bracket
                team1, team2, p1, p2, winner_index, *_ = match  
                winner = team1 if winner_index == 0 else team2

                const_obj = {
                    "team1": team1,
                    "team2": team2,
                    "p1": p1,
                    "p2": p2,
                    "winner": winner
                }

                if i < mid_point:
                    round_data["left"].append(const_obj)
                else:
                    round_data["right"].append(const_obj)
        
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
    
    return jsonify(["None"] + unique_teams)


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

    # format the bracket for rendering
    formatted_bracket = convert_bracket_format(raw_results)

    return render_template('results.html', selected_params=selected_params, 
                           results=raw_results, formatted_bracket=formatted_bracket,
                           score=score, percentile=percentile)

# route for analytics page
@app.route('/analytics')
def analytics():

    # retrieve stuff from session
    selected_params = session.get('selected_params', {})
    year = selected_params.get('year', 2024)
    user_score = session.get('score', None)
    bracket =  session.get('simulation_results', [])
    
    # get score info
    if user_score:
        sim_scores = odds_sim_scores[odds_sim_scores["year"] == year]["score"].tolist()
        public_user_avg = public_scores.loc[public_scores.year == year, "avg_user_score"].iloc[0]
        if np.isnan(public_user_avg):
            public_user_avg = None
        else:
            public_user_avg = round(public_user_avg * 10)

        seed_based_score = round((public_scores.loc[public_scores.year == year, "seed_based_score"].iloc[0])*10)
        points_possible = 1920
        score_hist_url = generate_score_distribution(user_score, sim_scores, public_user_avg)
    else:
        score_hist_url = None

    # get confidence info
    confidence_level = calculate_average_confidence(bracket)
    if confidence_level is not None:
        confidence_level = f"{confidence_level:.1f}%"
        confidence_color = get_confidence_color(confidence_level)
    else:
        confidence_level = "N/A"
        confidence_color = "#ffffff"
    most_confident, least_confident, confident_upsets = get_confidence_stuff(bracket)
    confidence_bar_url = generate_confidence_bar_plot(bracket)

    return render_template('analytics.html', 
                            distribution=sim_scores,
                            user_score=user_score,
                            bracket=bracket,
                            public_user_avg=public_user_avg,
                            seed_based_score=seed_based_score,
                            points_possible=points_possible,
                            score_histogram=score_hist_url,
                            confidence_level=confidence_level,
                            confidence_color=confidence_color,
                            most_confident_games=most_confident,
                            least_confident_games=least_confident,
                            confident_upsets=confident_upsets,
                            confidence_bar_url=confidence_bar_url
                        )


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
