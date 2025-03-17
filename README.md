# March-Madness-Simulator

A Flask-based web application that allows users to generate an optimized March Madness bracket based on their own preferences and analyze its performance. 

# Features

1. Bracket Simulation
    - Uses a predictive model to simulate matchups
    - Allows users to customize their bracket preferences:
        - Pick a specific winner
        - Choose a playstyle
        - Adjust risk level
    - Backtest preferences on tournament data back to 2008

2. Bracket Visualization
    - Displays the full 64-team bracket
    - Teams are colored on a red-to-green scale based on confidence of winning

3. Scoring System and Analytics
    - **User Score:** based on ESPN scoring system
        - Points double each round (10 for first round, 20 for second, etc.)
    - **Scoring and Analytics:**
        - Against historical user brackets
            - Gathered from 2010's data on marchmadness.com
            - Very limited access otherwise
        - Against 'chalk' brackets (picking the best seed)
        - Against 10,000 simulated brackets
            - I simulated 10k brackets based on each seeds historical success rate
            - Compares the user score to the distribution of simulated brackets
        - Confidence evaluation
            - Most and least confident games, most confident upsets
            - Confidence per round
            - Based on win probabilities produced by the model

# How It Works

1. Backend (Flask)
    - `app/app.py` handles routing and data processing
    - Uses data from `data/all_matchup_stats.parquet`
        - Scraped and aggregated in `app/utils/data_pipeline.py` (most comes from barttorvik.com)
    - Simulates the bracket in `app/models/bracket_model.py`
    - Stores and processes results in session data

2. The Simulator
    - Based on historical matchup data (all CBB games since 2008, regular-season included) and statistical modeling
        - Analyzes differences in important statistics like offensive/defensive efficiency, tempo, etc.
        - A single matchup predictor with XGBoost at its core
    - Passes winners on to the next round to play the other corresponding winner
    - Nuanced predictions:
        - Applies real bracket-making strategies, like limiting the success of teams that are predicted to have close matchups early in the tournament 
        - Adjusts feature weights, sample weights, and prediction thresholds based on user preferences
        - Weights tournament games 5x higher than regular season games in training
    - Evaluated and optimized with a customized hybrid metric
        - 0.7 * weighted_accuracy + 0.3 * bracket_score
        - Bracket score is too volatile to be the lone metric with points doubling each round. The championship game is worth 32 first round games.
        - The *weighted* part of accuracy puts more emphasis on getting close games right
    - Normally distributed randomness (std = 0.025, hard capped at 0.1) added to win probabilities to simulate real-world uncertainty (madness, if you will)... plus it's no fun if everyone gets the same bracket

3. Frontend (HTML, CSS, JavaScript)
    - `app/templates/home.html` holds the home page and offers/collects user preferences
    - `app/templates/results.html` displays the visualized bracket and user score (if applicable)
    - `app/templates/analytics.html` provides additional insights and comparisons
    - Uses JavaScript to dynamically update the matchups/insights based on year/preferences and the intentional randomness from the output

# Hosting

The simulator is hosted on Render at **[bracketsim.com](http://bracketsim.com)**.

I will be paying the service costs during the tournament, but the site will be down the rest of the year. You can still run it locally anytime by following the instructions below.

# Local Installation and Setup

1. Clone the Repository

```
$ git clone https://github.com/jacksonisidor/March-Madness-Simulator.git
$ cd march-madness-simulator
```

2. Install Dependencies

```
$ pip install -r requirements.txt
```

3. Run the App

```
$ python3 app/app.py
```

# Future Improvements

- User accounts to track multiple bracket attemps
- Real-time tournament updates with live scores
    - Live confidence updates
    - Currently, scores are only updated after tournament ends
- More advanced playstyle adjustments
    - A customizable slider instead of 3 options?
- Different type of model
    - Already have experimented with deep learning approaches
    - Particularly interested in a Siamese-esque architecture that compares each teams offense to the other teams defense, predicts a score for each, then compare... kind of like playing a real game
    - Pretty much at the maximum amount of data available unfortunately (~80,000 games)
- Get new *types* of features, like live tournament info
    - Location, weather type info might be valuable
    - Incorporate injuries
        - Difficult to quantify how losing a specific player will impact a specific team
        - Factor in percentage of total team minutes or points by current roster leading into the tournament
            - i.e. if Cooper Flagg is out, Duke loses ~30 mins/game, affects lineup strength and rotation.

# Contributors

- **Jackson Isidor** - Data Scientist and Developer
- **Alex Dekhtyar:** - Advisor

# License 

Available under MIT License. Open-source and always open to suggestions!