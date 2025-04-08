from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("STARTING")

# PULL IN TEAM STATS
# headers for the DataFrame taken from the website (the csv file does not download headers so i add them manually)
headers = ['', 'TEAM', 'ADJ OE', 'ADJ DE', 'BARTHAG', 'RECORD', 'WINS', 'GAMES', 'EFG', 'EFG D.', 
           'FT RATE', 'FT RATE D', 'TOV%', 'TOV% D', 'O REB%', 'OP OREB%', 'RAW T', 
           '2P %', '2P % D.', '3P %', '3P % D.', 'BLK %', 'BLKED %', 'AST %', 'OP AST %', 
           '3P RATE', '3P RATE D', 'ADJ. T', 'AVG HGT.', 'EFF. HGT.', 'EXP.', 'YEAR', 
           'PAKE', 'PASE', 'TALENT', '', 'FT%', 'OP. FT%', 'PPP OFF.', 'PPP DEF.', 
           'ELITE SOS', 'TEAM']

# loop through all the years, pull the data, and add them together
dfs = []
wabs = []
years = range(2008, 2026)
years = [year for year in years if year != 2020]
for year in years:

    # url of webpage containing data (type=R FOR REGULAR SEASON STATS ONLY, no data leakage)
    url = f'https://barttorvik.com/team-tables_each.php?tvalue=All&year={year}&sort=&t2value=None&oppType=All&conlimit=All&top=0&quad=4&mingames=0&toprk=0&venue=All&type=R&yax=3'

    # load the webpage using selenium driver and chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enable headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU usage 
    chrome_options.add_argument("--no-sandbox")  # Recommended for Docker 
    chrome_options.add_argument("--disable-dev-shm-usage")  # Reduce resource usage
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # wait for the table to appear on the page and get the source
    team_table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
    page_source = driver.page_source

    # parse the html content with BeautifulSoup and find the table
    soup = BeautifulSoup(page_source, 'html.parser')
    team_table = soup.find('table')

    # check to make sure the table was found
    if team_table:

        # extract data from the table
        data = []
        rows = team_table.find_all('tr')
            
        # extract the rows
        for row in rows[1:]:
            cells = row.find_all('td')
            row_data = [cell.get_text() for cell in cells]
            data.append(row_data)
            
        # convert data to a pd df and add to the list of dfs
        df = pd.DataFrame(data, columns=headers)
        dfs.append(df)

    else:
        print('Team table not found on the webpage.')
            
    # close the webdriver
    driver.quit()

    ## read in another table for WAB stat (i have the csv urls so no scraping needed)
    url = f'http://barttorvik.com/{year}_team_results.csv'
    extras = pd.read_csv(url)
    extras = extras.shift(axis=1)
    extras.drop(columns=["rank"], inplace=True)  
    wab_stats = extras[["team", "WAB"]]
    wab_stats = wab_stats.rename(columns={"team" : "TEAM"}).copy()
    wab_stats["YEAR"] = year
    wabs.append(wab_stats)

    print(f'Got all stats from {year}')

# concatenate the main stats DFs into a single DF
combined_stats = pd.concat(dfs, ignore_index=True)
combined_stats = combined_stats.T.drop_duplicates().T  # this should get rid of the second TEAM column

# concatenate the wab stats dfs
combined_wab = pd.concat(wabs, ignore_index=True)
# some more processing on the main stats
combined_stats = combined_stats.drop(columns=['']) # drop blank column
combined_stats = combined_stats.replace(',', '', regex=True) # get rid of commas from cells
numeric_columns = combined_stats.columns.difference(['TEAM', 'RECORD'])
combined_stats.replace(r'^\s*$', 0, regex=True, inplace=True) # found blank cells under TALENT, so just make it zero
combined_stats[numeric_columns] = combined_stats[numeric_columns].apply(pd.to_numeric) # convert everything to numeric

# merge the wab with the main stats
team_stats = pd.merge(combined_stats, combined_wab, on=["TEAM", "YEAR"], how="inner")

# Calculate win percentage
team_stats['win_percent'] = team_stats['WINS'] / team_stats['GAMES']


# make new variable combining offensive and defensive efficiency
team_stats['adjem'] = team_stats['ADJ OE'] - team_stats['ADJ DE']

print("Pulled in all stats")

## PULL IN REGULAR SEASON MATCHUPS

headers = ["DATE", "TYPE", "TEAM", "CONF.", "OPP.", "VENUE", "RESULT", "ADJ. O", 
           "ADJ. D", "T", "OEFF", "OEFG%", "OTO%", "OREB%", "OFTR", "DEFF", "DEFG%", "DTO%", 
           "DREB%", "DFTR"]

dfs = []
years = range(2008, 2026)
years = [year for year in years if year != 2020]

for year in years:
    url = f"http://barttorvik.com/getgamestats.php?year={year}&csv=1"
    year_matchups = pd.read_csv(url)
    year_matchups = year_matchups.iloc[:, :20]
    year_matchups.columns = headers
    dfs.append(year_matchups)

# concatenate the matchups into a single DF
combined_matchups = pd.concat(dfs, ignore_index=True)

# drop tourney games because we already have those in another dataframe and order is important for those
rs_matchups = combined_matchups[combined_matchups['TYPE'] != 3]

# Get only the variables I need. change some as needed
rs_matchups = rs_matchups[["DATE", "TEAM", "OPP.", "RESULT"]]
rs_matchups['RESULT'] = rs_matchups['RESULT'].str.contains('W') * 1
rs_matchups['DATE'] = rs_matchups['DATE'].apply(lambda x: int(x.split('/')[-1]) + 2001 
                                                            if int(x.split('/')[-3]) > 8 
                                                            else int(x.split('/')[-1]) + 2000)
rs_matchups = rs_matchups.rename(columns={
    "DATE" : "year",
    "TEAM" : "team_1",
    "OPP." : "team_2",
    "RESULT": "winner"
})

print("Pulled in RS matchups")

# READ IN TOURNAMENT MATCHUP DATA
tournament_matchups = pd.read_csv("data/tournament_matchups.csv")

# MERGE TOURNAMENT MATCHUPS WITH STATS
tourney_team_stats = pd.merge(tournament_matchups, team_stats, on=["TEAM", "YEAR"], how='left')

# use the same processing steps as before (combine rows, rename + drop columns)
tourney_matchup_stats = pd.DataFrame(columns=['year', 'team_1', 'seed_1', 'round_1', 'current_round', 'score_1',
                                     'team_2', 'seed_2', 'round_2', 'score_2'])
matchup_info_list = []

# iterate through data frame and jump 2 each iteration
for i in range(0, len(tourney_team_stats), 2):
    team1_info = tourney_team_stats.iloc[i]
    team2_info = tourney_team_stats.iloc[i+1]
    matchup_info = {
            'year': team1_info['YEAR'],
            'team_1': team1_info['TEAM'],
            'seed_1': team1_info['SEED'],
            'round_1': team1_info['ROUND'],
            'score_1' : team1_info['SCORE'],
            'score_2' : team2_info['SCORE'],
            'current_round': team1_info['CURRENT ROUND'],
            'team_2': team2_info['TEAM'],
            'seed_2': team2_info['SEED'],
            'round_2': team2_info['ROUND'],
            'badj_em_1': team1_info['adjem'],
            'badj_o_1': team1_info['ADJ OE'],
            'badj_d_1': team1_info['ADJ DE'],
            'wab_1': team1_info['WAB'],
            'barthag_1': team1_info['BARTHAG'],
            'efg_1': team1_info['EFG'],
            'efg_d_1': team1_info['EFG D.'],
            'ft_rate_1': team1_info['FT RATE'],
            'ft_rate_d_1': team1_info['FT RATE D'],
            'tov_percent_1': team1_info['TOV%'],
            'tov_percent_d_1': team1_info['TOV% D'],
            'o_reb_percent_1': team1_info['O REB%'],
            'op_o_reb_percent_1': team1_info['OP OREB%'],
            'blk_percent_1': team1_info['BLK %'],
            'ast_percent_1': team1_info['AST %'],
            'adj_tempo_1': team1_info['ADJ. T'],
            '3p_percent_1': team1_info['3P %'],
            '3p_rate_1': team1_info['3P RATE'],
            '2p_percent_1': team1_info['2P %'],
            '3p_percent_d_1': team1_info['3P % D.'],
            '2p_percent_d_1': team1_info['2P % D.'],
            'pppo_1': team1_info['PPP OFF.'],
            'pppd_1': team1_info['PPP DEF.'],
            'exp_1': team1_info['EXP.'],
            'eff_hgt_1': team1_info['EFF. HGT.'],
            'talent_1' : team1_info['TALENT'],
            'elite_sos_1': team1_info['ELITE SOS'],
            'win_percent_1': team1_info['win_percent'],
            'badj_em_2': team2_info['adjem'],
            'badj_o_2': team2_info['ADJ OE'],
            'badj_d_2': team2_info['ADJ DE'],
            'wab_2': team2_info['WAB'],
            'barthag_2': team2_info['BARTHAG'],
            'efg_2': team2_info['EFG'],
            'efg_d_2': team2_info['EFG D.'],
            'ft_rate_2': team2_info['FT RATE'],
            'ft_rate_d_2': team2_info['FT RATE D'],
            'tov_percent_2': team2_info['TOV%'],
            'tov_percent_d_2': team2_info['TOV% D'],
            'o_reb_percent_2': team2_info['O REB%'],
            'op_o_reb_percent_2': team2_info['OP OREB%'],
            'blk_percent_2': team2_info['BLK %'],
            'ast_percent_2': team2_info['AST %'],
            'adj_tempo_2': team2_info['ADJ. T'],
            '3p_percent_2': team2_info['3P %'],
            '3p_rate_2': team2_info['3P RATE'],
            '2p_percent_2': team2_info['2P %'],
            '3p_percent_d_2': team2_info['3P % D.'],
            '2p_percent_d_2': team2_info['2P % D.'],
            'pppo_2': team2_info['PPP OFF.'],
            'pppd_2': team2_info['PPP DEF.'],
            'exp_2': team2_info['EXP.'],
            'eff_hgt_2': team2_info['EFF. HGT.'],
            'talent_2' : team2_info['TALENT'],
            'elite_sos_2': team2_info['ELITE SOS'],
            'win_percent_2': team2_info['win_percent']
            }
    
    matchup_info_list.append(matchup_info)


tourney_matchup_stats = pd.concat([tourney_matchup_stats, pd.DataFrame(matchup_info_list)])
tourney_matchup_stats["winner"] = 1 * (tourney_matchup_stats["score_1"] > tourney_matchup_stats["score_2"])
tourney_matchup_stats.drop(columns=["score_1", "score_2"], inplace=True)

print("Tourney matchups + stats")

# MERGE REGULAR SEASON MATCHUPS WITH STATS

# change names in team_stats to match format of names in rs_matchups
team_stats = team_stats.rename(columns={
    'YEAR': 'year',
    'TEAM': 'team',
    'adjem' : 'badj_em',
    'ADJ OE': 'badj_o',
    'ADJ DE': 'badj_d',
    'WAB': 'wab',
    'BARTHAG': 'barthag',
    'EFG': 'efg',
    'EFG D.': 'efg_d',
    'FT RATE': 'ft_rate',
    'FT RATE D': 'ft_rate_d',
    'TOV%': 'tov_percent',
    'TOV% D': 'tov_percent_d',
    'O REB%': 'o_reb_percent',
    'OP OREB%': 'op_o_reb_percent',
    'BLK %': 'blk_percent',
    'AST %': 'ast_percent',
    'PPP OFF.': 'pppo',
    'PPP DEF.': 'pppd',
    'ADJ. T': 'adj_tempo',
    '3P %': '3p_percent',
    '3P RATE': '3p_rate',
    '2P %': '2p_percent',
    '3P % D.': '3p_percent_d',
    '2P % D.': '2p_percent_d',
    'EXP.': 'exp',
    'EFF. HGT.': 'eff_hgt',
    'TALENT': 'talent',
    'ELITE SOS': 'elite_sos',
    'win_percent': 'win_percent'
})

rs_matchups_with_stats = pd.merge(rs_matchups, team_stats, left_on=['team_1', 'year'], 
                                  right_on=['team', 'year'], how='left')
rs_matchups_with_stats.rename(columns=lambda x: x + '_1' if x not in ['team_1', 'team_2', 'year', 'winner'] 
                              else x, inplace=True)

print("Add team 1 stats to RS matchups")

rs_matchups_with_stats = pd.merge(rs_matchups_with_stats, team_stats, left_on=['team_2', 'year'], 
                                  right_on=['team', 'year'], how='left')
rs_matchups_with_stats.rename(columns=lambda x: x + '_2' if x not in ['team_1', 'team_2', 'year', 'winner'] 
                              and not x.endswith('_1') 
                              else x, inplace=True)

print("Add team 2 stats to RS matchups")

filtered_columns = [col for col in rs_matchups_with_stats.columns if col.islower()]
rs_matchups_with_stats = rs_matchups_with_stats[filtered_columns]

rs_matchups_with_stats.columns = ['year' if 'year' in col else col for col in rs_matchups_with_stats.columns]
rs_matchups_with_stats = rs_matchups_with_stats.loc[:,~rs_matchups_with_stats.columns.duplicated()].copy()

print("Remove unwanted columns from RS matchups")

# create a unique matchup id so i can drop the rows that are the same as others but with team_1 and team_2 flipped
# (make sure to not drop matchups that actually happened twice. duplicates are always next to each other)
rs_matchups_with_stats["matchup_id"] = rs_matchups_with_stats.apply(lambda row: str(row['year']) + 
                                                                    '-' + '-'.join(sorted([row['team_1'], row['team_2']])), 
                                                                    axis=1)
duplicate_mask = rs_matchups_with_stats["matchup_id"].eq(rs_matchups_with_stats["matchup_id"].shift())
non_consecutive_duplicates_mask = ~duplicate_mask
rs_matchups_with_stats = rs_matchups_with_stats[non_consecutive_duplicates_mask]
print("Add matchup id to RS data and drop duplicate games")

# add a column to each data set (tourney and regular season) to differentiate them post-merge
tourney_matchup_stats["type"] = "T" # tournament
rs_matchups_with_stats["type"] = "RS" # regular season

all_matchup_stats = pd.concat([tourney_matchup_stats, rs_matchups_with_stats], ignore_index=True)

print("add rs_matchups under tourney matchups")



# get the stat differences same as before
stat_variables = [
    'badj_em', 'badj_o', 'badj_d', 'wab', 'barthag', 'efg', 'efg_d', 'ft_rate', 'ft_rate_d', 'tov_percent',
    'tov_percent_d', 'adj_tempo', '3p_percent', '3p_rate', '3p_percent_d', '2p_percent', '2p_percent_d', 
    'o_reb_percent', 'op_o_reb_percent', 'blk_percent', 'ast_percent', 'pppo', 'pppd',
    'exp', 'eff_hgt', 'talent', 'elite_sos', 'win_percent'
]

# Create an empty DataFrame to store the stat differences
stat_diff_df = pd.DataFrame()

# Calculate stat differences for each variable
for variable in stat_variables:
    # Calculate stat difference
    stat_diff = all_matchup_stats[f'{variable}_1'] - all_matchup_stats[f'{variable}_2']
    # Add stat difference to the DataFrame
    stat_diff_df[f'{variable}_diff'] = stat_diff

# Concatenate the stat difference DataFrame with the existing DataFrame
all_matchup_stats = pd.concat([all_matchup_stats, stat_diff_df], axis=1)

# Add close call flags (only used for simulation strategy)
all_matchup_stats['close_call_1'] = False
all_matchup_stats['close_call_2'] = False

# EXPORT DF
directory = "data"
all_matchup_stats.to_parquet(directory + "/all_matchup_stats.parquet", index=False)

print("Finish")