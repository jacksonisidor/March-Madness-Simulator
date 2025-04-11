from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl
import numpy as np
import duckdb
from datetime import timedelta
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import time
start_time = time.time()
import gc
from threading import Lock # for progress tracking
snapshot_printed_years = set()
eos_printed_years = set()
printed_lock = Lock()


'''
This code is a MESS. I will spend some time soon cleaning it up,
but honestly there is not a lot of repeated operations to move to functions and
there is just a lot that needs to be done. 
It does work as is, but I will improve readability soon.
'''

chrome_options = Options()
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU usage 
chrome_options.add_argument("--no-sandbox")  # Recommended for Docker 
chrome_options.add_argument("--disable-dev-shm-usage")  # Reduce resource usage

# for tracking progress
checkpoint_time = start_time
def log_checkpoint(message):
    global checkpoint_time
    now = time.time()
    print(f"{message} in {now - checkpoint_time:.2f}s")
    checkpoint_time = now

# for scraping urls
def load_and_scrape_table(url):
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup.find('table')
    finally:
        driver.quit()

print("STARTING")

## PULL IN REGULAR SEASON MATCHUPS

headers = ["DATE", "TYPE", "TEAM", "CONF.", "OPP.", "VENUE", "RESULT", "ADJ. O", 
           "ADJ. D", "T", "OEFF", "OEFG%", "OTO%", "OREB%", "OFTR", "DEFF", "DEFG%", "DTO%", 
           "DREB%", "DFTR"]

matchup_dfs = []
years = range(2008, 2026)
years = [year for year in years if year != 2020]
 
for year in years:

    driver = webdriver.Chrome(options=chrome_options)

    print(f"Working on {year} regular season matchups")
    url = f"http://barttorvik.com/getgamestats.php?year={year}&csv=1"
    year_matchups = pd.read_csv(url)
    year_matchups = year_matchups.iloc[:, :20]
    year_matchups.columns = headers
    matchup_dfs.append(year_matchups)

# concatenate the matchups into a single DF
combined_matchups = pd.concat(matchup_dfs, ignore_index=True)
del matchup_dfs
gc.collect()

# drop tourney games because we already have those in another dataframe and order is important for those
rs_matchups = combined_matchups[combined_matchups['TYPE'] != 3]
del combined_matchups
gc.collect()

# Get only the variables I need. change some as needed
rs_matchups = rs_matchups[["DATE", "TEAM", "OPP.", "RESULT"]]
rs_matchups['RESULT'] = rs_matchups['RESULT'].str.contains('W') * 1
rs_matchups['exact_date'] = pd.to_datetime(rs_matchups['DATE'], format='%m/%d/%y')
rs_matchups['DATE'] = rs_matchups['DATE'].apply(lambda x: int(x.split('/')[-1]) + 2001 
                                                            if int(x.split('/')[-3]) > 8 
                                                            else int(x.split('/')[-1]) + 2000)
rs_matchups = rs_matchups.rename(columns={
    "DATE" : "year",
    "TEAM" : "team_1",
    "OPP." : "team_2",
    "RESULT": "winner"
})

# filter out any games before 2008 (weird stuff with the website for early season games in 08)
rs_matchups = rs_matchups[rs_matchups['exact_date'] > pd.Timestamp('2008-01-01')]

log_checkpoint("Pulled in all RS matchups")

## PULL IN TEAM SNAPSHOTS

# there are A LOT of snapshots to take, so parallelization speeds it up a bit

def fetch_snapshot_data(row, headers, wab_headers):
    year = row['year']
    game_date = row['exact_date']

    with printed_lock:
        if year not in snapshot_printed_years:
            print(f"Working on {year} snapshots")
            snapshot_printed_years.add(year)

    begin_date = f"{year-1}1101"
    end_date = (game_date - timedelta(days=1)).strftime("%Y%m%d")
    begin_date_mom = (game_date - timedelta(days=30)).strftime("%Y%m%d")

    snapshot_url = f"https://barttorvik.com/team-tables_each.php?tvalue=All&year={year}&sort=&t2value=None&oppType=All&conlimit=All&begin={begin_date}&end={end_date}&top=0&quad=4&mingames=0&toprk=0&venue=All&type=R&yax=3"
    wab_url = f"https://barttorvik.com/?year={year}&sort=&hteam=&t2value=&conlimit=All&state=All&begin={begin_date}&end={end_date}&top=0&revquad=0&quad=5&venue=All&type=R&mingames=0#"
    wab_mom_url = f"https://barttorvik.com/?year={year}&sort=&hteam=&t2value=&conlimit=All&state=All&begin={begin_date_mom}&end={end_date}&top=0&revquad=0&quad=5&venue=All&type=R&mingames=0#"

    snapshot_df, wab_df, mom_df = None, None, None

    for attempt in range(3):
        try:
            # Snapshot Table
            table = load_and_scrape_table(snapshot_url)
            if table:
                rows = table.find_all('tr')[1:]
                data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
                snapshot_df = pd.DataFrame(data, columns=headers)
                snapshot_df['year'] = year
                snapshot_df['exact_date'] = game_date
            break
        except Exception as e:
            print(f"[Snapshot ERROR] {year} {game_date} attempt {attempt+1}")
            time.sleep(1)

    for attempt in range(3):
        try:
            # WAB Table
            table = load_and_scrape_table(wab_url)
            if table:
                rows = table.find_all('tr')[1:]
                data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
                wab_df = pd.DataFrame(data, columns=wab_headers)
                wab_df = wab_df[["TEAM", "WAB"]]
                wab_df["YEAR"] = year
                wab_df["exact_date"] = game_date
            break
        except Exception as e:
            print(f"[WAB ERROR] {game_date} attempt {attempt+1}")
            time.sleep(1)

    for attempt in range(3):
        try:
            # Momentum WAB Table
            table = load_and_scrape_table(wab_mom_url)
            if table:
                rows = table.find_all('tr')[1:]
                data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
                mom_df = pd.DataFrame(data, columns=wab_headers)
                mom_df = mom_df[["TEAM", "WAB"]].rename(columns={"WAB": "recent_wab"})
                mom_df["YEAR"] = year
                mom_df["exact_date"] = game_date
            break
        except Exception as e:
            print(f"[MOMENTUM WAB ERROR] {game_date} attempt {attempt+1}")
            time.sleep(1)

    return snapshot_df, wab_df, mom_df

# headers for the DataFrame taken from the website
headers = ['', 'TEAM', 'ADJ OE', 'ADJ DE', 'BARTHAG', 'RECORD', 'WINS', 'GAMES', 'EFG', 'EFG D.', 
           'FT RATE', 'FT RATE D', 'TOV%', 'TOV% D', 'O REB%', 'OP OREB%', 'RAW T', 
           '2P %', '2P % D.', '3P %', '3P % D.', 'BLK %', 'BLKED %', 'AST %', 'OP AST %', 
           '3P RATE', '3P RATE D', 'ADJ. T', 'AVG HGT.', 'EFF. HGT.', 'EXP.', 'YEAR', 
           'PAKE', 'PASE', 'TALENT', '', 'FT%', 'OP. FT%', 'PPP OFF.', 'PPP DEF.', 
           'ELITE SOS', 'TEAM']
# define the headers for the WAB table (even tho we only want wab)
wab_headers = ["Rk", "TEAM", "Conf", "G", "Rec", "AdjOE", "AdjDE", "Barthag", "EFG%", 
                       "EFGD%", "TOR", "TORD", "ORB", "DRB", "FTR", "FTRD", "2P%","2P%D",
                       "3P%","3P%D","3PR O","3PR D","Adj T.","WAB"]

# get unique (year, exact_date) combinations from rs_matchups.
unique_snapshots = rs_matchups[['year', 'exact_date']].drop_duplicates()

snapshot_dfs = []
matchup_wabs = []
matchup_wabs_mom = []

rows = unique_snapshots.to_dict("records")
max_threads = min(8, len(rows)) 

# do the threading
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(fetch_snapshot_data, row, headers, wab_headers) for row in rows]

    for future in as_completed(futures):
        snapshot_df, wab_df, mom_df = future.result()
        if snapshot_df is not None:
            snapshot_dfs.append(snapshot_df)
        if wab_df is not None:
            matchup_wabs.append(wab_df)
        if mom_df is not None:
            matchup_wabs_mom.append(mom_df)

log_checkpoint("Pulled in all matchup snapshots")

# combine all snapshots into a single DataFrame
team_stats_time_sensitive = pd.concat(snapshot_dfs, ignore_index=True)

# concatenate the wab stats dfs
combined_wab = pd.concat(matchup_wabs, ignore_index=True)

# concatenate the momentum wab stats dfs
combined_wab_mom = pd.concat(matchup_wabs_mom, ignore_index=True)

del snapshot_dfs, matchup_wabs, matchup_wabs_mom, futures
gc.collect()

# some more processing on the main stats
team_stats_time_sensitive = team_stats_time_sensitive.drop(columns=['']) # drop blank column
team_stats_time_sensitive = team_stats_time_sensitive.replace(',', '', regex=True) # get rid of commas from cells
numeric_columns = team_stats_time_sensitive.columns.difference(['TEAM', 'RECORD'])
team_stats_time_sensitive.replace(r'^\s*$', 0, regex=True, inplace=True) # found blank cells under TALENT, so just make it zero
team_stats_time_sensitive[numeric_columns] = team_stats_time_sensitive[numeric_columns].apply(pd.to_numeric) # convert everything to numeric

# remove any duplicated column labels (keep the first occurrence)
team_stats_time_sensitive = team_stats_time_sensitive.loc[:, ~team_stats_time_sensitive.columns.duplicated()]
combined_wab = combined_wab.loc[:, ~combined_wab.columns.duplicated()]
combined_wab_mom = combined_wab_mom.loc[:, ~combined_wab_mom.columns.duplicated()]

# make sure types are correct and match
team_stats_time_sensitive["YEAR"] = team_stats_time_sensitive["YEAR"].astype(int)
team_stats_time_sensitive["exact_date"] = pd.to_datetime(team_stats_time_sensitive["exact_date"])
combined_wab["YEAR"] = combined_wab["YEAR"].astype(int)
combined_wab["exact_date"] = pd.to_datetime(combined_wab["exact_date"])
combined_wab_mom["YEAR"] = combined_wab_mom["YEAR"].astype(int)
combined_wab_mom["exact_date"] = pd.to_datetime(combined_wab_mom["exact_date"])
combined_wab["WAB"] = pd.to_numeric(combined_wab["WAB"], errors='coerce')
combined_wab_mom["recent_wab"] = pd.to_numeric(combined_wab_mom["recent_wab"], errors='coerce')
combined_wab['TEAM'] = combined_wab['TEAM'].str.replace('\xa0', ' ')
combined_wab['TEAM'] = combined_wab['TEAM'].str.extract(r'^(.+?)(?=\s+\d+\s+seed)', expand=False).fillna(combined_wab['TEAM']).str.strip()
combined_wab_mom['TEAM'] = combined_wab_mom['TEAM'].str.replace('\xa0', ' ')
combined_wab_mom['TEAM'] = combined_wab_mom['TEAM'].str.extract(r'^(.+?)(?=\s+\d+\s+seed)', expand=False).fillna(combined_wab_mom['TEAM']).str.strip()

# merge the wab with the main stats
rs_team_stats = pd.merge(team_stats_time_sensitive, combined_wab, on=["TEAM", "YEAR", "exact_date"], how="left")

# merge the momentum WAB snapshot into team_stats
rs_team_stats = pd.merge(rs_team_stats, combined_wab_mom, on=["TEAM", "YEAR", "exact_date"], how="left")

del team_stats_time_sensitive, combined_wab, combined_wab_mom
gc.collect()

# calculate win percentage
rs_team_stats['win_percent'] = rs_team_stats['WINS'] / rs_team_stats['GAMES']

# make new variable combining offensive and defensive efficiency
rs_team_stats['adjem'] = rs_team_stats['ADJ OE'] - rs_team_stats['ADJ DE']


## PULL IN END OF SZN TEAM STATS
# doing the same thing as above but stats for the whole regular season

eos_dfs, eos_wabs, eos_wabs_mom = [], [], []
for year in years:

    last_game_date = rs_matchups[rs_matchups['year'] == year]['exact_date'].max()
    begin_date_mom = (last_game_date - timedelta(days=30)).strftime("%Y%m%d")
    end_date = last_game_date.strftime("%Y%m%d")

    team_stats_url = f"https://barttorvik.com/team-tables_each.php?tvalue=All&year={year}&sort=&t2value=None&oppType=All&conlimit=All&top=0&quad=4&mingames=0&toprk=0&venue=All&type=R&yax=3"
    wab_url = f"https://barttorvik.com/?year={year}&sort=&hteam=&t2value=&conlimit=All&state=All&top=0&revquad=0&quad=5&venue=All&type=R&mingames=0#"
    wab_mom_url = f"https://barttorvik.com/?year={year}&sort=&hteam=&t2value=&conlimit=All&state=All&begin={begin_date_mom}&end={end_date}&top=0&revquad=0&quad=5&venue=All&type=R&mingames=0#"
    
    try:
        # Team stats
        table = load_and_scrape_table(team_stats_url)
        if table:
            rows = table.find_all('tr')[1:]
            data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
            df = pd.DataFrame(data, columns=headers)
            eos_dfs.append(df)
    except Exception as e:
        print(f"[EOS SNAPSHOT ERROR] {year}: {e}")

    try:
        # WAB Table
        table = load_and_scrape_table(wab_url)
        if table:
            rows = table.find_all('tr')[1:]
            data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
            df = pd.DataFrame(data, columns=wab_headers)[["TEAM", "WAB"]]
            df["YEAR"] = year
            eos_wabs.append(df)
    except Exception as e:
        print(f"[EOS WAB ERROR] {year}: {e}")

    try:
        # Momentum WAB Table
        table = load_and_scrape_table(wab_mom_url)
        if table:
            rows = table.find_all('tr')[1:]
            data = [[cell.get_text() for cell in row.find_all('td')] for row in rows]
            df = pd.DataFrame(data, columns=wab_headers)[["TEAM", "WAB"]]
            df = df.rename(columns={"WAB": "recent_wab"})
            df["YEAR"] = year
            eos_wabs_mom.append(df)
    except Exception as e:
        print(f"[EOS MOMENTUM WAB ERROR] {year}: {e}")

    print(f"Got all stats from {year}")

log_checkpoint("Pulled in all end-of-season stats")

combined_eos_stats = pd.concat(eos_dfs, ignore_index=True)
combined_eos_wabs = pd.concat(eos_wabs, ignore_index=True)
combined_eos_wab_moms = pd.concat(eos_wabs_mom, ignore_index=True)
del eos_dfs, eos_wabs, eos_wabs_mom
gc.collect()

combined_eos_stats = combined_eos_stats.drop(columns=['']) # drop blank column
combined_eos_stats = combined_eos_stats.replace(',', '', regex=True) # get rid of commas from cells
numeric_columns = combined_eos_stats.columns.difference(['TEAM', 'RECORD'])
combined_eos_stats.replace(r'^\s*$', 0, regex=True, inplace=True) # found blank cells under TALENT, so just make it zero
combined_eos_stats[numeric_columns] = combined_eos_stats[numeric_columns].apply(pd.to_numeric) # convert everything to numeric

combined_eos_stats = combined_eos_stats.loc[:, ~combined_eos_stats.columns.duplicated()]
combined_eos_wabs = combined_eos_wabs.loc[:, ~combined_eos_wabs.columns.duplicated()]
combined_eos_wab_moms = combined_eos_wab_moms.loc[:, ~combined_eos_wab_moms.columns.duplicated()]

combined_eos_stats["YEAR"] = combined_eos_stats["YEAR"].astype(int)
combined_eos_wabs["YEAR"] = combined_eos_wabs["YEAR"].astype(int)
combined_eos_wab_moms["YEAR"] = combined_eos_wab_moms["YEAR"].astype(int)
combined_eos_wabs["WAB"] = pd.to_numeric(combined_eos_wabs["WAB"], errors='coerce')
combined_eos_wab_moms["recent_wab"] = pd.to_numeric(combined_eos_wab_moms["recent_wab"], errors='coerce')
combined_eos_wabs['TEAM'] = combined_eos_wabs['TEAM'].str.replace('\xa0', ' ')
combined_eos_wabs['TEAM'] = combined_eos_wabs['TEAM'].str.extract(r'^(.+?)(?=\s+\d+\s+seed)', expand=False).fillna(combined_eos_wabs['TEAM']).str.strip()
combined_eos_wab_moms['TEAM'] = combined_eos_wab_moms['TEAM'].str.replace('\xa0', ' ')
combined_eos_wab_moms['TEAM'] = combined_eos_wab_moms['TEAM'].str.extract(r'^(.+?)(?=\s+\d+\s+seed)', expand=False).fillna(combined_eos_wab_moms['TEAM']).str.strip()

eos_team_stats = pd.merge(combined_eos_stats, combined_eos_wabs, on=["TEAM", "YEAR"], how="left")
eos_team_stats = pd.merge(eos_team_stats, combined_eos_wab_moms, on=["TEAM", "YEAR"], how="left")

del combined_eos_stats, combined_eos_wabs, combined_eos_wab_moms
gc.collect()

eos_team_stats['win_percent'] = eos_team_stats['WINS'] / eos_team_stats['GAMES']
eos_team_stats['adjem'] = eos_team_stats['ADJ OE'] - eos_team_stats['ADJ DE']


# READ IN TOURNAMENT MATCHUP DATA
tournament_matchups = pd.read_csv("data/tournament_matchups.csv")
tournament_matchups = tournament_matchups[tournament_matchups["YEAR"].isin(years)]

# MERGE TOURNAMENT MATCHUPS WITH STATS
tourney_team_stats = pd.merge(tournament_matchups, eos_team_stats, on=["TEAM", "YEAR"], how='left')
del tournament_matchups
gc.collect()

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
            'recent_wab_1': team1_info['recent_wab'],
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
            'recent_wab_2': team2_info['recent_wab'],
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

log_checkpoint("Tourney matchups + stats done")

# MERGE REGULAR SEASON MATCHUPS WITH STATS

# change names in team_stats to match format of names in rs_matchups
rs_team_stats = rs_team_stats.rename(columns={
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

rs_team_stats = rs_team_stats.loc[:, ~rs_team_stats.columns.duplicated()]
rs_matchups = rs_matchups.loc[:, ~rs_matchups.columns.duplicated()]

# merge the matchups with the stats
## use duckdb for merge because pandas is too slow
## i would like to do this throughout the pipeline, but dealing with the current bottleneck for now

con = duckdb.connect()
con.register("matchups", rs_matchups)
con.register("stats", rs_team_stats)

# sql join
query = """
    SELECT 
        m.*,
        {team1_cols},
        {team2_cols}
    FROM matchups m
    LEFT JOIN stats s1
        ON m.team_1 = s1.team AND m.year = s1.year AND m.exact_date = s1.exact_date
    LEFT JOIN stats s2
        ON m.team_2 = s2.team AND m.year = s2.year AND m.exact_date = s2.exact_date
"""

# build prefixed columns for each join
all_cols = [col for col in rs_team_stats.columns if col not in ['team', 'year', 'exact_date']]
team1_cols = ", ".join([f"s1.\"{col}\" AS \"{col}_1\"" for col in all_cols])
team2_cols = ", ".join([f"s2.\"{col}\" AS \"{col}_2\"" for col in all_cols])

# format into the sql query
full_query = query.format(team1_cols=team1_cols, team2_cols=team2_cols)

# now run it
rs_matchups_with_stats = con.execute(full_query).df()
con.close()

log_checkpoint("Added all team stats to RS matchups")

filtered_columns = [col for col in rs_matchups_with_stats.columns if col.islower()]
rs_matchups_with_stats = rs_matchups_with_stats[filtered_columns]

rs_matchups_with_stats.columns = ['year' if 'year' in col else col for col in rs_matchups_with_stats.columns]
rs_matchups_with_stats = rs_matchups_with_stats.loc[:,~rs_matchups_with_stats.columns.duplicated()].copy()

log_checkpoint("Remove unwanted columns from RS matchups")

# create a unique matchup id so i can drop the rows that are the same as others but with team_1 and team_2 flipped
# (make sure to not drop matchups that actually happened twice. duplicates are always next to each other)
rs_matchups_with_stats["matchup_id"] = rs_matchups_with_stats.apply(lambda row: str(row['exact_date']) + 
                                                                    '-' + '-'.join(sorted([row['team_1'], row['team_2']])), 
                                                                    axis=1)
rs_matchups_with_stats = rs_matchups_with_stats.drop_duplicates(subset=["matchup_id"])

log_checkpoint("Added matchup id to RS data and dropped duplicate games")

## BALANCE/SHUFFLE RS MATCHUPS 
# order matters for tourney games, so leave that alone

# randomly flip teams to reduce bias
np.random.seed(44)
flip_mask = np.random.rand(len(rs_matchups_with_stats)) < 0.5
flipped = rs_matchups_with_stats[flip_mask].copy()
non_flipped = rs_matchups_with_stats[~flip_mask].copy()
swap_columns = ['team_1', 'team_2', 'winner'] + [
    col for col in rs_matchups_with_stats.columns if col.endswith('_1') or col.endswith('_2')
]
for col in swap_columns:
    if col.endswith('_1'):
        twin = col.replace('_1', '_2')
        flipped[col], flipped[twin] = rs_matchups_with_stats.loc[flip_mask, twin], rs_matchups_with_stats.loc[flip_mask, col]
    elif col == 'team_1':
        flipped['team_1'], flipped['team_2'] = rs_matchups_with_stats.loc[flip_mask, 'team_2'], rs_matchups_with_stats.loc[flip_mask, 'team_1']
    elif col == 'winner':
        flipped['winner'] = 1 - rs_matchups_with_stats.loc[flip_mask, 'winner']

# combine back
rs_matchups_with_stats = pd.concat([flipped, non_flipped], ignore_index=True)

# enforce perfect balance of winner column
num_total = len(rs_matchups_with_stats)
num_to_flip = abs(rs_matchups_with_stats['winner'].sum() - num_total // 2)

# flip enough rows with excess winner value
if num_to_flip > 0:
    overrepresented_val = int(rs_matchups_with_stats['winner'].mean() > 0.5)
    to_correct = rs_matchups_with_stats[rs_matchups_with_stats['winner'] == overrepresented_val].sample(n=num_to_flip, random_state=42)
    corrected = to_correct.copy()

    for col in swap_columns:
        if col.endswith('_1'):
            twin = col.replace('_1', '_2')
            corrected[col], corrected[twin] = to_correct[twin], to_correct[col]
        elif col == 'team_1':
            corrected['team_1'], corrected['team_2'] = to_correct['team_2'], to_correct['team_1']
        elif col == 'winner':
            corrected['winner'] = 1 - to_correct['winner']

    # replace those rows
    rs_matchups_with_stats = rs_matchups_with_stats.drop(to_correct.index)
    rs_matchups_with_stats = pd.concat([rs_matchups_with_stats, corrected], ignore_index=True)

# final check
print("\nPost shuffle + correction check:")
print(rs_matchups_with_stats['winner'].value_counts(normalize=True))
print('\n')

# add a column to each data set (tourney and regular season) to differentiate them post-merge
tourney_matchup_stats["type"] = "T" # tournament
rs_matchups_with_stats["type"] = "RS" # regular season

## COMBINE TOURNEY AND REGULAR SEASON MATCHUPS

all_matchup_stats = pd.concat([tourney_matchup_stats, rs_matchups_with_stats], ignore_index=True)
del tourney_matchup_stats, rs_matchups_with_stats
gc.collect()

log_checkpoint("Added rs_matchups under tourney matchups")

# get the stat differences same as before
stat_variables = [
    'badj_em', 'badj_o', 'badj_d', 'wab', 'recent_wab', 'barthag', 'efg', 'efg_d', 'ft_rate', 'ft_rate_d', 'tov_percent',
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
del stat_diff_df
gc.collect()

# CLEAN UP
allowed_missing_cols = ['seed_1', 'seed_2', 'round_1', 'round_2', 'current_round', 'exact_date', 'matchup_id']
rows_with_allowed_nulls = all_matchup_stats.isnull() & all_matchup_stats.columns.to_series().apply(lambda col: col in allowed_missing_cols)
rows_with_other_nulls = all_matchup_stats.isnull() & ~all_matchup_stats.columns.to_series().apply(lambda col: col in allowed_missing_cols)
mask = ~rows_with_other_nulls.any(axis=1)
all_matchup_stats = all_matchup_stats[mask]

# EXPORT DF
directory = "data"
all_matchup_stats.to_parquet(directory + "/all_matchup_stats.parquet", index=False)

end_time = time.time()
print("\nTotal Elapsed time: {:.2f} seconds\n".format(end_time - start_time))