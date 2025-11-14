################
### PACKAGES ###
################

import csv
import os
import random
from functions import build_league, set_rng_seed, aggregate_team_box_scores, write_rank_csv
from classes import League

###################
### MAIN SCRIPT ###
###################

# Global seeding for reproducibility
SEED = 2026
set_rng_seed(SEED)   # NumPy RNG used inside simulation
random.seed(SEED)    # Python RNG used for coaches, shuffles, etc.

# Ensure data directory exists
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Organize by entity type
PLAYERS_DIR = os.path.join(DATA_DIR, 'players')
TEAMS_DIR = os.path.join(DATA_DIR, 'teams')
GAMES_DIR = os.path.join(DATA_DIR, 'games')
BOX_SCORES_DIR = os.path.join(GAMES_DIR, 'box-scores')

for directory in (PLAYERS_DIR, TEAMS_DIR, GAMES_DIR, BOX_SCORES_DIR):
    os.makedirs(directory, exist_ok=True)

# 1) Build a 32-team league
league = build_league(n_teams=32, n_lines=2, n_pairs=2, n_goalies=1)

# 2) Organize teams into divisions and conferences
league.organize_into_divisions_and_conferences(seed=SEED)

# 3) Build NHL-style schedule (82 games per team, imbalanced: division > conference > other conference)
league.build_schedule(shuffle=True, seed=SEED, group_weeks=True)
total_weeks = len(league.weeks)

# 4) Export rosters.csv (players)
with open(os.path.join(PLAYERS_DIR, 'rosters.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','coach','playstyle','player','position','creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'])
    for team in league.teams:
        for p in team.roster:
            total = p.creation + p.conversion + p.suppression + p.prevention + p.goalkeeping
            w.writerow([team.name, team.coach.name, team.coach.playstyle, p.name, p.position, 
                       f"{p.creation:.3f}", f"{p.conversion:.3f}", f"{p.suppression:.3f}", f"{p.prevention:.3f}", 
                       f"{p.goalkeeping:.3f}", f"{p.stamina:.3f}", f"{p.discipline:.3f}", f"{total:.3f}"])

# 5) Write schedule.csv (games)
with open(os.path.join(GAMES_DIR, 'schedule.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['matchweek','home_team','away_team'])
    for wk_idx, week in enumerate(league.weeks, start=1):
        for home, away in week:
            w.writerow([wk_idx, home.name, away.name])

# 6) Simulate season and collect results/PBP/box scores using League helpers
game_rows, pbp_rows, box_rows = league.simulate_schedule(pbp_weeks=total_weeks, generate_box_scores=True)

# 7) Compute standings up to full season
standings = league.get_standings(game_rows)

# 8) Write game-results.csv (games)
with open(os.path.join(GAMES_DIR, 'game-results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_score','away_score','went_ot','winner','loser'])
    w.writeheader()
    w.writerows(game_rows)

# 9) Write standings.csv (teams)
rows = []
for team_name, s in standings.items():
    gd = s['GF'] - s['GA']
    rows.append({
        'team': team_name,
        'GP': s['GP'], 'W': s['W'], 'L': s['L'], 'OTL': s['OTL'],
        'PTS': s['PTS'], 'GF': s['GF'], 'GA': s['GA'], 'GD': gd
    })
rows.sort(key=lambda r: (r['PTS'], r['GD'], r['GF']), reverse=True)
with open(os.path.join(TEAMS_DIR, 'standings.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','GP','W','L','OTL','PTS','GF','GA','GD'])
    w.writeheader()
    w.writerows(rows)

# 10) Write full-season play-by-play (games)
with open(os.path.join(GAMES_DIR, 'play-by-play.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','period','time_seconds','event_type','description','home_score','away_score','tag','home_on_ice','away_on_ice'])
    w.writeheader()
    w.writerows(pbp_rows)

# 11) Write line-level box scores
with open(os.path.join(BOX_SCORES_DIR, 'lines.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_line_type','home_pairing_type','away_line_type','away_pairing_type','home_goalie','away_goalie','went_ot','toi','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_assists','away_assists','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
    w.writeheader()
    w.writerows(box_rows)

# 11b) Generate and write team-level box scores (aggregated from line-level)
team_box_rows = aggregate_team_box_scores(box_rows)

with open(os.path.join(BOX_SCORES_DIR, 'games.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','went_ot','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_assists','away_assists','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
    w.writeheader()
    w.writerows(team_box_rows)

print()

# 12) Player rankings (players)
write_rank_csv(league, 'creation', 'creation', PLAYERS_DIR, 50)
write_rank_csv(league, 'conversion', 'conversion', PLAYERS_DIR, 50)
write_rank_csv(league, 'suppression', 'suppression', PLAYERS_DIR, 50)
write_rank_csv(league, 'prevention', 'prevention', PLAYERS_DIR, 50)
write_rank_csv(league, 'goalkeeping', 'goalkeeping', PLAYERS_DIR, 50)
write_rank_csv(league, 'stamina', 'stamina', PLAYERS_DIR, 50)
write_rank_csv(league, 'discipline', 'discipline', PLAYERS_DIR, 50)
print()

# 13) Team attribute sums (teams)
team_rows = league.get_teams()
# Sort by total_sum desc, then creation_sum as tie-breaker
team_rows.sort(key=lambda r: (r['total_sum'], r['creation_sum']), reverse=True)
with open(os.path.join(TEAMS_DIR, 'teams.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','coach','playstyle','conference','division','creation_sum','conversion_sum','suppression_sum','prevention_sum','goalkeeping_sum','stamina_sum','discipline_sum','total_sum','hfa_shot_creation_mult','hfa_xg_bonus','hfa_shot_suppression_mult','hfa_xg_suppression'])
    w.writeheader()
    w.writerows(team_rows)
print("Outputs written under data/: ")
print("  players/: rosters.csv, creation.csv, conversion.csv, suppression.csv, prevention.csv, goalkeeping.csv, stamina.csv, discipline.csv")
print("  teams/: teams.csv, standings.csv")
print("  games/: schedule.csv, game-results.csv, play-by-play.csv")
print("  games/box-scores/: lines.csv, games.csv")