################
### PACKAGES ###
################

import csv
import os
import random
from functions import build_league, set_rng_seed
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
SEASON_DIR = os.path.join(DATA_DIR, 'season')
BOX_SCORES_DIR = os.path.join(DATA_DIR, 'box-scores')
RANKINGS_DIR = os.path.join(DATA_DIR, 'rankings')

for directory in (SEASON_DIR, BOX_SCORES_DIR, RANKINGS_DIR):
    os.makedirs(directory, exist_ok=True)

# 1) Build a 30-team league
league = build_league(n_teams=30, n_lines=4, n_pairs=3, n_goalies=2)

# 2) Build schedule (weeks are grouped inside)
league.build_schedule(shuffle=True, seed=SEED, group_weeks=True)
total_weeks = len(league.weeks)

# 3) Export rosters.csv
with open(os.path.join(SEASON_DIR, 'rosters.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','coach','playstyle','player','position','creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'])
    for team in league.teams:
        for p in team.roster:
            total = p.creation + p.conversion + p.suppression + p.prevention + p.goalkeeping
            w.writerow([team.name, team.coach.name, team.coach.playstyle, p.name, p.position, 
                       f"{p.creation:.3f}", f"{p.conversion:.3f}", f"{p.suppression:.3f}", f"{p.prevention:.3f}", 
                       f"{p.goalkeeping:.3f}", f"{p.stamina:.3f}", f"{p.discipline:.3f}", f"{total:.3f}"])

# 4) Write schedule.csv from league.weeks
with open(os.path.join(SEASON_DIR, 'schedule.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['matchweek','home_team','away_team'])
    for wk_idx, week in enumerate(league.weeks, start=1):
        for home, away in week:
            w.writerow([wk_idx, home.name, away.name])

# 5) Simulate season and collect results/PBP/box scores using League helpers
game_rows, pbp_rows, box_rows = league.simulate_schedule(pbp_weeks=total_weeks, generate_box_scores=True)

# 6) Compute standings up to full season
standings = league.get_standings(game_rows)

# 7) Write game-results.csv
with open(os.path.join(SEASON_DIR, 'game-results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_score','away_score','went_ot','winner','loser'])
    w.writeheader()
    w.writerows(game_rows)

# 8) Write standings.csv (sorted)
rows = []
for team_name, s in standings.items():
    gd = s['GF'] - s['GA']
    rows.append({
        'team': team_name,
        'GP': s['GP'], 'W': s['W'], 'OTW': s['OTW'], 'L': s['L'], 'OTL': s['OTL'],
        'PTS': s['PTS'], 'GF': s['GF'], 'GA': s['GA'], 'GD': gd
    })
rows.sort(key=lambda r: (r['PTS'], r['GD'], r['GF']), reverse=True)
with open(os.path.join(SEASON_DIR, 'standings.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','GP','W','OTW','L','OTL','PTS','GF','GA','GD'])
    w.writeheader()
    w.writerows(rows)

# 9) Write full-season play-by-play
with open(os.path.join(SEASON_DIR, 'play-by-play.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','period','time_seconds','event_type','description','home_score','away_score','tag','home_on_ice','away_on_ice'])
    w.writeheader()
    w.writerows(pbp_rows)

# 10) Write line-level box scores
with open(os.path.join(BOX_SCORES_DIR, 'lines.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_line_type','home_pairing_type','away_line_type','away_pairing_type','went_ot','toi','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
    w.writeheader()
    w.writerows(box_rows)

# 10b) Generate and write team-level box scores (aggregated from line-level)
from collections import defaultdict
team_box_stats = defaultdict(lambda: {
    'home_shots': 0, 'away_shots': 0, 'home_xg': 0.0, 'away_xg': 0.0,
    'home_max_xg': 0.0, 'away_max_xg': 0.0, 'home_goals': 0, 'away_goals': 0,
    'home_penalties_taken': 0, 'away_penalties_taken': 0,
    'home_penalties_drawn': 0, 'away_penalties_drawn': 0,
    'home_penalty_minutes': 0, 'away_penalty_minutes': 0
})

# Aggregate line-level stats to team-level
for row in box_rows:
    game_id = row['game_id']
    if game_id not in team_box_stats:
        team_box_stats[game_id] = {
            'game_id': game_id,
            'week': row['week'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'went_ot': row['went_ot'],
            'home_shots': 0, 'away_shots': 0, 'home_xg': 0.0, 'away_xg': 0.0,
            'home_max_xg': 0.0, 'away_max_xg': 0.0, 'home_goals': 0, 'away_goals': 0,
            'home_penalties_taken': 0, 'away_penalties_taken': 0,
            'home_penalties_drawn': 0, 'away_penalties_drawn': 0,
            'home_penalty_minutes': 0, 'away_penalty_minutes': 0
        }
    
    team_box_stats[game_id]['home_shots'] += int(row['home_shots'])
    team_box_stats[game_id]['away_shots'] += int(row['away_shots'])
    team_box_stats[game_id]['home_xg'] += float(row['home_xg'])
    team_box_stats[game_id]['away_xg'] += float(row['away_xg'])
    team_box_stats[game_id]['home_max_xg'] = max(team_box_stats[game_id]['home_max_xg'], float(row['home_max_xg']))
    team_box_stats[game_id]['away_max_xg'] = max(team_box_stats[game_id]['away_max_xg'], float(row['away_max_xg']))
    team_box_stats[game_id]['home_goals'] += int(row['home_goals'])
    team_box_stats[game_id]['away_goals'] += int(row['away_goals'])
    team_box_stats[game_id]['home_penalties_taken'] += int(row['home_penalties_taken'])
    team_box_stats[game_id]['away_penalties_taken'] += int(row['away_penalties_taken'])
    team_box_stats[game_id]['home_penalties_drawn'] += int(row['home_penalties_drawn'])
    team_box_stats[game_id]['away_penalties_drawn'] += int(row['away_penalties_drawn'])
    team_box_stats[game_id]['home_penalty_minutes'] += int(row['home_penalty_minutes'])
    team_box_stats[game_id]['away_penalty_minutes'] += int(row['away_penalty_minutes'])

# Round and format team box scores
team_box_rows = []
for game_id in sorted(team_box_stats.keys()):
    stats = team_box_stats[game_id]
    team_box_rows.append({
        'game_id': stats['game_id'],
        'week': stats['week'],
        'home_team': stats['home_team'],
        'away_team': stats['away_team'],
        'went_ot': stats['went_ot'],
        'home_shots': stats['home_shots'],
        'away_shots': stats['away_shots'],
        'home_xg': round(stats['home_xg'], 4),
        'away_xg': round(stats['away_xg'], 4),
        'home_max_xg': round(stats['home_max_xg'], 4),
        'away_max_xg': round(stats['away_max_xg'], 4),
        'home_goals': stats['home_goals'],
        'away_goals': stats['away_goals'],
        'home_penalties_taken': stats['home_penalties_taken'],
        'away_penalties_taken': stats['away_penalties_taken'],
        'home_penalties_drawn': stats['home_penalties_drawn'],
        'away_penalties_drawn': stats['away_penalties_drawn'],
        'home_penalty_minutes': stats['home_penalty_minutes'],
        'away_penalty_minutes': stats['away_penalty_minutes'],
    })

with open(os.path.join(BOX_SCORES_DIR, 'games.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','went_ot','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
    w.writeheader()
    w.writerows(team_box_rows)

print()

# 11) Player rankings (via League)
def write_rank_csv(rank_type: str, key: str, n: int = 50) -> None:
    rows = []
    top = league.player_rankings(n, key)
    for i, row in enumerate(top, start=1):
        rows.append({ 'rank': i, **row })
    with open(os.path.join(RANKINGS_DIR, f'{rank_type}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank','player','position','team','creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'])
        w.writeheader()
        w.writerows(rows)

write_rank_csv('creation', 'creation', 50)
write_rank_csv('conversion', 'conversion', 50)
write_rank_csv('suppression', 'suppression', 50)
write_rank_csv('prevention', 'prevention', 50)
write_rank_csv('goalkeeping', 'goalkeeping', 50)
write_rank_csv('stamina', 'stamina', 50)
write_rank_csv('discipline', 'discipline', 50)
print()

# 12) Team attribute sums (via League)
team_rows = league.get_teams()
# Sort by total_sum desc, then creation_sum as tie-breaker
team_rows.sort(key=lambda r: (r['total_sum'], r['creation_sum']), reverse=True)
with open(os.path.join(SEASON_DIR, 'teams.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','coach','playstyle','creation_sum','conversion_sum','suppression_sum','prevention_sum','goalkeeping_sum','stamina_sum','discipline_sum','total_sum'])
    w.writeheader()
    w.writerows(team_rows)
print("Outputs written under data/: ")
print("  season/: rosters.csv, schedule.csv, game-results.csv, standings.csv, teams.csv, play-by-play.csv")
print("  box-scores/: lines.csv, games.csv")
print("  rankings/: creation-rankings.csv, conversion-rankings.csv, suppression-rankings.csv, prevention-rankings.csv, goalkeeping-rankings.csv, stamina-rankings.csv, discipline-rankings.csv")