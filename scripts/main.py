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
DATA_DIR = '../data'
os.makedirs(DATA_DIR, exist_ok=True)

# 1) Build a 30-team league
league = build_league(n_teams=30, n_lines=4, n_pairs=3, n_goalies=2)

# 2) Build schedule (weeks are grouped inside)
league.build_schedule(shuffle=True, seed=SEED, group_weeks=True)

# 3) Export rosters.csv
with open(os.path.join(DATA_DIR, 'rosters.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','coach','playstyle','player','position','offense','defense','stamina','discipline','total'])
    for team in league.teams:
        for p in team.roster:
            total = p.offense + p.defense
            w.writerow([team.name, team.coach.name, team.coach.playstyle, p.name, p.position, f"{p.offense:.3f}", f"{p.defense:.3f}", f"{p.stamina:.3f}", f"{p.discipline:.3f}", f"{total:.3f}"])

# 4) Write schedule.csv from league.weeks
with open(os.path.join(DATA_DIR, 'schedule.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['matchweek','home_team','away_team'])
    for wk_idx, week in enumerate(league.weeks, start=1):
        for home, away in week:
            w.writerow([wk_idx, home.name, away.name])

# 5) Simulate season and collect results/PBP using League helpers
game_rows, pbp_rows = league.simulate_schedule(pbp_weeks=5)

# 6) Compute standings up to full season
standings = league.get_standings(game_rows)

# 7) Write game_results.csv
with open(os.path.join(DATA_DIR, 'game_results.csv'), 'w', newline='') as f:
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
with open(os.path.join(DATA_DIR, 'standings.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','GP','W','OTW','L','OTL','PTS','GF','GA','GD'])
    w.writeheader()
    w.writerows(rows)

# 9) Write PBP for first 5 weeks
with open(os.path.join(DATA_DIR, 'pbp_first_5_weeks.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','period','time_seconds','event_type','description','home_score','away_score','tag','home_on_ice','away_on_ice'])
    w.writeheader()
    w.writerows(pbp_rows)

print()

# 10) Player rankings (via League)
def write_rank_csv(rank_type: str, key: str, n: int = 50) -> None:
    rows = []
    top = league.player_rankings(n, key)
    for i, row in enumerate(top, start=1):
        rows.append({ 'rank': i, **row })
    with open(os.path.join(DATA_DIR, f'{rank_type}_rankings.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank','player','position','team','offense','defense','stamina','discipline','total'])
        w.writeheader()
        w.writerows(rows)

write_rank_csv('offense', 'offense', 50)
write_rank_csv('defense', 'defense', 50)
write_rank_csv('stamina', 'stamina', 50)
write_rank_csv('discipline', 'discipline', 50)
print()

# 11) Team attribute sums (via League)
team_rows = league.get_teams()
# Sort by total_sum desc, then offense_sum as tie-breaker
team_rows.sort(key=lambda r: (r['total_sum'], r['offense_sum']), reverse=True)
with open(os.path.join(DATA_DIR, 'teams.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','coach','playstyle','offense_sum','defense_sum','stamina_sum','discipline_sum','total_sum'])
    w.writeheader()
    w.writerows(team_rows)
print(f"Outputs written to data/: "
      f"rosters.csv, schedule.csv, game_results.csv, standings.csv, pbp_first_5_weeks.csv, "
      f"offense_rankings.csv, defense_rankings.csv, stamina_rankings.csv, discipline_rankings.csv, teams.csv")