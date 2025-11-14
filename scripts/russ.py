################
### PACKAGES ###
################

import csv
import os
import random
from functions import build_league, set_rng_seed, aggregate_team_box_scores, write_rank_csv
from classes import League

###################
### RUSS SCRIPT ###
###################
# Simulates 5 seasons with the same schedule and parameters,
# but different simulation randomness for each season

# Base seed for league structure (same for all seasons)
BASE_SEED = 2026
NUM_SEASONS = 5

# Base data directory
BASE_DATA_DIR = 'data/russ'

# Ensure base directory exists
os.makedirs(BASE_DATA_DIR, exist_ok=True)

print("=" * 80)
print("RUSS: Simulating 5 seasons with same schedule, different randomness")
print("=" * 80)
print()

# Build league structure once (same for all seasons)
# Set seed for league building
random.seed(BASE_SEED)
set_rng_seed(BASE_SEED)

print(f"Building league structure with seed {BASE_SEED}...")
league = build_league(n_teams=32, n_lines=2, n_pairs=2, n_goalies=1)
league.organize_into_divisions_and_conferences(seed=BASE_SEED)
league.build_schedule(shuffle=True, seed=BASE_SEED, group_weeks=True)
total_weeks = len(league.weeks)

print(f"  - {len(league.teams)} teams")
print(f"  - {len(league.weeks)} weeks")
print(f"  - {len(league.schedule)} games")
print()

# Create shared directories for static data (same across all seasons)
shared_dir = os.path.join(BASE_DATA_DIR, 'shared')
shared_players_dir = os.path.join(BASE_DATA_DIR, 'players')

for directory in (shared_dir, shared_players_dir):
    os.makedirs(directory, exist_ok=True)

# Export static files (same for all seasons)
print("Exporting static files (players, teams, schedule)...")

# Export rosters.csv (players)
with open(os.path.join(shared_players_dir, 'rosters.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['team','coach','playstyle','player','position','creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'])
    for team in league.teams:
        for p in team.roster:
            total = p.creation + p.conversion + p.suppression + p.prevention + p.goalkeeping
            w.writerow([team.name, team.coach.name, team.coach.playstyle, p.name, p.position, 
                       f"{p.creation:.3f}", f"{p.conversion:.3f}", f"{p.suppression:.3f}", f"{p.prevention:.3f}", 
                       f"{p.goalkeeping:.3f}", f"{p.stamina:.3f}", f"{p.discipline:.3f}", f"{total:.3f}"])

# Write player rankings (players)
write_rank_csv(league, 'creation', 'creation', shared_players_dir, 50)
write_rank_csv(league, 'conversion', 'conversion', shared_players_dir, 50)
write_rank_csv(league, 'suppression', 'suppression', shared_players_dir, 50)
write_rank_csv(league, 'prevention', 'prevention', shared_players_dir, 50)
write_rank_csv(league, 'goalkeeping', 'goalkeeping', shared_players_dir, 50)
write_rank_csv(league, 'stamina', 'stamina', shared_players_dir, 50)
write_rank_csv(league, 'discipline', 'discipline', shared_players_dir, 50)

# Write teams.csv (shared)
team_rows = league.get_teams()
team_rows.sort(key=lambda r: (r['total_sum'], r['creation_sum']), reverse=True)
with open(os.path.join(shared_dir, 'teams.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','coach','playstyle','conference','division','creation_sum','conversion_sum','suppression_sum','prevention_sum','goalkeeping_sum','stamina_sum','discipline_sum','total_sum','hfa_shot_creation_mult','hfa_xg_bonus','hfa_shot_suppression_mult','hfa_xg_suppression'])
    w.writeheader()
    w.writerows(team_rows)

# Write schedule.csv (shared)
with open(os.path.join(shared_dir, 'schedule.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['matchweek','home_team','away_team'])
    for wk_idx, week in enumerate(league.weeks, start=1):
        for home, away in week:
            w.writerow([wk_idx, home.name, away.name])

print("  ✓ Static files exported")
print()

# Simulate each season (only simulation-dependent data)
for season_num in range(1, NUM_SEASONS + 1):
    print(f"Season {season_num}/{NUM_SEASONS}...")
    
    # Set different simulation seed for this season
    # Use BASE_SEED + season_num to ensure different randomness
    simulation_seed = BASE_SEED + (season_num * 1000)
    random.seed(simulation_seed)
    set_rng_seed(simulation_seed)
    
    # Create season-specific directories for simulation-dependent data
    # Organize by entity type: teams and games data
    season_teams_dir = os.path.join(BASE_DATA_DIR, f'season-{season_num}', 'teams')
    season_games_dir = os.path.join(BASE_DATA_DIR, f'season-{season_num}', 'games')
    season_box_scores_dir = os.path.join(season_games_dir, 'box-scores')
    
    for directory in (season_teams_dir, season_games_dir, season_box_scores_dir):
        os.makedirs(directory, exist_ok=True)
    
    # Simulate season with different randomness
    print(f"  Simulating games with seed {simulation_seed}...")
    game_rows, pbp_rows, box_rows = league.simulate_schedule(pbp_weeks=total_weeks, generate_box_scores=True)
    
    # Compute standings
    standings = league.get_standings(game_rows)
    
    # Write game-results.csv (games, season-specific)
    with open(os.path.join(season_games_dir, 'game-results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_score','away_score','went_ot','winner','loser'])
        w.writeheader()
        w.writerows(game_rows)
    
    # Write standings.csv (teams, season-specific)
    rows = []
    for team_name, s in standings.items():
        gd = s['GF'] - s['GA']
        rows.append({
            'team': team_name,
            'GP': s['GP'], 'W': s['W'], 'L': s['L'], 'OTL': s['OTL'],
            'PTS': s['PTS'], 'GF': s['GF'], 'GA': s['GA'], 'GD': gd
        })
    rows.sort(key=lambda r: (r['PTS'], r['GD'], r['GF']), reverse=True)
    with open(os.path.join(season_teams_dir, 'standings.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['team','GP','W','L','OTL','PTS','GF','GA','GD'])
        w.writeheader()
        w.writerows(rows)
    
    # Write play-by-play.csv (games, season-specific)
    with open(os.path.join(season_games_dir, 'play-by-play.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','period','time_seconds','event_type','description','home_score','away_score','tag','home_on_ice','away_on_ice'])
        w.writeheader()
        w.writerows(pbp_rows)
    
    # Write line-level box scores (games/box-scores, season-specific)
    with open(os.path.join(season_box_scores_dir, 'lines.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','home_line_type','home_pairing_type','away_line_type','away_pairing_type','home_goalie','away_goalie','went_ot','toi','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_assists','away_assists','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
        w.writeheader()
        w.writerows(box_rows)
    
    # Generate and write team-level box scores (games/box-scores, season-specific)
    team_box_rows = aggregate_team_box_scores(box_rows)
    with open(os.path.join(season_box_scores_dir, 'games.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['game_id','week','home_team','away_team','went_ot','home_shots','away_shots','home_xg','away_xg','home_max_xg','away_max_xg','home_goals','away_goals','home_assists','away_assists','home_penalties_taken','away_penalties_taken','home_penalties_drawn','away_penalties_drawn','home_penalty_minutes','away_penalty_minutes'])
        w.writeheader()
        w.writerows(team_box_rows)
    
    print(f"  ✓ Season {season_num} complete")
    print(f"    - {len(game_rows)} games simulated")
    print(f"    - {len(pbp_rows)} play-by-play events")
    print(f"    - {len(box_rows)} line-level box score rows")
    print()

print("=" * 80)
print("All 5 seasons complete!")
print(f"Output written to {BASE_DATA_DIR}/")
print("=" * 80)
print()
print("Directory structure:")
print(f"  {BASE_DATA_DIR}/")
print(f"    players/ (shared across all seasons)")
print(f"      - rosters.csv")
print(f"      - creation.csv, conversion.csv, suppression.csv, prevention.csv")
print(f"      - goalkeeping.csv, stamina.csv, discipline.csv")
print(f"    shared/ (shared across all seasons)")
print(f"      - teams.csv")
print(f"      - schedule.csv")
print()
for season_num in range(1, NUM_SEASONS + 1):
    print(f"    season-{season_num}/ (season-specific simulation data)")
    print(f"      teams/")
    print(f"        - standings.csv")
    print(f"      games/")
    print(f"        - game-results.csv")
    print(f"        - play-by-play.csv")
    print(f"        box-scores/")
    print(f"          - lines.csv")
    print(f"          - games.csv")

