################
### PACKAGES ###
################

from classes import Player, Team, League, Coach
from typing import List, Dict
import numpy as np
import random
from faker import Faker

###############
### HELPERS ###
###############

# Generate unique, title-free person names
def _fake_unique_name(fake: Faker, seen: set[str], male: bool = False) -> str:
    """Return a unique full name without titles using first/last components.

    Uses male first names if male=True; otherwise generic first names. Ensures
    uniqueness against the provided 'seen' set; appends a numeric suffix on rare collisions.
    """
    base = f"{fake.first_name_male()} {fake.last_name()}" if male else f"{fake.first_name()} {fake.last_name()}"
    if base not in seen:
        seen.add(base)
        return base
    # Resolve collision with a short numeric suffix
    i = 2
    while True:
        candidate = f"{base} {i}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        i += 1

# Central RNG for reproducibility across modules
RNG = np.random.default_rng()

def set_rng_seed(seed: int):
    """Set the global RNG seed for reproducible simulations."""
    global RNG
    RNG = np.random.default_rng(seed)

def create_coaches(n_teams: int) -> list[Coach]:
    """Create one coach per team with a random playstyle and unique title-free name."""
    playstyles = ["star-centric", "balanced", "complementary", "hyper-offensive", "hyper-defensive"]
    fake = Faker()
    # tie Faker to Python RNG seeding for determinism
    Faker.seed(random.randrange(1_000_000_000))
    coaches = []
    seen: set[str] = set()
    for _ in range(n_teams):
        playstyle = random.choice(playstyles)
        name = _fake_unique_name(fake, seen, male=False)
        coaches.append(Coach(name, playstyle))
    # reset unique tracker
    try:
        fake.unique.clear()
    except Exception:
        pass
    return coaches

def create_players(n_teams: int, n_lines: int, n_pairs: int, n_goalies: int) -> list[Player]:
    """Create enough players by position with unique Faker names to build n_teams squads.

    Players receive male first names to avoid unexpected titles (e.g., DDS), and names are
    constructed as First Last to ensure no prefixes/suffixes.
    """
    fake = Faker()
    # tie Faker to Python RNG seeding for determinism
    Faker.seed(random.randrange(1_000_000_000))
    players = []
    seen: set[str] = set()
    total_forwards = n_teams * n_lines * 3
    total_defenders = n_teams * n_pairs * 2
    total_goalies = n_teams * n_goalies

    # Forwards
    for _ in range(total_forwards):
        pname = _fake_unique_name(fake, seen, male=True)
        players.append(Player(
            name = pname,
            position = "F",
            creation = random.gauss(0, 1),
            conversion = random.gauss(0, 1),
            suppression = random.gauss(0, 1),
            prevention = random.gauss(0, 1),
            goalkeeping = 0.0,
            stamina = random.gauss(0, 1),
            discipline = random.gauss(0, 1)
        ))

    # Defensemen
    for _ in range(total_defenders):
        pname = _fake_unique_name(fake, seen, male=True)
        players.append(Player(
            name = pname,
            position = "D",
            creation = random.gauss(0, 1),
            conversion = random.gauss(0, 1),
            suppression = random.gauss(0, 1),
            prevention = random.gauss(0, 1),
            goalkeeping = 0.0,
            stamina = random.gauss(0, 1),
            discipline = random.gauss(0, 1)
        ))

    # Goalies
    for _ in range(total_goalies):
        pname = _fake_unique_name(fake, seen, male=True)
        players.append(Player(
            name = pname,
            position = "G",
            creation = 0.0,
            conversion = 0.0,
            suppression = 0.0,
            prevention = 0.0,
            goalkeeping = random.gauss(0, 1),
            stamina = random.gauss(0, 1),
            discipline = random.gauss(0, 1)
        ))

    # reset unique tracker
    try:
        fake.unique.clear()
    except Exception:
        pass

    return players

def draft_teams(n_teams: int, players: list[Player], n_lines: int, n_pairs: int, n_goalies: int, coaches: list[Coach]) -> list[Team]:
    """Draft teams without replacement from the provided player pool."""
    # Collect teams
    teams = []

    # Split by position
    available_forwards = [p for p in players if p.position == "F"]
    available_defenders = [p for p in players if p.position == "D"]
    available_goalies = [p for p in players if p.position == "G"]

    # Required counts per team
    required_forwards_per_team = n_lines * 3
    required_defenders_per_team = n_pairs * 2
    required_goalies_per_team = n_goalies

    # Ensure enough players exist overall
    if len(available_forwards) < n_teams * required_forwards_per_team:
        raise ValueError("Not enough forwards to build teams")
    if len(available_defenders) < n_teams * required_defenders_per_team:
        raise ValueError("Not enough defensemen to build teams")
    if len(available_goalies) < n_teams * required_goalies_per_team:
        raise ValueError("Not enough goalies to build teams")

    # Shuffle pools
    random.shuffle(available_forwards)
    random.shuffle(available_defenders)
    random.shuffle(available_goalies)

    # Use predefined list of 32 most hockey-centric countries
    # Organized by conference and division for geographic consistency
    HOCKEY_COUNTRIES = [
        # Eastern Conference
        # Atlantic Division
        "Canada", "United States", "Great Britain", "France",
        # Metropolitan Division  
        "Sweden", "Finland", "Norway", "Denmark",
        # Central Division
        "Russia", "Czech Republic", "Slovakia", "Belarus",
        # Pacific Division (Europe)
        "Germany", "Switzerland", "Austria", "Italy",
        # Western Conference
        # North Division
        "Latvia", "Estonia", "Lithuania", "Poland",
        # South Division
        "Slovenia", "Croatia", "Hungary", "Romania",
        # East Division
        "Kazakhstan", "Ukraine", "Japan", "South Korea",
        # West Division
        "Netherlands", "Belgium", "Spain", "China"
    ]
    
    if n_teams != len(HOCKEY_COUNTRIES):
        raise ValueError(f"Number of teams ({n_teams}) must match number of hockey countries ({len(HOCKEY_COUNTRIES)})")
    
    team_names = HOCKEY_COUNTRIES[:n_teams]

    # Build each team
    for i in range(n_teams):
        # Sample without replacement
        team_forwards = [available_forwards.pop() for _ in range(required_forwards_per_team)]
        team_defenders = [available_defenders.pop() for _ in range(required_defenders_per_team)]
        team_goalies = [available_goalies.pop() for _ in range(required_goalies_per_team)]
        # Compose roster
        roster = team_forwards + team_defenders + team_goalies
        
        # Sample team-specific home-ice advantage factors

        # - shot_creation_mult: avg 1.02
        # - xg_bonus: avg 0.0025
        # - shot_suppression_mult: avg 0.98
        # - xg_suppression: avg -0.0025
        hfa_shot_creation_mult = random.gauss(1.02, 0.01)  # ~2% boost on average, with variation
        hfa_xg_bonus = random.gauss(0.0025, 0.0005)  # ~0.0025 xG bonus on average
        hfa_shot_suppression_mult = random.gauss(0.98, 0.005)  # ~2% reduction to opponent shot rate
        hfa_xg_suppression = random.gauss(-0.0025, 0.0005)  # ~-0.0025 reduction to opponent xG
        
        # Create team
        teams.append(Team(
            name = team_names[i] if i < len(team_names) else f"Team {i+1}", 
            roster = roster,
            hfa_shot_creation_mult = hfa_shot_creation_mult,
            hfa_xg_bonus = hfa_xg_bonus,
            hfa_shot_suppression_mult = hfa_shot_suppression_mult,
            hfa_xg_suppression = hfa_xg_suppression,
            coach = coaches[i]
        ))
    # Return teams
    return teams


def build_league(n_teams: int, n_lines: int, n_pairs: int, n_goalies: int) -> League:
    """Create coaches, players, draft teams, and return a league."""
    # Create coaches and players, then draft teams
    coaches = create_coaches(n_teams)
    players = create_players(n_teams, n_lines, n_pairs, n_goalies)
    teams = draft_teams(n_teams, players, n_lines, n_pairs, n_goalies, coaches)
    return League(teams)


##############################
### DATA EXPORT HELPERS ###
##############################

def aggregate_team_box_scores(line_box_rows: List[Dict]) -> List[Dict]:
    """Aggregate line-level box scores to team-level box scores.
    
    Args:
        line_box_rows: List of dictionaries representing line-level box scores
        
    Returns:
        List of dictionaries representing team-level box scores
    """
    from collections import defaultdict
    
    team_box_stats = defaultdict(lambda: {
        'home_shots': 0, 'away_shots': 0, 'home_xg': 0.0, 'away_xg': 0.0,
        'home_max_xg': 0.0, 'away_max_xg': 0.0, 'home_goals': 0, 'away_goals': 0,
        'home_assists': 0, 'away_assists': 0,
        'home_penalties_taken': 0, 'away_penalties_taken': 0,
        'home_penalties_drawn': 0, 'away_penalties_drawn': 0,
        'home_penalty_minutes': 0, 'away_penalty_minutes': 0
    })
    
    # Aggregate line-level stats to team-level
    for row in line_box_rows:
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
                'home_assists': 0, 'away_assists': 0,
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
        team_box_stats[game_id]['home_assists'] += int(row['home_assists'])
        team_box_stats[game_id]['away_assists'] += int(row['away_assists'])
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
            'home_assists': stats['home_assists'],
            'away_assists': stats['away_assists'],
            'home_penalties_taken': stats['home_penalties_taken'],
            'away_penalties_taken': stats['away_penalties_taken'],
            'home_penalties_drawn': stats['home_penalties_drawn'],
            'away_penalties_drawn': stats['away_penalties_drawn'],
            'home_penalty_minutes': stats['home_penalty_minutes'],
            'away_penalty_minutes': stats['away_penalty_minutes'],
        })
    
    return team_box_rows


def write_rank_csv(league: League, rank_type: str, key: str, output_dir: str, n: int = 50) -> None:
    """Write player rankings to a CSV file.
    
    Args:
        league: League instance
        rank_type: Name for the ranking type (used in filename)
        key: Attribute to rank by ('creation', 'conversion', etc.)
        output_dir: Directory to write the CSV file
        n: Number of top players to include
    """
    import csv
    import os
    
    rows = []
    top = league.player_rankings(n, key)
    for i, row in enumerate(top, start=1):
        rows.append({ 'rank': i, **row })
    with open(os.path.join(output_dir, f'{rank_type}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank','player','position','team','creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'])
        w.writeheader()
        w.writerows(rows)