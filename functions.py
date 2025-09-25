from classes import Player, Team, League, Coach
import numpy as np
import random
from faker import Faker

# Central RNG for reproducibility across modules
RNG = np.random.default_rng()

def set_rng_seed(seed: int):
    """Set the global RNG seed for reproducible simulations."""
    global RNG
    RNG = np.random.default_rng(seed)

def create_coaches(n_teams: int) -> list[Coach]:
    """Create one coach per team with a random playstyle and unique Faker name."""
    playstyles = ["star-centric", "balanced", "complementary", "hyper-offensive", "hyper-defensive"]
    fake = Faker()
    # tie Faker to Python RNG seeding for determinism
    Faker.seed(random.randrange(1_000_000_000))
    coaches = []
    for _ in range(n_teams):
        playstyle = random.choice(playstyles)
        try:
            name = fake.unique.name()
        except Exception:
            name = fake.name()
        coaches.append(Coach(name, playstyle))
    # reset unique tracker
    try:
        fake.unique.clear()
    except Exception:
        pass
    return coaches

def create_players(n_teams: int, n_lines: int, n_pairs: int, n_goalies: int) -> list[Player]:
    """Create enough players by position with unique Faker names to build n_teams squads."""
    fake = Faker()
    # tie Faker to Python RNG seeding for determinism
    Faker.seed(random.randrange(1_000_000_000))
    players = []
    total_forwards = n_teams * n_lines * 3
    total_defenders = n_teams * n_pairs * 2
    total_goalies = n_teams * n_goalies

    # Forwards
    for _ in range(total_forwards):
        try:
            pname = fake.unique.name()
        except Exception:
            pname = fake.name()
        players.append(Player(
            name = pname,
            position = "F",
            offense = random.gauss(0, 1),
            defense = random.gauss(0, 1),
            stamina = random.gauss(0, 1),
            discipline = random.gauss(0, 1)
        ))

    # Defensemen
    for _ in range(total_defenders):
        try:
            pname = fake.unique.name()
        except Exception:
            pname = fake.name()
        players.append(Player(
            name = pname,
            position = "D",
            offense = random.gauss(0, 1),
            defense = random.gauss(0, 1),
            stamina = random.gauss(0, 1),
            discipline = random.gauss(0, 1)
        ))

    # Goalies
    for _ in range(total_goalies):
        try:
            pname = fake.unique.name()
        except Exception:
            pname = fake.name()
        players.append(Player(
            name = pname,
            position = "G",
            offense = random.gauss(0, 1),
            defense = random.gauss(0, 1),
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

    # Prepare unique team city names via Faker (required)
    team_names: list[str] = []
    fake = Faker()
    # Seed faker deterministically based on Python RNG to honor random.seed in main
    Faker.seed(random.randrange(1_000_000_000))
    used = set()
    tries = 0
    while len(team_names) < n_teams and tries < n_teams * 50:
        tries += 1
        try:
            city = fake.unique.city()
        except Exception:
            city = fake.city()
        if city not in used:
            used.add(city)
            team_names.append(city)
    # reset unique tracker for future calls
    try:
        fake.unique.clear()
    except Exception:
        pass

    # Build each team
    for i in range(n_teams):
        # Sample without replacement
        team_forwards = [available_forwards.pop() for _ in range(required_forwards_per_team)]
        team_defenders = [available_defenders.pop() for _ in range(required_defenders_per_team)]
        team_goalies = [available_goalies.pop() for _ in range(required_goalies_per_team)]
        # Compose roster
        roster = team_forwards + team_defenders + team_goalies
        # Create team
        teams.append(Team(
            name = team_names[i] if i < len(team_names) else f"Team {i+1}", 
            roster = roster,
            home_rink = random.gauss(0, 1),
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
### SIMPLE GAME SIMULATION ###
##############################

def _poisson_sample(lam: float) -> int:
    """Sample Poisson(lam) from the central RNG."""
    if lam <= 0:
        return 0
    return int(RNG.poisson(lam))


def simulate_game_simple(home: Team, away: Team, mean_goals: float = 3.0) -> dict:
    """Simulate a game from team-level strengths with a simple Poisson model."""
    def team_strength(t: Team) -> float:
        total = 0.0
        for p in t.roster:
            total += (p.offense + p.defense)
        return total / len(t.roster)

    lambda_home = mean_goals + team_strength(home) + getattr(home, "home_rink", 0.0)
    lambda_away = mean_goals + team_strength(away)

    home_goals = _poisson_sample(lambda_home)
    away_goals = _poisson_sample(lambda_away)

    return {
        "home": home.name,
        "away": away.name,
        "score": {"home": home_goals, "away": away_goals},
        "lambdas": {"home": lambda_home, "away": lambda_away},
    }

