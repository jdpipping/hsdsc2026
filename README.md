# 2026

Code and data for WSABI's 2026 High School Data Science Competition.

## Directory Guide

```
.
├── data/
│   ├── players/            # player data
│   │   ├── rosters.csv     # all players with attributes
│   │   ├── creation.csv    # top 50 players by creation
│   │   ├── conversion.csv  # top 50 players by conversion
│   │   ├── suppression.csv # top 50 players by suppression
│   │   ├── prevention.csv  # top 50 players by prevention
│   │   ├── goalkeeping.csv # top 50 players by goalkeeping
│   │   ├── stamina.csv     # top 50 players by stamina
│   │   └── discipline.csv  # top 50 players by discipline
│   ├── teams/              # team data
│   │   ├── teams.csv       # team attributes, HFA factors, conference/division
│   │   └── standings.csv   # final season standings
│   └── games/              # game data
│       ├── schedule.csv    # full season schedule
│       ├── game-results.csv # game outcomes
│       ├── play-by-play.csv # detailed game events
│       └── box-scores/     # box score data
│           ├── lines.csv   # line-level matchups and stats
│           └── games.csv   # game-level aggregated stats
├── scripts/
│   ├── main.py             # entry point – generates the league, runs season, dumps outputs
│   ├── classes.py          # core simulation classes (League, Game, Team, Player, etc.)
│   └── functions.py        # helpers for building league data
└── README.md
```

## Requirements

- Python 3.10+
- Standard library only (no external dependencies)

## Running the Simulation

1. Run the main script from the repository root:
   ```bash
   python3 scripts/main.py
   ```

2. Outputs will be written under `data/` according to the directory guide above.

## Output Files

### Players (`data/players/`)
- **rosters.csv**: All players with their attributes (creation, conversion, suppression, prevention, goalkeeping, stamina, discipline), team, coach, and playstyle
- **creation.csv, conversion.csv, etc.**: Top 50 players ranked by each attribute

### Teams (`data/teams/`)
- **teams.csv**: Team-level attributes (sums of player attributes), coach information, conference/division assignment, and home-ice advantage (HFA) factors
- **standings.csv**: Final season standings with wins, losses, points, goals for/against

### Games (`data/games/`)
- **schedule.csv**: Full season schedule with matchweek, home team, and away team
- **game-results.csv**: Game outcomes with scores, overtime status, winner, and loser
- **play-by-play.csv**: Detailed game events including goals, shots, penalties, etc.
- **box-scores/lines.csv**: Line-level matchup statistics (time on ice, shots, xG, goals, penalties, etc.)
- **box-scores/games.csv**: Game-level aggregated statistics (team totals for shots, xG, goals, penalties, etc.)