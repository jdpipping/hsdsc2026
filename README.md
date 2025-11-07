# 2026

Code and data for WSABI's 2026 High School Data Science Competition.

## Directory Guide

```
.
├── data/
│   ├── season/             # season-level outputs
│   │   ├── roster.csv
│   │   ├── schedule.csv
│   │   ├── game-results.csv
│   │   ├── standings.csv
│   │   ├── teams.csv
│   │   └── play-by-play.csv
│   ├── box-scores/         # line- and game-level box scores
│   │   ├── lines.csv
│   │   └── games.csv
│   └── rankings/           # player rankings by attribute
│       ├── creation.csv
│       ├── conversion.csv
│       ├── suppression.csv
│       ├── prevention.csv
│       ├── goalkeeping.csv
│       ├── stamina.csv
│       └── discipline.csv
├── scripts/
│   ├── main.py             # entry point – generates the league, runs season, dumps outputs
│   ├── classes.py          # core simulation classes (League, Game, Team, Player, etc.)
│   └──functions.py        # helpers for building league data
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