################
### PACKAGES ###
################

from typing import Dict, List, Optional, Set, Tuple, Union
import random
import math

# Goal model constants (per second baselines)
BASE_5V5 = 2.5 / 3600.0
BASE_PP_5V4 = 6.5 / 3600.0
BASE_SH_4V5 = 0.6 / 3600.0
BASE_PP_5V3 = 19.0 / 3600.0
BASE_SH_3V5 = 0.36 / 3600.0

# Pulled-goalie baselines (per second)
BASE_6V5_FOR = 6.5 / 3600.0      # team with 6 skaters attacking 5
BASE_5V6_FOR = 13.0 / 3600.0     # team with 5 shooting at empty net
BASE_6V4_FOR = 12.5 / 3600.0     # PP with pulled goalie (6v4)
BASE_4V6_FOR = 9.0 / 3600.0      # PK vs pulled goalie (4v6)

# Home-ice advantage offsets (per second, applied to baselines)
HFA_OFF = 0.15 / 3600.0
HFA_DEF = 0.15 / 3600.0

# Player influence scaling
STRENGTH_SCALE = 0.0001
TINY = 1e-12

# Penalty hazards
PEN_BASE = 6.0 / 3600.0
PEN_BETA = 0.40
PEN_NORM = math.exp(0.5 * PEN_BETA * PEN_BETA)
PEN_CLAMP_LO = 0.5
PEN_CLAMP_HI = 1.8


class Player:
    """Represents a hockey player (skater or goalie) with core attributes.

    Attributes:
        name: Player's display name.
        position: "F", "D", or "G".
        offense: Offensive skill.
        defense: Defensive skill.
        stamina: Measure of endurance.
        discipline: Measure of penalty discipline.
    """
    def __init__(self, name: str, position: str, offense: float, defense: float, stamina: float, discipline: float):
        self.name = name
        self.position = position
        self.offense = offense
        self.defense = defense
        self.stamina = stamina
        self.discipline = discipline

class Coach:
    """Represents a hockey coach with a specific playstyle."""
    def __init__(self, name: str, playstyle: str):
        self.name = name
        self.playstyle = playstyle  # "star-centric", "balanced", "complementary", "hyper-offensive", "hyper-defensive"
    
    def create_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Create player groupings based on coach's playstyle."""
        if self.playstyle == "star-centric":
            return self._star_centric_groupings(players, group_size)
        elif self.playstyle == "balanced":
            return self._balanced_groupings(players, group_size)
        elif self.playstyle == "complementary":
            return self._complementary_groupings(players, group_size)
        elif self.playstyle == "hyper-offensive":
            return self._hyper_offensive_groupings(players, group_size)
        elif self.playstyle == "hyper-defensive":
            return self._hyper_defensive_groupings(players, group_size)
    
    def create_lines(self, forwards: List[Player]) -> Dict[int, List[Player]]:
        """Create forward lines based on coach's playstyle."""
        return self.create_groupings(forwards, 3)
    
    def create_pairs(self, defensemen: List[Player]) -> Dict[int, List[Player]]:
        """Create defensive pairings based on coach's playstyle."""
        return self.create_groupings(defensemen, 2)
    
    def _star_centric_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Best players by sum of offensive and defensive attributes go to top groupings."""
        sorted_players = sorted(players, key=lambda p: p.offense + p.defense, reverse=True)
        groupings = {}
        for i in range(0, len(sorted_players), group_size):
            line_num = (i // group_size) + 1
            groupings[line_num] = sorted_players[i:i + group_size]
        return groupings
    
    def _balanced_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Make groupings as equal as possible in talent.

        Create exactly N = floor(len(players)/group_size) groups of size group_size
        using round-robin assignment to balance totals, avoiding 3x4 errors.
        """
        sorted_players = sorted(players, key=lambda p: p.offense + p.defense, reverse=True)
        num_groups = max(1, len(sorted_players) // group_size)
        groups: List[List[Player]] = [[] for _ in range(num_groups)]
        # Round-robin best-available into groups to balance sums
        for idx, p in enumerate(sorted_players[: num_groups * group_size]):
            groups[idx % num_groups].append(p)
        return {i + 1: grp for i, grp in enumerate(groups)}
    
    def _complementary_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Groupings contain a mix of offensive and defensive players."""
        offensive = sorted([p for p in players if p.offense > p.defense], key=lambda p: p.offense, reverse=True)
        defensive = sorted([p for p in players if p.defense >= p.offense], key=lambda p: p.defense, reverse=True)
        num_groups = max(1, len(players) // group_size)
        groups: List[List[Player]] = [[] for _ in range(num_groups)]
        # Build alternating pools of size group_size per group
        idx = 0
        while any([offensive, defensive]) and idx < num_groups * group_size:
            g = idx % num_groups
            # Alternate offense/defense per slot within group
            slot = len(groups[g])
            if slot % 2 == 0:
                pick = offensive.pop(0) if offensive else (defensive.pop(0) if defensive else None)
            else:
                pick = defensive.pop(0) if defensive else (offensive.pop(0) if offensive else None)
            if pick is None:
                break
            groups[g].append(pick)
            idx += 1
        # Trim to exact group_size
        groups = [grp[:group_size] for grp in groups]
        return {i + 1: grp for i, grp in enumerate(groups)}
    
    def _hyper_offensive_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Greedy search for best offensive attributes, stack groupings accordingly."""
        sorted_players = sorted(players, key=lambda p: p.offense, reverse=True)
        groupings = {}
        for i in range(0, len(sorted_players), group_size):
            line_num = (i // group_size) + 1
            groupings[line_num] = sorted_players[i:i + group_size]
        return groupings
    
    def _hyper_defensive_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Greedy search for best defensive attributes, stack groupings accordingly."""
        sorted_players = sorted(players, key=lambda p: p.defense, reverse=True)
        groupings = {}
        for i in range(0, len(sorted_players), group_size):
            line_num = (i // group_size) + 1
            groupings[line_num] = sorted_players[i:i + group_size]
        return groupings

class Team:
    """A team composed of players with helpers to access positions.

    Attributes:
        name: Team name.
        roster: List of Player instances.
        home_rink: Home rink advantage.
        coach: Coach instance.
        lines: Forward lines created by coach.
        pairs: Defensive pairings created by coach.
    """
    def __init__(self, name: str, roster: List[Player], home_rink: float, coach: 'Coach'):
        self.name = name
        self.roster = list(roster) if roster is not None else []
        self.home_rink = home_rink
        self.coach = coach
        self.lines = coach.create_lines(self.forwards())
        self.pairs = coach.create_pairs(self.defensemen())

    def forwards(self) -> List[Player]:
        return [p for p in self.roster if p.position == "F"]
    def defensemen(self) -> List[Player]:
        return [p for p in self.roster if p.position == "D"]
    def goalies(self) -> List[Player]:
        return [p for p in self.roster if p.position == "G"]

    def add_player(self, player: Player) -> None:
        self.roster.append(player)

    def remove_player(self, player: Player) -> None:
        self.roster.remove(player)

    def power_play_unit(self, unavailable: Optional[Set[Player]] = None) -> List[Player]:
        """Select a 5-skater power play unit from available players.

        Logic:
        - Rank all available skaters by offensive contribution (Player.offense)
        - Take top 5, but enforce: at least 1 defenseman and at most 2 defensemen
        - Goalies are excluded

        If constraints cannot be perfectly satisfied due to roster makeup, the method
        returns the best feasible selection given availability.
        """
        unavailable = unavailable or set()
        all_skaters = [p for p in self.roster if p.position != "G"]
        skaters = [p for p in all_skaters if p not in unavailable]
        if not skaters:
            return []

        # Determine target size based on number unavailable among skaters
        num_unavailable = sum(1 for p in all_skaters if p in unavailable)
        target_size = max(0, 5 - num_unavailable)
        if target_size == 0:
            return []

        skaters_sorted_off = sorted(skaters, key=lambda p: p.offense, reverse=True)
        selected: List[Player] = skaters_sorted_off[:target_size]

        # If fewer than target_size skaters are available, return what we have while trying
        # to satisfy constraints below where possible.
        max_size = min(target_size, len(skaters_sorted_off))
        selected = selected[:max_size]

        def is_def(player: Player) -> bool:
            return player.position == "D"

        def_count = sum(1 for p in selected if is_def(p))

        # Ensure at least 1 defenseman if any D are available overall.
        if def_count == 0:
            available_ds = [p for p in skaters_sorted_off if is_def(p) and p not in selected]
            if available_ds:
                # Replace the lowest-offense forward with the best available D
                forwards_in_selected = [p for p in selected if p.position == "F"]
                if forwards_in_selected:
                    lowest_fwd = min(forwards_in_selected, key=lambda p: p.offense)
                    best_d = available_ds[0]
                    # swap
                    selected.remove(lowest_fwd)
                    selected.append(best_d)
                    def_count = 1

        # Cap defensemen at 2 by replacing lowest-offense D with best available F
        while def_count > 2:
            ds_in_selected = sorted([p for p in selected if is_def(p)], key=lambda p: p.offense)
            forwards_available = [p for p in skaters_sorted_off if p.position == "F" and p not in selected]
            if not ds_in_selected or not forwards_available:
                break
            lowest_d = ds_in_selected[0]
            best_f = forwards_available[0]
            selected.remove(lowest_d)
            selected.append(best_f)
            def_count -= 1

        # If still no D (e.g., roster has 0 Ds), we accept the forward-only selection.

        # If we have capacity (<target_size due to availability), try to fill remaining with best offense
        # while respecting max 2 defensemen.
        if len(selected) < target_size:
            for candidate in skaters_sorted_off:
                if candidate in selected:
                    continue
                if candidate.position == "D" and def_count >= 2:
                    continue
                selected.append(candidate)
                if candidate.position == "D":
                    def_count += 1
                if len(selected) == target_size:
                    break

        # Return sorted by offense for readability/consistency
        return sorted(selected, key=lambda p: p.offense, reverse=True)

    def penalty_kill_unit(self, unavailable: Optional[Set[Player]] = None, num_skaters: int = 4) -> List[Player]:
        """Select a penalty kill unit with defensive constraints.

        Logic:
        - Rank all available skaters by defensive contribution (Player.defense)
        - Target size is (5 - number of unavailable skaters)
          (e.g., 4 when one penalized, 3 when two)
        - Enforce: at least 2 defensemen (if possible) and at most 3 defensemen
        - Goalies are excluded
        """
        unavailable = unavailable or set()
        all_skaters = [p for p in self.roster if p.position != "G"]
        skaters = [p for p in all_skaters if p not in unavailable]
        if not skaters:
            return []

        # Determine target size based on number unavailable among skaters
        num_unavailable = sum(1 for p in all_skaters if p in unavailable)
        target_size = max(0, 5 - num_unavailable)
        if target_size == 0:
            return []

        skaters_sorted_def = sorted(skaters, key=lambda p: p.defense, reverse=True)
        max_size = min(target_size, len(skaters_sorted_def))
        selected: List[Player] = skaters_sorted_def[:max_size]

        def is_def(player: Player) -> bool:
            return player.position == "D"

        def_count = sum(1 for p in selected if is_def(p))

        # Target counts given constraints and availability
        total_ds_available = sum(1 for p in skaters if is_def(p))
        min_ds = 2 if total_ds_available >= 2 else total_ds_available
        max_ds = min(3, max_size)

        # Ensure at least min_ds defensemen
        while def_count < min_ds:
            available_ds = [p for p in skaters_sorted_def if is_def(p) and p not in selected]
            if not available_ds:
                break
            # Replace the lowest-defense forward, if any
            forwards_in_selected = sorted([p for p in selected if p.position == "F"], key=lambda p: p.defense)
            if not forwards_in_selected:
                # If we have only defensemen selected but def_count < min_ds due to size, just add if capacity
                if len(selected) < max_size:
                    selected.append(available_ds[0])
                    def_count += 1
                break
            lowest_fwd = forwards_in_selected[0]
            best_d = available_ds[0]
            selected.remove(lowest_fwd)
            selected.append(best_d)
            def_count += 1

        # Cap at max_ds defensemen by replacing lowest-defense D with best available F
        while def_count > max_ds:
            ds_in_selected = sorted([p for p in selected if is_def(p)], key=lambda p: p.defense)
            forwards_available = [p for p in skaters_sorted_def if p.position == "F" and p not in selected]
            if not ds_in_selected or not forwards_available:
                break
            lowest_d = ds_in_selected[0]
            best_f = forwards_available[0]
            selected.remove(lowest_d)
            selected.append(best_f)
            def_count -= 1

        # Fill if we have fewer than requested due to availability
        if len(selected) < max_size:
            for candidate in skaters_sorted_def:
                if candidate in selected:
                    continue
                if candidate.position == "D" and def_count >= max_ds:
                    continue
                selected.append(candidate)
                if candidate.position == "D":
                    def_count += 1
                if len(selected) == max_size:
                    break

        # Return sorted by defense for readability/consistency
        return sorted(selected, key=lambda p: p.defense, reverse=True)

class League:
    """A collection of teams with basic management operations."""
    def __init__(self, teams: List[Team]):
        self.teams = list(teams) if teams is not None else []
        self.schedule: List[Tuple[Team, Team]] = []
        self.weeks: List[List[Tuple[Team, Team]]] = []

    def add_team(self, team: Team) -> None:
        self.teams.append(team)

    def remove_team(self, team: Team) -> None:
        self.teams.remove(team)

    def build_schedule(self, shuffle: bool = True, seed: Optional[int] = None, group_weeks: bool = True) -> List[Tuple[Team, Team]]:
        """Create a double round-robin schedule (home/away) for all teams."""
        games: List[Tuple[Team, Team]] = []
        n = len(self.teams)
        for i in range(n):
            for j in range(i + 1, n):
                games.append((self.teams[i], self.teams[j]))
                games.append((self.teams[j], self.teams[i]))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(games)
        self.schedule = games
        if group_weeks:
            self.weeks = self.build_weeks(self.schedule)
        return games

    def player_rankings(self, top_n: int, key: str) -> List[Dict]:
        """Return top N players across the league sorted by the given key.

        key should be one of: 'offense', 'defense', 'stamina', 'discipline', or 'total'.
        """
        rows: List[Dict] = []
        for team in self.teams:
            for p in team.roster:
                rows.append({
                    'team': team.name,
                    'player': p.name,
                    'position': p.position,
                    'offense': p.offense,
                    'defense': p.defense,
                    'stamina': p.stamina,
                    'discipline': p.discipline,
                    'total': p.offense + p.defense
                })
        if key not in {'offense','defense','stamina','discipline','total'}:
            key = 'offense'
        rows.sort(key=lambda r: r[key], reverse=True)
        return rows[:top_n]

    def get_teams(self) -> List[Dict]:
        """Return per-team summary: coach, playstyle, and sums of roster attributes."""
        out: List[Dict] = []
        for team in self.teams:
            off = sum(p.offense for p in team.roster)
            deff = sum(p.defense for p in team.roster)
            sta = sum(p.stamina for p in team.roster)
            dis = sum(p.discipline for p in team.roster)
            out.append({
                'team': team.name,
                'coach': team.coach.name,
                'playstyle': team.coach.playstyle,
                'offense_sum': off,
                'defense_sum': deff,
                'stamina_sum': sta,
                'discipline_sum': dis,
                'total_sum': off + deff
            })
        return out

    def simulate_game(self, home: Team, away: Team) -> Dict:
        """Simulate a single game between two teams and return the Game result dict."""
        g = Game(home, away)
        return g.simulate_game()

    def build_weeks(self, schedule: Optional[List[Tuple[Team, Team]]] = None) -> List[List[Tuple[Team, Team]]]:
        """Pack a schedule into matchweeks such that teams play at most once per week.

        Returns a list of weeks, where each week is a list of (home, away) tuples.
        """
        games = list(schedule) if schedule is not None else list(self.schedule)
        weeks: List[List[Tuple[Team, Team]]] = []
        remaining: List[Tuple[Team, Team]] = games
        while remaining:
            week: List[Tuple[Team, Team]] = []
            used: Set[Team] = set()
            nxt: List[Tuple[Team, Team]] = []
            for home, away in remaining:
                if (home in used) or (away in used):
                    nxt.append((home, away))
                    continue
                week.append((home, away))
                used.add(home)
                used.add(away)
            weeks.append(week)
            remaining = nxt
        return weeks

    def simulate_schedule(
        self,
        weeks: Optional[Union[List[int], int, Tuple[int, int], List[List[Tuple[Team, Team]]]]] = None,
        start_week: Optional[int] = None,
        end_week: Optional[int] = None,
        pbp_weeks: int = 0
    ) -> Tuple[List[Dict], List[Dict]]:
        """Simulate a season over the given weeks.

        Args:
            weeks: Optional pre-built weeks. If None, uses self.build_weeks().
            pbp_weeks: Export play-by-play rows for the first pbp_weeks weeks.

        Returns:
            (game_rows, pbp_rows) where game_rows contains one row per game with
            basic results, and pbp_rows contains play-by-play for the requested weeks.
        """

        game_rows: List[Dict] = []
        pbp_rows: List[Dict] = []

        # Determine base week matrix
        week_matrix: List[List[Tuple[Team, Team]]]
        if isinstance(weeks, list) and weeks and isinstance(weeks[0], list):
            week_matrix = weeks  # already a matrix
        else:
            week_matrix = self.weeks if self.weeks else self.build_weeks()

        # Determine which week indices to simulate (1-based)
        if isinstance(weeks, int):
            selected = [weeks]
        elif isinstance(weeks, tuple) and len(weeks) == 2:
            selected = list(range(weeks[0], weeks[1] + 1))
        elif isinstance(weeks, list) and weeks and isinstance(weeks[0], int):
            selected = list(weeks)
        else:
            if start_week is not None or end_week is not None:
                s = start_week if start_week is not None else 1
                e = end_week if end_week is not None else len(week_matrix)
                selected = list(range(s, e + 1))
            else:
                selected = list(range(1, len(week_matrix) + 1))

        # Keep only valid indices
        selected = [w for w in selected if 1 <= w <= len(week_matrix)]

        game_id = 0
        for wk_idx in selected:
            week = week_matrix[wk_idx - 1]
            for home, away in week:
                game_id += 1
                g = Game(home, away)
                result = g.simulate_game()
                home_name = home.name
                away_name = away.name
                hs = result['home_score']
                as_ = result['away_score']
                went_ot = any(e[1] == 'overtime_start' for e in g.events)
                winner = home_name if hs > as_ else away_name
                loser = away_name if hs > as_ else home_name

                game_rows.append({
                    'game_id': game_id,
                    'week': wk_idx,
                    'home_team': home_name,
                    'away_team': away_name,
                    'home_score': hs,
                    'away_score': as_,
                    'went_ot': int(went_ot),
                    'winner': winner,
                    'loser': loser
                })

                if (len(selected) and (wk_idx - min(selected) + 1) <= pbp_weeks) or (not selected and wk_idx <= pbp_weeks):
                    for e in g.events:
                        t = e[0]
                        et = e[1]
                        desc = e[2] if len(e) >= 3 else ''
                        hs_e = e[3] if len(e) >= 4 else None
                        as_e = e[4] if len(e) >= 5 else None
                        tag = e[5] if len(e) >= 6 else ''
                        home_on_ice = e[6] if len(e) >= 7 else []
                        away_on_ice = e[7] if len(e) >= 8 else []
                        period = int(t // 1200.0) + 1
                        pbp_rows.append({
                            'game_id': game_id,
                            'week': wk_idx,
                            'home_team': home_name,
                            'away_team': away_name,
                            'period': period,
                            'time_seconds': f"{t:.2f}",
                            'event_type': et,
                            'description': desc,
                            'home_score': hs_e,
                            'away_score': as_e,
                            'tag': tag,
                            'home_on_ice': '|'.join(home_on_ice) if home_on_ice else '',
                            'away_on_ice': '|'.join(away_on_ice) if away_on_ice else ''
                        })

        return game_rows, pbp_rows

    def get_standings(
        self,
        game_rows: List[Dict],
        by_week: bool = False,
        through_week: Optional[int] = None
    ) -> Union[Dict[str, Dict], Dict[int, List[Dict]]]:
        """Compute NHL-style standings from game rows.

        If by_week is False, returns a dict keyed by team name with cumulative totals.
        If by_week is True, returns a dict mapping week -> list of standings rows as of that week.
        """
        # Initialize accumulator
        base = {
            'GP': 0, 'W': 0, 'OTW': 0, 'L': 0, 'OTL': 0, 'PTS': 0, 'GF': 0, 'GA': 0
        }
        teams = [t.name for t in self.teams]
        cumulative: Dict[str, Dict] = {name: dict(base) for name in teams}

        # Optionally filter to games up through a given week
        rows_iter = game_rows
        if through_week is not None:
            rows_iter = [r for r in game_rows if int(r['week']) <= int(through_week)]

        if not by_week:
            for row in rows_iter:
                h = row['home_team']; a = row['away_team']
                hs = row['home_score']; as_ = row['away_score']
                ot = bool(row['went_ot'])
                cumulative[h]['GP'] += 1
                cumulative[a]['GP'] += 1
                cumulative[h]['GF'] += hs; cumulative[h]['GA'] += as_
                cumulative[a]['GF'] += as_; cumulative[a]['GA'] += hs
                if hs > as_:
                    if ot:
                        cumulative[h]['OTW'] += 1; cumulative[h]['PTS'] += 2
                        cumulative[a]['OTL'] += 1; cumulative[a]['PTS'] += 1
                    else:
                        cumulative[h]['W'] += 1; cumulative[h]['PTS'] += 2
                        cumulative[a]['L'] += 1
                else:
                    if ot:
                        cumulative[a]['OTW'] += 1; cumulative[a]['PTS'] += 2
                        cumulative[h]['OTL'] += 1; cumulative[h]['PTS'] += 1
                    else:
                        cumulative[a]['W'] += 1; cumulative[a]['PTS'] += 2
                        cumulative[h]['L'] += 1
            return cumulative

        # by_week cumulative snapshots
        snapshots: Dict[int, List[Dict]] = {}
        last_week = 0
        for row in rows_iter:
            wk = int(row['week'])
            # advance snapshots for skipped weeks if any
            for w in range(last_week + 1, wk):
                # store a deep-ish copy snapshot
                snap_rows: List[Dict] = []
                for team_name, s in cumulative.items():
                    gd = s['GF'] - s['GA']
                    snap_rows.append({
                        'team': team_name,
                        'GP': s['GP'], 'W': s['W'], 'OTW': s['OTW'], 'L': s['L'], 'OTL': s['OTL'],
                        'PTS': s['PTS'], 'GF': s['GF'], 'GA': s['GA'], 'GD': gd
                    })
                snap_rows.sort(key=lambda r: (r['PTS'], r['GD'], r['GF']), reverse=True)
                snapshots[w] = snap_rows
            last_week = wk

            # apply this game
            h = row['home_team']; a = row['away_team']
            hs = row['home_score']; as_ = row['away_score']
            ot = bool(row['went_ot'])
            cumulative[h]['GP'] += 1
            cumulative[a]['GP'] += 1
            cumulative[h]['GF'] += hs; cumulative[h]['GA'] += as_
            cumulative[a]['GF'] += as_; cumulative[a]['GA'] += hs
            if hs > as_:
                if ot:
                    cumulative[h]['OTW'] += 1; cumulative[h]['PTS'] += 2
                    cumulative[a]['OTL'] += 1; cumulative[a]['PTS'] += 1
                else:
                    cumulative[h]['W'] += 1; cumulative[h]['PTS'] += 2
                    cumulative[a]['L'] += 1
            else:
                if ot:
                    cumulative[a]['OTW'] += 1; cumulative[a]['PTS'] += 2
                    cumulative[h]['OTL'] += 1; cumulative[h]['PTS'] += 1
                else:
                    cumulative[a]['W'] += 1; cumulative[a]['PTS'] += 2
                    cumulative[h]['L'] += 1

            # snapshot at end of this week if this was the week's last game will be done below
            # We cannot know directly without week structure; simpler: after loop, ensure final week snapshot

        # ensure final week snapshot present
        if game_rows:
            max_week = max(int(r['week']) for r in game_rows)
            for w in range(1, max_week + 1):
                if w not in snapshots:
                    snap_rows = []
                    for team_name, s in cumulative.items():
                        gd = s['GF'] - s['GA']
                        snap_rows.append({
                            'team': team_name,
                            'GP': s['GP'], 'W': s['W'], 'OTW': s['OTW'], 'L': s['L'], 'OTL': s['OTL'],
                            'PTS': s['PTS'], 'GF': s['GF'], 'GA': s['GA'], 'GD': gd
                        })
                    snap_rows.sort(key=lambda r: (r['PTS'], r['GD'], r['GF']), reverse=True)
                    snapshots[w] = snap_rows

        return snapshots

class Period:
    """Represents a single period of a hockey game."""
    def __init__(self, period_number: int, duration_seconds: float, is_overtime: bool = False):
        self.period_number = period_number
        self.duration_seconds = duration_seconds
        self.is_overtime = is_overtime
        self.start_time = 0.0
        self.end_time = 0.0
        self.events = []
    
    def start_period(self, current_time: float) -> None:
        """Start the period at the given time."""
        self.start_time = current_time
        self.end_time = current_time + self.duration_seconds
        self.events.append((current_time, 'period_start', f'Start of Period {self.period_number}'))
    
    def end_period(self, current_time: float) -> None:
        """End the period at the given time."""
        self.events.append((current_time, 'period_end', f'End of Period {self.period_number}'))
    
    def is_finished(self, current_time: float) -> bool:
        """Check if the period is finished."""
        return current_time >= self.end_time

class Game:
    """Simulates a game between two teams using shift-based Poisson events."""
    def __init__(self, home_team: Team, away_team: Team):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.current_time = 0.0
        self.current_period = None
        self.home_on_ice = None
        self.away_on_ice = None
        self.home_penalties = []  # List of (player, time_remaining)
        self.away_penalties = []
        self.events = []  # List of (time, event_type, description)
        # Track current line/pair for rotation
        self.home_line_id = 1
        self.home_pair_id = 1
        self.away_line_id = 1
        self.away_pair_id = 1
        # Line-change deadlines (absolute seconds since game start)
        self.home_fwd_deadline = 0.0
        self.home_def_deadline = 0.0
        self.away_fwd_deadline = 0.0
        self.away_def_deadline = 0.0
        # Penalty / special-teams state
        self.penalized_team: Optional[str] = None  # set when one side is shorthanded
        # Active penalties per side: list of dicts with current segment end and metadata
        # {'player': Player|None, 'type': 'minor'|'double_minor'|'major', 'segments_left': int, 'segment_end': float}
        self.home_penalties: List[Dict] = []
        self.away_penalties: List[Dict] = []
        self.unavailable_home: Set[Player] = set()
        self.unavailable_away: Set[Player] = set()
        self.special_home_skaters: Optional[List[Player]] = None
        self.special_away_skaters: Optional[List[Player]] = None
        # Pulled goalie state: 'home'|'away'|None
        self.pulled_team: Optional[str] = None
    
    def start_shift(self) -> None:
        """Initialize a fresh shift: rebuild on-ice and resample all four clocks."""
        self._rebuild_on_ice_caches()
        self._resample_clocks()

    def _current_on_ice_names(self) -> Tuple[List[str], List[str]]:
        """Return current on-ice player name lists for home and away."""
        home_names = [p.name for p in getattr(self, 'home_players', [])]
        away_names = [p.name for p in getattr(self, 'away_players', [])]
        return home_names, away_names

    def _rebuild_on_ice_caches(self) -> None:
        """Rebuild on-ice units and caches without touching line-change clocks."""
        home_override = self.special_home_skaters
        away_override = self.special_away_skaters
        # If goalie is pulled for a side, construct a 6-skater unit using best extra attacker
        if self.pulled_team == 'home':
            home_override = self._compute_pulled_skaters('home')
        elif self.pulled_team == 'away':
            away_override = self._compute_pulled_skaters('away')
        self.home_players = self._build_on_ice_players(self.home_team, self.home_line_id, self.home_pair_id, self.home_team.goalies()[0], home_override)
        self.away_players = self._build_on_ice_players(self.away_team, self.away_line_id, self.away_pair_id, self.away_team.goalies()[0], away_override)
        # Cache scalars used in rate calculations
        self._home_off_sum = sum(p.offense for p in self.home_players if p.position != 'G')
        self._home_def_sum = sum(p.defense for p in self.home_players)
        self._away_off_sum = sum(p.offense for p in self.away_players if p.position != 'G')
        self._away_def_sum = sum(p.defense for p in self.away_players)
        # Discipline multipliers (normalized)
        hw = [math.exp(-PEN_BETA * p.discipline) for p in self.home_players if p.position != 'G']
        aw = [math.exp(-PEN_BETA * p.discipline) for p in self.away_players if p.position != 'G']
        self._home_pen_mult = (sum(hw)/len(hw) if hw else 1.0) / PEN_NORM
        self._away_pen_mult = (sum(aw)/len(aw) if aw else 1.0) / PEN_NORM
        self._home_pen_mult = min(PEN_CLAMP_HI, max(PEN_CLAMP_LO, self._home_pen_mult))
        self._away_pen_mult = min(PEN_CLAMP_HI, max(PEN_CLAMP_LO, self._away_pen_mult))

    def _resample_clocks(self, unit: Optional[str] = None) -> None:
        """Resample line-change clocks. If unit is None, resample all; else just that unit."""
        import random as _r
        # Determine if special teams is active (any side not at 5 skaters)
        active_home = sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time)
        active_away = sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time)
        special = (self.pulled_team is not None) or not (active_home == 0 and active_away == 0)
        fwd_low, fwd_high = (50.0, 70.0) if special else (30.0, 60.0)
        def_low, def_high = ((100.0, 120.0) if special else (40.0, 60.0))
        if unit is None:
            # Set absolute deadlines from current time
            self.home_fwd_deadline = self.current_time + _r.uniform(fwd_low, fwd_high)
            self.home_def_deadline = self.current_time + _r.uniform(def_low, def_high)
            self.away_fwd_deadline = self.current_time + _r.uniform(fwd_low, fwd_high)
            self.away_def_deadline = self.current_time + _r.uniform(def_low, def_high)
            return
        if unit == 'home_forward':
            self.home_fwd_deadline = self.current_time + _r.uniform(fwd_low, fwd_high)
        elif unit == 'home_defense':
            self.home_def_deadline = self.current_time + _r.uniform(def_low, def_high)
        elif unit == 'away_forward':
            self.away_fwd_deadline = self.current_time + _r.uniform(fwd_low, fwd_high)
        elif unit == 'away_defense':
            self.away_def_deadline = self.current_time + _r.uniform(def_low, def_high)

    def _enter_power_play(self, offending_team: str) -> None:
        """Initialize special teams units and penalty timer for a new minor penalty."""
        import random as _r
        # Sample penalty type per requested proportions
        r = random.random()
        if r < 0.985:
            ptype = 'minor'
            duration = 120.0
            segments = 1
        elif r < 0.985 + 0.010:
            ptype = 'double_minor'
            duration = 120.0
            segments = 2  # two consecutive minors
        else:
            ptype = 'major'
            duration = 300.0
            segments = 1
        segment_end = self.current_time + duration
        # Choose a penalized skater from current on-ice, fallback to roster
        if offending_team == 'home':
            penalized = self._select_penalized_skater('home')
            self.home_penalties.append({'player': penalized, 'type': ptype, 'segments_left': segments, 'segment_end': segment_end})
            if penalized is not None:
                self.unavailable_home.add(penalized)
            # PK for home, PP for away
            self.special_home_skaters = self.home_team.penalty_kill_unit(unavailable=self.unavailable_home)
            self.special_away_skaters = self.away_team.power_play_unit(unavailable=self.unavailable_away)
            # Label 4v4 or 3v3 correctly
            n_home = max(3, 5 - sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time))
            n_away = max(3, 5 - sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time))
            h_oi, a_oi = self._current_on_ice_names()
            if n_home == n_away and n_home < 5:
                self.events.append((self.current_time, 'pp_start', f'{n_home}v{n_away} starts', self.home_score, self.away_score, f'{n_home}v{n_away}', h_oi, a_oi))
            else:
                self.events.append((self.current_time, 'pp_start', 'Away power play starts (home shorthanded)', self.home_score, self.away_score, 'away_pp', h_oi, a_oi))
        else:
            penalized = self._select_penalized_skater('away')
            self.away_penalties.append({'player': penalized, 'type': ptype, 'segments_left': segments, 'segment_end': segment_end})
            if penalized is not None:
                self.unavailable_away.add(penalized)
            # PK for away, PP for home
            self.special_home_skaters = self.home_team.power_play_unit(unavailable=self.unavailable_home)
            self.special_away_skaters = self.away_team.penalty_kill_unit(unavailable=self.unavailable_away)
            n_home = max(3, 5 - sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time))
            n_away = max(3, 5 - sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time))
            h_oi, a_oi = self._current_on_ice_names()
            if n_home == n_away and n_home < 5:
                self.events.append((self.current_time, 'pp_start', f'{n_home}v{n_away} starts', self.home_score, self.away_score, f'{n_home}v{n_away}', h_oi, a_oi))
            else:
                self.events.append((self.current_time, 'pp_start', 'Home power play starts (away shorthanded)', self.home_score, self.away_score, 'home_pp', h_oi, a_oi))

    def _end_one_penalty(self, side: str) -> None:
        """Expire the oldest minor for the given side ('home' or 'away')."""
        if side == 'home' and self.home_penalties:
            # oldest = with smallest segment_end > current_time
            active = [p for p in self.home_penalties if p['segment_end'] > self.current_time]
            if active:
                oldest = min(active, key=lambda d: d['segment_end'])
                if oldest['type'] == 'major':
                    # Majors do not end on PP goals
                    pass
                elif oldest['type'] == 'double_minor' and oldest['segments_left'] > 1:
                    oldest['segments_left'] -= 1
                    oldest['segment_end'] = self.current_time + 120.0
                else:
                    # remove this penalty
                    self.home_penalties.remove(oldest)
                    p = oldest.get('player')
                    if p is not None:
                        self.unavailable_home.discard(p)
        elif side == 'away' and self.away_penalties:
            active = [p for p in self.away_penalties if p['segment_end'] > self.current_time]
            if active:
                oldest = min(active, key=lambda d: d['segment_end'])
                if oldest['type'] == 'major':
                    pass
                elif oldest['type'] == 'double_minor' and oldest['segments_left'] > 1:
                    oldest['segments_left'] -= 1
                    oldest['segment_end'] = self.current_time + 120.0
                else:
                    self.away_penalties.remove(oldest)
                    p = oldest.get('player')
                    if p is not None:
                        self.unavailable_away.discard(p)
        # If no active penalties remain for a side, log full strength
        # Announce correct state at this moment
        n_home = max(3, 5 - sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time))
        n_away = max(3, 5 - sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time))
        h_oi, a_oi = self._current_on_ice_names()
        if n_home == 5 and n_away == 5:
            self.events.append((self.current_time, 'pp_end', 'Back to full strength (5v5)', self.home_score, self.away_score, 'full', h_oi, a_oi))
        elif n_home == n_away:
            self.events.append((self.current_time, 'pp_end', f'{n_home}v{n_away} continues', self.home_score, self.away_score, f'{n_home}v{n_away}', h_oi, a_oi))
        elif n_home > n_away:
            self.events.append((self.current_time, 'pp_end', 'Home power play begins', self.home_score, self.away_score, 'home_pp', h_oi, a_oi))
        else:
            self.events.append((self.current_time, 'pp_end', 'Away power play begins', self.home_score, self.away_score, 'away_pp', h_oi, a_oi))
        # Recompute special units based on remaining penalties
        self._recompute_special_units()

    def _recompute_special_units(self) -> None:
        """Set special team skaters based on current penalty stacks."""
        active_home = sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time)
        active_away = sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time)
        n_home = max(3, 5 - active_home)
        n_away = max(3, 5 - active_away)
        self.penalized_team = None
        # Even strength
        if n_home == 5 and n_away == 5:
            self.special_home_skaters = None
            self.special_away_skaters = None
            return
        # Home advantage (home has more skaters)
        if n_home > n_away:
            self.penalized_team = 'away'
            self.special_home_skaters = self.home_team.power_play_unit(unavailable=self.unavailable_home)
            # Away PK size determined by n_away
            self.special_away_skaters = self.away_team.penalty_kill_unit(unavailable=self.unavailable_away, num_skaters=n_away)
        elif n_away > n_home:
            self.penalized_team = 'home'
            self.special_away_skaters = self.away_team.power_play_unit(unavailable=self.unavailable_away)
            self.special_home_skaters = self.home_team.penalty_kill_unit(unavailable=self.unavailable_home, num_skaters=n_home)
        else:
            # 4v4 or 3v3: choose balanced units (use PK selector for target size)
            self.special_home_skaters = self.home_team.penalty_kill_unit(unavailable=self.unavailable_home, num_skaters=n_home)
            self.special_away_skaters = self.away_team.penalty_kill_unit(unavailable=self.unavailable_away, num_skaters=n_away)

    def _rotate_forward(self, team: str) -> None:
        """Advance forward line for given team id pointer."""
        if team == 'home':
            self.home_line_id = (self.home_line_id % len(self.home_team.lines)) + 1
        else:
            self.away_line_id = (self.away_line_id % len(self.away_team.lines)) + 1

    def _rotate_defense(self, team: str) -> None:
        """Advance defensive pair for given team id pointer."""
        if team == 'home':
            self.home_pair_id = (self.home_pair_id % len(self.home_team.pairs)) + 1
        else:
            self.away_pair_id = (self.away_pair_id % len(self.away_team.pairs)) + 1

    def _select_penalized_skater(self, team: str) -> Optional[Player]:
        """Select a penalized skater on-ice weighted by indiscipline (lower discipline → higher chance).

        We transform discipline (standard normal) to nonnegative weights with a softplus-like mapping:
        weight = 1 + max(0, -discipline).
        If no on-ice skaters are available (should not happen), fallback to roster with same logic.
        """
        import random as _r
        if team == 'home':
            pool = [p for p in self.home_players if p.position != 'G'] if hasattr(self, 'home_players') and self.home_players else (self.home_team.forwards() + self.home_team.defensemen())
        else:
            pool = [p for p in self.away_players if p.position != 'G'] if hasattr(self, 'away_players') and self.away_players else (self.away_team.forwards() + self.away_team.defensemen())
        if not pool:
            return None
        # Exponential weighting across full range so every discipline point matters
        beta = 0.55  # exp(+/- 2*beta) ≈ 3x odds across 2 SDs
        weights = [float(__import__('math').exp(-beta * p.discipline)) for p in pool]
        total = sum(weights)
        if total <= 0:
            return _r.choice(pool)
        r = _r.random() * total
        acc = 0.0
        for p, w in zip(pool, weights):
            acc += w
            if r <= acc:
                return p
        return pool[-1]
    
    def _build_on_ice_players(self, team: 'Team', f_line_id: int, d_pair_id: int, goalie: Player, skaters_override: Optional[List[Player]] = None) -> List[Player]:
        """Compose the on-ice unit: 5 skaters + goalie, honoring special-teams overrides."""
        if skaters_override:
            # If this side has the goalie pulled and an explicit 6-skater list is provided,
            # do not add the goalie back in.
            if (team is self.home_team and self.pulled_team == 'home' and len(skaters_override) >= 6) or \
               (team is self.away_team and self.pulled_team == 'away' and len(skaters_override) >= 6):
                return list(skaters_override)
            return list(skaters_override) + [goalie]
        forwards = team.lines[f_line_id]
        defensemen = team.pairs[d_pair_id]
        return forwards + defensemen + [goalie]

    def _compute_pulled_skaters(self, side: str) -> List[Player]:
        """Return 6 skaters for the pulled-goalie side using PP-style logic with an extra attacker.

        Base the unit on current special-team overrides if present; otherwise use current line+pair.
        Then add the best available offensive skater from the roster who is not already on the ice.
        """
        if side == 'home':
            team = self.home_team
            base = list(self.special_home_skaters) if self.special_home_skaters else (self.home_team.lines[self.home_line_id] + self.home_team.pairs[self.home_pair_id])
        else:
            team = self.away_team
            base = list(self.special_away_skaters) if self.special_away_skaters else (self.away_team.lines[self.away_line_id] + self.away_team.pairs[self.away_pair_id])
        base = [p for p in base if p.position != 'G']
        # Candidate pool: all skaters not already on-ice
        candidates = [p for p in team.roster if p.position != 'G' and p not in base]
        extra = max(candidates, key=lambda p: p.offense) if candidates else []
        if extra:
            unit = base + [extra]
        else:
            unit = base
        # Ensure size is at most 6
        return unit[:6]
    
    def calculate_event_rates(self) -> Dict[str, float]:
        """Calculate all event rates for current on-ice situation."""
        # Use cached on-ice sums from shift start
        home_off = self._home_off_sum
        home_def = self._home_def_sum
        away_off = self._away_off_sum
        away_def = self._away_def_sum
        
        # Baselines per manpower state (per second per team)
        base_5v5 = BASE_5V5
        base_pp = BASE_PP_5V4
        base_sh = BASE_SH_4V5
        base_5v3_adv = BASE_PP_5V3
        base_3v5_def = BASE_SH_3V5

        # Home-ice advantage (per second): split into for/against effects.
        home_off_bonus = HFA_OFF
        home_def_bonus = HFA_DEF
        
        # Select baselines based on current manpower (supports stacks, pulled goalie)
        active_home = sum(1 for e in getattr(self, 'home_penalty_expires', []) if e > self.current_time)
        active_away = sum(1 for e in getattr(self, 'away_penalty_expires', []) if e > self.current_time)
        n_home = max(3, 5 - active_home)
        n_away = max(3, 5 - active_away)
        # Pulled goalie overrides (late game): 6v5 or 5v6, and 6v4/4v6 when penalties
        if self.pulled_team == 'home':
            n_home = 6
        elif self.pulled_team == 'away':
            n_away = 6

        if n_home == 5 and n_away == 5:
            base_home = base_5v5
            base_away = base_5v5
        elif n_home == 5 and n_away == 4:
            base_home = base_pp
            base_away = base_sh
        elif n_home == 4 and n_away == 5:
            base_home = base_sh
            base_away = base_pp
        elif n_home == 5 and n_away == 3:
            base_home = base_5v3_adv
            base_away = base_3v5_def
        elif n_home == 3 and n_away == 5:
            base_home = base_3v5_def
            base_away = base_5v3_adv
        elif n_home == 6 and n_away == 5:
            base_home = BASE_6V5_FOR
            base_away = BASE_5V6_FOR
        elif n_home == 5 and n_away == 6:
            base_home = BASE_5V6_FOR
            base_away = BASE_6V5_FOR
        elif n_home == 6 and n_away == 4:
            base_home = BASE_6V4_FOR
            base_away = BASE_4V6_FOR
        elif n_home == 4 and n_away == 6:
            base_home = BASE_4V6_FOR
            base_away = BASE_6V4_FOR
        else:
            # 4v4, 3v3: treat as even-strength baseline
            base_home = base_5v5
            base_away = base_5v5

        # Apply home-ice advantage: boost home scoring; suppress away scoring
        base_home += home_off_bonus
        base_away = max(0.0, base_away - home_def_bonus)

        # Goal rate = baseline + (team offense - opponent defense)
        lambda_home_goal = max(TINY, base_home + STRENGTH_SCALE * home_off - STRENGTH_SCALE * away_def)
        lambda_away_goal = max(TINY, base_away + STRENGTH_SCALE * away_off - STRENGTH_SCALE * home_def)
        
        # Penalty rates: base 6 per team per 60 minutes, modulated by on-ice discipline
        # Use normalized exponential weights so league average stays centered at 1.
        base_pen = PEN_BASE
        lambda_home_pen = base_pen * self._home_pen_mult
        lambda_away_pen = base_pen * self._away_pen_mult
        
        return {
            'home_goal': lambda_home_goal,
            'away_goal': lambda_away_goal,
            'home_penalty': lambda_home_pen,
            'away_penalty': lambda_away_pen
        }
    
    def simulate_shift(self) -> None:
        """Simulate one shift using Poisson events."""
        # Late-game pulled-goalie strategy per spec using period 3 and period_remaining ≤ 120
        # Compute precise remaining time in current period using Period.end_time.
        if self.current_period is not None:
            period_remaining = max(0.0, self.current_period.end_time - self.current_time)
        else:
            period_remaining = 0.0
        if (self.current_period and self.current_period.period_number == 3) and period_remaining <= 120.0:
            diff = self.home_score - self.away_score
            if self.pulled_team is None:
                if diff < 0 and abs(diff) <= 2:
                    self.pulled_team = 'home'
                    # Log goalie pulled
                    h_oi, a_oi = self._current_on_ice_names()
                    self.events.append((self.current_time, 'goalie_pulled', 'Home pulls the goalie for an extra attacker', self.home_score, self.away_score, 'home_pulled', h_oi, a_oi))
                elif diff > 0 and abs(diff) <= 2:
                    self.pulled_team = 'away'
                    # Log goalie pulled
                    h_oi, a_oi = self._current_on_ice_names()
                    self.events.append((self.current_time, 'goalie_pulled', 'Away pulls the goalie for an extra attacker', self.home_score, self.away_score, 'away_pulled', h_oi, a_oi))
                if self.pulled_team is not None:
                    self._rebuild_on_ice_caches()
        
        rates = self.calculate_event_rates()
        # Only stochastic (goal/penalty) compete with deterministic line-change/period boundaries
        total_rate = rates['home_goal'] + rates['away_goal'] + rates['home_penalty'] + rates['away_penalty']
        
        if total_rate <= 0:
            return
        
        # Sample next stochastic event time (goals/penalties)
        import random
        delta_event = random.expovariate(total_rate) if total_rate > 0 else float('inf')
        
        # Deterministic boundaries: line changes (uniform windows) and period end
        # Compute remaining times from absolute deadlines
        rem_hf = max(0.0, self.home_fwd_deadline - self.current_time)
        rem_hd = max(0.0, self.home_def_deadline - self.current_time)
        rem_af = max(0.0, self.away_fwd_deadline - self.current_time)
        rem_ad = max(0.0, self.away_def_deadline - self.current_time)
        boundaries = [
            ('home_forward', rem_hf),
            ('home_defense', rem_hd),
            ('away_forward', rem_af),
            ('away_defense', rem_ad),
            ('period_end', period_remaining)
        ]
        # Add earliest penalty expiration boundary for each side if active
        future_home = [p['segment_end'] for p in self.home_penalties if p['segment_end'] > self.current_time]
        future_away = [p['segment_end'] for p in self.away_penalties if p['segment_end'] > self.current_time]
        if future_home:
            boundaries.append(('penalty_end_home', min(future_home) - self.current_time))
        if future_away:
            boundaries.append(('penalty_end_away', min(future_away) - self.current_time))
        boundary_kind, boundary_time = min(boundaries, key=lambda x: x[1])
        
        if delta_event < boundary_time:
            # Stochastic event occurs first: end shift
            self.current_time += delta_event
            
            # Choose which event
            event_choice = random.choices(
                ['home_goal','away_goal','home_penalty','away_penalty'],
                weights=[rates['home_goal'],rates['away_goal'],rates['home_penalty'],rates['away_penalty']],
                k=1
            )[0]
            
            self._handle_event(event_choice)
        else:
            # Boundary occurs first; clamp to period end to avoid any drift beyond 3600 in regulation
            if self.current_period is not None:
                boundary_time = min(boundary_time, max(0.0, self.current_period.end_time - self.current_time))
            self.current_time += boundary_time
            if boundary_kind == 'period_end':
                # Let period logic handle the end; do not rotate here
                return
            if boundary_kind == 'penalty_end_home':
                self._end_one_penalty('home')
                self._handle_line_change()
                return
            if boundary_kind == 'penalty_end_away':
                self._end_one_penalty('away')
                self._handle_line_change()
                return
            # Apply specific line change and start a new shift (resamples clocks)
            if boundary_kind == 'home_forward':
                self._handle_line_change(team='home', unit='forward')
            elif boundary_kind == 'home_defense':
                self._handle_line_change(team='home', unit='defense')
            elif boundary_kind == 'away_forward':
                self._handle_line_change(team='away', unit='forward')
            elif boundary_kind == 'away_defense':
                self._handle_line_change(team='away', unit='defense')
    
    def _handle_event(self, event_type: str) -> None:
        """Handle a specific event type."""
        if event_type == 'home_goal':
            self.home_score += 1
            context = 'home_pp_goal' if self.penalized_team == 'away' else ('away_pp_against' if self.penalized_team == 'home' else 'even')
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'goal', f'Home team scores! {self.home_score}-{self.away_score}', self.home_score, self.away_score, context, h_oi, a_oi))
            # If away was shorthanded, release oldest away minor (oldest-minor rule)
            if self.penalized_team == 'away':
                self._end_one_penalty('away')
            # If away had pulled and conceded EN against them, revert to normal
            if self.pulled_team == 'away' and self.penalized_team is None:
                self.pulled_team = None
                h_oi, a_oi = self._current_on_ice_names()
                self.events.append((self.current_time, 'goalie_in', 'Away goalie returns after empty-net against', self.home_score, self.away_score, 'away_in', h_oi, a_oi))
                self._rebuild_on_ice_caches()
            # If home had pulled and just tied the game, revert to normal
            if self.pulled_team == 'home' and self.home_score == self.away_score:
                self.pulled_team = None
                h_oi, a_oi = self._current_on_ice_names()
                self.events.append((self.current_time, 'goalie_in', 'Home goalie returns after tying the game', self.home_score, self.away_score, 'home_in', h_oi, a_oi))
                self._rebuild_on_ice_caches()
            self._handle_line_change()  # New shift after goal
            
        elif event_type == 'away_goal':
            self.away_score += 1
            context = 'away_pp_goal' if self.penalized_team == 'home' else ('home_pp_against' if self.penalized_team == 'away' else 'even')
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'goal', f'Away team scores! {self.home_score}-{self.away_score}', self.home_score, self.away_score, context, h_oi, a_oi))
            # If home was shorthanded, release oldest home minor
            if self.penalized_team == 'home':
                self._end_one_penalty('home')
            # If home had pulled and conceded EN against them, revert to normal
            if self.pulled_team == 'home' and self.penalized_team is None:
                self.pulled_team = None
                h_oi, a_oi = self._current_on_ice_names()
                self.events.append((self.current_time, 'goalie_in', 'Home goalie returns after empty-net against', self.home_score, self.away_score, 'home_in', h_oi, a_oi))
                self._rebuild_on_ice_caches()
            # If away had pulled and just tied the game, revert to normal
            if self.pulled_team == 'away' and self.home_score == self.away_score:
                self.pulled_team = None
                h_oi, a_oi = self._current_on_ice_names()
                self.events.append((self.current_time, 'goalie_in', 'Away goalie returns after tying the game', self.home_score, self.away_score, 'away_in', h_oi, a_oi))
                self._rebuild_on_ice_caches()
            self._handle_line_change()  # New shift after goal
            
        elif event_type == 'home_penalty':
            # Simplified penalty handling
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'penalty', 'Home team penalty', self.home_score, self.away_score, 'home_penalty', h_oi, a_oi))
            # Enter power play state before changing lines so special teams deploy
            self._enter_power_play('home')
            self._handle_line_change()
            
        elif event_type == 'away_penalty':
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'penalty', 'Away team penalty', self.home_score, self.away_score, 'away_penalty', h_oi, a_oi))
            self._enter_power_play('away')
            self._handle_line_change()
            
        elif event_type in ['home_line_change', 'away_line_change', 'home_def_change', 'away_def_change']:
            # Team-specific line change events
            if event_type == 'home_line_change':
                self._handle_line_change(team='home', unit='forward')
            elif event_type == 'away_line_change':
                self._handle_line_change(team='away', unit='forward')
            elif event_type == 'home_def_change':
                self._handle_line_change(team='home', unit='defense')
            elif event_type == 'away_def_change':
                self._handle_line_change(team='away', unit='defense')
    
    def _handle_line_change(self, team: Optional[str] = None, unit: Optional[str] = None) -> None:
        """Apply a line change and update on-ice units.

        Args:
            team: 'home' or 'away' to change a specific team; None changes both.
            unit: 'forward' or 'defense' to change a specific unit; None changes both.
        """
        if team is None:
            # Natural shift end or reset after events: rotate both teams, both units
            self._rotate_forward('home')
            self._rotate_defense('home')
            self._rotate_forward('away')
            self._rotate_defense('away')
            self._rebuild_on_ice_caches()
            self._resample_clocks()
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'line_change', 'Both teams change forwards and defense', self.home_score, self.away_score, '', h_oi, a_oi))
            return

        # Team-specific change
        if unit in (None, 'forward'):
            self._rotate_forward(team)
        if unit in (None, 'defense'):
            self._rotate_defense(team)
        # Rebuild on-ice and resample only the changed unit's clock
        self._rebuild_on_ice_caches()
        unit_key = f"{team}_{'forward' if unit=='forward' else 'defense'}"
        self._resample_clocks(unit_key)
        label_team = 'Home' if team == 'home' else 'Away'
        label_unit = 'forward line' if unit == 'forward' else ('defensive pair' if unit == 'defense' else 'lines and pairs')
        h_oi, a_oi = self._current_on_ice_names()
        self.events.append((self.current_time, 'line_change', f'{label_team} {label_unit} change', self.home_score, self.away_score, '', h_oi, a_oi))
    
    def simulate_game(self) -> Dict:
        """Simulate the entire game."""
        self.start_shift()
        
        # Simulate 3 regulation periods
        for period_num in range(1, 4):
            period = Period(period_num, 1200.0, is_overtime=False)  # 20 minutes = 1200 seconds
            self._simulate_period(period)
        
        # Check if game is tied after regulation
        if self.home_score == self.away_score:
            self._simulate_overtime()
        
        return {
            'home_team': self.home_team.name,
            'away_team': self.away_team.name,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'events': self.events
        }
    
    def _simulate_period(self, period: Period) -> None:
        """Simulate a single period."""
        self.current_period = period
        # Stagger start slightly to avoid same-timestamp with prior end
        start_t = self.current_time + (1e-6 if self.events and self.events[-1][1] == 'period_end' else 0.0)
        period.start_period(start_t)
        h_oi, a_oi = self._current_on_ice_names()
        self.events.append((start_t, 'period_start', f'Start of Period {period.period_number}', self.home_score, self.away_score, '', h_oi, a_oi))
        
        # Simulate shifts until period ends
        while not period.is_finished(self.current_time):
            self.simulate_shift()
        
        # End the period
        period.end_period(self.current_time)
        h_oi, a_oi = self._current_on_ice_names()
        self.events.append((self.current_time, 'period_end', f'End of Period {period.period_number}', self.home_score, self.away_score, '', h_oi, a_oi))
    
    def _simulate_overtime(self) -> None:
        """Simulate sudden-death overtime with repeated periods until a goal is scored."""
        ot_idx = 1
        while self.home_score == self.away_score:
            # Each OT period is 20 minutes sudden-death
            overtime = Period(3 + ot_idx, 1200.0, is_overtime=True)
            self.current_period = overtime
            overtime.start_period(self.current_time)
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'overtime_start', f'Overtime {ot_idx} begins - sudden death', self.home_score, self.away_score, '', h_oi, a_oi))

            while not overtime.is_finished(self.current_time):
                self.simulate_shift()
                if self.home_score != self.away_score:
                    h_oi, a_oi = self._current_on_ice_names()
                    self.events.append((self.current_time, 'overtime_goal', 'Overtime goal - game over!', self.home_score, self.away_score, '', h_oi, a_oi))
                    return

            # Period ended without a goal; mark end and loop to another OT period
            overtime.end_period(self.current_time)
            h_oi, a_oi = self._current_on_ice_names()
            self.events.append((self.current_time, 'period_end', f'End of Overtime {ot_idx}', self.home_score, self.away_score, '', h_oi, a_oi))
            ot_idx += 1