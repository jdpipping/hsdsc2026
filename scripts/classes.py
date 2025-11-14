################
### PACKAGES ###
################

from typing import Dict, List, Optional, Set, Tuple, Union
import random
import math

# Shot rate baselines (per second) - shots on goal (SOG) per situation
# Rates are per minute, converted to per second by dividing by 60
# Format: (a, b) -> (shot_rate_for_team_with_a_skaters, shot_rate_for_team_with_b_skaters)
SHOT_RATE_BASELINES = {
    (3, 3): (0.418 / 60.0, 0.418 / 60.0),
    (3, 4): (0.198 / 60.0, 0.792 / 60.0),
    (3, 5): (0.115 / 60.0, 1.353 / 60.0),
    (3, 6): (0.072 / 60.0, 1.920 / 60.0),
    (4, 3): (0.792 / 60.0, 0.198 / 60.0),
    (4, 4): (0.462 / 60.0, 0.462 / 60.0),
    (4, 5): (0.221 / 60.0, 0.754 / 60.0),
    (4, 6): (0.104 / 60.0, 1.566 / 60.0),
    (5, 3): (1.353 / 60.0, 0.115 / 60.0),
    (5, 4): (0.754 / 60.0, 0.221 / 60.0),
    (5, 5): (0.400 / 60.0, 0.400 / 60.0),
    (5, 6): (0.174 / 60.0, 0.900 / 60.0),
    (6, 3): (1.920 / 60.0, 0.072 / 60.0),
    (6, 4): (1.566 / 60.0, 0.104 / 60.0),
    (6, 5): (0.900 / 60.0, 0.174 / 60.0),
}

# xG baselines (per shot on goal) - expected goals per shot on goal (xGOT) by situation
# Format: (a, b) -> (xG_for_team_with_a_skaters, xG_for_team_with_b_skaters)
XG_BASELINES = {
    (3, 3): (0.105, 0.105),
    (3, 4): (0.085, 0.129),
    (3, 5): (0.061, 0.218),
    (3, 6): (0.050, 0.225),
    (4, 3): (0.129, 0.085),
    (4, 4): (0.105, 0.105),
    (4, 5): (0.076, 0.146),
    (4, 6): (0.071, 0.172),
    (5, 3): (0.218, 0.061),
    (5, 4): (0.146, 0.076),
    (5, 5): (0.095, 0.095),
    (5, 6): (0.117, 0.163),
    (6, 3): (0.225, 0.050),
    (6, 4): (0.172, 0.071),
    (6, 5): (0.163, 0.117),
}

# Home-ice advantage multipliers
# Penalty rate: boost home team's penalty draw rate (e.g., +5-10%)
# Note: Shot creation, xG bonus, and suppression are now team-specific (see Team.hfa_* attributes)
HFA_PENALTY_MULT = 1.075  # ~7.5% boost to home team's penalty draw rate

# Player influence scaling
SHOT_RATE_SCALE = 0.00035         # Scaling for creation/suppression on shot rates
XG_SCALE = 0.006                # Scaling for conversion/prevention on xG
GOALIE_SUPPRESSION_SCALE = XG_SCALE * 2 # Scaling for goalkeeping effect on xG
TINY = 1e-12

# Penalty hazards
PEN_BASE = 6.0 / 3600.0
PEN_BETA = 0.40
PEN_NORM = math.exp(0.5 * PEN_BETA * PEN_BETA)
PEN_CLAMP_LO = 0.5
PEN_CLAMP_HI = 1.8

# Assist distributions by manpower situation
# Format: (a, b) -> ({prob_0_assists, prob_1_assist, prob_2_assists} for team with a skaters,
#                    {prob_0_assists, prob_1_assist, prob_2_assists} for team with b skaters)
ASSIST_DISTRIBUTIONS = {
    (3, 3): ([0.12, 0.48, 0.40], [0.12, 0.48, 0.40]),
    (3, 4): ([0.20, 0.50, 0.30], [0.10, 0.40, 0.50]),
    (3, 5): ([0.28, 0.48, 0.24], [0.07, 0.35, 0.58]),
    (3, 6): ([0.60, 0.32, 0.08], [0.05, 0.33, 0.62]),
    (4, 3): ([0.10, 0.40, 0.50], [0.20, 0.50, 0.30]),
    (4, 4): ([0.11, 0.31, 0.58], [0.11, 0.31, 0.58]),
    (4, 5): ([0.25, 0.45, 0.30], [0.05, 0.25, 0.70]),
    (4, 6): ([0.55, 0.35, 0.10], [0.02, 0.18, 0.80]),
    (5, 3): ([0.07, 0.35, 0.58], [0.28, 0.48, 0.24]),
    (5, 4): ([0.05, 0.25, 0.70], [0.25, 0.45, 0.30]),
    (5, 5): ([0.09, 0.22, 0.69], [0.09, 0.22, 0.69]),
    (5, 6): ([0.45, 0.37, 0.18], [0.07, 0.20, 0.73]),
    (6, 3): ([0.05, 0.33, 0.62], [0.60, 0.32, 0.08]),
    (6, 4): ([0.02, 0.18, 0.80], [0.55, 0.35, 0.10]),
    (6, 5): ([0.07, 0.20, 0.73], [0.45, 0.37, 0.18]),
}

def sample_assists(n_scoring_team: int, n_opposing_team: int) -> int:
    """Sample number of assists (0, 1, or 2) for a goal based on manpower situation.
    
    Args:
        n_scoring_team: Number of skaters on the team that scored
        n_opposing_team: Number of skaters on the opposing team
    
    Returns:
        Number of assists: 0, 1, or 2
    """
    # Clamp to valid range (3-6 skaters)
    n_scoring_team = max(3, min(6, n_scoring_team))
    n_opposing_team = max(3, min(6, n_opposing_team))
    
    # Get distribution for scoring team
    situation = (n_scoring_team, n_opposing_team)
    if situation in ASSIST_DISTRIBUTIONS:
        dist_for_scoring, dist_for_opposing = ASSIST_DISTRIBUTIONS[situation]
        probs = dist_for_scoring
    else:
        # Fallback to 5v5 distribution
        probs = ASSIST_DISTRIBUTIONS[(5, 5)][0]
    
    # Sample from distribution
    r = random.random()
    if r < probs[0]:
        return 0
    elif r < probs[0] + probs[1]:
        return 1
    else:
        return 2


class Player:
    """Represents a hockey player (skater or goalie) with core attributes.

    Attributes:
        name: Player's display name.
        position: "F", "D", or "G".
        creation: Ability to generate shots (shot creation, rate modifier).
        conversion: Ability to create high-xG shots (shot conversion, xG quality modifier).
        suppression: Ability to suppress shot rates (shot suppression, rate reduction).
        prevention: Ability to reduce xG quality of shots that occur (goal prevention, xG reduction).
        goalkeeping: Ability to save shots (only for goalies, reduces xG).
        stamina: Measure of endurance.
        discipline: Measure of penalty discipline.
    """
    def __init__(self, name: str, position: str, 
                 creation: float = 0.0, conversion: float = 0.0,
                 suppression: float = 0.0, prevention: float = 0.0,
                 goalkeeping: float = 0.0,
                 stamina: float = 0.0, discipline: float = 0.0):
        self.name = name
        self.position = position
        self.creation = creation
        self.conversion = conversion
        self.suppression = suppression
        self.prevention = prevention
        self.goalkeeping = goalkeeping if position == "G" else 0.0
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
        """Best players by sum of all attributes go to top groupings."""
        sorted_players = sorted(players, key=lambda p: p.creation + p.conversion + p.suppression + p.prevention, reverse=True)
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
        sorted_players = sorted(players, key=lambda p: p.creation + p.conversion + p.suppression + p.prevention, reverse=True)
        num_groups = max(1, len(sorted_players) // group_size)
        groups: List[List[Player]] = [[] for _ in range(num_groups)]
        # Round-robin best-available into groups to balance sums
        for idx, p in enumerate(sorted_players[: num_groups * group_size]):
            groups[idx % num_groups].append(p)
        return {i + 1: grp for i, grp in enumerate(groups)}
    
    def _complementary_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Groupings contain a mix of offensive and defensive players."""
        # Offensive = players with high creation+conversion
        # Defensive = players with high suppression+prevention
        offensive_score = lambda p: p.creation + p.conversion
        defensive_score = lambda p: p.suppression + p.prevention
        offensive = sorted([p for p in players if offensive_score(p) > defensive_score(p)], key=offensive_score, reverse=True)
        defensive = sorted([p for p in players if defensive_score(p) >= offensive_score(p)], key=defensive_score, reverse=True)
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
        """Greedy search for best offensive attributes (creation+conversion), stack groupings accordingly."""
        sorted_players = sorted(players, key=lambda p: p.creation + p.conversion, reverse=True)
        groupings = {}
        for i in range(0, len(sorted_players), group_size):
            line_num = (i // group_size) + 1
            groupings[line_num] = sorted_players[i:i + group_size]
        return groupings
    
    def _hyper_defensive_groupings(self, players: List[Player], group_size: int) -> Dict[int, List[Player]]:
        """Greedy search for best defensive attributes (suppression+prevention), stack groupings accordingly."""
        sorted_players = sorted(players, key=lambda p: p.suppression + p.prevention, reverse=True)
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
        hfa_shot_creation_mult: Home-ice advantage multiplier for shot creation rate (league avg ~1.02).
        hfa_xg_bonus: Home-ice advantage bonus to xG per shot (league avg ~0.0025).
        hfa_shot_suppression_mult: Home-ice advantage multiplier to suppress opponent shot rate (league avg ~0.98).
        hfa_xg_suppression: Home-ice advantage reduction to opponent xG per shot (league avg ~-0.0025).
        coach: Coach instance.
        lines: Forward lines created by coach.
        pairs: Defensive pairings created by coach.
    """
    def __init__(self, name: str, roster: List[Player], 
                 hfa_shot_creation_mult: float, hfa_xg_bonus: float,
                 hfa_shot_suppression_mult: float, hfa_xg_suppression: float,
                 coach: 'Coach'):
        self.name = name
        self.roster = list(roster) if roster is not None else []
        self.hfa_shot_creation_mult = hfa_shot_creation_mult
        self.hfa_xg_bonus = hfa_xg_bonus
        self.hfa_shot_suppression_mult = hfa_shot_suppression_mult
        self.hfa_xg_suppression = hfa_xg_suppression
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
        - Rank all available skaters by offensive contribution (creation + conversion)
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

        offensive_score = lambda p: p.creation + p.conversion
        skaters_sorted_off = sorted(skaters, key=offensive_score, reverse=True)
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
                    lowest_fwd = min(forwards_in_selected, key=offensive_score)
                    best_d = available_ds[0]
                    # swap
                    selected.remove(lowest_fwd)
                    selected.append(best_d)
                    def_count = 1

        # Cap defensemen at 2 by replacing lowest-offense D with best available F
        while def_count > 2:
            ds_in_selected = sorted([p for p in selected if is_def(p)], key=offensive_score)
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

        # Return sorted by offensive score for readability/consistency
        return sorted(selected, key=offensive_score, reverse=True)

    def penalty_kill_unit(self, unavailable: Optional[Set[Player]] = None, num_skaters: int = 4) -> List[Player]:
        """Select a penalty kill unit with defensive constraints.

        Logic:
        - Rank all available skaters by defensive contribution (suppression + prevention)
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

        defensive_score = lambda p: p.suppression + p.prevention
        skaters_sorted_def = sorted(skaters, key=defensive_score, reverse=True)
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
            forwards_in_selected = sorted([p for p in selected if p.position == "F"], key=defensive_score)
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
            ds_in_selected = sorted([p for p in selected if is_def(p)], key=defensive_score)
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

        # Return sorted by defensive score for readability/consistency
        return sorted(selected, key=defensive_score, reverse=True)

class Division:
    """A division containing 4 teams."""
    def __init__(self, name: str, teams: List[Team]):
        self.name = name
        self.teams = list(teams)
        if len(self.teams) != 4:
            raise ValueError(f"Division must have exactly 4 teams, got {len(self.teams)}")

class Conference:
    """A conference containing 4 divisions."""
    def __init__(self, name: str, divisions: List[Division]):
        self.name = name
        self.divisions = list(divisions)
        if len(self.divisions) != 4:
            raise ValueError(f"Conference must have exactly 4 divisions, got {len(self.divisions)}")
    
    def get_teams(self) -> List[Team]:
        """Return all teams in this conference."""
        teams = []
        for division in self.divisions:
            teams.extend(division.teams)
        return teams

class League:
    """A collection of teams with basic management operations."""
    def __init__(self, teams: List[Team], divisions: Optional[List[Division]] = None, conferences: Optional[List[Conference]] = None):
        self.teams = list(teams) if teams is not None else []
        self.divisions: List[Division] = divisions if divisions else []
        self.conferences: List[Conference] = conferences if conferences else []
        self.schedule: List[Tuple[Team, Team]] = []
        self.weeks: List[List[Tuple[Team, Team]]] = []
    
    def organize_into_divisions_and_conferences(self, seed: Optional[int] = None) -> None:
        """Organize 32 teams into 8 divisions (4 teams each) and 2 conferences (4 divisions each).
        
        Uses predefined geographic organization based on hockey-centric countries.
        """
        if len(self.teams) != 32:
            raise ValueError(f"League must have exactly 32 teams, got {len(self.teams)}")
        
        # Predefined organization of hockey countries by division
        # Eastern Conference (Traditional hockey powers)
        EASTERN_ATLANTIC = ["Canada", "United States", "Great Britain", "France"]  # Atlantic/North America
        EASTERN_NORDIC = ["Sweden", "Finland", "Norway", "Denmark"]  # Nordic countries
        EASTERN_SLAVIC = ["Russia", "Czech Republic", "Slovakia", "Belarus"]  # Slavic/Eastern European
        EASTERN_ALPINE = ["Germany", "Switzerland", "Austria", "Italy"]  # Alpine/Central European
        
        # Western Conference (Emerging hockey nations)
        WESTERN_BALTIC = ["Latvia", "Estonia", "Lithuania", "Poland"]  # Baltic region
        WESTERN_DANUBE = ["Slovenia", "Croatia", "Hungary", "Romania"]  # Danube/Central European
        WESTERN_ASIA_PACIFIC = ["Kazakhstan", "Ukraine", "Japan", "South Korea"]  # Asia-Pacific/Eurasian
        WESTERN_WESTERN_EUROPE = ["Netherlands", "Belgium", "Spain", "China"]  # Western Europe/Asia
        
        # Create a mapping from team name to team object
        team_dict = {team.name: team for team in self.teams}
        
        # Helper function to get teams by name list
        def get_teams_by_names(names: List[str]) -> List[Team]:
            teams_list = []
            for name in names:
                if name not in team_dict:
                    raise ValueError(f"Team '{name}' not found in league. Available teams: {list(team_dict.keys())}")
                teams_list.append(team_dict[name])
            return teams_list
        
        # Create divisions with predefined teams and geographic names
        divisions = [
            Division("Atlantic", get_teams_by_names(EASTERN_ATLANTIC)),
            Division("Nordic", get_teams_by_names(EASTERN_NORDIC)),
            Division("Slavic", get_teams_by_names(EASTERN_SLAVIC)),
            Division("Alpine", get_teams_by_names(EASTERN_ALPINE)),
            Division("Baltic", get_teams_by_names(WESTERN_BALTIC)),
            Division("Danube", get_teams_by_names(WESTERN_DANUBE)),
            Division("Asia-Pacific", get_teams_by_names(WESTERN_ASIA_PACIFIC)),
            Division("Western Europe", get_teams_by_names(WESTERN_WESTERN_EUROPE))
        ]
        
        # Create 2 conferences (4 divisions each)
        conference1 = Conference("Eastern", divisions[0:4])
        conference2 = Conference("Western", divisions[4:8])
        
        self.divisions = divisions
        self.conferences = [conference1, conference2]
    
    def get_team_division(self, team: Team) -> Optional[Division]:
        """Get the division for a given team."""
        for division in self.divisions:
            # Compare by name since team objects might be different instances
            for div_team in division.teams:
                if div_team.name == team.name:
                    return division
        return None
    
    def get_team_conference(self, team: Team) -> Optional[Conference]:
        """Get the conference for a given team."""
        for conference in self.conferences:
            # Compare by name since team objects might be different instances
            for conf_team in conference.get_teams():
                if conf_team.name == team.name:
                    return conference
        return None

    def add_team(self, team: Team) -> None:
        self.teams.append(team)

    def remove_team(self, team: Team) -> None:
        self.teams.remove(team)

    def build_schedule(self, shuffle: bool = True, seed: Optional[int] = None, group_weeks: bool = True) -> List[Tuple[Team, Team]]:
        """Create an NHL-style imbalanced schedule (82 games per team).
        
        Schedule breakdown per team:
        - Division opponents (3 teams): 2 teams × 5 games + 1 team × 4 games = 14 games
        - Conference non-division opponents (12 teams): 12 teams × 3 games = 36 games
        - Other conference opponents (16 teams): 16 teams × 2 games = 32 games
        Total: 14 + 36 + 32 = 82 games
        """
        if len(self.teams) != 32:
            raise ValueError(f"NHL-style schedule requires exactly 32 teams, got {len(self.teams)}")
        
        if not self.divisions or not self.conferences:
            raise ValueError("League must have divisions and conferences. Call organize_into_divisions_and_conferences() first.")
        
        games: List[Tuple[Team, Team]] = []
        rng = random.Random(seed) if seed is not None else random
        
        # Track games per team pair (symmetric)
        pair_game_counts: Dict[Tuple[Team, Team], int] = {}
        
        # Assign division games symmetrically
        # In each division (4 teams), each team plays 3 opponents and needs 14 games total
        # Each team needs: 2 opponents × 5 games + 1 opponent × 4 games = 14 games
        # Mathematical constraint: For 4 teams with 6 pairs, we need exactly 4 pairs at 5 games and 2 pairs at 4 games
        # This ensures: 4 teams × (5+5+4) = 56 total game assignments, which equals (4×5 + 2×4) × 2 = 56 ✓
        for division in self.divisions:
            teams = sorted(division.teams, key=lambda t: t.name)  # Sort for consistency
            rng_div = random.Random(seed + hash(division.name)) if seed is not None else random
            
            # Create all 6 pairs in division
            pairs = []
            for i, team1 in enumerate(teams):
                for team2 in teams[i+1:]:
                    pair_key = tuple(sorted([team1, team2], key=lambda t: t.name))
                    pairs.append((team1, team2, pair_key))
            
            # Initialize all pairs to 5 games
            for t1, t2, pk in pairs:
                pair_game_counts[pk] = 5
            
            # Now we need to reduce 2 pairs to 4 games, ensuring each team still gets 14 games
            # Strategy: For each team, ensure it has exactly 2 pairs at 5 and 1 pair at 4
            # We'll assign which opponent each team plays 4 times
            team_4_game_opponents = {}
            for team in teams:
                opponents = [t for t in teams if t != team]
                # Randomly pick which opponent this team plays 4 times (others get 5)
                opponent_4 = rng_div.choice(opponents)
                team_4_game_opponents[team] = opponent_4
            
            # Convert team assignments to pair counts
            # For each pair, if either team wants to play the other 4 times, set to 4
            # Otherwise keep at 5
            for t1, t2, pk in pairs:
                # If team1 wants to play team2 4 times, or team2 wants to play team1 4 times
                if team_4_game_opponents.get(t1) == t2 or team_4_game_opponents.get(t2) == t1:
                    pair_game_counts[pk] = 4
                else:
                    pair_game_counts[pk] = 5
            
            # Verify each team has exactly 14 division games
            team_totals = {team: 0 for team in teams}
            for t1, t2, pk in pairs:
                count = pair_game_counts[pk]
                team_totals[t1] += count
                team_totals[t2] += count
            
            # If verification fails, use a known valid pattern that guarantees 14 games per team
            # Pattern for teams A, B, C, D (in sorted order):
            #   (A,B)=5, (A,C)=5, (A,D)=4  → A: 5+5+4=14
            #   (B,C)=4, (B,D)=5           → B: 5+4+5=14
            #   (C,D)=5                    → C: 5+4+5=14, D: 4+5+5=14
            if any(total != 14 for total in team_totals.values()):
                # Reset all pairs
                for t1, t2, pk in pairs:
                    pair_game_counts[pk] = 5
                
                # Apply known valid pattern
                # (teams[0], teams[3]) = 4 games
                pair_key_0_3 = tuple(sorted([teams[0], teams[3]], key=lambda t: t.name))
                pair_game_counts[pair_key_0_3] = 4
                # (teams[1], teams[2]) = 4 games  
                pair_key_1_2 = tuple(sorted([teams[1], teams[2]], key=lambda t: t.name))
                pair_game_counts[pair_key_1_2] = 4
                # All other pairs remain at 5 games
        
        # Now assign conference and other conference games (symmetric)
        for team in self.teams:
            team_division = self.get_team_division(team)
            team_conference = self.get_team_conference(team)
            
            if not team_division or not team_conference:
                continue
            
            # 2. Conference non-division opponents (12 teams): 3 games each = 36 games
            conference_teams = team_conference.get_teams()
            conference_non_division = [t for t in conference_teams if t != team and t not in team_division.teams]
            
            for opponent in conference_non_division:
                pair_key = tuple(sorted([team, opponent], key=lambda t: t.name))
                if pair_key not in pair_game_counts:
                    pair_game_counts[pair_key] = 3
            
            # 3. Other conference opponents (16 teams): 2 games each = 32 games
            other_conference = self.conferences[1] if team_conference == self.conferences[0] else self.conferences[0]
            other_conference_teams = other_conference.get_teams()
            
            for opponent in other_conference_teams:
                pair_key = tuple(sorted([team, opponent], key=lambda t: t.name))
                if pair_key not in pair_game_counts:
                    pair_game_counts[pair_key] = 2
        
        # Build games from pair_game_counts, ensuring home/away balance
        for pair_key, total_games in pair_game_counts.items():
            team1, team2 = pair_key
            
            # Determine which team should be home first (use team name as tiebreaker)
            home_first = team1 if team1.name < team2.name else team2
            away_first = team2 if home_first == team1 else team1
            
            # Create games with home/away balance
            for i in range(total_games):
                # Alternate home/away, starting with home_first
                if i % 2 == 0:
                    games.append((home_first, away_first))
                else:
                    games.append((away_first, home_first))
        
        # Shuffle the schedule if requested
        if shuffle:
            if seed is not None:
                rng = random.Random(seed)
                rng.shuffle(games)
            else:
                random.shuffle(games)
        
        self.schedule = games
        if group_weeks:
            self.weeks = self.build_weeks(self.schedule)
        return games

    def player_rankings(self, top_n: int, key: str) -> List[Dict]:
        """Return top N players across the league sorted by the given key.

        key should be one of: 'creation', 'conversion', 'suppression', 'prevention', 'goalkeeping', 'stamina', 'discipline', or 'total'.
        """
        rows: List[Dict] = []
        for team in self.teams:
            for p in team.roster:
                rows.append({
                    'team': team.name,
                    'player': p.name,
                    'position': p.position,
                    'creation': p.creation,
                    'conversion': p.conversion,
                    'suppression': p.suppression,
                    'prevention': p.prevention,
                    'goalkeeping': p.goalkeeping,
                    'stamina': p.stamina,
                    'discipline': p.discipline,
                    'total': p.creation + p.conversion + p.suppression + p.prevention + p.goalkeeping
                })
        if key not in {'creation','conversion','suppression','prevention','goalkeeping','stamina','discipline','total'}:
            key = 'creation'
        rows.sort(key=lambda r: r[key], reverse=True)
        return rows[:top_n]

    def get_teams(self) -> List[Dict]:
        """Return per-team summary: coach, playstyle, sums of roster attributes, HFA factors, and division/conference."""
        out: List[Dict] = []
        for team in self.teams:
            creation_sum = sum(p.creation for p in team.roster)
            conversion_sum = sum(p.conversion for p in team.roster)
            suppression_sum = sum(p.suppression for p in team.roster)
            prevention_sum = sum(p.prevention for p in team.roster)
            goalkeeping_sum = sum(p.goalkeeping for p in team.roster)
            sta = sum(p.stamina for p in team.roster)
            dis = sum(p.discipline for p in team.roster)
            
            # Get division and conference info
            team_division = self.get_team_division(team)
            team_conference = self.get_team_conference(team)
            division_name = team_division.name if team_division else None
            conference_name = team_conference.name if team_conference else None
            
            out.append({
                'team': team.name,
                'coach': team.coach.name,
                'playstyle': team.coach.playstyle,
                'conference': conference_name,
                'division': division_name,
                'creation_sum': creation_sum,
                'conversion_sum': conversion_sum,
                'suppression_sum': suppression_sum,
                'prevention_sum': prevention_sum,
                'goalkeeping_sum': goalkeeping_sum,
                'stamina_sum': sta,
                'discipline_sum': dis,
                'total_sum': creation_sum + conversion_sum + suppression_sum + prevention_sum + goalkeeping_sum,
                'hfa_shot_creation_mult': team.hfa_shot_creation_mult,
                'hfa_xg_bonus': team.hfa_xg_bonus,
                'hfa_shot_suppression_mult': team.hfa_shot_suppression_mult,
                'hfa_xg_suppression': team.hfa_xg_suppression
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
        pbp_weeks: int = 0,
        generate_box_scores: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Simulate a season over the given weeks.

        Args:
            weeks: Optional pre-built weeks. If None, uses self.build_weeks().
            pbp_weeks: Export play-by-play rows for the first pbp_weeks weeks.
            generate_box_scores: Whether to generate box score rows (default: True).

        Returns:
            (game_rows, pbp_rows, box_rows) where game_rows contains one row per game with
            basic results, pbp_rows contains play-by-play for the requested weeks, and
            box_rows contains box score rows aggregated by line/pairing matchups.
        """
        
        game_rows: List[Dict] = []
        pbp_rows: List[Dict] = []
        box_rows: List[Dict] = []

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
                
                # Generate box scores for all games
                if generate_box_scores:
                    game_box_rows = g.generate_box_scores(game_id, wk_idx)
                    box_rows.extend(game_box_rows)

        return game_rows, pbp_rows, box_rows

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
        
        # Cache attributes for shot rate calculations (creation affects shot rates, suppression reduces them)
        self._home_creation_sum = sum(p.creation for p in self.home_players if p.position != 'G')
        self._home_suppression_sum = sum(p.suppression for p in self.home_players if p.position != 'G')
        self._away_creation_sum = sum(p.creation for p in self.away_players if p.position != 'G')
        self._away_suppression_sum = sum(p.suppression for p in self.away_players if p.position != 'G')
        
        # Cache attributes for xG calculations (conversion increases xG, prevention reduces it)
        self._home_conversion_sum = sum(p.conversion for p in self.home_players if p.position != 'G')
        self._home_prevention_sum = sum(p.prevention for p in self.home_players if p.position != 'G')
        self._away_conversion_sum = sum(p.conversion for p in self.away_players if p.position != 'G')
        self._away_prevention_sum = sum(p.prevention for p in self.away_players if p.position != 'G')
        
        # Goalie goalkeeping (only for goalies on ice, reduces xG)
        home_goalie = next((p for p in self.home_players if p.position == 'G'), None)
        away_goalie = next((p for p in self.away_players if p.position == 'G'), None)
        self._home_goalie_goalkeeping = home_goalie.goalkeeping if home_goalie else 0.0
        self._away_goalie_goalkeeping = away_goalie.goalkeeping if away_goalie else 0.0
        
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
        extra = max(candidates, key=lambda p: p.creation + p.conversion) if candidates else []
        if extra:
            unit = base + [extra]
        else:
            unit = base
        # Ensure size is at most 6
        return unit[:6]
    
    def calculate_event_rates(self) -> Dict[str, float]:
        """Calculate all event rates for current on-ice situation."""
        # Use cached on-ice sums from shift start
        home_creation = self._home_creation_sum
        home_suppression = self._home_suppression_sum
        away_creation = self._away_creation_sum
        away_suppression = self._away_suppression_sum
        
        # Select shot rate baselines based on current manpower (supports stacks, pulled goalie)
        active_home = sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time)
        active_away = sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time)
        n_home = max(3, 5 - active_home)
        n_away = max(3, 5 - active_away)
        # Pulled goalie overrides (late game): 6v5 or 5v6, and 6v4/4v6 when penalties
        if self.pulled_team == 'home':
            n_home = 6
        elif self.pulled_team == 'away':
            n_away = 6

        # Get shot rate baselines for both teams based on n_home:n_away situation
        situation = (n_home, n_away)
        if situation in SHOT_RATE_BASELINES:
            base_home_shots, base_away_shots = SHOT_RATE_BASELINES[situation]
        else:
            # Fallback to 5v5 if situation not covered
            base_home_shots, base_away_shots = SHOT_RATE_BASELINES[(5, 5)]

        # Apply home-ice advantage to shot creation: multiply home team's baseline with team-specific multiplier
        base_home_shots *= self.home_team.hfa_shot_creation_mult
        # Apply home team's shot suppression to opponent's baseline
        base_away_shots *= self.home_team.hfa_shot_suppression_mult

        # Shot rate = baseline + (team creation - opponent suppression)
        lambda_home_shot = max(TINY, base_home_shots + SHOT_RATE_SCALE * home_creation - SHOT_RATE_SCALE * away_suppression)
        lambda_away_shot = max(TINY, base_away_shots + SHOT_RATE_SCALE * away_creation - SHOT_RATE_SCALE * home_suppression)
        
        # Penalty rates: base 6 per team per 60 minutes, modulated by on-ice discipline
        # Use normalized exponential weights so league average stays centered at 1.
        base_pen = PEN_BASE
        # Apply home-ice advantage to penalty rate: multiply home team's penalty draw rate
        lambda_home_pen = base_pen * self._home_pen_mult * HFA_PENALTY_MULT
        lambda_away_pen = base_pen * self._away_pen_mult
        
        return {
            'home_shot': lambda_home_shot,
            'away_shot': lambda_away_shot,
            'home_penalty': lambda_home_pen,
            'away_penalty': lambda_away_pen
        }
    
    def _get_xg_baseline(self, for_team: str) -> float:
        """Get xG baseline for current situation for the specified team.
        
        Args:
            for_team: 'home' or 'away'
        
        Returns:
            Baseline xG value for the current manpower situation
        """
        active_home = sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time)
        active_away = sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time)
        n_home = max(3, 5 - active_home)
        n_away = max(3, 5 - active_away)
        
        # Pulled goalie overrides
        if self.pulled_team == 'home':
            n_home = 6
        elif self.pulled_team == 'away':
            n_away = 6
        
        # Get baseline xG based on situation
        situation = (n_home, n_away)
        if situation in XG_BASELINES:
            xg_home, xg_away = XG_BASELINES[situation]
            base_xg = xg_home if for_team == 'home' else xg_away
        else:
            # Fallback to 5v5 if situation not covered
            base_xg = XG_BASELINES[(5, 5)][0] if for_team == 'home' else XG_BASELINES[(5, 5)][1]
        
        return base_xg
    
    def _process_shot(self, team: str) -> bool:
        """Process a shot event: calculate xG and determine if it becomes a goal.
        
        Args:
            team: 'home' or 'away' - the team taking the shot
        
        Returns:
            True if shot becomes a goal, False otherwise
        """
        import random
        
        # Get baseline xG for this situation
        base_xg = self._get_xg_baseline(team)
        
        # Calculate xG modifiers from player attributes
        if team == 'home':
            # Offensive conversion increases xG
            conversion_modifier = XG_SCALE * self._home_conversion_sum
            # Opponent prevention reduces xG
            prevention_modifier = XG_SCALE * self._away_prevention_sum
            # Opponent goalie reduces xG (unless goalie is pulled)
            if self.pulled_team != 'away':
                goalie_modifier = GOALIE_SUPPRESSION_SCALE * self._away_goalie_goalkeeping
            else:
                goalie_modifier = 0.0  # No goalie when pulled
            # Apply home-ice advantage: team-specific xG bonus
            home_xg_bonus = self.home_team.hfa_xg_bonus
        else:  # away
            conversion_modifier = XG_SCALE * self._away_conversion_sum
            prevention_modifier = XG_SCALE * self._home_prevention_sum
            if self.pulled_team != 'home':
                goalie_modifier = GOALIE_SUPPRESSION_SCALE * self._home_goalie_goalkeeping
            else:
                goalie_modifier = 0.0
            # Apply home team's xG suppression to away team's shots
            home_xg_bonus = self.home_team.hfa_xg_suppression
        
        # Calculate final xG: baseline + modifiers + home advantage
        xg = base_xg + conversion_modifier - prevention_modifier - goalie_modifier + home_xg_bonus
        
        # Clamp xG to [0, 1]
        xg = max(0.0, min(1.0, xg))
        
        # Convert shot to goal with probability = xG
        is_goal = random.random() < xg
        
        # Record shot event with xG value
        h_oi, a_oi = self._current_on_ice_names()
        context = 'home_pp_shot' if (team == 'home' and self.penalized_team == 'away') else \
                  ('away_pp_shot' if (team == 'away' and self.penalized_team == 'home') else 'even')
        
        # Add xG to event tuple (extend event format)
        self.events.append((
            self.current_time, 
            'shot', 
            f'{team.capitalize()} team shot (xG: {xg:.3f})', 
            self.home_score, 
            self.away_score, 
            context, 
            h_oi, 
            a_oi,
            xg  # Add xG as 9th element
        ))
        
        return is_goal
    
    def simulate_shift(self) -> None:
        """Simulate one shift using Poisson events."""
        # Late-game pulled-goalie strategy per spec using period 3 and period_remaining ≤ 120
        # Compute precise remaining time in current period using Period.end_time.
        if self.current_period is not None:
            period_remaining = max(0.0, self.current_period.end_time - self.current_time)
        else:
            period_remaining = 0.0
        if (self.current_period and self.current_period.period_number == 3) and period_remaining <= 150.0:
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
        # Only stochastic (shots/penalties) compete with deterministic line-change/period boundaries
        total_rate = rates['home_shot'] + rates['away_shot'] + rates['home_penalty'] + rates['away_penalty']
        
        if total_rate <= 0:
            return
        
        # Sample next stochastic event time (shots/penalties)
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
                ['home_shot','away_shot','home_penalty','away_penalty'],
                weights=[rates['home_shot'],rates['away_shot'],rates['home_penalty'],rates['away_penalty']],
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
    
    def _get_manpower_state(self) -> Tuple[int, int]:
        """Get current manpower state (home_skaters, away_skaters).
        
        Returns:
            Tuple of (home_skaters, away_skaters), accounting for penalties and pulled goalie
        """
        active_home = sum(1 for p in self.home_penalties if p['segment_end'] > self.current_time)
        active_away = sum(1 for p in self.away_penalties if p['segment_end'] > self.current_time)
        n_home = max(3, 5 - active_home)
        n_away = max(3, 5 - active_away)
        
        # Pulled goalie overrides
        if self.pulled_team == 'home':
            n_home = 6
        elif self.pulled_team == 'away':
            n_away = 6
        
        return n_home, n_away
    
    def _handle_event(self, event_type: str) -> None:
        """Handle a specific event type."""
        if event_type == 'home_shot':
            # Process shot: calculate xG and determine if it becomes a goal
            is_goal = self._process_shot('home')
            
            if is_goal:
                # Shot becomes a goal
                self.home_score += 1
                context = 'home_pp_goal' if self.penalized_team == 'away' else ('away_pp_against' if self.penalized_team == 'home' else 'even')
                h_oi, a_oi = self._current_on_ice_names()
                # Get manpower state and sample assists
                n_home, n_away = self._get_manpower_state()
                assists = sample_assists(n_home, n_away)
                self.events.append((self.current_time, 'goal', f'Home team scores! {self.home_score}-{self.away_score}', self.home_score, self.away_score, context, h_oi, a_oi, assists))
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
            
        elif event_type == 'away_shot':
            # Process shot: calculate xG and determine if it becomes a goal
            is_goal = self._process_shot('away')
            
            if is_goal:
                # Shot becomes a goal
                self.away_score += 1
                context = 'away_pp_goal' if self.penalized_team == 'home' else ('home_pp_against' if self.penalized_team == 'away' else 'even')
                h_oi, a_oi = self._current_on_ice_names()
                # Get manpower state and sample assists
                n_home, n_away = self._get_manpower_state()
                assists = sample_assists(n_away, n_home)  # Note: scoring team is away, so n_away comes first
                self.events.append((self.current_time, 'goal', f'Away team scores! {self.home_score}-{self.away_score}', self.home_score, self.away_score, context, h_oi, a_oi, assists))
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
            # Enter power play state before changing lines so special teams deploy
            # This will create the penalty entry with type info
            self._enter_power_play('home')
            # Get the penalty type and minutes from the most recent penalty
            if self.home_penalties:
                latest_penalty = self.home_penalties[-1]
                ptype = latest_penalty['type']
                # Convert penalty type to minutes
                if ptype == 'minor':
                    penalty_minutes = 2
                elif ptype == 'double_minor':
                    penalty_minutes = 4
                elif ptype == 'major':
                    penalty_minutes = 5
                else:
                    penalty_minutes = 2  # default
            else:
                penalty_minutes = 2  # fallback
            self.events.append((self.current_time, 'penalty', 'Home team penalty', self.home_score, self.away_score, 'home_penalty', h_oi, a_oi, penalty_minutes))
            self._handle_line_change()
            
        elif event_type == 'away_penalty':
            h_oi, a_oi = self._current_on_ice_names()
            # Enter power play state before changing lines so special teams deploy
            self._enter_power_play('away')
            # Get the penalty type and minutes from the most recent penalty
            if self.away_penalties:
                latest_penalty = self.away_penalties[-1]
                ptype = latest_penalty['type']
                # Convert penalty type to minutes
                if ptype == 'minor':
                    penalty_minutes = 2
                elif ptype == 'double_minor':
                    penalty_minutes = 4
                elif ptype == 'major':
                    penalty_minutes = 5
                else:
                    penalty_minutes = 2  # default
            else:
                penalty_minutes = 2  # fallback
            self.events.append((self.current_time, 'penalty', 'Away team penalty', self.home_score, self.away_score, 'away_penalty', h_oi, a_oi, penalty_minutes))
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
    
    def _get_line_type(self, team: str, line_id: int) -> str:
        """Get line type classification: 'top' for line 1, 'secondary' for others."""
        return 'top' if line_id == 1 else 'secondary'
    
    def _get_pairing_type(self, team: str, pair_id: int) -> str:
        """Get pairing type classification: 'top' for pair 1, 'secondary' for others."""
        return 'top' if pair_id == 1 else 'secondary'
    
    def generate_box_scores(self, game_id: int, week: int) -> List[Dict]:
        """Generate box score rows from game events.
        
        Args:
            game_id: Game ID number
            week: Week number
        
        Returns:
            List of box score row dictionaries
        """
        box_rows = []
        
        # Track stats per matchup
        # Key: (home_line_type, home_pair_type, away_line_type, away_pair_type)
        matchup_stats = {}
        
        # Track current matchup and time
        current_matchup = None
        prev_time = 0.0
        prev_matchup = None
        
        # Track PP/PK state
        in_pp = False
        pp_team = None  # 'home' or 'away' or None
        
        # Track current line/pair IDs (maintain game state during box score generation)
        # Start with line 1, pair 1 for both teams
        current_home_line_id = 1
        current_home_pair_id = 1
        current_away_line_id = 1
        current_away_pair_id = 1
        
        # Track current goalies (start with first goalie from each team's roster)
        # Will be set to None when goalie is pulled
        current_home_goalie = self.home_team.goalies()[0].name if self.home_team.goalies() else None
        current_away_goalie = self.away_team.goalies()[0].name if self.away_team.goalies() else None
        prev_home_goalie = current_home_goalie
        prev_away_goalie = current_away_goalie
        
        # Process events chronologically
        for i, event in enumerate(self.events):
            event_time = event[0]
            event_type = event[1] if len(event) > 1 else ''
            tag = event[5] if len(event) > 5 else ''
            
            # Update line/pair IDs based on line_change events
            if event_type == 'line_change':
                # Parse the description to determine which team/unit changed
                desc = str(event[2]) if len(event) > 2 else ''
                if 'Home forward line' in desc or ('Home' in desc and 'forward' in desc.lower() and 'defense' not in desc.lower()):
                    # Home forward line change
                    current_home_line_id = (current_home_line_id % len(self.home_team.lines)) + 1
                elif 'Home defensive pair' in desc or ('Home' in desc and ('defense' in desc.lower() or 'defensive' in desc.lower())):
                    # Home defensive pair change
                    current_home_pair_id = (current_home_pair_id % len(self.home_team.pairs)) + 1
                elif 'Away forward line' in desc or ('Away' in desc and 'forward' in desc.lower() and 'defense' not in desc.lower()):
                    # Away forward line change
                    current_away_line_id = (current_away_line_id % len(self.away_team.lines)) + 1
                elif 'Away defensive pair' in desc or ('Away' in desc and ('defense' in desc.lower() or 'defensive' in desc.lower())):
                    # Away defensive pair change
                    current_away_pair_id = (current_away_pair_id % len(self.away_team.pairs)) + 1
                elif 'Both teams change' in desc:
                    # Both teams change - rotate all
                    current_home_line_id = (current_home_line_id % len(self.home_team.lines)) + 1
                    current_home_pair_id = (current_home_pair_id % len(self.home_team.pairs)) + 1
                    current_away_line_id = (current_away_line_id % len(self.away_team.lines)) + 1
                    current_away_pair_id = (current_away_pair_id % len(self.away_team.pairs)) + 1
            
            # Update PP/PK state BEFORE determining matchup
            old_in_pp = in_pp
            if event_type == 'pp_start':
                in_pp = True
                if 'home_pp' in tag or 'Home' in str(event[2]):
                    pp_team = 'home'
                elif 'away_pp' in tag or 'Away' in str(event[2]):
                    pp_team = 'away'
            elif event_type == 'pp_end' or (event_type == 'period_end' and in_pp):
                in_pp = False
                pp_team = None
            
            # Determine current matchup
            # During PP/PK, use aggregated matchup (PP/PP vs PK/PK)
            if in_pp:
                if pp_team == 'home':
                    current_matchup = ('PP', 'PP', 'PK', 'PK')
                elif pp_team == 'away':
                    current_matchup = ('PK', 'PK', 'PP', 'PP')
                else:
                    # Fallback: use previous PP/PK matchup if available
                    if prev_matchup and ('PP' in str(prev_matchup) or 'PK' in str(prev_matchup)):
                        if prev_matchup[0] == 'PP' or prev_matchup[1] == 'PP':
                            current_matchup = ('PP', 'PP', 'PK', 'PK')
                        elif prev_matchup[2] == 'PP' or prev_matchup[3] == 'PP':
                            current_matchup = ('PK', 'PK', 'PP', 'PP')
                        else:
                            current_matchup = prev_matchup
                    else:
                        current_matchup = ('PP', 'PP', 'PK', 'PK')
            else:
                # Regular 5v5: use tracked line/pair IDs
                home_line_type = self._get_line_type('home', current_home_line_id)
                home_pair_type = self._get_pairing_type('home', current_home_pair_id)
                away_line_type = self._get_line_type('away', current_away_line_id)
                away_pair_type = self._get_pairing_type('away', current_away_pair_id)
                current_matchup = (home_line_type, home_pair_type, away_line_type, away_pair_type)
            
            # Calculate time elapsed since last event (attributed to previous matchup)
            time_elapsed = event_time - prev_time
            # Attribute time to previous matchup (or current if this is the first event)
            matchup_to_credit = prev_matchup if prev_matchup is not None else current_matchup
            if time_elapsed > 0 and matchup_to_credit is not None:
                # Initialize matchup stats if needed
                if matchup_to_credit not in matchup_stats:
                    matchup_stats[matchup_to_credit] = {
                        'toi': 0.0,
                        'home_shots': 0,
                        'away_shots': 0,
                        'home_xg': 0.0,
                        'away_xg': 0.0,
                        'home_max_xg': 0.0,
                        'away_max_xg': 0.0,
                        'home_goals': 0,
                        'away_goals': 0,
                        'home_assists': 0,
                        'away_assists': 0,
                        'home_penalties_taken': 0,
                        'away_penalties_taken': 0,
                        'home_penalties_drawn': 0,
                        'away_penalties_drawn': 0,
                        'home_penalty_minutes': 0,
                        'away_penalty_minutes': 0,
                        'home_goalie': prev_home_goalie,  # Use previous goalie state during this time period
                        'away_goalie': prev_away_goalie,
                    }
                
                # Attribute time to matchup
                matchup_stats[matchup_to_credit]['toi'] += time_elapsed
            
            # Update goalie tracking based on goalie_pulled/goalie_in events
            # Do this AFTER attributing time but BEFORE initializing current matchup
            if event_type == 'goalie_pulled':
                if 'home_pulled' in tag or 'Home' in str(event[2]):
                    current_home_goalie = None
                elif 'away_pulled' in tag or 'Away' in str(event[2]):
                    current_away_goalie = None
            elif event_type == 'goalie_in':
                # When goalie returns, set back to the first goalie (since we don't track goalie rotation)
                if 'home_in' in tag or 'Home' in str(event[2]):
                    current_home_goalie = self.home_team.goalies()[0].name if self.home_team.goalies() else None
                elif 'away_in' in tag or 'Away' in str(event[2]):
                    current_away_goalie = self.away_team.goalies()[0].name if self.away_team.goalies() else None
            
            # Initialize matchup stats if needed
            if current_matchup not in matchup_stats:
                matchup_stats[current_matchup] = {
                    'toi': 0.0,
                    'home_shots': 0,
                    'away_shots': 0,
                    'home_xg': 0.0,
                    'away_xg': 0.0,
                    'home_max_xg': 0.0,
                    'away_max_xg': 0.0,
                    'home_goals': 0,
                    'away_goals': 0,
                    'home_assists': 0,
                    'away_assists': 0,
                    'home_penalties_taken': 0,
                    'away_penalties_taken': 0,
                    'home_penalties_drawn': 0,
                    'away_penalties_drawn': 0,
                    'home_penalty_minutes': 0,
                    'away_penalty_minutes': 0,
                    'home_goalie': current_home_goalie,
                    'away_goalie': current_away_goalie,
                }
            
            # Process event types
            if event_type == 'shot':
                # Shot event - check which team
                if 'Home' in str(event[2]) or 'home' in tag.lower():
                    matchup_stats[current_matchup]['home_shots'] += 1
                    if len(event) > 8:
                        xg = event[8]
                        matchup_stats[current_matchup]['home_xg'] += xg
                        matchup_stats[current_matchup]['home_max_xg'] = max(
                            matchup_stats[current_matchup]['home_max_xg'], xg
                        )
                elif 'Away' in str(event[2]) or 'away' in tag.lower():
                    matchup_stats[current_matchup]['away_shots'] += 1
                    if len(event) > 8:
                        xg = event[8]
                        matchup_stats[current_matchup]['away_xg'] += xg
                        matchup_stats[current_matchup]['away_max_xg'] = max(
                            matchup_stats[current_matchup]['away_max_xg'], xg
                        )
            
            elif event_type == 'goal':
                # Goal event - check which team scored
                home_score = event[3] if len(event) > 3 else 0
                away_score = event[4] if len(event) > 4 else 0
                # Get assists from event (9th element, if present)
                assists = event[8] if len(event) > 8 else 0
                # Determine which team scored by comparing to previous scores
                if i > 0:
                    prev_home = self.events[i-1][3] if len(self.events[i-1]) > 3 else 0
                    prev_away = self.events[i-1][4] if len(self.events[i-1]) > 4 else 0
                    if home_score > prev_home:
                        matchup_stats[current_matchup]['home_goals'] += 1
                        matchup_stats[current_matchup]['home_assists'] += assists
                    elif away_score > prev_away:
                        matchup_stats[current_matchup]['away_goals'] += 1
                        matchup_stats[current_matchup]['away_assists'] += assists
            
            elif event_type == 'penalty':
                # Penalty event - track penalty minutes assessed
                penalty_minutes = event[8] if len(event) > 8 else 2  # Default to 2 if not found
                
                if 'home_penalty' in tag or 'Home' in str(event[2]):
                    matchup_stats[current_matchup]['home_penalties_taken'] += 1
                    matchup_stats[current_matchup]['home_penalty_minutes'] += penalty_minutes
                    # Away team drew the penalty (home took it)
                    matchup_stats[current_matchup]['away_penalties_drawn'] += 1
                elif 'away_penalty' in tag or 'Away' in str(event[2]):
                    matchup_stats[current_matchup]['away_penalties_taken'] += 1
                    matchup_stats[current_matchup]['away_penalty_minutes'] += penalty_minutes
                    # Home team drew the penalty (away took it)
                    matchup_stats[current_matchup]['home_penalties_drawn'] += 1
            
            prev_time = event_time
            prev_matchup = current_matchup
            prev_home_goalie = current_home_goalie
            prev_away_goalie = current_away_goalie
        
        # Check if game went to OT
        went_ot = any(e[1] == 'overtime_start' for e in self.events)
        
        # Add remaining time from last event to end of game
        # Game ends at 3600 seconds (3 periods of 1200s each) for non-OT games
        # For OT games, find the actual end time from the last event
        if self.events:
            last_event_time = self.events[-1][0]
            game_end_time = 3600.0 if not went_ot else last_event_time
            
            # Add remaining time to the last matchup
            if prev_matchup is not None and game_end_time > last_event_time:
                remaining_time = game_end_time - last_event_time
                if prev_matchup not in matchup_stats:
                    matchup_stats[prev_matchup] = {
                        'toi': 0.0,
                        'home_shots': 0,
                        'away_shots': 0,
                        'home_xg': 0.0,
                        'away_xg': 0.0,
                        'home_max_xg': 0.0,
                        'away_max_xg': 0.0,
                        'home_goals': 0,
                        'away_goals': 0,
                        'home_assists': 0,
                        'away_assists': 0,
                        'home_penalties_taken': 0,
                        'away_penalties_taken': 0,
                        'home_penalties_drawn': 0,
                        'away_penalties_drawn': 0,
                        'home_penalty_minutes': 0,
                        'away_penalty_minutes': 0,
                        'home_goalie': prev_home_goalie,  # Use previous goalie state during this time period
                        'away_goalie': prev_away_goalie,
                    }
                matchup_stats[prev_matchup]['toi'] += remaining_time
        
        # Convert matchup stats to box score rows
        for matchup, stats in matchup_stats.items():
            home_line_type, home_pair_type, away_line_type, away_pair_type = matchup
            
            box_rows.append({
                'game_id': game_id,
                'week': week,
                'home_team': self.home_team.name,
                'away_team': self.away_team.name,
                'home_line_type': home_line_type,
                'home_pairing_type': home_pair_type,
                'away_line_type': away_line_type,
                'away_pairing_type': away_pair_type,
                'home_goalie': stats.get('home_goalie'),
                'away_goalie': stats.get('away_goalie'),
                'went_ot': 1 if went_ot else 0,
                'toi': round(stats['toi'], 2),
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
        
        return box_rows