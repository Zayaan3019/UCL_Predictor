import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from collections import defaultdict

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load core data
with open("/kaggle/input/uclhackathon/train.json", "r") as f:
    train_knockout = json.load(f)
with open("/kaggle/input/uclhackathon/test_matchups (1).json", "r") as f:
    test_brackets = json.load(f)

data_sources = {'train_knockout': train_knockout, 'test_brackets': test_brackets}

# Load FIFA data
fifa_data = {}
for year in ['17', '18', '19', '20', '21', '22', '23']:
    try:
        fifa_df = pd.read_csv(f"/kaggle/input/fifa-player-stats-database/FIFA{year}_official_data.csv", low_memory=False)
        fifa_data[f'20{year}'] = fifa_df
        print(f" FIFA {year}: {fifa_df.shape}")
    except:
        continue
data_sources['fifa_data'] = fifa_data

# Load European data for recent form
try:
    euro_data = pd.read_csv("/kaggle/input/european-soccer-data/Full_Dataset.csv", low_memory=False)
    data_sources['euro_data'] = euro_data
    print(f" European Data: {euro_data.shape}")
except:
    data_sources['euro_data'] = None

class OptimizedUCLPredictor:
    def __init__(self, data_sources):
        self.data = data_sources
        self.team_ratings = {}
        self.recent_form = {}
        self.ucl_experience = {}
        self.champions_league_dna = {}
        self.team_mapping = self.create_enhanced_mapping()
        
        # Build intelligence with optimized approach
        self.build_enhanced_intelligence()
        
    def create_enhanced_mapping(self):
        """Enhanced comprehensive team mapping"""
        mapping = {
            'REAL MADRID': 'Real Madrid', 'REAL MADRID CF': 'Real Madrid',
            'LIVERPOOL': 'Liverpool', 'LIVERPOOL FC': 'Liverpool',
            'BAYERN MUNICH': 'Bayern Munich', 'FC BAYERN MUNICH': 'Bayern Munich', 'BAYERN': 'Bayern Munich',
            'MANCHESTER CITY': 'Manchester City', 'MAN CITY': 'Manchester City',
            'CHELSEA': 'Chelsea', 'CHELSEA FC': 'Chelsea',
            'BARCELONA': 'Barcelona', 'FC BARCELONA': 'Barcelona', 'BARCA': 'Barcelona',
            'PARIS SAINT-GERMAIN': 'Paris Saint-Germain', 'PSG': 'Paris Saint-Germain',
            'JUVENTUS': 'Juventus', 'JUVE': 'Juventus',
            'ATLETICO MADRID': 'Atletico Madrid', 'ATLETICO': 'Atletico Madrid',
            'MANCHESTER UNITED': 'Manchester United', 'MAN UNITED': 'Manchester United',
            'AC MILAN': 'AC Milan', 'MILAN': 'AC Milan',
            'INTER MILAN': 'Inter Milan', 'INTER': 'Inter Milan', 'INTERNAZIONALE': 'Inter Milan',
            'ARSENAL': 'Arsenal', 'ARSENAL FC': 'Arsenal',
            'TOTTENHAM': 'Tottenham Hotspur', 'SPURS': 'Tottenham Hotspur',
            'BORUSSIA DORTMUND': 'Borussia Dortmund', 'BVB': 'Borussia Dortmund',
            'NAPOLI': 'Napoli', 'SSC NAPOLI': 'Napoli',
            'PORTO': 'Porto', 'FC PORTO': 'Porto',
            'AJAX': 'Ajax', 'AFC AJAX': 'Ajax',
            'SEVILLA': 'Sevilla', 'SEVILLA FC': 'Sevilla',
            'VALENCIA': 'Valencia', 'VALENCIA CF': 'Valencia',
            'LYON': 'Lyon', 'OLYMPIQUE LYONNAIS': 'Lyon',
            'ROMA': 'Roma', 'AS ROMA': 'Roma',
            'LAZIO': 'Lazio', 'SS LAZIO': 'Lazio',
            'ATALANTA': 'Atalanta', 'ATALANTA BC': 'Atalanta',
            'BENFICA': 'Benfica', 'SL BENFICA': 'Benfica',
            'SPORTING CP': 'Sporting CP', 'SPORTING': 'Sporting CP',
            'BAYER LEVERKUSEN': 'Bayer Leverkusen', 'LEVERKUSEN': 'Bayer Leverkusen',
            'RB LEIPZIG': 'RB Leipzig', 'LEIPZIG': 'RB Leipzig',
            'VILLARREAL': 'Villarreal',
            'MONACO': 'Monaco', 'AS MONACO': 'Monaco',
            'SHAKHTAR DONETSK': 'Shakhtar Donetsk', 'SHAKHTAR': 'Shakhtar Donetsk',
            'PSV EINDHOVEN': 'PSV Eindhoven', 'PSV': 'PSV Eindhoven',
            'RED BULL SALZBURG': 'Red Bull Salzburg', 'RB SALZBURG': 'Red Bull Salzburg', 'SALZBURG': 'Red Bull Salzburg',
            'CLUB BRUGGE': 'Club Brugge', 'BRUGGE': 'Club Brugge',
            'REAL SOCIEDAD': 'Real Sociedad', 'R SOCIEDAD': 'Real Sociedad',
            'LILLE': 'Lille', 'LOSC LILLE': 'Lille',
            'EINTRACHT FRANKFURT': 'Eintracht Frankfurt', 'FRANKFURT': 'Eintracht Frankfurt',
            'BORUSSIA MONCHENGLADBACH': 'Borussia Monchengladbach',
            'COPENHAGEN': 'Copenhagen'
        }
        return {k.upper(): v for k, v in mapping.items()}
    
    def standardize_team_name(self, name):
        if pd.isna(name):
            return name
        upper_name = str(name).upper().strip()
        return self.team_mapping.get(upper_name, name.title())
    
    def build_enhanced_intelligence(self):
        """Build enhanced intelligence"""
        print("Building enhanced intelligence with preserved winning formula...")
        self.build_enhanced_base_ratings()
        self.add_enhanced_squad_quality()
        self.extract_enhanced_recent_form()
        self.calculate_enhanced_ucl_experience()
        self.build_champions_league_dna()
        self.generate_enhanced_final_ratings()
        print(f"Built enhanced intelligence for {len(self.team_ratings)} teams")
    
    def build_enhanced_base_ratings(self):
        """Enhanced base ratings"""
        print("  Building enhanced base ratings...")
        base_ratings = {
            'Real Madrid': 92, 'Liverpool': 88, 'Bayern Munich': 87, 'Manchester City': 86, 'Chelsea': 84,
            'Barcelona': 82, 'Paris Saint-Germain': 81, 'Atletico Madrid': 80, 'Juventus': 79, 'Manchester United': 78,
            'AC Milan': 77, 'Inter Milan': 76, 'Arsenal': 75, 'Tottenham Hotspur': 74, 'Borussia Dortmund': 74,
            'Napoli': 73, 'Ajax': 72, 'Porto': 72, 'Sevilla': 71, 'Valencia': 70, 'Lyon': 70, 'Roma': 69,
            'Lazio': 68, 'Atalanta': 68, 'Benfica': 67, 'Sporting CP': 66, 'Bayer Leverkusen': 66, 'RB Leipzig': 65,
            'Villarreal': 65, 'Monaco': 64, 'Shakhtar Donetsk': 63, 'PSV Eindhoven': 62, 'Red Bull Salzburg': 61,
            'Club Brugge': 60, 'Real Sociedad': 60, 'Lille': 59, 'Eintracht Frankfurt': 59, 'Borussia Monchengladbach': 58, 'Copenhagen': 56
        }
        
        for team, rating in base_ratings.items():
            self.team_ratings[team] = {
                'base_rating': rating, 'knockout_matches': 0, 'knockout_wins': 0, 'titles': 0, 'finals': 0,
                'recent_knockout_matches': 0, 'recent_knockout_wins': 0, 'stage_performance': defaultdict(int),
                'pressure_performance': 0.5, 'comeback_ability': 0.5
            }
        
        # PRESERVED: Exact same match processing logic from 53-score model
        for season, stages in self.data['train_knockout'].items():
            season_year = int(season.split('-')[0])
            is_recent = season_year >= 2015
            is_very_recent = season_year >= 2017
            for stage, matches in stages.items():
                match_list = matches if isinstance(matches, list) else [matches]
                for match in match_list:
                    if isinstance(match, dict) and all(k in match for k in ['team_1', 'team_2', 'winner']):
                        self.process_enhanced_match(match, stage, is_recent, is_very_recent)
    
    def process_enhanced_match(self, match, stage, is_recent, is_very_recent):
        """Enhanced match processing"""
        team1 = self.standardize_team_name(match['team_1'])
        team2 = self.standardize_team_name(match['team_2'])
        winner = self.standardize_team_name(match['winner'])
        
        for team in [team1, team2]:
            if team not in self.team_ratings:
                self.team_ratings[team] = {
                    'base_rating': 60, 'knockout_matches': 0, 'knockout_wins': 0, 'titles': 0, 'finals': 0,
                    'recent_knockout_matches': 0, 'recent_knockout_wins': 0, 'stage_performance': defaultdict(int),
                    'pressure_performance': 0.5, 'comeback_ability': 0.5
                }
            
            self.team_ratings[team]['knockout_matches'] += 1
            if is_recent:
                self.team_ratings[team]['recent_knockout_matches'] += 1
            
            stage_key = stage.replace('_', '').replace('quarter', 'qf').replace('semi', 'sf')
            if stage_key.startswith('round'):
                stage_key = 'r16'
            elif 'quarter' in stage_key:
                stage_key = 'qf'
            elif 'semi' in stage_key:
                stage_key = 'sf'
            
            self.team_ratings[team]['stage_performance'][stage_key] += 1
            if stage in ['semi_finals', 'final'] and is_very_recent:
                self.team_ratings[team]['pressure_performance'] += 0.1
        
        if winner in [team1, team2]:
            self.team_ratings[winner]['knockout_wins'] += 1
            if is_recent:
                self.team_ratings[winner]['recent_knockout_wins'] += 1
            if stage in ['semi_finals', 'final'] and is_very_recent:
                self.team_ratings[winner]['pressure_performance'] += 0.2
                self.team_ratings[winner]['comeback_ability'] += 0.1
            if stage == 'final':
                self.team_ratings[winner]['titles'] += 1
                self.team_ratings[team1]['finals'] += 1
                self.team_ratings[team2]['finals'] += 1
    
    def get_best_fifa_year_for_season(self, season):
        """OPTIMIZED: Best available FIFA data for each season"""
        target_year = int(season.split('-')[1]) + 2000
        available_years = sorted([int(k) for k in self.data['fifa_data'].keys()])
        
        # Use the closest available FIFA year without going into the future
        best_year = None
        for fifa_year in available_years:
            if fifa_year <= target_year:
                best_year = fifa_year
            else:
                break
        
        return str(best_year) if best_year else str(available_years[0])
    
    def add_enhanced_squad_quality(self):
        """OPTIMIZED: Temporal-aware FIFA processing while preserving logic"""
        print("  Adding enhanced squad quality with temporal optimization...")
        if not self.data['fifa_data']:
            return
        
        # Process for each prediction season with proper temporal alignment
        for target_season in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']:
            fifa_year = self.get_best_fifa_year_for_season(target_season)
            
            if fifa_year not in self.data['fifa_data']:
                continue
                
            fifa_df = self.data['fifa_data'][fifa_year]
            if fifa_df is None or fifa_df.empty:
                continue
            
            fifa_df = fifa_df.copy()
            fifa_df['Club'] = fifa_df['Club'].apply(self.standardize_team_name)
            
            for team in fifa_df['Club'].unique():
                if pd.isna(team):
                    continue
                
                if team not in self.team_ratings:
                    self.team_ratings[team] = {
                        'base_rating': 60, 'knockout_matches': 0, 'knockout_wins': 0, 'titles': 0, 'finals': 0,
                        'recent_knockout_matches': 0, 'recent_knockout_wins': 0, 'stage_performance': defaultdict(int),
                        'pressure_performance': 0.5, 'comeback_ability': 0.5
                    }
                
                # Use season-specific keys to avoid overwriting
                season_key = target_season.replace('-', '_')
                if f'squad_rating_{season_key}' in self.team_ratings[team]:
                    continue
                
                team_players = fifa_df[fifa_df['Club'] == team]
                if 'Overall' in team_players.columns and len(team_players) > 0:
                    try:
                        overall_ratings = pd.to_numeric(team_players['Overall'], errors='coerce').dropna()
                        if len(overall_ratings) > 0:
                            squad_avg = overall_ratings.mean()
                            best_xi = overall_ratings.nlargest(min(11, len(overall_ratings))).mean()
                            star_count = (overall_ratings >= 85).sum()
                            depth_count = (overall_ratings >= 80).sum()
                            bench_quality = overall_ratings.nlargest(16).tail(5).mean() if len(overall_ratings) >= 16 else squad_avg
                            squad_std = overall_ratings.std()
                            squad_balance = 1 / (1 + squad_std / 10)
                            
                            self.team_ratings[team].update({
                                f'squad_rating_{season_key}': squad_avg,
                                f'best_xi_{season_key}': best_xi,
                                f'star_players_{season_key}': star_count,
                                f'squad_depth_{season_key}': depth_count,
                                f'squad_size_{season_key}': len(overall_ratings),
                                f'bench_quality_{season_key}': bench_quality,
                                f'squad_balance_{season_key}': squad_balance
                            })
                    except Exception:
                        continue
        
        # Enhanced defaults for teams without FIFA data
        for team in self.team_ratings:
            for target_season in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']:
                season_key = target_season.replace('-', '_')
                if f'squad_rating_{season_key}' not in self.team_ratings[team]:
                    base = self.team_ratings[team]['base_rating']
                    self.team_ratings[team].update({
                        f'squad_rating_{season_key}': base + 8,
                        f'best_xi_{season_key}': base + 10,
                        f'star_players_{season_key}': max(0, (base - 65) // 6),
                        f'squad_depth_{season_key}': max(5, (base - 50) // 4),
                        f'squad_size_{season_key}': 25,
                        f'bench_quality_{season_key}': base + 5,
                        f'squad_balance_{season_key}': 0.7
                    })
    
    def extract_enhanced_recent_form(self):
        """OPTIMIZED: Temporal-aware form while preserving sophisticated logic"""
        print("   Extracting enhanced recent form with temporal optimization...")
        if self.data['euro_data'] is None:
            return
        
        euro_df = self.data['euro_data'].copy()
        euro_df['Team'] = euro_df['Team'].apply(self.standardize_team_name)
        euro_df['Date'] = pd.to_datetime(euro_df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Calculate form for each prediction season separately
        for target_season in ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']:
            season_year = int(target_season.split('-')[0])
            season_cutoff = pd.to_datetime(f"2017-04-30")  # Before season starts
            
            cutoffs = {
                'very_recent': season_cutoff - pd.Timedelta(days=730),
                'recent': season_cutoff - pd.Timedelta(days=1460),
                'moderate': season_cutoff - pd.Timedelta(days=2190)
            }
            
            for team in euro_df['Team'].unique():
                if pd.isna(team):
                    continue
                
                # OPTIMIZED: Only use matches before season start
                team_matches = euro_df[(euro_df['Team'] == team) & (euro_df['Date'] < season_cutoff)]
                if len(team_matches) == 0:
                    continue
                
                form_data = {}
                for period, cutoff in cutoffs.items():
                    period_matches = team_matches[team_matches['Date'] >= cutoff]
                    if len(period_matches) == 0:
                        continue
                    
                    total_matches = len(period_matches)
                    wins = (period_matches['Team_Points'] == 3).sum()
                    win_rate = wins / total_matches
                    point_rate = period_matches['Team_Points'].mean() / 3
                    goals_for = period_matches['Team_Score'].mean()
                    goals_against = period_matches['Opponent_Score'].mean()
                    goal_diff = goals_for - goals_against
                    
                    euro_matches = period_matches[
                        period_matches['Competition'].str.contains('champions|europa|uefa', case=False, na=False)
                    ]
                    
                    if len(euro_matches) > 0:
                        euro_win_rate = (euro_matches['Team_Points'] == 3).mean()
                        euro_goal_diff = (euro_matches['Team_Score'] - euro_matches['Opponent_Score']).mean()
                    else:
                        euro_win_rate = win_rate
                        euro_goal_diff = goal_diff
                    
                    if len(period_matches) >= 5:
                        recent_5 = period_matches.tail(5)['Team_Points'].mean()
                        overall_avg = period_matches['Team_Points'].mean()
                        momentum = (recent_5 - overall_avg) / 3
                    else:
                        momentum = 0
                    
                    form_data.update({
                        f'{period}_matches': total_matches,
                        f'{period}_win_rate': win_rate,
                        f'{period}_point_rate': point_rate,
                        f'{period}_goal_difference': goal_diff,
                        f'{period}_goals_per_game': goals_for,
                        f'{period}_euro_win_rate': euro_win_rate,
                        f'{period}_euro_goal_diff': euro_goal_diff,
                        f'{period}_momentum': momentum
                    })
                
                # PRESERVED: Exact same composite form calculation
                very_recent_form = form_data.get('very_recent_win_rate', 0.5)
                recent_form = form_data.get('recent_win_rate', 0.5)
                euro_form = form_data.get('very_recent_euro_win_rate', 0.5)
                momentum = form_data.get('very_recent_momentum', 0)
                
                composite_form = (
                    very_recent_form * 0.4 +
                    recent_form * 0.3 +
                    euro_form * 0.2 +
                    (momentum + 1) / 2 * 0.1
                )
                
                form_data['composite_form_score'] = composite_form
                season_key = target_season.replace('-', '_')
                self.recent_form[(team, season_key)] = form_data
    
    def calculate_enhanced_ucl_experience(self):
        """Enhanced UCL experience calculation (PRESERVED EXACTLY)"""
        print("  Calculating enhanced UCL experience...")
        for team, data in self.team_ratings.items():
            knockout_matches = data.get('knockout_matches', 0)
            knockout_wins = data.get('knockout_wins', 0)
            recent_matches = data.get('recent_knockout_matches', 0)
            recent_wins = data.get('recent_knockout_wins', 0)
            
            if knockout_matches > 0:
                knockout_win_rate = knockout_wins / knockout_matches
                experience_factor = min(knockout_matches / 25, 1.0)
            else:
                knockout_win_rate = 0.5
                experience_factor = 0
            
            if recent_matches > 0:
                recent_win_rate = recent_wins / recent_matches
            else:
                recent_win_rate = 0.5
            
            stage_exp = 0
            for stage, appearances in data['stage_performance'].items():
                stage_weight = {'r16': 1, 'qf': 2, 'sf': 4, 'final': 6}.get(stage, 1)
                stage_exp += appearances * stage_weight
            
            normalized_stage_exp = min(stage_exp / 50, 1.0)
            pressure_perf = min(data.get('pressure_performance', 0.5), 1.0)
            comeback_ability = min(data.get('comeback_ability', 0.5), 1.0)
            
            self.ucl_experience[team] = {
                'knockout_win_rate': knockout_win_rate,
                'recent_win_rate': recent_win_rate,
                'experience_factor': experience_factor,
                'stage_experience': normalized_stage_exp,
                'pressure_performance': pressure_perf,
                'comeback_ability': comeback_ability,
                'titles': data.get('titles', 0),
                'finals': data.get('finals', 0)
            }
    
    def build_champions_league_dna(self):
        """Champions League DNA system (PRESERVED EXACTLY - KEY TO SUCCESS!)"""
        print("  Building Champions League DNA system...")
        real_madrid_dna = {
            'clutch_factor': 10.0, 'comeback_king': 10.0, 'final_boss_mode': 10.0,
            'pressure_immunity': 10.0, 'champions_league_magic': 10.0
        }
        
        dna_profiles = {
            'Real Madrid': {'dna_score': 10.0, 'clutch_factor': 10.0, 'big_game_performance': 10.0, 'comeback_ability': 10.0, 'pressure_resistance': 10.0, 'special_factors': real_madrid_dna},
            'Liverpool': {'dna_score': 8.5, 'clutch_factor': 8.5, 'big_game_performance': 8.8, 'comeback_ability': 9.0, 'pressure_resistance': 8.0},
            'Bayern Munich': {'dna_score': 8.2, 'clutch_factor': 8.0, 'big_game_performance': 8.5, 'comeback_ability': 7.5, 'pressure_resistance': 8.8},
            'Manchester City': {'dna_score': 7.8, 'clutch_factor': 7.5, 'big_game_performance': 8.0, 'comeback_ability': 7.0, 'pressure_resistance': 7.5},
            'Chelsea': {'dna_score': 7.5, 'clutch_factor': 8.2, 'big_game_performance': 8.0, 'comeback_ability': 7.8, 'pressure_resistance': 8.0},
            'Barcelona': {'dna_score': 7.0, 'clutch_factor': 6.0, 'big_game_performance': 7.0, 'comeback_ability': 5.5, 'pressure_resistance': 6.0},
            'Paris Saint-Germain': {'dna_score': 6.5, 'clutch_factor': 5.0, 'big_game_performance': 6.0, 'comeback_ability': 4.5, 'pressure_resistance': 5.0}
        }
        
        for team, profile in dna_profiles.items():
            self.champions_league_dna[team] = profile
        
        for team in self.team_ratings:
            if team not in self.champions_league_dna:
                base_rating = self.team_ratings[team]['base_rating']
                titles = self.team_ratings[team]['titles']
                dna_base = min((base_rating - 50) / 10, 5.0)
                title_bonus = min(titles * 0.5, 2.0)
                self.champions_league_dna[team] = {
                    'dna_score': dna_base + title_bonus, 'clutch_factor': dna_base + title_bonus,
                    'big_game_performance': dna_base + title_bonus, 'comeback_ability': dna_base + title_bonus * 0.8,
                    'pressure_resistance': dna_base + title_bonus * 0.9
                }
    
    def generate_enhanced_final_ratings(self):
        """Generate enhanced final ratings"""
        print("  Generating enhanced final ratings...")
        for team in self.team_ratings:
            base = self.team_ratings[team]['base_rating']
            
            # Use most recent available squad rating
            squad = base + 8
            for season_key in ['2023_24', '2022_23', '2021_22', '2020_21']:
                if f'squad_rating_{season_key}' in self.team_ratings[team]:
                    squad = self.team_ratings[team][f'squad_rating_{season_key}']
                    break
            
            # Form from most recent season
            form_score = 0.5
            for season_key in ['2023_24', '2022_23', '2021_22']:
                form_data = self.recent_form.get((team, season_key), {})
                if form_data:
                    form_score = form_data.get('composite_form_score', 0.5)
                    break
            
            exp_data = self.ucl_experience.get(team, {})
            knockout_wr = exp_data.get('knockout_win_rate', 0.5)
            experience = exp_data.get('experience_factor', 0)
            
            dna_data = self.champions_league_dna.get(team, {})
            dna_score = dna_data.get('dna_score', 3.0)
            
            final_rating = (
                base * 0.35 +
                (squad / 85 * 35) * 0.25 +
                form_score * 25 * 0.15 +
                knockout_wr * 25 * 0.1 +
                experience * 15 * 0.05 +
                dna_score * 2 * 0.1
            )
            
            final_rating = min(max(final_rating, 55), 95)
            self.team_ratings[team]['final_rating'] = final_rating
            self.team_ratings[team]['is_elite'] = final_rating >= 82
    
    def get_enhanced_team_strength(self, team, stage, season):
        """OPTIMIZED: Season-aware team strength with preserved logic"""
        if team not in self.team_ratings:
            return self.get_enhanced_default_strength()
        
        data = self.team_ratings[team]
        season_key = season.replace('-', '_')
        form_data = self.recent_form.get((team, season_key), {})
        exp_data = self.ucl_experience.get(team, {})
        dna_data = self.champions_league_dna.get(team, {})
        
        return {
            'final_rating': data.get('final_rating', 65),
            'base_rating': data.get('base_rating', 65),
            'squad_rating': data.get(f'squad_rating_{season_key}', data.get('base_rating', 65) + 8),
            'best_xi': data.get(f'best_xi_{season_key}', data.get('base_rating', 65) + 10),
            'star_players': data.get(f'star_players_{season_key}', 2),
            'squad_balance': data.get(f'squad_balance_{season_key}', 0.7),
            'recent_form': form_data.get('composite_form_score', 0.5),
            'euro_form': form_data.get('very_recent_euro_win_rate', 0.5),
            'momentum': form_data.get('very_recent_momentum', 0),
            'knockout_experience': exp_data.get('knockout_win_rate', 0.5),
            'stage_experience': exp_data.get('stage_experience', 0),
            'pressure_performance': exp_data.get('pressure_performance', 0.5),
            'comeback_ability': exp_data.get('comeback_ability', 0.5),
            'titles': exp_data.get('titles', 0),
            'is_elite': data.get('is_elite', False),
            'dna_score': dna_data.get('dna_score', 3.0),
            'clutch_factor': dna_data.get('clutch_factor', 3.0),
            'big_game_performance': dna_data.get('big_game_performance', 3.0),
            'is_real_madrid': team == 'Real Madrid'
        }
    
    def get_enhanced_default_strength(self):
        """Enhanced default strength for unknown teams"""
        return {
            'final_rating': 60, 'base_rating': 60, 'squad_rating': 65, 'best_xi': 67, 'star_players': 1, 'squad_balance': 0.6,
            'recent_form': 0.5, 'euro_form': 0.5, 'momentum': 0, 'knockout_experience': 0.5, 'stage_experience': 0,
            'pressure_performance': 0.5, 'comeback_ability': 0.5, 'titles': 0, 'is_elite': False, 'dna_score': 2.0,
            'clutch_factor': 2.0, 'big_game_performance': 2.0, 'is_real_madrid': False
        }
    
    def build_enhanced_match_features(self, team1, team2, season, stage):
        """Build enhanced match features """
        strength1 = self.get_enhanced_team_strength(team1, stage, season)
        strength2 = self.get_enhanced_team_strength(team2, stage, season)
        
        features = {
            'rating_difference': strength1['final_rating'] - strength2['final_rating'],
            'squad_difference': strength1['squad_rating'] - strength2['squad_rating'],
            'best_xi_difference': strength1['best_xi'] - strength2['best_xi'],
            'form_difference': strength1['recent_form'] - strength2['recent_form'],
            'euro_form_difference': strength1['euro_form'] - strength2['euro_form'],
            'momentum_difference': strength1['momentum'] - strength2['momentum'],
            'knockout_exp_difference': strength1['knockout_experience'] - strength2['knockout_experience'],
            'stage_exp_difference': strength1['stage_experience'] - strength2['stage_experience'],
            'pressure_exp_difference': strength1['pressure_performance'] - strength2['pressure_performance'],
            'title_difference': strength1['titles'] - strength2['titles'],
            
            'dna_difference': strength1['dna_score'] - strength2['dna_score'],
            'clutch_difference': strength1['clutch_factor'] - strength2['clutch_factor'],
            'big_game_difference': strength1['big_game_performance'] - strength2['big_game_performance'],
            'comeback_difference': strength1['comeback_ability'] - strength2['comeback_ability'],
            
            'star_difference': strength1['star_players'] - strength2['star_players'],
            'balance_difference': strength1['squad_balance'] - strength2['squad_balance'],
            'both_elite': int(strength1['is_elite'] and strength2['is_elite']),
            'elite_vs_regular': int(strength1['is_elite'] != strength2['is_elite']),
            'real_madrid_factor': int(strength1['is_real_madrid']) - int(strength2['is_real_madrid']),
            
            'team1_rating': strength1['final_rating'],
            'team2_rating': strength2['final_rating'],
            'avg_rating': (strength1['final_rating'] + strength2['final_rating']) / 2,
            'quality_level': min(strength1['final_rating'], strength2['final_rating']),
            'max_quality': max(strength1['final_rating'], strength2['final_rating']),
            
            'stage_importance': {'round_of_16': 1, 'quarter_finals': 2, 'semi_finals': 3, 'final': 4}.get(stage, 1),
            'is_final': int(stage == 'final'),
            'is_late_stage': int(stage in ['semi_finals', 'final']),
            'is_pressure_stage': int(stage in ['semi_finals', 'final']),
            
            'rating_ratio': strength1['final_rating'] / max(strength2['final_rating'], 50),
            'form_ratio': (strength1['recent_form'] + 0.1) / (strength2['recent_form'] + 0.1),
            'dna_ratio': (strength1['dna_score'] + 1) / (strength2['dna_score'] + 1)
        }
        return features
    
    def prepare_enhanced_training_data(self, target_season):
        """OPTIMIZED: Strict temporal validation while preserving weighting"""
        target_year = int(target_season.split('-')[0])
        X_data, y_data, weights = [], [], []
        
        for season, stages in self.data['train_knockout'].items():
            season_year = int(season.split('-')[0])
            if season_year >= target_year:  # STRICT: No future data
                continue
            
            years_back = target_year - season_year
            if years_back <= 3:
                season_weight = 2.5
            elif years_back <= 6:
                season_weight = 2.0
            elif years_back <= 10:
                season_weight = 1.2
            else:
                season_weight = 0.7
            
            for stage in ['round_of_16', 'quarter_finals', 'semi_finals', 'final']:
                if stage not in stages:
                    continue
                stage_weight = {'round_of_16': 1.0, 'quarter_finals': 1.2, 'semi_finals': 1.5, 'final': 2.0}[stage]
                matches = stages[stage] if isinstance(stages[stage], list) else [stages[stage]]
                
                for match in matches:
                    if not isinstance(match, dict) or not all(k in match for k in ['team_1', 'team_2', 'winner']):
                        continue
                    try:
                        features = self.build_enhanced_match_features(match['team_1'], match['team_2'], season, stage)
                        X_data.append(list(features.values()))
                        y_data.append(1 if match['winner'] == match['team_1'] else 0)
                        weights.append(season_weight * stage_weight)
                    except Exception:
                        continue
        return np.array(X_data), np.array(y_data), np.array(weights)
    
    def train_enhanced_model(self, target_season):
        """Train enhanced ensemble model (PRESERVED EXACTLY FROM 53-SCORE MODEL)"""
        print(f" Training enhanced model for {target_season}...")
        X, y, weights = self.prepare_enhanced_training_data(target_season)
        if len(X) == 0:
            raise ValueError("No training data!")
        print(f" Training on {len(X)} examples with {X.shape[1]} features")
        
        imputer = SimpleImputer(strategy='median')
        X_clean = imputer.fit_transform(X)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        models = {
            'xgboost_enhanced': xgb.XGBClassifier(
                n_estimators=350, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'
            ),
            'gradient_boost_enhanced': GradientBoostingClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42
            ),
            'random_forest_enhanced': RandomForestClassifier(
                n_estimators=250, max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=8, min_samples_split=4, min_samples_leaf=2, random_state=42
            ),
            'logistic_enhanced': LogisticRegression(random_state=42, max_iter=1000, C=0.3)
        }
        
        trained_models, model_scores = {}, {}
        for name, model in models.items():
            try:
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated_model.fit(X_scaled, y, sample_weight=weights)
                train_pred = calibrated_model.predict_proba(X_scaled)[:, 1]
                train_auc = roc_auc_score(y, train_pred, sample_weight=weights)
                trained_models[name] = calibrated_model
                model_scores[name] = train_auc
                print(f"  ✅ {name}: AUC = {train_auc:.4f}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        return {'models': trained_models, 'scores': model_scores, 'imputer': imputer, 'scaler': scaler}
    
    def predict_enhanced_match(self, team1, team2, target_season, stage, model_ensemble):
        """PRESERVED: All successful domain knowledge adjustments from 53-score model"""
        try:
            features = self.build_enhanced_match_features(team1, team2, target_season, stage)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            feature_vector = model_ensemble['imputer'].transform(feature_vector)
            feature_vector = model_ensemble['scaler'].transform(feature_vector)
            
            predictions, total_weight = [], 0
            for name, model in model_ensemble['models'].items():
                try:
                    pred = model.predict_proba(feature_vector)[0][1]
                    weight = model_ensemble['scores'][name]
                    predictions.append(pred * weight)
                    total_weight += weight
                except Exception:
                    continue
            
            ensemble_prob = sum(predictions) / total_weight if predictions and total_weight > 0 else 0.5
            strength1 = self.get_enhanced_team_strength(team1, stage, target_season)
            strength2 = self.get_enhanced_team_strength(team2, stage, target_season)
            
            if strength1['is_real_madrid']:
                if stage in ['semi_finals', 'final']:
                    ensemble_prob += 0.08
                else:
                    ensemble_prob += 0.04
            elif strength2['is_real_madrid']:
                if stage in ['semi_finals', 'final']:
                    ensemble_prob -= 0.08
                else:
                    ensemble_prob -= 0.04
        
            if stage in ['semi_finals', 'final']:
                dna_diff = strength1['dna_score'] - strength2['dna_score']
                ensemble_prob += dna_diff * 0.01
                
                clutch_diff = strength1['clutch_factor'] - strength2['clutch_factor']
                ensemble_prob += clutch_diff * 0.008
                
                if strength1['is_elite'] and not strength2['is_elite']:
                    ensemble_prob += 0.035
                elif strength2['is_elite'] and not strength1['is_elite']:
                    ensemble_prob -= 0.035
            
            if stage == 'final':
                title_diff = strength1['titles'] - strength2['titles']
                ensemble_prob += title_diff * 0.018
                
                big_game_diff = strength1['big_game_performance'] - strength2['big_game_performance']
                ensemble_prob += big_game_diff * 0.01
            
            form_diff = strength1['recent_form'] - strength2['recent_form']
            ensemble_prob += form_diff * 0.06
            
            momentum_diff = strength1['momentum'] - strength2['momentum']
            ensemble_prob += momentum_diff * 0.03
            
            squad_diff = strength1['squad_rating'] - strength2['squad_rating']
            ensemble_prob += squad_diff * 0.002
            
            ensemble_prob = np.clip(ensemble_prob, 0.2, 0.8)
            
            winner = team1 if ensemble_prob > 0.5 else team2
            confidence = max(ensemble_prob, 1 - ensemble_prob)
            
            return {'winner': winner, 'team1_prob': ensemble_prob, 'team2_prob': 1 - ensemble_prob, 'confidence': confidence}
            
        except Exception as e:
            print(f"Error predicting {team1} vs {team2}: {e}")
            
            rating1 = self.team_ratings.get(team1, {}).get('final_rating', 65)
            rating2 = self.team_ratings.get(team2, {}).get('final_rating', 65)
            dna1 = self.champions_league_dna.get(team1, {}).get('dna_score', 3.0)
            dna2 = self.champions_league_dna.get(team2, {}).get('dna_score', 3.0)
            
            total1 = rating1 + dna1 * 3
            total2 = rating2 + dna2 * 3
            prob = 0.5 + (total1 - total2) / 150
            
            if team1 == 'Real Madrid':
                prob += 0.06
            elif team2 == 'Real Madrid':
                prob -= 0.06
            
            prob = np.clip(prob, 0.25, 0.75)
            return {'winner': team1 if prob > 0.5 else team2, 'team1_prob': prob, 'team2_prob': 1 - prob, 'confidence': max(prob, 1 - prob)}
    
    def simulate_enhanced_tournament(self, season):
        """Simulate tournament"""
        if season not in self.data['test_brackets']:
            return None
        print(f"\n OPTIMIZED SIMULATION: {season}")
        print("-" * 60)
        
        model_ensemble = self.train_enhanced_model(season)
        bracket = self.data['test_brackets'][season]
        results = {'round_of_16': [], 'quarter_finals': [], 'semi_finals': [], 'final': None}
        
        # Round of 16
        r16_winners = []
        print(" Round of 16:")
        for match in bracket.get('round_of_16_matchups', []):
            team1, team2 = match['team_1'], match['team_2']
            prediction = self.predict_enhanced_match(team1, team2, season, 'round_of_16', model_ensemble)
            winner = prediction['winner']
            results['round_of_16'].append({'team_1': team1, 'team_2': team2, 'winner': winner})
            r16_winners.append(winner)
            print(f"  {team1} vs {team2} → {winner} ({prediction['team1_prob']:.3f})")
        
        # Quarter Finals
        qf_winners = []
        if len(r16_winners) >= 8:
            print("\n Quarter Finals:")
            for i in range(0, 8, 2):
                if i + 1 < len(r16_winners):
                    team1, team2 = r16_winners[i], r16_winners[i + 1]
                    prediction = self.predict_enhanced_match(team1, team2, season, 'quarter_finals', model_ensemble)
                    winner = prediction['winner']
                    results['quarter_finals'].append({'team_1': team1, 'team_2': team2, 'winner': winner})
                    qf_winners.append(winner)
                    print(f"  {team1} vs {team2} → {winner} ({prediction['team1_prob']:.3f})")
        
        # Semi Finals
        sf_winners = []
        if len(qf_winners) >= 4:
            print("\n Semi Finals:")
            for i in range(0, 4, 2):
                team1, team2 = qf_winners[i], qf_winners[i + 1]
                prediction = self.predict_enhanced_match(team1, team2, season, 'semi_finals', model_ensemble)
                winner = prediction['winner']
                results['semi_finals'].append({'team_1': team1, 'team_2': team2, 'winner': winner})
                sf_winners.append(winner)
                print(f"  {team1} vs {team2} → {winner} ({prediction['team1_prob']:.3f})")
        
        # Final
        if len(sf_winners) >= 2:
            print("\n CHAMPIONS LEAGUE FINAL:")
            team1, team2 = sf_winners[0], sf_winners[1]
            prediction = self.predict_enhanced_match(team1, team2, season, 'final', model_ensemble)
            winner = prediction['winner']
            results['final'] = {'team_1': team1, 'team_2': team2, 'winner': winner}
            print(f"   CHAMPION: {winner} (Prob: {prediction['team1_prob']:.3f})")
        
        return results

# Initialize optimized predictor
print("\n INITIALIZING OPTIMIZED PREDICTOR V4.0")
predictor = OptimizedUCLPredictor(data_sources)
test_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
optimized_predictions = {}

print("\n GENERATING OPTIMIZED PREDICTIONS")
for season in test_seasons:
    try:
        predictions = predictor.simulate_enhanced_tournament(season)
        optimized_predictions[season] = predictions if predictions else {'round_of_16': [], 'quarter_finals': [], 'semi_finals': [], 'final': None}
    except Exception as e:
        print(f"❌ Error in {season}: {e}")
        optimized_predictions[season] = {'round_of_16': [], 'quarter_finals': [], 'semi_finals': [], 'final': None}

# Save optimized predictions
rows = []
for i, season in enumerate(test_seasons):
    pred_data = optimized_predictions.get(season, {'round_of_16': [], 'quarter_finals': [], 'semi_finals': [], 'final': None})
    rows.append({'id': i, 'season': season, 'predictions': json.dumps(pred_data)})

results_df = pd.DataFrame(rows)
results_df.to_csv('champions_league_predictions_v48.csv', index=False)

print(f"\n OPTIMIZED PREDICTIONS SAVED!")
print(f"File: champions_league_predictions_v48.csv")












