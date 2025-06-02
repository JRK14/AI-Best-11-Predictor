import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
import gspread
from google.oauth2.service_account import Credentials
import sys  # Import sys for command-line arguments
import os  # Import os for path handling
warnings.filterwarnings('ignore')

class Dream11Predictor:
    def __init__(self, match_number=None):  # Add match_number parameter with default None
        self.match_number = match_number  # Store match_number as instance variable
        self.batting_stats = pd.read_csv('Batting stats (csv).csv')
        self.bowling_stats = pd.read_csv('ipl bowling (csv).csv')
        self.schedule = pd.read_csv('ipl schedule 2025 (csv).csv')
        self.pitches = pd.read_csv('ipl pitches (csv).csv')
        self.stadiums = pd.read_csv('ipl stadium data (csv).csv')
        self.conditions = pd.read_csv('ipl ml conditions.txt', sep='\t')
        self.xgb_model = None
        self.playing_11 = None
        self.current_match = None
        self.setup_google_sheets()
        
    def get_current_ist_time(self):
        # Get current UTC time
        utc_now = datetime.utcnow()
        # Convert to IST (UTC+5:30)
        ist_now = utc_now + pd.Timedelta(hours=5, minutes=30)
        return ist_now
        
    def setup_google_sheets(self):
        try:
            # Define the scope
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            
            # Create credentials
            credentials = Credentials.from_service_account_file('credentials.json', scopes=scope)
            
            # Authorize the client
            self.gc = gspread.authorize(credentials)
            
            # Open the spreadsheet with the new URL
            spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1q4oH4qCTAGi0O7pbUCjhCUh1Wv3jsouukHG7i6rVC1A/edit'
            # No console output
            self.spreadsheet = self.gc.open_by_url(spreadsheet_url)
            
            try:
                # First try to access the main squad worksheet
                # No console output
                worksheet = self.spreadsheet.worksheet('SquadData_AllTeams')
                
                # If a specific match number was provided, use it directly
                if self.match_number is not None:
                    # Match numbers in spreadsheet are 1-based, so we use the provided number directly
                    match_number = self.match_number
                    
                    # Get the match details from the schedule
                    if 1 <= match_number <= len(self.schedule):
                        # Adjust for 0-based indexing in DataFrame
                        selected_match = self.schedule.iloc[match_number - 1]
                    else:
                        # Fall back to time-based selection
                        self.match_number = None
                else:
                    # Original time-based match selection logic
                    # Get current date and time in IST
                    current_datetime_ist = self.get_current_ist_time()
                    current_time_ist = current_datetime_ist.time()
                    current_date_ist = current_datetime_ist.date()
                    
                    # Convert schedule dates to datetime and extract time
                    self.schedule['Date'] = pd.to_datetime(self.schedule['Date']).dt.date
                    self.schedule['Time'] = pd.to_datetime(self.schedule['Time (IST)'], format='%I:%M %p').dt.time
                    
                    # Find matches for today
                    today_matches = self.schedule[self.schedule['Date'] == current_date_ist]
                    
                    if len(today_matches) > 0:
                        # Sort matches by time
                        today_matches = today_matches.sort_values('Time')
                        
                        # Check for ongoing match
                        current_match = None
                        next_match = None
                        
                        for _, match in today_matches.iterrows():
                            match_time = match['Time']
                            match_datetime = datetime.combine(current_date_ist, match_time)
                            
                            # Calculate time since match start in hours
                            time_since_start = (current_datetime_ist - match_datetime).total_seconds() / 3600
                            
                            # If match started less than 1 hour ago, it's ongoing
                            if 0 <= time_since_start <= 1:
                                current_match = match
                                break
                            # If match starts within next 20 minutes
                            elif time_since_start < 0 and abs(time_since_start) * 60 <= 20:
                                next_match = match
                                break
                        
                        if current_match is not None:
                            # Use the ongoing match
                            selected_match = current_match
                        elif next_match is not None:
                            # Use the upcoming match (within 20 minutes)
                            selected_match = next_match
                        else:
                            # Find the next match of the day
                            upcoming_matches = today_matches[today_matches['Time'] > current_time_ist]
                            if len(upcoming_matches) > 0:
                                selected_match = upcoming_matches.iloc[0]
                            else:
                                # No more matches today, get tomorrow's first match
                                selected_match = self.schedule[self.schedule['Date'] > current_date_ist].iloc[0]
                    else:
                        # If no matches today, find the next match
                        selected_match = self.schedule[self.schedule['Date'] > current_date_ist].iloc[0]
                    
                    match_number = selected_match.name + 1  # Adding 1 because index is 0-based
                
                # Set current match info
                self.current_match = {
                    'number': match_number,
                    'home_team': selected_match['Home Team'],
                    'away_team': selected_match['Away Team'],
                    'date': selected_match['Date'],
                    'time': selected_match['Time (IST)']
                }
                
                # Get playing 11 data
                self.playing_11 = self.get_playing_11(match_number)
                
            except gspread.exceptions.WorksheetNotFound as e:
                self.playing_11 = None
                self.current_match = None
        except Exception as e:
            self.playing_11 = None
            self.current_match = None
        
    def preprocess_data(self):
        # Convert date columns to datetime
        self.schedule['Date'] = pd.to_datetime(self.schedule['Date'], format='%d-%b-%y')
        
        # Rename columns to match
        self.batting_stats = self.batting_stats.rename(columns={'PLAYER': 'Player', 'TEAM': 'Team'})
        self.bowling_stats = self.bowling_stats.rename(columns={'PLAYER': 'Player', 'TEAM': 'Team'})
        
        # Merge batting and bowling stats
        self.player_stats = pd.merge(
            self.batting_stats, 
            self.bowling_stats, 
            on=['Player', 'Team'], 
            how='outer'
        )
        
        # Fill NaN values
        self.player_stats = self.player_stats.fillna(0)
        
        # Map player types to roles
        def map_role(player_type):
            if isinstance(player_type, str):
                if 'Wicket Keeper' in player_type:
                    return 'WK'
                elif 'Fast Bowler' in player_type or 'Spinner' in player_type:
                    return 'BOWL'
                elif 'All Rounder' in player_type:
                    return 'AR'
                elif 'Openor' in player_type or 'Middle Order' in player_type:
                    return 'BAT'
            return 'BAT'  # Default to batsman if role is unclear
        
        self.player_stats['Role'] = self.player_stats['Player Type'].apply(map_role)
        
        # Calculate base points
        self.player_stats['Base_Points'] = (
            self.player_stats['RUNS'] * 1 + 
            self.player_stats['4S'] * 0.5 + 
            self.player_stats['6S'] * 1 + 
            self.player_stats['Wickets'] * 25 +
            self.player_stats['3 Wickets in Innings'] * 8
        )
        
        # Prepare features for XGBoost
        self.prepare_features()
        
        # Train XGBoost model
        self.train_xgboost()
        
    def prepare_features(self):
        # Create feature set for XGBoost
        features = self.player_stats.copy()
        
        # Convert categorical variables to numerical
        features['Role_Code'] = features['Role'].map({'WK': 0, 'BAT': 1, 'AR': 2, 'BOWL': 3})
        
        # Select relevant features
        feature_columns = [
            'INNS', 'RUNS', 'AVG', '30S', '50S', '100S', '4S', '6S', 'SR',
            'Overs', 'Runs Conceded', 'Wickets', 'Average', '3 Wickets in Innings',
            'Economy Rate', 'Strike Rate', 'Role_Code', 'Base_Points'
        ]
        
        # Handle missing values
        for col in feature_columns:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                features[col] = features[col].fillna(0)
        
        self.features = features[feature_columns]
        self.target = features['Base_Points']
        
    def train_xgboost(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        
        # Create and train XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Modified fit method without early_stopping_rounds parameter
        # This is compatible with older versions of XGBoost
        self.xgb_model.fit(X_train, y_train)
        
    def predict_points(self, player_data):
        # Prepare player data for prediction
        player_features = player_data[self.features.columns].copy()
        
        # Make prediction
        predicted_points = self.xgb_model.predict(player_features)
        
        # Add some randomness to simulate match conditions
        random_factor = np.random.normal(1, 0.1, size=len(predicted_points))
        predicted_points = predicted_points * random_factor
        
        return predicted_points
        
    def get_todays_match(self):
        if self.current_match:
            return self.current_match['home_team'], self.current_match['away_team']
        return None, None
    
    def predict_best_11(self):
        team1, team2 = self.get_todays_match()
        if team1 is None:
            return "No match scheduled for today"
        
        # If playing 11 is available, use only those players
        if self.playing_11 is not None:
            playing_11_df = pd.DataFrame(self.playing_11)
            
            # Filter only playing players
            playing_11_df = playing_11_df[playing_11_df['IsPlaying'] == 'PLAYING']
            
            # Create a mapping of player names
            player_name_map = {}
            for _, row in self.player_stats.iterrows():
                player_name = row['Player']
                for _, p11_row in playing_11_df.iterrows():
                    if player_name.lower() in p11_row['Player Name'].lower() or p11_row['Player Name'].lower() in player_name.lower():
                        player_name_map[p11_row['Player Name']] = player_name
                        break
            
            # Get stats for playing 11 players using the name mapping
            team_players = []
            for _, p11_row in playing_11_df.iterrows():
                player_name = p11_row['Player Name']
                if player_name in player_name_map:
                    player_stats = self.player_stats[self.player_stats['Player'] == player_name_map[player_name]].copy()
                    if not player_stats.empty:
                        # Update role based on playing 11 data
                        player_stats.loc[:, 'Role'] = p11_row['Role']
                        team_players.append(player_stats)
            
            if team_players:
                team_players = pd.concat(team_players)
            else:
                # No playing players found in the match data, using all available players from both teams
                team_players = self.player_stats[
                    (self.player_stats['Team'] == team1) | 
                    (self.player_stats['Team'] == team2)
                ].copy()
        else:
            # Get all players from both teams
            team_players = self.player_stats[
                (self.player_stats['Team'] == team1) | 
                (self.player_stats['Team'] == team2)
            ].copy()
        
        # Add Role_Code for prediction
        team_players['Role_Code'] = team_players['Role'].map({'WK': 0, 'BAT': 1, 'AR': 2, 'BOWL': 3})
        
        # Predict points using XGBoost
        team_players['Predicted_Points'] = self.predict_points(team_players)
        
        # Sort by predicted points
        team_players = team_players.sort_values('Predicted_Points', ascending=False)
        
        # Select best 11 players following Dream11 rules
        wk = team_players[team_players['Role'] == 'WK'].head(1)
        batsmen = team_players[team_players['Role'] == 'BAT'].head(5)
        all_rounders = team_players[team_players['Role'] == 'AR'].head(2)
        bowlers = team_players[team_players['Role'] == 'BOWL'].head(3)
        
        # Combine all players
        best_11 = pd.concat([wk, batsmen, all_rounders, bowlers])
        
        # If we don't have enough players, try to fill with highest scoring players from any role
        remaining_slots = 11 - len(best_11)
        if remaining_slots > 0:
            used_players = set(best_11['Player'])
            remaining_players = team_players[~team_players['Player'].isin(used_players)]
            additional_players = remaining_players.head(remaining_slots)
            best_11 = pd.concat([best_11, additional_players])
        
        # Select captain and vice-captain (highest predicted points)
        best_11 = best_11.sort_values('Predicted_Points', ascending=False)
        best_11['Captain'] = False
        best_11['Vice_Captain'] = False
        best_11.iloc[0, best_11.columns.get_loc('Captain')] = True
        best_11.iloc[1, best_11.columns.get_loc('Vice_Captain')] = True
        
        return best_11[['Player', 'Team', 'Role', 'Predicted_Points', 'Captain', 'Vice_Captain']]
    
    def display_team(self):
        best_11 = self.predict_best_11()
        if isinstance(best_11, str):
            # Skip console output
            return
        
        # Save to file only, no console output
        self.save_predictions_to_csv(best_11)
        
    def save_predictions_to_csv(self, best_11):
        try:
            # Get user's Downloads folder path
            home_dir = os.path.expanduser("~")
            downloads_folder = os.path.join(home_dir, "Downloads")
            
            # Create downloads folder if it doesn't exist (for Docker environment)
            if not os.path.exists(downloads_folder):
                os.makedirs(downloads_folder)
            
            # Format date for filename (YYYYMMDD format)
            if isinstance(self.current_match['date'], str):
                match_date = pd.to_datetime(self.current_match['date']).strftime('%Y%m%d')
            else:
                match_date = self.current_match['date'].strftime('%Y%m%d')
            
            # Create match-specific filename
            match_number = self.current_match['number']
            home_team = self.current_match['home_team']
            away_team = self.current_match['away_team']
            
            # Fixed format for the output filename with match number
            match_filename = f"Vit_Gladiators_Match{match_number}_{home_team}vs{away_team}_{match_date}.csv"
            
            # Standard output filename
            standard_filename = "Vit_Gladiators_Output.csv"
            
            # Full paths for saving
            match_filepath = os.path.join(downloads_folder, match_filename)
            standard_filepath = os.path.join(downloads_folder, standard_filename)
            
            # Also save to current directory (for Docker environment)
            local_match_filepath = match_filename
            local_standard_filepath = standard_filename
            
            # Create a copy of the dataframe and remove the Predicted_Points column
            output_df = best_11.copy()
            if 'Predicted_Points' in output_df.columns:
                output_df = output_df.drop('Predicted_Points', axis=1)
                
            # Combine Captain and Vice_Captain columns into a single C/VC column
            output_df['C/VC'] = 'NA'
            for idx, row in output_df.iterrows():
                if row['Captain']:
                    output_df.at[idx, 'C/VC'] = 'C'
                elif row['Vice_Captain']:
                    output_df.at[idx, 'C/VC'] = 'VC'
            
            # Drop the original Captain, Vice_Captain, and Role columns
            output_df = output_df.drop(['Captain', 'Vice_Captain', 'Role'], axis=1)
            
            # Save to both the downloads folder and current directory
            output_df.to_csv(match_filepath, index=False)
            output_df.to_csv(standard_filepath, index=False)
            output_df.to_csv(local_match_filepath, index=False)
            output_df.to_csv(local_standard_filepath, index=False)
            
            # Minimal output - only confirmation
            print(f"Predictions saved to: {standard_filename}")
        except Exception as e:
            print(f"Error saving predictions to file: {str(e)}")

    def get_team_data(self, team_name):
        try:
            # Get the squad data
            squad_worksheet = self.spreadsheet.worksheet('SquadData_AllTeams')
            squad_data = squad_worksheet.get_all_records()
            
            # Filter for the specified team
            team_data = [player for player in squad_data if player['Team'] == team_name]
            
            # Get the playing 11 if available
            playing_11 = self.get_playing_11(self.current_match['number'])
            
            if playing_11:
                # Filter team data to only include players in the playing 11
                playing_11_names = [player['Name'] for player in playing_11]
                team_data = [player for player in team_data if player['Name'] in playing_11_names]
                print(f"\nUsing confirmed playing 11 for {team_name}")
            else:
                print(f"\nWARNING: No playing 11 data available for match {self.current_match['number']}")
                print(f"Using full squad data for {team_name} - this may include players not in the final playing 11")
            
            return team_data
        except Exception as e:
            print(f"\nERROR: Could not get team data for {team_name}: {str(e)}")
            return []

    def get_playing_11(self, match_number):
        try:
            # Try both formats of the worksheet name (with and without leading space)
            try:
                worksheet = self.spreadsheet.worksheet(f'Match_{match_number}')
            except gspread.exceptions.WorksheetNotFound:
                try:
                    worksheet = self.spreadsheet.worksheet(f' Match_{match_number}')
                except gspread.exceptions.WorksheetNotFound:
                    return None
            
            playing_11 = worksheet.get_all_records()
            
            # Filter out substitutes and non-playing players
            playing_11 = [player for player in playing_11 
                         if player.get('IsPlaying', '').strip() == 'PLAYING']
            
            if not playing_11:
                return None
                
            # Map player names to match the format in stats
            for player in playing_11:
                player['Player'] = player['Player Name']
                # Map player type to role
                if player['Player Type'] == 'WK':
                    player['Role'] = 'WK'
                elif player['Player Type'] == 'BAT':
                    player['Role'] = 'BAT'
                elif player['Player Type'] == 'ALL':
                    player['Role'] = 'AR'
                elif player['Player Type'] == 'BOWL':
                    player['Role'] = 'BOWL'
                else:
                    player['Role'] = 'BAT'  # Default to BAT if unknown
            
            return playing_11
        except Exception as e:
            return None

if __name__ == "__main__":
    # Check if match number is provided as command-line argument
    match_number = None
    
    if len(sys.argv) > 1:
        try:
            match_number = int(sys.argv[1])
            # No verbose output
        except ValueError:
            # No verbose output
            pass
    
    # Create predictor instance with match number if provided
    predictor = Dream11Predictor(match_number=match_number)
    predictor.preprocess_data()
    predictor.display_team() 