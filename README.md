# IPL Dream11 Predictor

A machine learning model that predicts optimal Dream11 fantasy cricket teams for Indian Premier League (IPL) matches.

## Overview

This project uses historical player data, match conditions, and stadium information to predict which players are likely to perform well in upcoming IPL matches. It applies machine learning techniques to recommend an optimal Dream11 fantasy team selection.

## Features

- Predicts the best 11 players for Dream11 fantasy teams
- Considers player statistics (batting/bowling)
- Factors in pitch conditions and stadium characteristics
- Analyzes team matchups and historical performance
- Adjusts predictions based on playing 11 information from Google Sheets
- Supports Docker for easy deployment

## Data Sources

The model uses several data sources:
- Player batting statistics (`Batting stats (csv).csv`)
- Player bowling statistics (`ipl bowling (csv).csv`)
- IPL 2025 match schedule (`ipl schedule 2025 (csv).csv`)
- Pitch characteristics (`ipl pitches (csv).csv`)
- Stadium data (`ipl stadium data (csv).csv`)
- ML condition settings (`ipl ml conditions.txt`)

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - pandas==1.3.5
  - numpy==1.21.5
  - scikit-learn==1.0.2
  - xgboost==1.5.0
  - gspread==5.1.0
  - google-auth==2.6.0
  - google-auth-oauthlib==0.5.1
  - google-auth-httplib2==0.1.0

## Setup & Usage

### Local Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your Google Sheets credentials in `credentials.json`
4. Run the prediction model:
   ```
   # For the next upcoming match
   python3 dream11_predictor.py
   
   # For a specific match number (e.g., match #38)
   python3 dream11_predictor.py 38
   ```

### Docker Installation

1. Build the Docker image:
   ```
   docker build -t vit_gladiators_gameathon .
   ```

2. Run the container:
   ```
   # For the next upcoming match
   docker run --rm -v ~/Downloads:/data vit_gladiators_gameathon
   
   # For a specific match number (e.g., match #5)
   docker run --rm -v ~/Downloads:/data vit_gladiators_gameathon 5
   ```

   This will save output files to your Downloads directory.

### Command Reference

```bash
# Python execution
python3 dream11_predictor.py [match_number]

# Docker execution
docker build -t vit_gladiators_gameathon .
docker run --rm -v ~/Downloads:/data vit_gladiators_gameathon [match_number]
```

## Output

The model generates:
- A CSV file with predicted optimal Dream11 team selections
- Points estimates for each selected player
- Team composition following Dream11 rules (captain, vice-captain, etc.)

## How It Works

The model:
1. Loads and preprocesses player statistics
2. Retrieves playing 11 information from Google Sheets
3. Uses XGBoost to predict player performance
4. Optimizes team selection based on Dream11 rules
5. Outputs the best predicted team

## License

MIT License

## Last Updated
Verified tar file update: 2023-05-17 