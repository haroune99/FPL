from database import fixtures_23_24, clubs, merged_gws_23_24, teams_23_24
import pandas as pd


def get_top_performers(df, metric, top_n=10):
     return df.nlargest(top_n, metric)[['name', metric, 'position', 'team']]

def khibra(df, selected_col, threshold=10, top_n=10):
    
    differentials = df[(df[selected_col] < threshold)]
    return differentials.nlargest(top_n, 'total_points')[['name', 'total_points', selected_col]]

def find_team_next_difficulty(team_name,gameweek_min,gameweek_max):

    filtered_fixtures = fixtures_23_24[
        ((fixtures_23_24['team_a_name'] == team_name) | (fixtures_23_24['team_h_name'] == team_name)) 
        & (fixtures_23_24['event'].between(gameweek_min, gameweek_max))
    ]

    difficulty = 0

    for _, fixture in filtered_fixtures.iterrows():
        if fixture['team_a_name'] == team_name:
            difficulty += fixture['team_a_difficulty']
        else:
            difficulty += fixture['team_h_difficulty']

    if len(filtered_fixtures) > 0:
        difficulty = difficulty/len(filtered_fixtures)
    else:
        difficulty = 2 

    return difficulty

def suggest_team_to_pick_from(gameweek_min,gameweek_max):
    to_suggest = []
    
    for name in clubs:
        difficulty = find_team_next_difficulty(name, gameweek_min, gameweek_max)
        if difficulty <= 2.75:
            to_suggest.append({'Team Name': name, 'Difficulty': difficulty})

    return pd.DataFrame(to_suggest)

def suggest_players(gameweek_min, gameweek_max, position):

    best_teams = suggest_team_to_pick_from(gameweek_min, gameweek_max)['Team Name'].tolist()

    players = merged_gws_23_24[
        (merged_gws_23_24['team'].isin(best_teams)) &
        (merged_gws_23_24['position'] == position) &
        (merged_gws_23_24['GW'] < gameweek_min)
    ]

    columns = ['assists', 'expected_goals_conceded', 'clean_sheets', 'creativity', 'expected_assists',
               'expected_goal_involvements', 'expected_goals', 'goals_conceded', 'goals_scored',
               'ict_index', 'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards',
               'saves', 'total_points', 'yellow_cards']

    players = players.groupby(['name']).mean(numeric_only=True)[columns]

    if position == 'DEF':
        players = players.sort_values(
            by=['total_points', 'clean_sheets', 'expected_goals_conceded', 'goals_conceded', 'yellow_cards', 'own_goals'],
            ascending=[False, False, True, False, True, False]
        )
    elif position == 'MID':
        players = players.sort_values(
            by=['total_points', 'expected_goals', 'expected_assists', 'goals_scored', 'assists', 'ict_index', 'yellow_cards'],
            ascending=[False, False, False, False, False, False, True]
        )
    elif position == 'FWD':
        players = players.sort_values(
            by=['total_points', 'expected_goals', 'expected_assists', 'goals_scored', 'assists', 'ict_index', 'yellow_cards'],
            ascending=[False, False, False, False, False, False, True]
        )

    return players.index

def suggest_captaincy(team, gameweek):

    players = merged_gws_23_24[
        (merged_gws_23_24['name'].isin(team)) & (merged_gws_23_24['GW'] <= gameweek)
    ]

    players = players.merge(
        teams_23_24,
        left_on='opponent_team',
        right_on='id',
        how='left',
        suffixes=('', '_opponent')
    )

    players['captaincy_score'] = (
        (players['total_points'].fillna(0) * 0.5 +
         players['xP'].fillna(0) * 0.5) /
        players['strength'].fillna(1)
    )

    players = players.sort_values(by=['captaincy_score'], ascending=False)

    return players['name'][:10]

  
