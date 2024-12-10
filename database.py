import pandas as pd # type: ignore
import re
import psycopg2
import requests
import sqlalchemy
from sqlalchemy import create_engine
from psycopg2 import sql

merged_gws_23_24 = pd.read_csv('/Users/harouneaaffoute/Desktop/PL fantasy/Fantasy-Premier-League-master/data/2023-24/gws/merged_gw_new.csv')
fixtures_23_24 = pd.read_csv('/Users/harouneaaffoute/Desktop/PL fantasy/Fantasy-Premier-League-master/data/2023-24/fixtures_new.csv')
teams_23_24 = pd.read_csv('/Users/harouneaaffoute/Desktop/PL fantasy/Fantasy-Premier-League-master/data/2023-24/teams_new.csv')
clubs = teams_23_24['name']


merged_gws_23_24['kickoff_time'] = pd.to_datetime(merged_gws_23_24['kickoff_time'])
fixtures_23_24['kickoff_time'] = pd.to_datetime(fixtures_23_24['kickoff_time'])
fixtures_23_24 = fixtures_23_24.merge(teams_23_24[['team_id', 'name']], left_on='team_a', right_on='team_id', how='left'
).rename(columns={'name': 'team_a_name'})
fixtures_23_24 = fixtures_23_24.merge(teams_23_24[['team_id', 'name']], left_on='team_h', right_on='team_id', how='left'
).rename(columns={'name': 'team_h_name'})


merged_gws_23_24['value'] = merged_gws_23_24['value']/10
merged_gws_23_24['selected'] = ((merged_gws_23_24['selected']/(merged_gws_23_24.groupby('GW')['selected'].transform('sum')))*100).round(1)
merged_gws_23_24['selected'] = merged_gws_23_24['selected'].fillna(0)
merged_gws_23_24['croq_index'] = merged_gws_23_24['goals_scored']/merged_gws_23_24['expected_goals']
merged_gws_23_24['croq_index'] = merged_gws_23_24['croq_index'].fillna(0)
merged_gws_23_24['efficiency'] = merged_gws_23_24['total_points']/merged_gws_23_24['value']
merged_gws_23_24['efficiency'] = merged_gws_23_24['efficiency'].fillna(0)


conn = psycopg2.connect(
            host = "localhost",
            database = "FPL",
            user = "postgres")

cur = conn.cursor()

engine = create_engine('postgresql+psycopg2://postgres@localhost:5432/FPL')


merged_gws_23_24.to_sql('merged_gws_23_24', engine, if_exists='replace', index=False)
fixtures_23_24.to_sql('fixtures_23_24', engine, if_exists='replace', index=False)
teams_23_24.to_sql('teams_23_24', engine, if_exists='replace', index=False)
clubs.to_sql('clubs', engine, if_exists='replace', index=False)
