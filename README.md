# Dota2Predict

A program to give a prediction on a Dota2 game depending on input parameters (team heroes composition, team names).
The project has 3 data files for test purposes, for the Machine Learning Model to train on.

1) The program converts and purifies input data, creates a Machine Learning model, train the model.

2) The program displays accuracy scores of team, corresponding to the data you fed it.

3) Then, the program prompts you with the game's parameters:
	- Hero composition of both teams
	- Team names

4) The program gives a prediction on weather team_1 will win or lose.

## Data sources

Data was collected with the open source Dota2 data platform (https://www.opendota.com/).
Data currently used for the project:
- Pro matches from 26 January 2020 to 15 May 2020
- Pro matches from 17 April 2020 to 15 May 2020
- Pro matches from 01 August 2019 to 15 May 2020

New data can be fetched using SQL requests listed in the sql_request_folder.

## Requirements

Python3 is required for the program to work.

## Configuration

The Machine Learning model is trained based on a .json data file.
This file can be changed before launching the program, inside DataExtractor.py

Defaults files can be found in the resource_data dir.

## Usage

### Launch the program

```bash
python3 Dota2Predict.py
```
Team names will be displayed.

### Enter heroes composition for Team1 and Team2

Hero names to type in can be found in dota2_heroes.csv

### Enter Team1 and Team2 names

Team names to type in were displayed previously.

## Issues

Machine Learning model seems to be overfitting, and accuracy between Dota2 Pro teams seems to vary too much.
