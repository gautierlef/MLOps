import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def clean_dataset():
    pd.options.mode.chained_assignment = None

    player_pergame = pd.read_csv("../data/01_raw/Player Per Game.csv")
    player_allstar = pd.read_csv("../data/01_raw/All-Star Selections.csv")
    player_awards = pd.read_csv("../data/01_raw/Player Award Shares.csv")
    team_stats = pd.read_csv("../data/01_raw/Team Summaries.csv")
    advanced_stats = pd.read_csv("../data/01_raw/Advanced.csv")

    # Adding a column to indicate whether a player was an all-star in that season.
    player_pergame["all_star"] = False
    star_season = []
    for i in range(len(player_allstar)):
        star = player_allstar.at[i, 'player']
        season = player_allstar.at[i, 'season']
        star_season.append([star, season])

    for i in range(len(player_pergame)):
        player = player_pergame.at[i, 'player']
        season = player_pergame.at[i, 'season']
        player_season = [player, season]
        if player_season in star_season:
            player_pergame.at[i, 'all_star'] = True

    # Adding column for winning percentage.
    team_stats['win_perc'] = team_stats.apply(lambda x: x['w'] / (x['w'] + x['l']), axis=1)
    team_stats = team_stats[['season', 'abbreviation', 'win_perc', 'w', 'l']]
    team_player_stats = pd.merge(player_pergame, team_stats, how='left', left_on=['season', 'tm'],
                                 right_on=['season', 'abbreviation'])

    # Removing all data prior to 1980
    data_modern = team_player_stats[team_player_stats['season'] >= 1980]
    data_modern.describe()

    # Reformatting data so that the index is the player ID and season and dealing with missing data due to combining
    # datasets.
    data_modern.loc[:]['season'] = pd.Categorical(data_modern['season'])
    data_modern = data_modern[data_modern['win_perc'].notna()]
    data_modern = data_modern[data_modern['abbreviation'].notna()]

    # Adding column to indicate what conference a player is in.
    east = ['MIL', 'TOT', 'ATL', 'IND', 'ORL',
            'BOS', 'DET', 'CHI', 'BRK', 'WAS', 'MIA', 'CHO', 'NYK',
            'CLE', 'TOR', 'PHI', 'CHA', 'NJN', 'CHH', 'WSB', 'KCK', ]
    west = ['MIN', 'DAL', 'DEN', 'OKC', 'SAC', 'SAS', 'HOU', 'LAC', 'GSW',
            'POR', 'LAL', 'PHO', 'MEM', 'NOP', 'UTA', 'NOH', 'SEA', 'NOK', 'VAN', 'SDC']

    data_modern['conf'] = 'east'
    for index, row in data_modern.iterrows():
        team = row['tm']
        if team in west:
            data_modern.at[index, 'conf'] = 'west'

    # Creating a column to indicate the percentage of games a player played.
    data_modern['total_games'] = data_modern.apply(lambda x: x['w'] + x['l'], axis=1)
    data_modern['games_pct'] = data_modern.apply(lambda x: x['g'] / x['total_games'], axis=1)

    # Missing data will be filled in with the median in the column. Also removing the 2023 season as that data
    # will be used for the final predictions.
    data_model1 = data_modern.copy()
    data_model1 = data_model1[data_model1['season'] != 2023]
    data_model1 = data_model1.fillna(data_model1.median(numeric_only=True))
    f = open("../data/primary/primary.csv", "a")
    f.write(data_model1.to_csv())
    f.close()
    return data_model1


def training(data_model1):
    def transform(x_cols):
        return make_column_transformer(
            (OneHotEncoder(), ['pos']),
            (StandardScaler(), x_cols[1:]),  # Everything except for 'pos'
        )

    def get_train_test(data, test_size):
        global train
        global test
        train_years = list(range(1980,2023))
        test_years = []
        for i in range(test_size):
            year = random.choice(train_years)
            train_years.remove(year)
            test_years.append(year)

        train = data.loc[data['season'].isin(train_years)]
        test = data.loc[data['season'].isin(test_years)]

        print("Test Seasons:")
        print(test_years)

    get_train_test(data_model1, 6)

    # Fitting and scoring the logistic regression model.
    x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game", "drb_per_game",
              "trb_per_game",
              "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game", "pts_per_game", "win_perc",
              "ft_per_game"]
    trans = transform(x_cols)

    y_col = "all_star"

    model = Pipeline([
        ("tr", trans),
        ("lr", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])

    model.fit(train[x_cols], train[y_col])
    model.score(test[x_cols], test[y_col])

    def prediction():
        def rank_predict(test, model):
            test["prob_all_star"] = model.predict_proba(test[x_cols])[:, 1]
            test['rank'] = test.groupby(['season', 'conf'])['prob_all_star'].rank(ascending=False)
            test['pred_all_star'] = np.where((test['rank'] <= 12.0), True, False)

        rank_predict(test, model)

        finalmodel = Pipeline([
            ("tr", trans),
            ("lr",
             LogisticRegression(fit_intercept=False, max_iter=1000, C=0.5455594781168515, class_weight="balanced")),
        ])
        data_modern = data_modern.fillna(data_modern.median(numeric_only=True))
        train = data_modern[data_modern['season'] != 2023]
        test = data_modern[data_modern['season'] == 2023]
        finalmodel.fit(train[x_cols], train[y_col])
        rank_predict(test, finalmodel)

        test['rank'] = test['rank'].astype(int)
        print("2023 ALL-STAR PREDICTIONS:")
        print("Eastern Conference:")
        print((test[(test["pred_all_star"] == True) & (test["conf"] == 'east')].sort_values('rank')[
            ['player', 'rank']]).to_string(index=False))
        print()
        print("Western Conference:")
        print((test[(test["pred_all_star"] == True) & (test["conf"] == 'west')].sort_values('rank')[
            ['player', 'rank']]).to_string(index=False))