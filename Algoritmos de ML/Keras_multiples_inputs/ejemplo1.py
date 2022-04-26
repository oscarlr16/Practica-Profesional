import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 8)

games_season = pd.read_csv('games_season.csv')
games_season.head()

games_tourney = pd.read_csv('games_tourney.csv')
games_tourney.head()