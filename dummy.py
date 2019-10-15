# -*- coding: utf-8 -*-
"""
Income Prediction

Importing Data
Data imported from Kaggle competition https://www.kaggle.com/c/tcdml1920-income-ind
Pandas will be used to read csv files
"""

import pandas as pd

data = {} # all data in one place



data.keys()

"""Now let's take a look at the data we have so far."""

for file in data:
  print(file)
  data[file].info()
  print('\n')

"""## Data Cleaning
Currently we have three dataframes to deal with, so I'll start with the 'Seasons_Stats' dataframe.
"""

# Have to remove the first column of the dataframe

data['Seasons_Stats'] = data['Seasons_Stats'].drop(data['Seasons_Stats'].columns[0], axis=1)

data['Seasons_Stats'].head()

"""Taking a look at the head of the 'Seasons_Stats' data, I need to also deal with the data type of 'year' as well as figure out what to do with the lack of advanced data for the players from ye olden days."""

data['Seasons_Stats']['Year'].head()
data['Seasons_Stats']['Year'].isnull().values.any()
#data['Seasons_Stats']['Year'] = data['Seasons_Stats']['Year'].astype('int64')

"""Looks like there are some NaN values in the year column."""

data['Seasons_Stats']['Year'].isnull().sum()

data['Seasons_Stats'][data['Seasons_Stats']['Year'].isnull() == True]

"""I've found the rows where the 'year' column is NaN, and it appears that in those rows every single column is also NaN, which makes our lives a bit easier because we can just delete those directly."""

data['Seasons_Stats'] = data['Seasons_Stats'].dropna(subset=['Year'])

data['Seasons_Stats']['Year'].isnull().sum()

# Nice! No more pesky NaN values in the 'year' column!

data['Seasons_Stats']['Year'] = data['Seasons_Stats']['Year'].astype('int64')

# Now I can convert the years into integers

data['Seasons_Stats'].head()

"""Because of the error that can come about through inflation and time-related issues, I need to cut down on the scope of this dataframe. First, I'll take the subset of player stats that is 1980 or later. Why? Because the 3-point line was first added between 1979 and 1980, thus, a lot of advanced stats would not have existed beforehand.
Because I am cutting out nearly 30 years of NBA history, this NBA salary prediction algorithm would only make sense for 1980 and later. (Which is fine because that's when basketball actually started to be entertaining)
"""

data['Seasons_Stats'] = data['Seasons_Stats'][data['Seasons_Stats']['Year'] > 1980]

data['Seasons_Stats'].head()

"""Let's get rid of some columns that either don't make sense or aren't important."""

data['Seasons_Stats'] = data['Seasons_Stats'].drop(['GS', 'blanl', 'blank2'], axis=1)
data['Seasons_Stats'].head()

"""Now might be a good time to learn what the column names actually mean. Some of these abbreviations are difficult to decipher so I'll figure that out now."""

data['Seasons_Stats'].columns

"""Here are the meanings of the statistics:
*   'Pos' - position
*   'Tm' - team
*   'G' - games played
*   'MP' - minutes played
*   'PER' - player efficiency rating
*   'TS%' - true shooting percentage (weights 3-pointers higher)
*   '3PAr' - 3-point attempt rate
*   'Ftr' - free throw attempt rate
*   'ORB%' - offensive rebound percentage
*   'DRB%' - defensive rebound percentage
*   'TRB%' - total rebound percentage
*   'AST%' - assist percentage
*   'STL%' - steal percentage
*   'BLK%' - block percentage
*   'TOV%' - turnover percentage
*   'USG%' - usage rate
*   'OWS' - offensive win shares
*   'DWS' - defensive win shares
*   'WS' - win shares
*   'WS/48' - win shares over 48 minutes
*   'OBPM' - offensive box plus/minus
*   'DBPM' - defensive box plus/minus
*   'BPM' - box plus/minus
*   'VORP' - value over replacement player
*   'FG' - field goals made
*   'FGA' - field goals attempted
*   'FG%' - field goal percentage'
*   '3P' - 3-pointers made
*   '3PA' - 3-pointers attempted
*   '3P%' - 3-point percentage
*   '2P' - 2-pointers made
*   '2PA' - 2-pointers attempted
*   '2P%' - 2-point percentage'
*   'eFG%' - effective field goal percentage
*   'FT' - free throws made
*   'FTA' - free throws attempted
*   'FT%' - free throw percentage
*   'ORB' - offensive rebounds
*   'DRB' - defensive rebounds
*   'TRB' - total rebounds
*   'AST' - assists
*   'STL' - steals
*   'BLK' - blocks
*   'TOV' - turnovers
*   'PF' - personal fouls
*   'PTS' - points
Before I can clean up other datasets and eventually join a few to make a big dataframe, I need to make sure that there are no more annoying NaN values lurking in the data
"""

data['Seasons_Stats'].count()

# The total number of rows is 18570, but some of the columns are missing values. To simplify everything, I'm just going to turn those missing values to the value of zero.

data['Seasons_Stats'] = data['Seasons_Stats'].fillna(0)

data['Seasons_Stats'].count()

# That's much better.

"""## Salary Data
Lesson learned: data preparation and data cleaning are WITHOUT A DOUBT the most tedious and time-consuming part of data science.
After actually looking (briefly) through the datasets, I see that I need to perform some data wizardry with the 1990-2018 dataframe to extract important salary information.
As mentioned earlier, because of the effect of inflation and time-related issues, I can't just put all of the salaries together and then use machine learning algorithms on those. I need a way to standardize all of the data. That's why I decided to create some metrics like: player's salary as proportion of team payroll, team's payroll as proportion of total payroll (market size), player's salary as proportion of total payroll in the NBA.
Time to manipulate some more data.
"""

data['1990_to_2018'].head()

data['1990_to_2018'].count()

"""It took me a while, but I managed to create some new features based on this dataset, including: team payroll, total NBA payroll, and year.
At first I just downloaded the file into Excel and did some Excel formulas to generate the values, but that was too inefficient, so I created a robust Python script that automated the task VERY quickly.
Here is the [notebook](https://colab.research.google.com/drive/1SV3Q8RLvyQvobya2SVshq9_D4qYUForU) on which I did this.
Now it's time to bring in the new and improved salary data.
"""

url = 'https://raw.githubusercontent.com/jerrytigerxu/NBA-Salary-Prediction/master/data/nba_salaries.csv'

data['salaries'] = pd.read_csv(url)

data['salaries'] = data['salaries'].drop(data['salaries'].columns[0], axis=1)

data['salaries'].head()

"""## Feature Engineering
Now is where the fun really begins!
Let's make some more features that will aid us in our analysis of what are the most predictors of a high salary.
A few I'll make right off the bat are:
*   Salary as proportion of team payroll (player leverage)
*   Salary as proportion of total NBA payroll (league weight)
*   Team payroll as proportion of total NBA payroll (team market size)
*   Region of US
"""

# Creating a Player Leverage column

data['salaries']['Player Leverage'] = data['salaries']['Salary'] / data['salaries']['Team Payroll']

data['salaries'].head()

# Creating a League Weight column
data['salaries']['League Weight'] = data['salaries']['Salary'] / data['salaries']['Total NBA Payroll']

data['salaries'].head()

# Creating a Team Market Size column
data['salaries']['Team Market Size'] = data['salaries']['Team Payroll'] / data['salaries']['Total NBA Payroll']

data['salaries'].head(50)

# Just for fun, using the League Weight column, I can quickly pinpoint which player was paid the most in a particular year

highest = data['salaries']['League Weight'] == max(data['salaries'][data['salaries']['Year'] == 1991]['League Weight'])

data['salaries'][highest]['Player']

# Let's find the find the highest paid players for every year

start_year = 1991

for i in range(17):
  highest = data['salaries']['League Weight'] == max(data['salaries'][data['salaries']['Year'] == (1991+i)]['League Weight'])

  print(data['salaries'][highest]['Player'].values)

"""Now let's create the US Region column.
I'll make it simple and divide up the country (as well as Toronto) into four regions based on how the US government does it: Northeast, Midwest, South, and West
**Northeast**: New York, Boston, Brooklyn, Philadelphia, Toronto
**Midwest**: Cleveland, Detroit, Indiana, Milwaukee, Minnesota, Chicago
**South**: Atlanta, Charlotte, Miami, Orlando, New Orleans, Washington, Dallas, Oklahoma City, Houston, San Antonio, Memphis
**West**: Golden State, Denver, Utah, Sacramento, Los Angeles Lakers, Los Angeles Clippers, Phoenix, Portland
"""

data['salaries']['Team'].unique()

data['salaries']['US Region'] = 0

# Dummy entry

data['salaries'].head()

for rowNum in range(len(data['salaries'].index.values)):
  if data['salaries'].iloc[rowNum]['Team'] in ['New York Knicks', 'Boston Celtic', 'Brooklyn Nets', 'Philadelphia 76ers', 'Toronto Raptors']:
    data['salaries'].loc[rowNum, 'US Region'] = 'Northeast'
  elif data['salaries'].iloc[rowNum]['Team'] in ['Cleveland Caveliers', 'Detroit Pistons', 'Indiana Pacers', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'Chicago Bulls']:
    data['salaries'].loc[rowNum, 'US Region'] = 'Midwest'
  elif data['salaries'].iloc[rowNum]['Team'] in ['Atlanta Hawks', 'Charlotte Hornets', 'Miami Heat', 'Orlando Magic', 'New Orleans Pelicans', 'Washington Wizards', 'Dallas Mavericks', 'Oklahoma City Thunder', 'Houston Rockets', 'San Antonio Spurs', 'Memphis Grizzlies']:
    data['salaries'].loc[rowNum, 'US Region'] = 'South'
  elif data['salaries'].iloc[rowNum]['Team'] in ['Golden State Warriors', 'Denver Nuggets', 'Utah Jazz', 'Sacramento Kings', 'Los Angeles Lakers', 'Los Angeles Clippers', 'Phoenix Suns', 'Portland Trail Blazers']:
    data['salaries'].loc[rowNum, 'US Region'] = 'West'

data['salaries'].head(100)

"""## Merging Datasets
Now it's time to put together the two datasets that we've spent a lot of time working on!
Because the salary data starts in the 90s, I'm going to have to cut off another decade off of my 'Seasons_Stats' dataframe. This might hurt some of the analysis because it only encompasses the modern era of the NBA, but because of the proportion features I've created, much of the effect of time should (hopefully) be mitigated.
"""

data['player_stats'] = pd.merge(data['Seasons_Stats'], data['salaries'], how='left', left_on=['Year', 'Player'], right_on=['Year', 'Player'])

data['player_stats'].head()

data['player_stats'] = data['player_stats'][data['player_stats']['Year'] > 1990]

data['player_stats'].head(200)

"""Very nice!
There are a few rows where the 'salaries' data didn't contain the players listed in the 'Seasons_Stats' dataframe, so for the sake of simplicity (because those players probably weren't that big of a deal anyways) I'll just get rid of those rows
"""

data['player_stats'] = data['player_stats'].dropna()

data['player_stats'].count()

# No NA values!

data['player_stats'].head(500)

# Small nitpicking to just remove the extra team column.

data['player_stats'] = data['player_stats'].drop(columns=['Tm'])

data['player_stats'].head()

"""## Dealing with Categorical Variables
We've got a few variables that are not numerical and thus cannot be directly worked with in any machine learning algorithm that works with continuous data.
The variables we have to turn into dummy variables are Position, US Region, and Team
"""

# Creating dummy variables for Position first
positions = pd.get_dummies(data['player_stats']['Pos'])
data['player_stats'] = pd.concat([data['player_stats'], positions], axis=1)

data['player_stats'].head()

data['player_stats']['Pos'].unique()

"""Something very interesting that comes up for positions is that some players play multiple positions in their careers, so the dummy variables reflected that. Perhaps that might actually make a difference in the analysis?"""

# Creating dummy variables for US Region
regions = pd.get_dummies(data['player_stats']['US Region'])
data['player_stats'] = pd.concat([data['player_stats'], regions], axis=1)

data['player_stats'].head()

# Creating dummy variables for Team
positions = pd.get_dummies(data['player_stats']['Team'])
data['player_stats'] = pd.concat([data['player_stats'], positions], axis=1)

data['player_stats'].head()

# There are a LOT of dummy variables for 'Team'

"""## Exploratory Data Analysis
Finally, FINALLY we can explore this data and find some juicy relationships!
*NOTE: Because there are SO many variables to account for, my EDA could have been more extensive. Feel free to play around with the data in the .ipynb notebook or in the Python file!*
"""

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set_palette('Spectral')
sns.set_style('whitegrid')

"""As a quick refresher, here are all of the features and what they mean.
*   'Pos' - position
*   'G' - games played
*   'MP' - minutes played
*   'PER' - player efficiency rating
*   'TS%' - true shooting percentage (weights 3-pointers higher)
*   '3PAr' - 3-point attempt rate
*   'Ftr' - free throw attempt rate
*   'ORB%' - offensive rebound percentage
*   'DRB%' - defensive rebound percentage
*   'TRB%' - total rebound percentage
*   'AST%' - assist percentage
*   'STL%' - steal percentage
*   'BLK%' - block percentage
*   'TOV%' - turnover percentage
*   'USG%' - usage rate
*   'OWS' - offensive win shares
*   'DWS' - defensive win shares
*   'WS' - win shares
*   'WS/48' - win shares over 48 minutes
*   'OBPM' - offensive box plus/minus
*   'DBPM' - defensive box plus/minus
*   'BPM' - box plus/minus
*   'VORP' - value over replacement player
*   'FG' - field goals made
*   'FGA' - field goals attempted
*   'FG%' - field goal percentage'
*   '3P' - 3-pointers made
*   '3PA' - 3-pointers attempted
*   '3P%' - 3-point percentage
*   '2P' - 2-pointers made
*   '2PA' - 2-pointers attempted
*   '2P%' - 2-point percentage'
*   'eFG%' - effective field goal percentage
*   'FT' - free throws made
*   'FTA' - free throws attempted
*   'FT%' - free throw percentage
*   'ORB' - offensive rebounds
*   'DRB' - defensive rebounds
*   'TRB' - total rebounds
*   'AST' - assists
*   'STL' - steals
*   'BLK' - blocks
*   'TOV' - turnovers
*   'PF' - personal fouls
*   'PTS' - points
*   'Salary' - the annual salary for the particular player in that particular year
*   'Team' - the team that the particular player played in for that particular year
*  'Team Payroll' - the total amount of money spent on player salaries for that particular team for that particular year
*   'Total NBA Payroll' - the total amount of money spent on player salaries for the entire NBA in that particular year
*   'Player Leverage' - the proportion of the total team payroll that particular player had in that particular year
*   'League Weight' - the proportion of the total NBA payroll that particular player had in that particular year
*   'Team Market Size' - the proportion of the total NBA payroll that particular team had in that particular year
*   'US Region' - the region in which that team is located
Let's explore some relationships between the variables
"""

sns.jointplot(x='Salary', y='League Weight', data=data['player_stats'])

"""This may seem silly at first to plot league weight with salary because they basically are explaining the same thing, mainly, the amount of money a player is getting relative to the rest of the league. However, looking at this plot, we can see why the Salary variable might not be the best predictor variable. The near straight lines that can be inferred from the plot show that League Weight and Salary are almost perfectly correlated, but because of inflation and other time-related factors, the salary amount that would give a player a league weight of 0.01 in one year might give another player (or the same player) a completely different league weight in another year.
The time value of money is powerful and could really mess up our analysis if we're not careful.
**For now, we should focus on predicting the League Weight variable.**
"""

# Comparing win shares per 48 minutes with league weight

sns.jointplot(x='WS/48', y='League Weight', data=data['player_stats'])

# Comparing total points with league weight

sns.jointplot(x='PTS', y='League Weight', data=data['player_stats'])

# Comparing VORP with League Weight

sns.jointplot(x='VORP', y='League Weight', data=data['player_stats'])

# Comparing USG% with League Weight

sns.jointplot(x='USG%', y='League Weight', data=data['player_stats'])

# Here is a strange outlier

data['player_stats'][data['player_stats']['USG%'] == data['player_stats']['USG%'].max()]

# Getting rid of this outlier
data['player_stats'] = data['player_stats'].drop(8227)

"""## Predictive Model Building
Let's get to the best part, the machine learning algorithms!
For this set of data, trying to predict a continuous variable, it might be best to use multiple regression, though we could also use decision trees or random forest.
### Train Test Split
Let's split the data into training and testing data
"""

from sklearn.model_selection import train_test_split

# The predictor variable is 'League Weight', and since League Weight is calculated from salary and the related variables, I'll remove the salary-related variables

colnames = list(data['player_stats'])

colnames.remove('Player')
colnames.remove('Pos')
colnames.remove('US Region')
colnames.remove('Salary')
colnames.remove('League Weight')
colnames.remove('Total NBA Payroll')
colnames.remove('Team Payroll')
colnames.remove('Team')
colnames.remove('Player Leverage')
print(colnames)

# Here we are trying to predict the league weight of the players

y = data['player_stats']['League Weight']
X = data['player_stats'][colnames]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

"""### Regression
Let's start simple for our predictive model and just use linear regression. It may be the most basic algorithm, but sometimes simple yields the best results.
"""

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

# Let's take a look at our coefficients
# WARNING: There will be a LOT

print('Coefficients: \n', lm.coef_)

# Which variable had the biggest impact on our model?
import numpy as np

highest = np.argmax(lm.coef_)
value = np.amax(lm.coef_)
print(value)


var = colnames[highest]
print(var)

"""Now this is interesting! Something seemingly unimportant like what position a player plays can actually have a pretty big impact!
Just the position of center has a coefficient of more than 3%! (Of course, this is based on the linear regression model so it might not be fully accurate)
### Predicting Test Data for Regression
Let's see how our model performed!
"""

predictions = lm.predict(X_test)

# Here we'll plot our predictions versus the actual values

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

"""### Evaluation of Regression Model
Let's get some concrete numbers on the performance of our model. We'll use the metrics: mean absolute error, mean squared error, and root mean squared error.
Later we'll compare these numbers between all of the algorithms we use in this project.
"""

from sklearn import metrics


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

"""### Decision Tree Model
Though decision trees are mainly used for categorical data because the structure of it allows for either distinctions for non-continuous data, it can still be used here. Let's see how it stacks up!
"""

from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()

dtree.fit(X_train, y_train)

"""### Evaluation of Decision Tree Model"""

dt_predictions = dtree.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, dt_predictions))
print('MSE:', metrics.mean_squared_error(y_test, dt_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dt_predictions)))

"""### Random Forest Model"""

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)

"""### Evaluation of Random Forest Model"""

rfr_predictions = rfr.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, rfr_predictions))
print('MSE:', metrics.mean_squared_error(y_test, rfr_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_predictions)))

"""## Comparison of Models
We have three models: linear regression, decision trees, and random forests. Let's compare all of the numbers to see which ones are most accurate.
"""

print('Linear Regression: ' )
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('\n')

print('Decision Tree Model: ' )
print('MAE:', metrics.mean_absolute_error(y_test, dt_predictions))
print('MSE:', metrics.mean_squared_error(y_test, dt_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dt_predictions)))
print('\n')

print('Random Forest Model: ')
print('MAE:', metrics.mean_absolute_error(y_test, rfr_predictions))
print('MSE:', metrics.mean_squared_error(y_test, rfr_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_predictions)))

"""Based on these numbers, it looks like linear regression actually did the best! Like I mentioned before, sometimes the simplest method leads to the best results.
*NOTE: I didn't do any tuning of the hyperparameters for the decision tree and random forest models so perhaps I didn't get the optimal models for them. Feel free to tune the models to improve them!*
# Potential Improvements
Finally, we've covered the entire basic data science workflow by simply hunting down an answer for a simple yet difficult question to answer: what factors determine who gets paid a higher NBA salary?
This process was quite extensive, and thus, there is much room for improvement. With this project, I've prepared a template for any of you to improve upon and yield even better results than I did. Here are a few things (not comprehensive) that could be improved:
### 1. Dealing with the time value of money more effectively
Because of the fact that the value of money changed over time, I knew that I couldn't compare salaries from the 1990s to the salaries of players today. Instead of converting the numbers to a single adjusted number, I opted to created a metric called League Weight that would make all of the player salaries relative to one another in terms of who has a greater proportion of the total payroll. Maybe this wasn't the best way to deal with the problem. Feel free to find a better way to adjust the salary numbers.
### 2. Getting more over-arching data
I only used data from 1990 to the present because any salary data and advanced statistics were sparse and difficult to find for the years earlier. Before 1979 there wasn't even a 3-point line! Nonetheless, for people who are less impatient than me, don't be afraid to look for more historic data that might allow for a more comprehensive analysis of the salary data of NBA history.
### 3. Adding more features
I used a LOT of variables in this project, but the significance of the variables to the final outcome definitely differed a lot. There are many other advanced stats in the NBA that I didn't use that others could add.
### 4. Using more machine learning algorithms
I only used three machine learning algorithms, and all of them were pretty simple. I considered using neural networks but that would have required a lot of hyperparameter tuning.
### 5. Fine-tuning the algorithms already used
Even amongst the few algorithms I used in this project, for decision trees and random forests, I could definitely have tuned the models to make them more effective.
"""
