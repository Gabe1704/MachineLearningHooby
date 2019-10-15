import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import csv



class DataFrameImputer(TransformerMixin):
#Impute NaNs using scikit-learn SimpleImpute Class and getting rid of garbage data

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def ChangeStringToDummyNumbers(dataset):
    dataset = pd.get_dummies(dataset, dummy_na=True)
    return dataset

def PlotGraph(data):
    sns.set()
    sns.set_palette('Spectral')
    sns.set_style('whitegrid')

    sns.jointplot(x='Income in EUR', y='Body Height [cm]', data=dataset)

def LinearRegression(data1, data2):
    #Linear Regression Algorithm
    lm = LinearRegression()

    lm.fit(data1, data2)

    highest = np.argmax(lm.coef_)
    value = np.amax(lm.coef_)
    print(value)


    var = colnames[highest]
    print(var)

    predictions = lm.predict(X_test)

    # Here we'll plot our predictions versus the actual values

    plt.scatter(y_test, predictions)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')


def main():
    #file = "tcd ml 2019-20 income prediction training (with labels).csv"
    #test = "tcd ml 2019-20 income prediction test (without labels)"

    #dataset = {} # all data in one place
    #dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    #dataset2 = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

    reader = csv.reader(open("tcd ml 2019-20 income prediction training (with labels).csv"))
    reader1 = csv.reader(open("tcd ml 2019-20 income prediction test (without labels).csv"))
    f = open("combined.csv", "w")
    writer = csv.writer(f)

    for row in reader:
        writer.writerow(row)
    for row in reader1:
        writer.writerow(row)
    f.close()

    dataset = pd.read_csv("combined.csv")

    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    dataset['Income'] = dataset['Income'].fillna(0)
    print(dataset)


if __name__ == '__main__':
  main()
