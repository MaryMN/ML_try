import os
import argparse
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics




def main(args):
    # create the outputs folder
    os.makedirs('outputs', exist_ok=True)
    
    # Log arguments
   dataset = pd.read_csv('D:\lab1\lab3\Datasets\data.txt')


    dataset.shape
    dataset.head()
    dataset.describe()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    print(regressor.intercept_)

    print(regressor.coef_)

    y_pred = regressor.predict(X_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df
    print(df.head())


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "model.pkl"
    joblib.dump(svm_model, os.path.join('outputs', model_file_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter of the error term')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
