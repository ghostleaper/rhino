#!wget https://raw.githubusercontent.com/andmon97/ATPTennisMatchPredictions/main/Data/all_matches_final.csv

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import math

import warnings
warnings.filterwarnings('ignore')


from sklearn import metrics

from scipy.special import legendre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv('all_matches_final.csv')

players = ['Player1_id','Player1_name','Player2_id','Player2_name']

to_drop = ['tourney_id','tourney_date','score']

cat_cols = ['tourney_name','surface','draw_size','tourney_level','Player1_entry','Player1_hand','Player1_ioc',
            'Player2_entry','Player2_hand','Player2_ioc','best_of','round']


num_cols = ['match_num','Player1_seed','Player1_ht','Player1_age','Player1_rank','Player1_rank_points',
            'Player2_seed','Player2_ht','Player2_age','Player2_rank','Player2_rank_points','minutes','w_ace','w_df',
            'w_svpt','w_1stIn','w_1stWon','w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced','l_ace','l_df','l_svpt','l_1stIn',
            'l_1stWon','l_2ndWon','l_SvGms','l_bpSaved','l_bpFaced']

df = df.drop(to_drop, axis=1)

for i in cat_cols:
    df[i] = df[i].replace(np.NaN, df[i].mode()[0])
for i in num_cols:
    print(i)
    df[i] = df[i].replace(np.NaN,df[i].mean())
def convertCatToNum(dff):
    dff_new = pd.get_dummies(dff, columns=cat_cols) 
    return dff_new
df = convertCatToNum(df)
df.head()
def normalize(dff,col_name_list):
    result = dff.copy()
    for feature_name in col_name_list:
        max_value = dff[feature_name].max()
        min_value = dff[feature_name].min()
        result[feature_name] = (dff[feature_name] - min_value) / (max_value - min_value)
    return result


df = normalize(df,num_cols)
df.head()
num_bins = 10
for i in num_cols:
    plt.hist(df[i],num_bins,density=True, stacked= True,facecolor='blue',alpha=0.7)
    plt.ylabel(i)
    plt.title("Data Distribution for " + i)
    plt.legend()
    plt.grid()
    plt.show()
sns.distplot(df[num_cols[2]], color='b', bins=100, kde_kws={"color": "k", "lw": 3, "label": num_cols[2]},
             hist_kws={'alpha': 0.4});

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPRegressor

Y = pd.DataFrame(df['y'])
df = df.drop(['y'], axis=1)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train = X_train.drop(['Player1_id'], axis=1)
X_train = X_train.drop(['Player1_name'], axis=1)
X_train = X_train.drop(['Player2_id'], axis=1)
X_train = X_train.drop(['Player2_name'], axis=1)

test_names = []
for index, row in X_test.iterrows():
    v = []
    v.append(row['Player1_name'])
    v.append(row['Player2_name'])
    test_names.append(v)

X_test = X_test.drop(['Player1_id'], axis=1)
X_test = X_test.drop(['Player1_name'], axis=1)
X_test = X_test.drop(['Player2_id'], axis=1)
X_test = X_test.drop(['Player2_name'], axis=1)

# import tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# use keras API
model = tf.keras.Sequential()

# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Input(shape=X_train.shape[1]))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.6))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=128)
test_loss, test_acc = model.evaluate(X_test, y_test)

y_pred_test = model.predict(X_test)
y_pred_test = [int(i > .5) for i in y_pred_test]

print(classification_report(y_test, y_pred_test))

model.save('my_model.h5')
