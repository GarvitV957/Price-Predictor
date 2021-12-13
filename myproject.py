import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

df=pd.read_csv("data.csv")

print(df.info())                # info of dataset
print(df.describe())            # describe properties like count, mean, std, min, max


# spliting into test and training data
train , test = train_test_split(df,test_size=.3,random_state=42)
# print(f"Rows in train set: {len(train)}\nRows in test set {len(test)}\n")

# Stratified Shuffle Split w.r.t 'CHAS' attribute so that the ratio of classes is same for both train and test data
split=StratifiedShuffleSplit(n_splits=1, test_size=.3, random_state=42)
for i,j in split.split(df,df['CHAS']):
    strat_train=df.loc[i]
    strat_test=df.loc[j]

house = strat_train.copy()

# Correlation of MEDV attribute with others
corr_matrix = house.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
print (corr_matrix)
# MAXIMUM correlation is coming with 'RM' attribute which implies that with change in RM value, MEDV will change drastically.
plt.scatter(house['RM'],house['MEDV'],alpha=.8)
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.show()

house = strat_train.drop("MEDV", axis=1)
house_price = strat_train["MEDV"].copy()

house["RM"].fillna(house["RM"].median())            # filling NaN entries with median
print("Shape of dataset : ",house.shape)
print()

# print(house.describe())

from sklearn.impute import SimpleImputer

# imputer = SimpleImputer(strategy="median")
# imputer.fit_transform(house)

# print(imputer.statistics_)
# house_tr = pd.DataFrame(imputer, columns=house.columns)
# print(house_tr.describe())

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

house_tr = my_pipeline.fit_transform(house)
print(house_tr.shape)
print()

# Performing on different models so as to get a desired model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def regr(model_name):
    model_name.fit(house_tr, house_price)
    # computing rmse and mape
    housing_predictions = model_name.predict(house_tr)
    mse = mean_squared_error(house_price, housing_predictions)
    rmse = np.sqrt(mse)                                                             # RMSE
    mape= mean_absolute_percentage_error(house_price, housing_predictions)          # MAPE

    print("RMSE: ",rmse)
    print("MAPE: ",mape)
    print()

    # Validating which model has least rmse
    from sklearn.model_selection import cross_val_score
    scores_rmse = cross_val_score(model, house_tr, house_price, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores_rmse)

    def print_scores(scores):
        print("Scores:", scores)
        print("Mean: ", scores.mean())
        print("Standard deviation: ", scores.std())

    print_scores(rmse_scores)


model_lr = LinearRegression()
model_dtr = DecisionTreeRegressor()
model = RandomForestRegressor()

# Linear Regression model
print("Parameters by Linear Regression")
regr(model_lr)
print()

# Decision Tree Regression
print("Parameters by Decision Tree Regressor")
regr(model_dtr)
print()

# Random Forest Regressor
print("Parameters by Random Forest Regressor")
regr(model)
print()

''' It is observed that RMSE and MAPE we get by Random Forset Regressor is less than the other two,
 so we'll use Random Forest Regressor model for our price prediction
 '''

model.fit(house_tr, house_price)

# Saving our model
from joblib import dump, load
dump(model, 'house.joblib')

import pickle
pickle.dump(model,open('model.pkl','wb'))

# Testing our model on test dataset
X_test = strat_test.drop("MEDV", axis=1)
Y_test = strat_test["MEDV"].copy()
X_test_prepared = my_pipeline.fit_transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("RMSE on test dataset: ",final_rmse)