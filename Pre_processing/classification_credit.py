# import of libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# !pip install plotly --upgrade

# data import
base_credit = pd.read_csv('/content/credit_data.csv')

# to view the first records
base_credit.head(10)

# to generate some statistics
base_credit.describe()

# to make filters
base_credit[base_credit['income'] >= 69995.685578]
base_credit[base_credit['clientid'] == 423]
base_credit[base_credit['loan'] <= 1.377630]
base_credit[base_credit['clientid'] == 866]
np.unique(base_credit['default'], return_counts = True)

# visualization of graphics
sns.countplot(x=base_credit['default'])
plt.hist(x = base_credit['age'])
plt.hist(x = base_credit['income'])
plt.hist(x = base_credit['loan'])
graphics = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
graphics.show()

# identifying records with negative ages
base_credit.loc[base_credit['age'] < 0]


# TECHNIQUE TO CORRECT RECORDS WITH NEGATIVE AGES
# FILLING INCONSISTENT VALUES MANUALLY WITH AVERAGE
base_credit.mean()
base_credit['age'].mean()
base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
base_credit.loc[base_credit['age'] < 0]
base_credit.head(27)

# graphics without the influence of records with negative ages
px.scatter_matrix(base_credit, dimensions=['age','income','loan'], color= 'default')

# identifying records with null values
base_credit.isnull().sum()
base_credit.loc[pd.isnull(base_credit['age'])]
base_credit['age'].fillna(base_credit['age'].mean(), inplace = True)
base_credit.loc[pd.isnull(base_credit['age'])]

# graphs without the influence of records with zero ages (blank records)
px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color = 'default')
base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]

# CREATING THE VARIABLES THAT WILL RECEIVE THE FORECASTS AND THE CLASS
# variable X that will receive the forecasters
X_credit = base_credit.iloc[:, 1:4].values
X_credit

type(X_credit)

# variable y that will receive the class
y_credit = base_credit.iloc[:, 4].values
y_credit

type(y_credit)

# SCALING OF ATTRIBUTES
# Standardisation 
# Normalization
X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()
X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()

from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min()
X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max()
X_credit
