#!/usr/bin/env python
# coding: utf-8

# #Importing relevant libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')


# #Importing the customer support data

# In[4]:


data = pd.read_csv('Customer_support_data.csv', encoding='latin1')


# #Data description

# In[5]:


data.info()


# In[6]:


data.head()


# #Dropping connected_handling_time column

# In[7]:


data=data.drop('connected_handling_time',axis=1)


# #Checking null values

# In[8]:


data.isnull().sum()


# #Removing null values

# In[9]:


data=data.dropna()
display(data.info())


# #Data visualisation

# In[10]:


px.defaults.template = "plotly_dark"

# Visualizing distribution of CSAT scores by agent shift interactively
csat_distribution_by_shift_fig = px.box(data, x='Agent Shift', y='CSAT Score', color='Agent Shift',
                                         title='Distribution of CSAT Scores by Agent Shift',
                                         labels={'Agent Shift': 'Agent Shift', 'CSAT Score': 'CSAT Score'})

# Customizing layout
csat_distribution_by_shift_fig.update_layout(
    xaxis=dict(title='Agent Shift'),
    yaxis=dict(title='CSAT Score'),
    font=dict(family='Arial', size=12, color='white')
)

# Showing interactive graph
csat_distribution_by_shift_fig.show()


# # Creating pie chart of channel distribution

# In[11]:


px.defaults.template = "plotly_dark"


pie_chart_fig = px.pie(data, names='channel_name', title='Channel Distribution',
                       labels={'channel_name': 'Channel'})

# Customize layout
pie_chart_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Showing interactive graph
pie_chart_fig.show()


# 
# # Creating histogram of agent tenure bucket distribution

# In[12]:


tenure_bucket_dist_fig = px.histogram(data, x='Tenure Bucket', title='Agent Tenure Bucket Distribution',
                                      labels={'Tenure Bucket': 'Tenure Bucket'},
                                      color='Tenure Bucket')

# Customize layout
tenure_bucket_dist_fig.update_layout(font=dict(family='Arial', size=12, color='white'))

# Show interactive graph
tenure_bucket_dist_fig.show()


# # Sunburst chart of category and sub-category distribution

# In[13]:


sunburst_chart_fig = px.sunburst(data, path=['category', 'Sub-category'], title='Category and Sub-category Distribution',
                                 labels={'category': 'Category', 'Sub-category': 'Sub-category'})
sunburst_chart_fig.update_layout(font=dict(family='Arial', size=15, color='white'))

# Show interactive graph
sunburst_chart_fig.show()


# # Preparing dataset for implementing machine learning algorithms

# #Defining target *variable*

# In[14]:


X=data.drop(['CSAT Score'], axis=1)
y=data[['CSAT Score']]


# #Converting categorical variables

# In[15]:


X = X.apply(lambda x: pd.factorize(x)[0])


# #Training and testing

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.25, random_state=42)


# #Decision Tree model implementation
# 

# In[17]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)


# In[18]:


confusion_matrix(y_test, y_pred)


# In[19]:


accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))


# # Feature Engineering

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Customer_support_data.csv', encoding='latin1')

# Drop irrelevant columns and rows with missing values
data.drop(['Unique id', 'Order_id', 'order_date_time', 'Customer Remarks', 'Survey_response_Date'], axis=1, inplace=True)
data.dropna(inplace=True)

# Define features and target variable
X = data.drop(['CSAT Score'], axis=1)
y = data['CSAT Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define numerical and categorical features
numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# # Hypermeter Tuning

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint

# Load the dataset
data = pd.read_csv('Customer_support_data.csv', encoding='latin1')

# Drop irrelevant columns and rows with missing values
data.drop(['Unique id', 'Order_id', 'order_date_time', 'Customer Remarks', 'Survey_response_Date'], axis=1, inplace=True)
data.dropna(inplace=True)

# Define features and target variable
X = data.drop(['CSAT Score'], axis=1)
y = data['CSAT Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define numerical and categorical features
numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

# Define hyperparameters to tune
param_dist = {
    'classifier__n_estimators': randint(100, 1000),
    'classifier__max_features': ['auto', 'sqrt'],
    'classifier__max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 20),
    'classifier__bootstrap': [True, False]
}

# Define RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Make predictions using the best model
y_pred = random_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[20]:


tree.plot_tree(dtc)


# # Convoluted Neural Network model implementation

# In[70]:


# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[80]:


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1)
])


# In[81]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[88]:


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[87]:


loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Customer_support_data.csv', encoding='latin1')

# Drop irrelevant columns and rows with missing values
data.drop(['Unique id', 'Order_id', 'order_date_time', 'Customer Remarks', 'Survey_response_Date'], axis=1, inplace=True)
data.dropna(inplace=True)

# Define features and target variable
X = data.drop(['CSAT Score'], axis=1)
y = data['CSAT Score']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the CNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", loss)


# In[12]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

