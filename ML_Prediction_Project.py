              #Predicting User Behavior on Dating Apps Using Machine Learning        

import pandas as pd # For data processing

#Load the dataset
df=pd.read_csv("C:\\Users\\Win 10\\Downloads\\dating_app_behavior_dataset.csv")

# Display 1st 5 rows
df.head()

# Display basic information about the dataset
df.info()

# Get statistical summary for numerical columns
df.describe()
# insights
# 1.There are 50000 rows in each columns.
# Avg time spent on app is 150min/day.
# Avg active hour is 11.52~11.
# std is around 86 ie,there is a high variation.
# min&max app usage time in b/w 0 & 300 ie,some users don't use the app,while others are very active.
# middle 50% spend in b/w 74 & 225 minutes,their last active hour in b/w 5 & 18.
# most users spent around 150 min with high variation.
# most of them active in daytime,while others active in midnight.

# Check for missing values in the dataset
df.isnull().sum()
#There is no missing values.

# Check for duplicate rows
df.duplicated().sum()
# There is no duplicate rows.

# Check data types
df.dtypes

# EDA
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For advanced visualizations

# Histplot:App Usage Time Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['app_usage_time_min'], kde=True, color='blue', bins=20)
plt.title('Distribution of App Usage Time')
plt.xlabel('App Usage Time (min)')
plt.ylabel('Frequency')
plt.show()
# insights
# 1.Most of the users spend wide range of time in the app from 0 to 300 minutes.
# 2.There is no common usage of time.
# 3.App usage time is spread evenly.

# Countplot:Gender Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='gender', palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
# insights
# 1.All genders have almost same count in the dataset.
# 2.There is a slight variation between them. But overall well cleared.
# 3.This reduce gender bias.

# Countplot:Match Outcome Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df,x='match_outcome',palette='Set2')
plt.title('Match Outcome Distribution')
plt.xlabel('Match Outcome')
plt.ylabel('Count')
plt.show()
# insights
# 1.All outcomes have same counts.
# 2.Slightly more users have 'One-Sided Like'.
# 3.Match outcomes are well balanced.It is good for training a classification model.

# Boxplot:Sexual Orientation vs App Usage Time
plt.figure(figsize=(8, 5))
sns.boxplot(x='sexual_orientation', y='app_usage_time_min', data=df, palette="Set2")
plt.title('Sexual Orientation vs App Usage Time')
plt.xlabel('Sexual Orientation')
plt.ylabel('App Usage Time')
plt.show()
# insights
# 1.App usage time is similar at all sexual orientations.
# 2.Avg app usage time is around 150 minutes.
# 3.Some users spend very low ,others spend more time in all groups.

# Piechart:Swipe Time Of Day Distribution
if 'swipe_time_of_day' in df.columns:
    plt.figure(figsize=(8,8))
    df['swipe_time_of_day'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='coolwarm', wedgeprops={'edgecolor': 'black'})
    plt.title('Swipe Time Of Day Distribution')
    plt.show()
# insights
# 1. After midnight swipes are common(ie,~17%).
# 2. Early morning,Late night,Evening & morning shows some similar swipe times (ie, ~16.6%-16.5%).
# 3.Most of the users spend time After midnight.This signifies young / night active users.
# This chart shows a balaced usage among all times.

# Line Chart:Mutual Matches with Likes Received
df_grouped = df.groupby('likes_received')['mutual_matches'].sum()
plt.figure(figsize=(10,5))
plt.plot(df_grouped.index, df_grouped.values, marker='o', linestyle='-', color='teal')
plt.title('Trend of Mutual Matches with Likes Received')
plt.xlabel('Likes Received')
plt.ylabel('Mutual Matches')
plt.xticks(rotation=45)
plt.show()  
# insights
# 1.Mutuial matches increses rapidly at the initial stage with Likes received.
# 2.Between 25-50 likes,mutual matches starts to slow down.
# 3.Afterwards,there is a peak in 125 likes.
# 4.In certain points leads to a significant increase in mutual matches.
# 5.So, more likes refers to more matches at 1st and then the increase in matches slows down.

# Scatter plot:Relationship b/w Profile pics count & Likes received
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['profile_pics_count'], y=df['likes_received'], data=df,color='crimson')  
plt.xlabel('Profile pics count')
plt.ylabel('Likes received')
plt.title('Relationship between Profile Pictures and Likes Received')
plt.show()
# insights
# 1.Users with more profile pics recieved more likes.
# 2.There is an upvard trend shows.No.of profile pics increases as likes also increases.
# 3.Visual presence plays a major role in App engaging. 

# Overall Summary of EDA:
# Users spend various amounts of time on the app, with no clear favorite time.
# The gender and match outcomes are balanced. 
# People use the app differently, no matter of their sexual orientation, most users swipe late at night.
# More likes lead to more mutual matches, but after a certain no.of likes, the increase slows down.
# Users with more profile pictures tend to get more likes.


# Data Preprocessing for Modeling
# Logistic Regression
# Use when the dependent variable is category.
# Here the dependent variable(target) is match_outcome , all other columns are independent variables(features)
 
# Features and Target
X = df.drop('match_outcome', axis=1)  # Features (independent variables)
y = df['match_outcome']  # Target variable (dependent variable)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

# Feature and target split
X = df_encoded.drop('match_outcome', axis=1)
y = df_encoded['match_outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

# Fit & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Best model comparison
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    performance[name] = acc

best_model = max(performance, key=performance.get)
print("\nBest Performing Model:", best_model)
print("Accuracy:", performance[best_model])

# Model Evaluation Insights 
# 1. All models gave an accuracy of around 10%, which is very low.
# 2. This suggests the models are not able to predict match outcomes correctly.
# 3. It may be because the input features are not very useful for prediction.
# 4. Random Forest performed slightly better than other models (10.08% accuracy).
# 5. Confusion matrix shows that predictions are spread randomly across classes.
# 6. This looks like a 10-class classification problem with balanced classes.
# 7. But models behave like guessing randomly, meaning we need better features.

# -------------------- Final Conclusion --------------------

# In this project, we analyzed dating app user behavior and tried to predict match outcomes using machine learning.
# We performed data cleaning, EDA, and applied multiple classification models like Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Naive Bayes.
# All models have a low accuracy of around 10%, indicating that the current features are not strong predictors of match outcome.
# Random Forest showed slightly better performance among the models.
# Overall, the results suggest that more relevant features are needed to improve prediction performance.




