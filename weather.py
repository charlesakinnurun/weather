# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# Data Loading

# %%
df = pd.read_csv("weather.csv")
print(df.head().to_string())
print(df.info())

# %% [markdown]
# Data Preprocesing and Feature Engineering

# %%
# Data Preprocesing is critical for machine learning models
# First, we need to convert the categorical target variable ("weather") into a numerical labels,
# as most Ml algorithms require numerical input. We'll use th LabelEncoder for this
le = LabelEncoder()
df["weather_encoded"] = le.fit_transform(df["weather"])

# Next, we select the features (X) and the target (y)
# We'll drop the original "weather" column and the "date" column, as it's not directly
# useful for the specific model without futher feature engineering
features = ["precipitation","temp_max","temp_min","wind"]
target = "weather_encoded"

X = df[features]
y = df[target]

print("Shape of the features (X):",X.shape)
print("Shape of target (y):",y.shape)

# %% [markdown]
# Data Splitting

# %%
# We'll split the data into the training sets to evaluate our model on unseen data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Number of samples in training set:",len(X_train))
print("Number of samples in testing set:",len(X_test))

# %% [markdown]
# Model Training

# %%
# We initialize our Decision Tree Classifier model and train it using the training data
print("Training the Decision Tree Classifier model")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
print("Model Training Complete")

# %% [markdown]
# Model Evaluation

# %%
# We make predictions on the test set and evaluate the model's performance using
# accuracy and a detailed classification report
print("Evaluating the model")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred,target_names=le.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Clasification Report:",report)

# %% [markdown]
# Visualization

# %%
# A confusion matrix helps visualize how well our model performed on each class
# The rows represent the actual classes, and the columns represent the predicted classes
# The diagonal values show the number of correct predictions for each class

print("Plotting confusion matrix.....")
cm = confusion_matrix(y_test,y_pred,labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=le.classes_)
plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(True)
plt.show()

# %% [markdown]
# Making a New Prediction

# %%
# We can use the trained model to predict the weather for a new data point.
# Note: The values are hypothetical for demonstration
new_data = pd.DataFrame([[5.0, 10.0, 5.0, 3.0]], columns=features)
predicted_label = model.predict(new_data)
predicted_weather = le.inverse_transform(predicted_label)

print(f"Example prediction for a new data point")
print(f"Features: precipitation=5.0,temp_max=10.0,temp_min=5.0,wind=3.0")
print(f"Predicted Weather: {predicted_weather[0]}")


