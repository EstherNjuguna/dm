#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn .ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score,KFold
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read data
df =  pd.read_excel("diabetes.xlsx")
df.head()


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


# df.dropna(inplace=True)


# In[7]:


#Visualisation of all variables before imputation
df.hist(figsize=(20,25), bins=15)
plt.show()


# In[8]:


diabetes_counts = df['Diabetes '].value_counts()

# Plot the counts
plt.figure(figsize=(8, 6))
ax = diabetes_counts.plot(kind='bar', color=['skyblue', 'blue'])
plt.title('Distribution of Diabetes Status')
plt.xlabel('Diabetes Status')
plt.ylabel('Frequency')
plt.xticks(ticks=range(len(diabetes_counts.index)), labels=['Non-Diabetic (0.0)', 'Diabetic (1.0)'], rotation=0)

# Adding value labels on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 8),
                textcoords='offset points')

plt.show()


# In[9]:


# Step 1: Identify columns with numerical types and low unique values
df.select_dtypes(include=['number'])
categorical_like_columns = [col for col in df.columns if (df[col].dtype == 'float' or df[col].dtype == 'int') and df[col].nunique() < 10]

# Step 2: Add specific columns to the list
categorical_like_columns = categorical_like_columns + [ 'race', 'household_size', 'insurance','gender','income']

# Step 3: Convert identified columns to 'category' type
for col in categorical_like_columns:
    df[col] = df[col].astype('category')

# Display the DataFrame info to confirm changes
print(df.info())



# In[10]:


df['glucose'] = df['glucose'] * 0.0555


# Missing values

# In[11]:


null_percentages = df.isnull().sum()/len(df)*100
null_percentages[null_percentages > 60]


# In[12]:


df.info()


# In[13]:


df= df.rename(columns={'Diabetes ': 'Diabetes'})


# In[14]:


numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

# Set a threshold for high correlation
high_corr_threshold = 0.9
low_corr_threshold = 0.1

# Find pairs of highly correlated features
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [(column, idx) for column in upper.columns for idx in upper.index if abs(upper[column][idx]) > high_corr_threshold]



# Drop one feature from each pair of highly correlated features
features_to_drop = set([pair[1] for pair in high_corr_pairs] )

print(f"High correlation pairs: {high_corr_pairs}")
print(f"Features to drop: {features_to_drop}")

# Create a reduced dataframe with the selected features removed
df_reduced = df.drop(columns=features_to_drop)
print(f"Remaining columns: {df_reduced.columns.tolist()}")


# In[15]:


df_dm = df.drop(columns = [ 'seqn','fvc', 'fev1', 'fev1_fvc_ratio','insurance', 'private_insur',
      'medicare', 'medicaid', 'military_insur','gen_health', 'no_insurance','asthma', 'pulse','first', 'diabetes','last','days_active',  'cancer', 'drugs', 'copd','glucose.1',])


# In[16]:


# Step 1: Identify columns with numerical types and low unique values
df_dm.select_dtypes(include=['number'])
categorical_like_columns = [col for col in df_dm.columns if (df_dm[col].dtype == 'float' or df_dm[col].dtype == 'int') and df_dm[col].nunique() < 10]

# Step 2: Add specific columns to the list
categorical_like_columns = categorical_like_columns + ['race', 'household_size', 'gender','income']

# Step 3: Convert identified columns to 'category' type
for col in categorical_like_columns:
    df_dm[col] = df_dm[col].astype('category')

# Display the DataFrame info to confirm changes
print(df_dm.info())


# In[17]:


df_dm.info()


# In[18]:


df.shape


# In[19]:


df_dm.isna().sum()


# In[20]:


# Define the imputer
imputer = KNNImputer(n_neighbors=5)

# List of columns to impute
columns_to_impute = df_dm.drop(columns=['Diabetes']).columns

# Fit the imputer on the dataset
imputer.fit(df_dm[columns_to_impute])

# Check if all columns exist in the DataFrame
missing_columns = [col for col in columns_to_impute if col not in df_dm.columns]
if missing_columns:
    raise ValueError(f"Columns missing from the DataFrame: {missing_columns}")

# Transform the dataset
df_dm[columns_to_impute] = imputer.fit_transform(df_dm[columns_to_impute])


# Replace remaining NaNs with zeros
df_dm.replace(np.nan, 0, inplace=True)
df_dm.dropna(inplace=True)


# In[21]:


df_dm.isna().sum()


# In[22]:


df_dm


# In[ ]:





# In[23]:


# df_dm['glucose'] = df_dm['glucose'] * 0.0555


# In[24]:


df_dm.info()


# In[25]:


df_dm.sample(5)


# In[26]:


scaler = StandardScaler()

# Created a copy of original data (named it=df1)
df1 = df_dm.copy(deep=True)
df1.select_dtypes(include=['number'])
categorical_like_columns = [col for col in df1.columns if (df1[col].dtype == 'float' or df1[col].dtype == 'int') and df1[col].nunique() < 10]

# Step 2: Add specific columns to the list
categorical_like_columns = categorical_like_columns + ['race', 'household_size', 'gender','income']

# Step 3: Convert identified columns to 'category' type
for col in categorical_like_columns:
    df1[col] = df1[col].astype('category')
columns_to_scale = [col for col in df1.columns if df1[col].dtype not in ['category', 'object']]
df1[columns_to_scale] = scaler.fit_transform(df1[columns_to_scale])


# In[27]:


df1.sample(5)


# In[28]:


df1.select_dtypes('category')


# In[29]:


df1.describe()


# ## Model training

# In[30]:


X = df1.drop(columns='Diabetes')
y = df1['Diabetes']


# In[31]:


y.isna().sum()


# In[32]:


print(f"X_train_htn shape: {X.shape}")
print(f"y_train_htn shape: {y.shape}")


# In[33]:


X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3, random_state=42)


# In[34]:


smote = SMOTE(sampling_strategy='auto', random_state=42)
x_train_rez,y_train_rez= smote.fit_resample(X=X_train,y=y_train)


# In[35]:


x_train_rez.shape


# In[36]:


y_train_rez.shape


# In[37]:


y_test.shape


# In[38]:


balanced_count = y_train_rez.value_counts()
# Plot the bar chart
plt.figure(figsize=(8, 6))
ax = balanced_count.plot(kind='bar', color=['skyblue', 'blue'])
plt.title('Distribution of Diabetes status in a balanced datas set')
plt.xlabel('Diabetes Status')
plt.ylabel('Frequency')
plt.show()


# In[ ]:





# In[ ]:





# In[39]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced')

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to data
grid_search.fit(x_train_rez, y_train_rez)

# Get the best parameters and model
print("Best Parameters: ", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

# Evaluate best model
predictions_dm = best_rf_model.predict(X_test)
accuracy_dm = accuracy_score(y_test, predictions_dm)
print("Tuned DM Model Accuracy:", accuracy_dm)
print("\nClassification Report:")
print(classification_report(y_test, predictions_dm))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions_dm)

# Plotting the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[40]:


from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(best_rf_model, x_train_rez, y_train_rez, cv=5)

# Print average cross-validation score
print("Average Cross-Validation Score: ", cv_scores.mean())


# In[41]:


import pickle

# Save the model using pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved using pickle.")

# Step 3: Load the model and scaler using pickle
with open('random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
def load_model_and_scaler():
    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Make predictions using the loaded model
predictions_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, predictions_loaded)
print("Loaded Model Accuracy:", accuracy_loaded)


# In[42]:


from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Assuming y_test is the true labels and y_proba is the predicted probabilities
y_proba = best_rf_model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

# Plotting the ROC Curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plotting the Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})', color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()


# In[43]:


selected_features = [ 'age', 'weight_kg', 'height_cm', 'bmi', 'sys_bp', 'dia_bp',
       'glucose']
# Create a new DataFrame with only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Retrain the model using the selected features
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
model_selected = RandomForestClassifier( bootstrap=False, 
    max_depth=10, 
    max_features='sqrt', 
    min_samples_leaf=2, 
    min_samples_split=5, 
    n_estimators=300, class_weight='balanced'


)  

model_selected.fit(X_train_selected, y_train)

# Evaluate the retrained model
from sklearn.metrics import classification_report, accuracy_score

y_pred_selected = model_selected.predict(X_test_selected)
print("Accuracy of Retrained Model: ", accuracy_score(y_test, y_pred_selected))
print(classification_report(y_test, y_pred_selected))


# In[44]:


import joblib

# Save the trained model to a file
joblib.dump(model_selected, 'random_forest_model1.pkl')

print("Model saved to random_forest_model.pkl")


# ## Xgboost

# In[45]:


import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3],  # for imbalanced classes
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, enable_categorical=True)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Print best parameters and evaluate the model
print("Best Parameters: ", grid_search.best_params_)
best_xgb_model = grid_search.best_estimator_

# Predict and evaluate the performance on test data
y_pred = best_xgb_model.predict(X_test)
print("Accuracy of Retrained XGBoost Model: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[46]:


selected_features = [ 'age', 'weight_kg', 'height_cm', 'bmi', 'sys_bp', 'dia_bp',
       'glucose']
xgb_model = xgb.XGBClassifier(
    random_state=42, 
    scale_pos_weight=1,
    enable_categorical=True,
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=2,
    n_estimators=300,
    subsample=0.8
)

# Create a new DataFrame with only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Fit the model
xgb_model.fit(X_train_selected, y_train)

# Evaluate the retrained model
y_pred_selected = xgb_model.predict(X_test_selected)

# Print the accuracy and classification report
print("Accuracy of Retrained Model: ", accuracy_score(y_test, y_pred_selected))
print(classification_report(y_test, y_pred_selected))


# In[47]:


# from sklearn.svm import SVC

# # Initialize SVM model with class_weight='balanced' to handle imbalance
# svm_model = SVC(class_weight='balanced', random_state=42)

# # Fit the model
# svm_model.fit(x_train_rez, y_train_rez)

# # Make predictions
# predictions_svm = svm_model.predict(X_test)
# accuracy_svm = accuracy_score(y_test, predictions_svm)
# print("SVM Model Accuracy:", accuracy_svm)

# # Classification Report
# print(classification_report(y_test, predictions_svm))


# In[48]:


from sklearn.ensemble import VotingClassifier

# Create a voting classifier with different models
voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='hard')

# Fit the voting classifier
voting_clf.fit(x_train_rez, y_train_rez)

# Make predictions
predictions_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, predictions_voting)
print("Voting Classifier Accuracy:", accuracy_voting)

# Classification Report
print(classification_report(y_test, predictions_voting))


# In[49]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Create base models (e.g., RandomForest, XGBoost, SVM)
base_learners = [
    ('rf', RandomForestClassifier(class_weight='balanced')),
    ('xgb', XGBClassifier(scale_pos_weight=1,enable_categorical=True)),
    # ('svm', SVC(class_weight='balanced', probability=True))
]

# Create a meta-model (Logistic Regression in this case)
meta_model = LogisticRegression()

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Fit the stacking model
stacking_clf.fit(x_train_rez, y_train_rez)

# Make predictions and evaluate
predictions_stacking = stacking_clf.predict(X_test)
accuracy_stacking = accuracy_score(y_test, predictions_stacking)
print("Stacking Classifier Accuracy:", accuracy_stacking)


# In[50]:


# from imblearn.under_sampling import NearMiss
# from imblearn.over_sampling import SMOTE
# from collections import Counter

# # Initialize NearMiss (you can choose NearMiss-1, NearMiss-2, NearMiss-3)
# nearmiss = NearMiss(sampling_strategy='auto', version=1)  # NearMiss-1 by default

# # Apply NearMiss to undersample the majority class
# X_train_nearmiss, y_train_nearmiss = nearmiss.fit_resample(X_train, y_train)

# # Check the class distribution after NearMiss
# print("Class distribution after NearMiss:", Counter(y_train_nearmiss))

# # Train your model with the resampled data
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train_nearmiss, y_train_nearmiss)

# # Evaluate model performance on the test set
# y_pred = model.predict(X_test)
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))


# In[51]:


# from imblearn.under_sampling import TomekLinks
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from collections import Counter

# # Apply Tomek Links to remove majority class samples
# tomek = TomekLinks(sampling_strategy='auto')
# X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)

# # Check the class distribution after Tomek Links
# print("Class distribution after Tomek Links:", Counter(y_train_tomek))

# # Train a RandomForest model on the Tomek Links resampled data
# rf_model_tomek = RandomForestClassifier(random_state=42)
# rf_model_tomek.fit(X_train_tomek, y_train_tomek)

# # Make predictions on the test set
# predictions_tomek = rf_model_tomek.predict(X_test)

# # Evaluate the model
# accuracy_tomek = accuracy_score(y_test, predictions_tomek)
# print("Accuracy after Tomek Links:", accuracy_tomek)

# # Classification Report
# print("Classification Report after Tomek Links:")
# print(classification_report(y_test, predictions_tomek))


# In[52]:


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_train)

# Split the data into training and validation sets
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Check the shape and labels
print("X_train_dl shape:", X_train_dl.shape)
print("y_train_dl shape:", y_train_dl.shape)
print("Unique classes in y_train_dl:", np.unique(y_train_dl))

# Ensure the data is in numpy array format
X_train_dl = np.array(X_train_dl)
X_val_dl = np.array(X_val_dl)
y_train_dl = np.array(y_train_dl)
y_val_dl = np.array(y_val_dl)

# Check data types and ensure they are numeric
print("Data type of X_train_dl:", X_train_dl.dtype)
print("Data type of y_train_dl:", y_train_dl.dtype)

# Create the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train_dl.shape[1]),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_dl, y_train_dl, epochs=30, batch_size=64, validation_data=(X_val_dl, y_val_dl), class_weight={0: 1, 1: 3})

# Evaluate the model on test data
X_test_scaled = scaler.transform(X_test)
predictions_dl = model.predict(X_test_scaled)
predictions_dl = (predictions_dl > 0.5).astype(int)

# Print accuracy and classification report
accuracy_dl = accuracy_score(y_test, predictions_dl)
print("Deep Learning Model Accuracy:", accuracy_dl)
print("Classification Report:")
print(classification_report(y_test, predictions_dl))


# In[53]:


import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Assuming StandardScaler was used

# Load the saved model and scaler for diabetes prediction
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler used during model training

# Streamlit app structure
st.title("Health Risk Prediction App")

# Diabetes Prediction Section
st.header("Diabetes Prediction with Selected Features")

# Input form for user data (Diabetes)
with st.form("user_input_form"):
    age = st.number_input("Age", min_value=0, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
    height_cm = st.number_input("Height (cm)", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    sys_bp = st.number_input("Systolic Blood Pressure", min_value=0.0, step=0.1)
    dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, step=0.1)
    glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
    
    # Submit button for diabetes prediction
    submitted = st.form_submit_button("Predict Diabetes Risk")

if submitted:
    # Prepare the input data for diabetes prediction
    input_data = pd.DataFrame({
        'age': [age],
        'weight_kg': [weight_kg],
        'height_cm': [height_cm],
        'bmi': [bmi],
        'sys_bp': [sys_bp],
        'dia_bp': [dia_bp],
        'glucose': [glucose]
    })

    # Apply the same scaling as was done during training
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction for diabetes
    prediction = model.predict(input_data_scaled)
    
    # Display the result for diabetes prediction
    if prediction[0] == 1:
        st.success("The model predicts a risk of diabetes.")
    else:
        st.success("The model predicts no risk of diabetes.")

# Depression Risk Assessment Section (PH9)
st.header("Depression Risk Assessment (PH9)")

# Section for PH9 questions
st.subheader("Please answer the following questions:")

# PH9 Questions
ph9_answers = {
    "Little interest or pleasure in doing things?": st.radio("1. Little interest or pleasure in doing things?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling down, depressed, or hopeless?": st.radio("2. Feeling down, depressed, or hopeless?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Trouble falling or staying asleep, or sleeping too much?": st.radio("3. Trouble falling or staying asleep, or sleeping too much?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling tired or having little energy?": st.radio("4. Feeling tired or having little energy?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Poor appetite or overeating?": st.radio("5. Poor appetite or overeating?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down?": st.radio("6. Feeling bad about yourself, or that you are a failure, or have let yourself or your family down?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Trouble concentrating on things, such as reading the newspaper or watching television?": st.radio("7. Trouble concentrating on things, such as reading the newspaper or watching television?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?": st.radio("8. Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Thoughts that you would be better off dead, or of hurting yourself in some way?": st.radio("9. Thoughts that you would be better off dead, or of hurting yourself in some way?", ("Not at all", "Several days", "More than half the days", "Nearly every day"))
}

# Mapping responses to numeric values for scoring
score_map = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Calculate the total score based on responses
if st.button("Assess Depression Risk"):
    ph9_score = sum([score_map[answer] for answer in ph9_answers.values()])
    
    # Depression Risk Classification based on PH9 score
    if ph9_score < 5:
        risk_level = "Low risk of depression."
    elif 5 <= ph9_score < 15:
        risk_level = "Moderate risk of depression."
    else:
        risk_level = "High risk of depression."
    
    # Display the result for depression risk assessment
    st.write(f"Total PH9 Score: {ph9_score}")
    st.write(risk_level)

