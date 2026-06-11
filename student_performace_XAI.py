#Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
#Importing SHAP
import shap
#Importing LIME
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
#Loading Dataset and Displaying First 5 Rows
df = pd.read_csv('student-por.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
df.head(5)
# Create binary target variable (pass if G3 >= 10, else fail)
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
# Dropping raw grade columns
df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)
# Preparing Data for Modelling
X = df_encoded.drop('pass', axis=1)
y = df_encoded['pass']
# Splitting the Data to Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Evaluation Function
def evaluate_model(model, X_test, y_test, title="Model"):
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
else None
print(f"\n{title} Classification Report:\n", classification_report(y_test,
y_pred))
print(f"{title} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
if y_prob is not None:
auc = roc_auc_score(y_test, y_prob)
print(f"{title} ROC-AUC Score: {auc:.4f}")
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"{title} (AUC = {auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
#Random Forest Models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test, "RF Default")
rf_trial1 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_trial1.fit(X_train, y_train)
evaluate_model(rf_trial1, X_test, y_test, "RF Trial 1")
rf_trial2 = RandomForestClassifier(n_estimators=100, min_samples_split=5,
class_weight='balanced', random_state=42)
rf_trial2.fit(X_train, y_train)
evaluate_model(rf_trial2, X_test, y_test, "RF Trial 2")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_sm = RandomForestClassifier(random_state=42)
rf_sm.fit(X_train_sm, y_train_sm)
evaluate_model(rf_sm, X_test, y_test, "RF SMOTE")
#SVM Models
svm_param_grid = {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 1]}
gs_svm_no_smote = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3,
n_jobs=-1)
gs_svm_no_smote.fit(X_train, y_train)
evaluate_model(gs_svm_no_smote.best_estimator_, X_test, y_test, "SVM No SMOTE")
gs_svm_smote = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3, n_jobs=-1)
gs_svm_smote.fit(X_train_sm, y_train_sm)
evaluate_model(gs_svm_smote.best_estimator_, X_test, y_test, "SVM SMOTE")
svm_trial1 = SVC(C=0.1, kernel='linear', probability=True)
svm_trial1.fit(X_train, y_train)
evaluate_model(svm_trial1, X_test, y_test, "SVM Trial 1 (C=0.1, linear)")
svm_trial2 = SVC(C=10, kernel='rbf', gamma='scale', probability=True)
svm_trial2.fit(X_train, y_train)
evaluate_model(svm_trial2, X_test, y_test, "SVM Trial 2 (C=10, RBF)")
#MLP Models
mlp_param_grid = {
'hidden_layer_sizes': [(50,), (50,50)],
'activation': ['relu'],
'learning_rate_init': [0.001],
'solver': ['adam'],
'max_iter': [300]
}
gs_mlp_no_smote = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid,
cv=3, n_jobs=-1)
gs_mlp_no_smote.fit(X_train, y_train)
evaluate_model(gs_mlp_no_smote.best_estimator_, X_test, y_test, "MLP No SMOTE")
gs_mlp_smote = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid, cv=3,
n_jobs=-1)
gs_mlp_smote.fit(X_train_sm, y_train_sm)
evaluate_model(gs_mlp_smote.best_estimator_, X_test, y_test, "MLP SMOTE")
mlp_trial1 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
learning_rate_init=0.001, max_iter=300, random_state=42)
mlp_trial1.fit(X_train, y_train)
evaluate_model(mlp_trial1, X_test, y_test, "MLP Trial 1 (100 units)")
mlp_trial2 = MLPClassifier(hidden_layer_sizes=(50, 50), activation='tanh',
learning_rate_init=0.01, max_iter=300, random_state=42)
mlp_trial2.fit(X_train, y_train)
evaluate_model(mlp_trial2, X_test, y_test, "MLP Trial 2 (50,50 tanh)")
#Summary Table
def summarize_models(models, names):
results = []
for model, name in zip(models, names):
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
results.append([name, acc, f1, rec, prec])
return pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "Recall",
"Precision"])
final_models = [rf, rf_trial1, rf_trial2, rf_sm, gs_svm_no_smote.best_estimator_,
gs_svm_smote.best_estimator_, svm_trial1, svm_trial2,
gs_mlp_no_smote.best_estimator_, gs_mlp_smote.best_estimator_, mlp_trial1,
mlp_trial2]
final_names = ["RF Default", "RF Trial 1", "RF Trial 2", "RF SMOTE", "SVM No
SMOTE", "SVM SMOTE", "SVM Trial 1", "SVM Trial 2", "MLP No SMOTE", "MLP SMOTE",
"MLP Trial 1", "MLP Trial 2"]
summary_df = summarize_models(final_models, final_names)
print("\nModel Comparison Summary:\n")
print(summary_df)
#WITH FEATURE ENGINEERING
#Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,
roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
#Loading of Dataset and Preprocessing
df = pd.read_csv('student-por.csv', encoding='utf-8')
df.columns = df.columns.str.strip()
assert 'G3' in df.columns
# Define engineered features and their descriptions
engineered_features = {
"study_efficiency": "Study time divided by travel time; measures how
efficiently a student uses their time.",
"parental_education_avg": "Average of mother's and father's education levels;
indicates home academic support.",
"support_score": "Sum of various support indicators (family support, school
support, tutoring, activities).",
"alcohol_score": "Total alcohol consumption score (weekday + weekend).",
"stress_level": "Sum of study time, travel time, and absences; used as a proxy
for academic or life stress.",
"commitment_ratio": "Ratio of study time to free time; shows how much a student
prioritizes studying.",
"high_risk_flag": "Binary indicator; 1 if absences > 10 and past class failures
> 1, else 0."
}
# Convert to a DataFrame
engineered_features_df = pd.DataFrame(
list(engineered_features.items()),
columns=["Engineered Feature", "Description"]
)
# Display the table
engineered_features_df
# Create binary target variable (pass if G3 >= 10, else fail)
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
# Dropping raw grade columns
df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
# Implementing Feature Engineering (adding new features)
# Ensuring traveltime is at least 1
df['traveltime'] = df['traveltime'].replace(0, 1)
df['study_efficiency'] = df['studytime'] / (df['traveltime'] + 1)
if 'Medu' in df.columns and 'Fedu' in df.columns:
df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
# Mapping support columns from yes/no to 1/0
support_columns = ['schoolsup', 'famsup', 'paid', 'activities']
for col in support_columns:
if df[col].dtype == 'object':
df[col] = df[col].map({'yes': 1, 'no': 0})
df['support_score'] = df[support_columns].sum(axis=1)
if 'Dalc' in df.columns and 'Walc' in df.columns:
df['alcohol_score'] = df['Dalc'] + df['Walc']
df['stress_level'] = df['studytime'] + df['traveltime'] + df['absences']
df['commitment_ratio'] = df['studytime'] / (df['freetime'] + 1)
df['high_risk'] = ((df['absences'] > 10) & (df['failures'] > 1)).astype(int)
# One-hot encoding after feature engineering
df_encoded = pd.get_dummies(df, drop_first=True)
# Preparing Data for Modelling
X = df_encoded.drop('pass', axis=1)
y = df_encoded['pass']
# Get feature names from X DataFrame before scaling
feature_names = list(X.columns)
# Splitting the Data to Test and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Evaluation Function
def evaluate_model(model, X_test, y_test, title="Model"):
y_pred = model.predict(X_test)
# Check if predict_proba is available
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
else None
print(f"\n{title} Classification Report:\n", classification_report(y_test,
y_pred))
print(f"{title} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
if y_prob is not None:
auc = roc_auc_score(y_test, y_prob)
print(f"{title} ROC-AUC Score: {auc:.4f}")
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"{title} (AUC = {auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
#Random Forest Models
rf_trial1 = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42)
rf_trial1.fit(X_train, y_train)
evaluate_model(rf_trial1, X_test, y_test, "RF Trial 1")
rf_trial2 = RandomForestClassifier(n_estimators=300, max_depth=5,
min_samples_split=20, class_weight='balanced_subsample', random_state=52)
rf_trial2.fit(X_train, y_train)
evaluate_model(rf_trial2, X_test, y_test, "RF Trial 2")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_sm = RandomForestClassifier(random_state=42)
rf_sm.fit(X_train_sm, y_train_sm)
evaluate_model(rf_sm, X_test, y_test, "RF SMOTE")
#SVM Models
svm_param_grid = {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 1]}
gs_svm_no_smote = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3,
n_jobs=-1)
gs_svm_no_smote.fit(X_train, y_train)
evaluate_model(gs_svm_no_smote.best_estimator_, X_test, y_test, "SVM No SMOTE")
gs_svm_smote = GridSearchCV(SVC(probability=True), svm_param_grid, cv=3, n_jobs=-1)
gs_svm_smote.fit(X_train_sm, y_train_sm)
evaluate_model(gs_svm_smote.best_estimator_, X_test, y_test, "SVM SMOTE")
svm_trial1 = SVC(C=0.1, kernel='linear', probability=True)
svm_trial1.fit(X_train, y_train)
evaluate_model(svm_trial1, X_test, y_test, "SVM Trial 1 (C=0.1, linear)")
svm_trial2 = SVC(C=10, kernel='rbf', gamma='scale', probability=True)
svm_trial2.fit(X_train, y_train)
evaluate_model(svm_trial2, X_test, y_test, "SVM Trial 2 (C=10, RBF)")
#MLP Models
mlp_param_grid = {'hidden_layer_sizes': [(50,), (50, 50)],'activation':
['relu'],'learning_rate_init': [0.001],'solver': ['adam'],'max_iter': [300]}
gs_mlp_no_smote = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid,
cv=3, n_jobs=-1)
gs_mlp_no_smote.fit(X_train, y_train)
evaluate_model(gs_mlp_no_smote.best_estimator_, X_test, y_test, "MLP No SMOTE")
gs_mlp_smote = GridSearchCV(MLPClassifier(random_state=42), mlp_param_grid, cv=3,
n_jobs=-1)
gs_mlp_smote.fit(X_train_sm, y_train_sm)
evaluate_model(gs_mlp_smote.best_estimator_, X_test, y_test, "MLP SMOTE")
mlp_trial1 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
learning_rate_init=0.001, max_iter=300, random_state=42)
mlp_trial1.fit(X_train, y_train)
evaluate_model(mlp_trial1, X_test, y_test, "MLP Trial 1 (100 units)")
mlp_trial2 = MLPClassifier(hidden_layer_sizes=(50, 50), activation='tanh',
learning_rate_init=0.01, max_iter=300, random_state=42)
mlp_trial2.fit(X_train, y_train)
evaluate_model(mlp_trial2, X_test, y_test, "MLP Trial 2 (50,50 tanh)")
#Summary Table
def summarize_models(models, names):
results = []
for model, name in zip(models, names):
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
results.append([name, acc, f1, rec, prec])
return pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "Recall",
"Precision"])
final_models = [rf_trial1, rf_trial2, rf_sm,
gs_svm_no_smote.best_estimator_, gs_svm_smote.best_estimator_,
svm_trial1, svm_trial2,
gs_mlp_no_smote.best_estimator_, gs_mlp_smote.best_estimator_,
mlp_trial1, mlp_trial2]
final_names = ["RF Trial 1", "RF Trial 2", "RF SMOTE",
"SVM No SMOTE", "SVM SMOTE",
"SVM Trial 1", "SVM Trial 2",
"MLP No SMOTE", "MLP SMOTE",
"MLP Trial 1", "MLP Trial 2"]
summary_df = summarize_models(final_models, final_names)
print("\nModel Comparison Summary:\n")
print(summary_df)
#SHAP PLOTTING
import shap
shap.initjs()
#SHAP Plotting
X_test_df = pd.DataFrame(X_test, columns=X.columns)
rf_explainer = shap.TreeExplainer(rf)
# SHAP value extraction
rf_shap_values_all = rf_explainer.shap_values(X_test_df)
rf_shap_values_positive = rf_shap_values_all[:, :, 1]
# Confirm shapes
print("X_test_df shape:", X_test_df.shape)
print("SHAP values (positive class) shape:", rf_shap_values_positive.shape)
# Generate interactive SHAP force plot for one instance
instance_idx = 2
force_plot = shap.force_plot(
rf_explainer.expected_value[1],
rf_shap_values_positive[instance_idx],
X_test_df.iloc[instance_idx],
matplotlib=False # Interactive HTML version
)
shap.initjs()
# Save the interactive plot clearly to an HTML file
shap.save_html("shap_force_plot.html", force_plot)
shap.initjs()
# SHAP Summary plot with clearer formatting
plt.figure(figsize=(12, 8))
shap.summary_plot(
rf_shap_values_positive,
X_test_df,
plot_type="dot",
show=True,
max_display=15 # Shows clearly only top 15 features for readability
)
#LIME PLOTTING
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
# Setup for LIME using the saved feature names
class_names = ['Fail', 'Pass']
# Create LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
training_data=X_train,
feature_names=feature_names,
class_names=class_names,
mode='classification'
)
# Explain a Prediction
instance_index = 2
exp = explainer.explain_instance(
data_row=X_test[instance_index],
predict_fn=rf_trial1.predict_proba,
num_features=10
)
# LIME Explanation
# Displaying inline
#exp.show_in_notebook(show_table=True)
#Displaying static matplotlib chart
fig = exp.as_pyplot_figure()
plt.title(f'LIME Explanation for Student #{instance_index}')
plt.tight_layout()
