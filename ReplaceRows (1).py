#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split as tts, KFold as KF, GridSearchCV as GSCV
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.decomposition import PCA as pca
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score as acc, precision_score as prec, recall_score as rec, f1_score as f1
from sklearn.pipeline import Pipeline as Pipe
from sklearn.ensemble import StackingClassifier as Stack
from time import time as timer
from joblib import Parallel as ParallelLib, delayed as delay
import warnings

warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("heart.csv")
df.head() #shows the first 5 rows


# In[6]:


df.shape


# In[8]:


# Extract unique values for 'ca' and 'thal' columns
unique_ca_values = df['ca'].unique()
unique_thal_values = df['thal'].unique()

# Print the unique values
print("Unique values for 'ca':", unique_ca_values)
print("Unique values for 'thal':", unique_thal_values)


# In[10]:


# Count the occurrences of each integer value in the 'ca' column
value_counts = df['ca'].value_counts()

# Print the results
print(value_counts)


# In[12]:


# Count the occurrences of each integer value in the 'ca' column
value_counts = df['thal'].value_counts()

# Print the results
print(value_counts)


# In[14]:


# Count the number of occurrences of each value in the 'ca' column
ca_value_counts = df['ca'].value_counts()

# Extract the counts for the values 0 and 1
ca_4_count = ca_value_counts.get(4, 0)

print("Number of 4 values in 'ca':", ca_4_count)


# In[16]:


# Count the number of occurrences of each value in the 'ca' column
thal_value_counts = df['thal'].value_counts()

# Extract the counts for the values 0 and 1
thal_0_count = thal_value_counts.get(0, 0)

print("Number of 0 values in 'thal':", thal_0_count)


# In[18]:


# Clean 'ca' column
df['ca'] = df['ca'].replace(4, df['ca'].mode()[0])  # Replace 4 with the most common value

# Clean 'thal' column
df['thal'] = df['thal'].replace(0, df['thal'].mode()[0])  # Replace 0 with the most common value

# Verify the changes
print(df['ca'].value_counts())
print(df['thal'].value_counts())


# In[20]:


df.shape


# In[22]:


#gives overall information of the dataset
df.info()


# In[24]:


df.dtypes


# In[26]:


#check for null values in the dataset
df.isnull().sum()


# In[28]:


print(df.describe())


# In[30]:


df["target"].value_counts()


# In[32]:


# Count of each class in the target variable
target_dis = df['target'].value_counts()

# Plotting the distribution
plt.figure(figsize=(8, 5))
target_dis.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Target Variable')
plt.xlabel('Heart Disease Presence (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[34]:


df["sex"].value_counts()


# In[36]:


# Shows the Distribution of Heat Diseases with respect to male and female
dis_fig=px.histogram(df, 
                 x="target",
                 color="sex",
                 hover_data=df.columns,
                 title="Distribution of Heart Diseases",
                 barmode="group")
dis_fig.show()


# In[38]:


# Generate the correlation matrix
corr_mat = df.corr()

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(corr_mat, cmap='coolwarm')

# Add annotations
for i in range(corr_mat.shape[0]):
    for j in range(corr_mat.shape[1]):
        text = ax.text(j, i, f'{corr_mat.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black")

# Set tick labels
ax.set_xticks(np.arange(len(corr_mat.columns)))
ax.set_yticks(np.arange(len(corr_mat.columns)))
ax.set_xticklabels(corr_mat.columns)
ax.set_yticklabels(corr_mat.columns)

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

plt.title("Correlation Heatmap", fontsize=20)
plt.tight_layout()
plt.show()


# In[40]:


num_rows = 5
num_cols = 3

plt.figure(figsize=(15, 15))
for i, col in enumerate(df.columns, 1):
    plt.subplot(num_rows, num_cols, i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(df[col], kde=True)
    plt.tight_layout()
    plt.plot()
plt.show()


# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import multiprocessing
from joblib import Parallel, delayed

from sklearn.model_selection import GridSearchCV as GSCV, StratifiedKFold as SKF, train_test_split as tts, cross_val_score as cv_score, learning_curve as lc
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.decomposition import PCA as PCA
from sklearn.feature_selection import SelectKBest as SelectK, f_classif
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC, StackingClassifier as Stack
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import accuracy_score as acc, precision_score as prec, recall_score as rec, f1_score as f1
from sklearn.pipeline import Pipeline as Pipe

# Feature selection
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X_data = df[feature_columns]
X_data = pd.get_dummies(X_data, drop_first=True)
y_data = df['target']

# Splitting the data into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = tts(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
X_train, X_val, y_train, y_val = tts(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

def train_and_evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test):
    start_time = time()
    model.fit(X_train, y_train)
    training_time = time() - start_time
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_acc = acc(y_val, y_val_pred)
    val_prec = prec(y_val, y_val_pred, average='weighted')
    val_rec = rec(y_val, y_val_pred, average='weighted')
    val_f1 = f1(y_val, y_val_pred, average='weighted')
    
    # Testing metrics
    start_time = time()
    y_test_pred = model.predict(X_test)
    testing_time = time() - start_time
    test_acc = acc(y_test, y_test_pred)
    test_prec = prec(y_test, y_test_pred, average='weighted')
    test_rec = rec(y_test, y_test_pred, average='weighted')
    test_f1 = f1(y_test, y_test_pred, average='weighted')
    
    return {
        'val_accuracy': val_acc, 'val_precision': val_prec,
        'val_recall': val_rec, 'val_f1': val_f1,
        'test_accuracy': test_acc, 'test_precision': test_prec,
        'test_recall': test_rec, 'test_f1': test_f1,
        'training_time': training_time, 'testing_time': testing_time
    }

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = lc(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def process_model(model_name, pipeline, params):
    print(f"Training and evaluating {model_name}...")
    
    # Hyperparameter tuning using GridSearch
    grid_search = GSCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Training and evaluation
    model_results = train_and_evaluate(best_model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Cross-validation scores
    cv_scores = cv_score(best_model, X_train_val, y_train_val, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Plot learning curve
    plt.figure(figsize=(20, 5))
    plot_learning_curve(best_model, f'Learning Curve for {model_name}', X_train_val, y_train_val, cv=5, n_jobs=-1)
    plt.savefig(f'{model_name}_learning_curve.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': best_params,
        'val_accuracy': model_results['val_accuracy'],
        'val_precision': model_results['val_precision'],
        'val_recall': model_results['val_recall'],
        'val_f1': model_results['val_f1'],
        'test_accuracy': model_results['test_accuracy'],
        'test_precision': model_results['test_precision'],
        'test_recall': model_results['test_recall'],
        'test_f1': model_results['test_f1'],
        'cv_mean_accuracy': cv_mean,
        'cv_std_accuracy': cv_std,
        'training_time': model_results['training_time'],
        'testing_time': model_results['testing_time']
    }

# Define pipelines and parameters

# Pipeline for Logistic Regression
lr_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', LR(random_state=42))
])
lr_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2']
}

# Pipeline for Naive Bayes
nb_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', GNB())
])
nb_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
}

# Pipeline for Decision Tree
dt_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', DTC(random_state=42))
])
dt_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__max_depth': [3, 5, 7],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}

# Pipeline for Random Forest
rf_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', RFC(random_state=42))
])
rf_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5, 7],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}

# Pipeline for Gradient Boosting
gb_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', GBC(random_state=42))
])
gb_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__n_estimators': [50, 100],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.01, 0.1]
}

# Pipeline for Stacking Classifier
stack_pipe = Pipe([
    ('scaler', Scaler()),
    ('feature_selection', SelectK(score_func=f_classif)),
    ('pca', PCA()),
    ('clf', Stack(
        estimators=[('rf', RFC(random_state=42)), 
                    ('gb', GBC(random_state=42))],
        final_estimator=LR(random_state=42)))
])
stack_params = {
    'feature_selection__k': [5, 8, 10],
    'pca__n_components': [3, 5, 7],
    'clf__cv': [3, 5],
    'clf__final_estimator__C': [0.1, 1.0, 10.0]
}

model_list = [
    ('Decision Tree', dt_pipe, dt_params),
    ('Random Forest', rf_pipe, rf_params),
    ('Gradient Boosting', gb_pipe, gb_params),
    ('Stacking Classifier', stack_pipe, stack_params),
    ('Logistic Regression', lr_pipe, lr_params),
    ('Naive Bayes', nb_pipe, nb_params)
]

# Use all available cores, but leave one free
num_cores = multiprocessing.cpu_count() - 1

# Parallel processing of models
results = Parallel(n_jobs=num_cores)(delayed(process_model)(model_name, pipeline, params) for model_name, pipeline, params in model_list)

# Display results
for result in results:
    print(f"Model: {result['model_name']}")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Validation Accuracy: {result['val_accuracy']:.2f}")
    print(f"Validation Precision: {result['val_precision']:.2f}")
    print(f"Validation Recall: {result['val_recall']:.2f}")
    print(f"Validation F1 Score: {result['val_f1']:.2f}")
    print(f"Testing Accuracy: {result['test_accuracy']:.2f}")
    print(f"Testing Precision: {result['test_precision']:.2f}")
    print(f"Testing Recall: {result['test_recall']:.2f}")
    print(f"Testing F1 Score: {result['test_f1']:.2f}")
    print(f"Cross-validation Mean Accuracy: {result['cv_mean_accuracy']:.2f}")
    print(f"Cross-validation Std Accuracy: {result['cv_std_accuracy']:.2f}")
    print(f"Training Time: {result['training_time']:.3f} seconds")
    print(f"Testing Time: {result['testing_time']:.3f} seconds\n")

# Feature importance analysis for Random Forest
rf_result = next(result for result in results if result['model_name'] == 'Random Forest')
rf_model = rf_result['best_model']

# Check if the model has feature_importances_ attribute
if hasattr(rf_model.named_steps['clf'], 'feature_importances_'):
    feature_importance = rf_model.named_steps['clf'].feature_importances_
    
    # Get feature names
    if 'feature_selection' in rf_model.named_steps:
        feature_names = rf_model.named_steps['feature_selection'].get_feature_names_out()
    else:
        feature_names = X_data.columns
    
    # Sort features by importance
    feature_importance_sorted = sorted(zip(feature_importance, feature_names), reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar([x[1] for x in feature_importance_sorted], [x[0] for x in feature_importance_sorted])
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Top 5 important features:")
    for importance, name in feature_importance_sorted[:5]:
        print(f"{name}: {importance:.4f}")
else:
    print("Feature importance is not available for this model.")


# In[44]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def plot_learning_curves(classifiers, X, y, cv=5, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(12, 8))
    
    # Define a list of easily distinguishable colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (name, classifier) in enumerate(classifiers):
        train_sizes, train_scores, test_scores = learning_curve(
            classifier, X, y, cv=cv, n_jobs=-1, 
            train_sizes=train_sizes, scoring='accuracy'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        
        color = colors[i % len(colors)]  # Cycle through colors if more classifiers than colors
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color=color, label=f"{name} (Training)")
        plt.plot(train_sizes, test_scores_mean, 's--', color=color, label=f"{name} (Validation)")
    
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Learning Curves for All Classifiers", fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('learning_curves_all_classifiers.png', dpi=300, bbox_inches='tight')

# Prepare the classifiers using the best models from the results
classifiers = [(result['model_name'], result['best_model']) for result in results]

# Plot learning curves
plot_learning_curves(classifiers, X_train_val, y_train_val)

# Find the Gradient Boosting result
gb_result = next(result for result in results if result['model_name'] == 'Gradient Boosting')

# Save the Gradient Boosting model
import joblib
joblib.dump(gb_result['best_model'], 'gradient_boosting_model.joblib')
print("Gradient Boosting model saved as 'gradient_boosting_model.joblib'")
# In[ ]:





