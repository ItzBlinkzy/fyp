import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

# setup
warnings.filterwarnings('ignore')

# load data
df = pd.read_csv('./dataset/transaction_dataset.csv')
print("Original dataset shape:", df.shape)

# drop identifiers
drop_cols = ['Address', 'Index', 'Unnamed: 0']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# handle duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()

# separate features and target
X = df.drop('FLAG', axis=1)
y = df['FLAG']

# identify feature types
categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric = X.select_dtypes(include=[np.number]).columns.tolist()

# --- feature selection ---

# remove zero variance features
numeric_data = X[numeric]
variance_selector = VarianceThreshold(threshold=0.01)
numeric_filtered = pd.DataFrame(
    variance_selector.fit_transform(numeric_data),
    columns=numeric_data.columns[variance_selector.get_support()],
    index=X.index
)

# check target correlation (leakage detection)
print("\nChecking correlation with target:")
for col in numeric_filtered.columns:
    corr = numeric_filtered[col].corr(y)
    if abs(corr) > 0.8:
        print(f"WARNING - High correlation: {col}: {corr:.3f}")

# remove highly correlated features
corr_matrix = numeric_filtered.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
numeric_final = numeric_filtered.drop(columns=high_corr_pairs)

# reassemble dataset
X_final = pd.concat([numeric_final, X[categorical]], axis=1)

# check for suspicious aggregated keywords
suspicious_keywords = ['total', 'balance', 'sum', 'aggregate']
suspicious_features = [col for col in numeric_final.columns if any(k in col.lower() for k in suspicious_keywords)]
if suspicious_features:
    print(f"\nSuspicious features found: {suspicious_features[:10]}")

# update feature lists
categorical_final = X_final.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_final_list = X_final.select_dtypes(include=[np.number]).columns.tolist()

# stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, stratify=y, test_size=0.30, random_state=42
)

# preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_final_list),
    ('cat', categorical_transformer, categorical_final)
])

# pipeline with smote and model
pipeline = ImbPipeline([
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=3)),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42, penalty='l2'))
])

# hyperparameter grid
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1.0, 5.0],
    'smote__k_neighbors': [3, 5]
}

# grid search execution
print("\nStarting Grid Search...")
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV F1 score: {grid.best_score_:.4f}")

# evaluation
y_pred = grid.best_estimator_.predict(X_test)
y_pred_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# feature importance analysis
coefficients = grid.best_estimator_.named_steps['clf'].coef_[0]
feature_names = numeric_final_list.copy()

if categorical_final:
    try:
        cat_features = grid.best_estimator_.named_steps['preproc'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_final)
        feature_names.extend(cat_features)
    except:
        pass

if len(feature_names) >= len(coefficients):
    feature_importance = pd.DataFrame({
        'feature': feature_names[:len(coefficients)],
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    print("\nTop 15 features:")
    print(feature_importance.head(15)[['feature', 'coefficient']])
    feature_importance.to_csv('feature_importance.csv', index=False)