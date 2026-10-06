"""
train.py - Model training, cross-validation, hyperparameter tuning, and pipeline construction.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib


def create_model_pipeline(preprocessor, classifier):
    """Combine preprocessor ColumnTransformer and classifier into a single sklearn Pipeline."""
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def train_baseline_logistic_regression(X_train, y_train, preprocessor, class_weight='balanced', random_state=42):
    """Train baseline Logistic Regression classifier pipeline."""
    clf = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        C=1.0
    )
    pipeline = create_model_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train, preprocessor, class_weight='balanced', random_state=42):
    """Train Random Forest Classifier pipeline."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight=class_weight,
        random_state=random_state
    )
    pipeline = create_model_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_gradient_boosting(X_train, y_train, preprocessor, random_state=42):
    """Train Gradient Boosting / XGBoost style Classifier pipeline."""
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=random_state
    )
    pipeline = create_model_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)
    return pipeline


def tune_logistic_regression(X_train, y_train, preprocessor, random_state=42):
    """Perform GridSearchCV hyperparameter tuning on training set for Logistic Regression."""
    pipeline = create_model_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=random_state))
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__class_weight': [None, 'balanced']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def save_trained_pipeline(pipeline, filepath: str):
    """Save trained pipeline artifact to disk."""
    joblib.dump(pipeline, filepath)


def load_trained_pipeline(filepath: str):
    """Load trained pipeline artifact from disk."""
    return joblib.load(filepath)
