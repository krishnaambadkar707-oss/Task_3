"""
run.py - Main entry point to run the Customer Churn Prediction workflow.
"""

import sys
import os
import pandas as pd
from src.preprocessing import (
    load_raw_data,
    audit_data,
    clean_data,
    get_feature_target_split,
    build_preprocessing_pipeline,
    prepare_stratified_split
)
from src.train import (
    train_baseline_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    save_trained_pipeline
)
from src.evaluate import (
    evaluate_model,
    evaluate_threshold_range,
    extract_feature_importance
)


def main():
    print("=" * 80)
    print("STARTING CUSTOMER CHURN PREDICTION SYSTEM RUN")
    print("=" * 80)

    # 1. Load Raw Data
    raw_path = os.path.join("data", "raw", "telco_customer_churn.csv")
    print(f"\n[Step 1] Loading Raw Data from: {raw_path}")
    df_raw = load_raw_data(raw_path)
    
    # 2. Audit Data
    audit_results = audit_data(df_raw)
    print("\n--- Data Audit Summary ---")
    print(f"Dataset Dimensions: {audit_results['shape'][0]} rows x {audit_results['shape'][1]} columns")
    print(f"Duplicates: {audit_results['duplicates']}")
    print(f"Churn Distribution: {audit_results['churn_distribution']}")
    print(f"Churn Rate: {audit_results['churn_rate_pct']}%")
    print(f"TotalCharges Blanks Found: {audit_results['total_charges_blanks']}")

    # 3. Clean Data
    print("\n[Step 2] Cleaning Dataset & Handling Missing Values...")
    df_clean = clean_data(df_raw)
    
    # Save processed data
    proc_path = os.path.join("data", "processed", "telco_churn_cleaned.csv")
    df_clean.to_csv(proc_path, index=False)
    print(f"Cleaned dataset saved to: {proc_path}")

    # 4. Feature / Target Split & Train-Test Split
    print("\n[Step 3] Splitting Features (X) & Target (y)...")
    X, y = get_feature_target_split(df_clean, target_col='Churn')
    X_train, X_test, y_train, y_test = prepare_stratified_split(X, y, test_size=0.20, random_state=42)
    print(f"Train Set Size: {X_train.shape[0]} samples")
    print(f"Test Set Size:  {X_test.shape[0]} samples")

    # 5. Build Preprocessing Pipeline
    print("\n[Step 4] Constructing ColumnTransformer Preprocessing Pipeline...")
    preprocessor = build_preprocessing_pipeline(X_train)

    # 6. Train Models
    print("\n[Step 5] Training Classification Models...")
    print("  - Training Baseline Logistic Regression (Balanced)...")
    model_logreg = train_baseline_logistic_regression(X_train, y_train, preprocessor)

    print("  - Training Random Forest Classifier...")
    model_rf = train_random_forest(X_train, y_train, preprocessor)

    print("  - Training Gradient Boosting / XGBoost Classifier...")
    model_gb = train_gradient_boosting(X_train, y_train, preprocessor)

    # 7. Evaluate Models
    print("\n[Step 6] Evaluating Model Performance on Held-Out Test Data...")
    metrics_logreg = evaluate_model(model_logreg, X_test, y_test)
    metrics_rf = evaluate_model(model_rf, X_test, y_test)
    metrics_gb = evaluate_model(model_gb, X_test, y_test)

    print("\n" + "=" * 80)
    print("FINAL TEST METRICS EVALUATION")
    print("=" * 80)
    print(f"1. Logistic Regression: Accuracy={metrics_logreg['accuracy']}, Recall={metrics_logreg['recall']}, Precision={metrics_logreg['precision']}, ROC-AUC={metrics_logreg['roc_auc']}")
    print(f"2. Random Forest:       Accuracy={metrics_rf['accuracy']}, Recall={metrics_rf['recall']}, Precision={metrics_rf['precision']}, ROC-AUC={metrics_rf['roc_auc']}")
    print(f"3. Gradient Boosting:   Accuracy={metrics_gb['accuracy']}, Recall={metrics_gb['recall']}, Precision={metrics_gb['precision']}, ROC-AUC={metrics_gb['roc_auc']}")
    print("=" * 80)

    # 8. Threshold Analysis
    print("\n[Step 7] Running Decision Threshold Analysis (0.10 to 0.90) on Gradient Boosting...")
    df_thresholds = evaluate_threshold_range(model_gb, X_test, y_test)
    print(df_thresholds.to_string(index=False))

    # 9. Extract Feature Importance
    print("\n[Step 8] Top 10 Churn Feature Drivers:")
    df_imp = extract_feature_importance(model_gb)
    print(df_imp.head(10).to_string(index=False))

    # 10. Save Best Model Pipeline
    model_out_path = os.path.join("outputs", "best_churn_pipeline.joblib")
    os.makedirs("outputs", exist_ok=True)
    save_trained_pipeline(model_gb, model_out_path)
    print(f"\n[Step 9] Best Trained Pipeline Saved to: {model_out_path}")

    print("\n" + "=" * 80)
    print("SUCCESS: RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
