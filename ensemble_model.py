
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data_preprocessing import (
    DATA_PATH_UCI,
    ORIGINAL_CATEGORICAL_FEATURES,
    ORIGINAL_NUMERIC_FEATURES,
    build_preprocessor,
    drop_unnecessary_columns,
    load_data,
)
from feature_engineering import add_pca_components_to_split, add_synthetic_features
from utils import save_model_results_to_csv

RANDOM_STATE = 42
SOFT_VOTE_WEIGHTS = {
    'rf': 1.0,
    'svm': 0.5,
    'knn': 2}


def get_metrics_row(model_name, y_true, y_pred):
    return {
        'Model': model_name,
        'Acc': accuracy_score(y_true, y_pred),
        'Prec': precision_score(y_true, y_pred, zero_division=0),
        'Rec': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }


def run_targeted_ensemble(X, y):
    base_numeric_features = list(ORIGINAL_NUMERIC_FEATURES)
    base_categorical_features = list(ORIGINAL_CATEGORICAL_FEATURES)

    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Model 1: Baseline Random Forest (no feature engineering, no PCA)
    rf_preprocessor = build_preprocessor(
        base_numeric_features,
        base_categorical_features,
        scale_numeric=False
    )
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', rf_preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ))
    ])

    # Model 2: SVM after feature engineering
    X_train_fe, fe_cat_features, fe_num_features = add_synthetic_features(
        X_train_base,
        base_categorical_features,
        base_numeric_features
    )
    X_test_fe, _, _ = add_synthetic_features(
        X_test_base,
        base_categorical_features,
        base_numeric_features
    )

    svm_preprocessor = build_preprocessor(
        fe_num_features,
        fe_cat_features,
        scale_numeric=True
    )
    svm_pipeline = Pipeline(steps=[
        ('preprocessor', svm_preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='auto', probability=True),)
    ])

    # Model 3: KNN after adding PCA components
    pca_base_features = ['age', 'trestbps', 'chol']
    X_train_pca, X_test_pca, pca_columns = add_pca_components_to_split(
        X_train_base,
        X_test_base,
        pca_base_features,
        n_components=3
    )
    knn_numeric_features = base_numeric_features + pca_columns
    knn_preprocessor = build_preprocessor(
        knn_numeric_features,
        base_categorical_features,
        scale_numeric=True
    )
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', knn_preprocessor),
        ('classifier', KNeighborsClassifier(
            n_neighbors=5,
            p=2,
            weights='distance',
            algorithm='brute'
        ))
    ])

    rf_pipeline.fit(X_train_base, y_train)
    svm_pipeline.fit(X_train_fe, y_train)
    knn_pipeline.fit(X_train_pca, y_train)

    rf_proba = rf_pipeline.predict_proba(X_test_base)[:, 1]
    # svm_scores = svm_pipeline.decision_function(X_test_fe)
    # svm_proba = 1.0 / (1.0 + np.exp(-svm_scores))
    svm_proba = svm_pipeline.predict_proba(X_test_fe)[:, 1]
    knn_proba = knn_pipeline.predict_proba(X_test_pca)[:, 1]

    rf_pred = (rf_proba >= 0.5).astype(int)
    svm_pred = svm_pipeline.predict(X_test_fe)
    knn_pred = (knn_proba >= 0.5).astype(int)

    weighted_proba = np.average(
        np.vstack([rf_proba, svm_proba, knn_proba]),
        axis=0,
        weights=[
            SOFT_VOTE_WEIGHTS['rf'],
            SOFT_VOTE_WEIGHTS['svm'],
            SOFT_VOTE_WEIGHTS['knn']
        ]
    )
    ensemble_pred = (weighted_proba >= 0.5).astype(int)

    results = [
        get_metrics_row('Baseline Random Forest', y_test, rf_pred),
        get_metrics_row('Feature Engineered SVM', y_test, svm_pred),
        get_metrics_row('PCA KNN', y_test, knn_pred),
        get_metrics_row('Ensemble (RF + FE-SVM + PCA-KNN)',
                        y_test, ensemble_pred)
    ]

    results_df = pd.DataFrame(results)
    print(results_df)
    print(f"Weighted soft-vote weights: {SOFT_VOTE_WEIGHTS}")
    save_model_results_to_csv(
        results_df, 'Ensemble Scenario (RF FE-SVM PCA-KNN)')

    return results_df


def main():
    df = load_data(DATA_PATH_UCI)
    df = drop_unnecessary_columns(df)

    X = df.drop(columns=['target'])
    y = df['target']

    run_targeted_ensemble(X, y)


if __name__ == '__main__':
    main()
