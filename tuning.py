import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from data_preprocessing import load_data, drop_unnecessary_columns, ORIGINAL_NUMERIC_FEATURES, ORIGINAL_CATEGORICAL_FEATURES, build_preprocessor
from utils import save_model_results_to_csv   

RANDOM_STATE = 42

# Hyper parameter tuning for the baseline models
def get_tuning_setup():
    models_and_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__class_weight': [None, 'balanced'],
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 5, 10],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__class_weight': [None, 'balanced']
            }
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto'],
                'classifier__class_weight': [None, 'balanced']
            }
        ),
        "KNN": (
            KNeighborsClassifier(algorithm='brute'),
            {
                'classifier__n_neighbors': [3, 5, 7, 9, 11],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['minkowski'],
                'classifier__p': [1, 2]
            }
        )
    }
    return models_and_grids


def tune_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, scenario_name):
    print(f"\n---Hyperparameter Tuning and Evaluation for {scenario_name}---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_results = []

    models_and_grids = get_tuning_setup()

    for model_name, (model, param_grid) in models_and_grids.items():
        print(f"\nTuning {model_name}...")

        # Apply the specific tree-friendly preprocessor for Random Forest
        if model_name in ["Random Forest", "XGBoost"]:
            preprocessor_tree = build_preprocessor(ORIGINAL_NUMERIC_FEATURES, ORIGINAL_CATEGORICAL_FEATURES, scale_numeric=False)
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor_tree),
                ('classifier', model)
            ])
        else:
            # Apply standard preprocessor (with scaling) for everything else
            preprocessor_scaled = build_preprocessor(
                ORIGINAL_NUMERIC_FEATURES, ORIGINAL_CATEGORICAL_FEATURES, scale_numeric=True)
            model_pipeline = Pipeline(steps=[
                ('preprocessor',    preprocessor_scaled),
                ('classifier', model)
            ])

        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=1,
            refit=True,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        all_results.append({
            'Model': model_name,
            'Best CV F1': grid_search.best_score_,
            'Test Acc': accuracy_score(y_test, y_pred),
            'Test Prec': precision_score(y_test, y_pred, zero_division=0),
            'Test Rec': recall_score(y_test, y_pred, zero_division=0),
            'Test F1': f1_score(y_test, y_pred, zero_division=0),
            'Best Params': grid_search.best_params_
        })

    results_df = pd.DataFrame(all_results).sort_values(
        by='Test F1', ascending=False)
    print(results_df[['Model', 'Best CV F1', 'Test Acc',
          'Test Prec', 'Test Rec', 'Test F1']])

    print("\nBest Parameters by Model:")
    for _, row in results_df.iterrows():
        print(f"{row['Model']}: {row['Best Params']}")

    best_row = results_df.iloc[0]
    print("\nBest Performing Model:")
    print(f"Model: {best_row['Model']}")
    print(f"Test F1: {best_row['Test F1']:.4f}")
    print(f"Best Params: {best_row['Best Params']}")

    return results_df


if __name__ == "__main__":
    df = load_data()
    df = drop_unnecessary_columns(df)

    X = df.drop(columns=['target'])
    y = df['target']

    preprocessor = build_preprocessor(
        ORIGINAL_NUMERIC_FEATURES, ORIGINAL_CATEGORICAL_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results_df = tune_and_evaluate_models(X_train, X_test, y_train,
                                          y_test, preprocessor, "Hyper Tuning Baseline Scenario with UCI Data")
    save_model_results_to_csv(results_df, "Hyper Tuning with UCI Data")