import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from data_preprocessing import load_data, drop_unnecessary_columns, DATA_PATH_UCI, ORIGINAL_NUMERIC_FEATURES, ORIGINAL_CATEGORICAL_FEATURES, build_preprocessor
from utils import save_model_results_to_csv

RANDOM_STATE = 42


def add_synthetic_features(df, categorical_features, numeric_features):
    df = df.copy()
    updated_categorical_features = categorical_features.copy()
    updated_numeric_features = numeric_features.copy()

    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 40, 55, 70, 100],
        labels=['young', 'mid', 'senior', 'elder']
    )

    df['bp_category'] = pd.cut(
        df['trestbps'],
        bins=[0, 119.9, 129.9, 139.9, 300],
        labels=['Normal', 'Elevated', 'Stage_1_HTN', 'Stage_2_HTN']
    )

    df['chol_group'] = pd.cut(
        df['chol'],
        bins=[0, 200, 240, 600],
        labels=['normal', 'borderline', 'high']
    )

    updated_categorical_features.extend(
        ['age_group', 'bp_category', 'chol_group'])

    df['hr_competence_ratio'] = df['thalch'] / (220 - df['age'])

    df['severe_ischemia'] = ((df['exang'] == 1) & (
        df['oldpeak'] > 2.0)).astype(int)

    updated_numeric_features.extend(['hr_competence_ratio', 'severe_ischemia'])

    return df, updated_categorical_features, updated_numeric_features


def add_pca_components_to_split(X_train, X_test, pca_features, n_components=3):
    train_copy = X_train.copy()
    test_copy = X_test.copy()

    available_features = [
        col for col in pca_features if col in train_copy.columns and col in test_copy.columns
    ]
    if not available_features:
        print('PCA skipped: no valid PCA base features found in split data.')
        return train_copy, test_copy, []

    # Coerce to numeric before scaling/PCA. Non-numeric values become NaN and are imputed.
    train_numeric_df = train_copy[available_features].apply(
        pd.to_numeric, errors='coerce')
    test_numeric_df = test_copy[available_features].apply(
        pd.to_numeric, errors='coerce')

    imputer = SimpleImputer(strategy='median')
    train_numeric = imputer.fit_transform(train_numeric_df)
    test_numeric = imputer.transform(test_numeric_df)

    max_components = min(n_components, len(
        available_features), train_numeric.shape[0])
    if max_components < 1:
        print('PCA skipped: not enough samples/features for requested components.')
        return train_copy, test_copy, []

    scaler = StandardScaler()
    pca = PCA(n_components=max_components, random_state=RANDOM_STATE)

    train_scaled = scaler.fit_transform(train_numeric)
    test_scaled = scaler.transform(test_numeric)

    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    pca_columns = [f'pca_{idx + 1}' for idx in range(max_components)]
    for idx, column_name in enumerate(pca_columns):
        train_copy[column_name] = train_pca[:, idx]
        test_copy[column_name] = test_pca[:, idx]

    explained = ', '.join(
        f'{ratio:.3f}' for ratio in pca.explained_variance_ratio_)
    print(f'PCA explained variance ratios: {explained}')

    return train_copy, test_copy, pca_columns


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    numeric_steps = [('imputer', KNNImputer(n_neighbors=5))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


def get_models():
    # Each tuple contains (model_name, estimator, preprocessor_key).
    print("Getting models")
    return [
        (
            'Logistic Regression',
            LogisticRegression(max_iter=2000, class_weight='balanced'),
            'scaled'
        ),
        (
            'Random Forest',
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=2,
                # max_features='sqrt',
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            'unscaled'
        ),
        (
            'Support Vector Machine',
            SVC(kernel='rbf', C=1.0, gamma='auto', probability=True),
            'scaled'
        ),
        (
            'K-Nearest Neighbors',
            KNeighborsClassifier(n_neighbors=5, p=2,
                                 weights='distance', algorithm='brute'),
            'scaled'
        )
    ]


def evaluate_models(X_train, X_test, y_train, y_test, scenario_name, models, preprocessors):
    print(f'--- Evaluating Scenario: {scenario_name} ---')
    results = []

    for model_name, estimator, preprocessor_key in models:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessors[preprocessor_key]),
            ('classifier', estimator)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            'Model': model_name,
            'Acc': accuracy_score(y_test, y_pred),
            'Prec': precision_score(y_test, y_pred),
            'Rec': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


def run_scenario(base_X, y, scenario_name, use_feature_engineering=False, use_pca=False):
    scenario_X = base_X.copy()
    scenario_numeric_features = list(ORIGINAL_NUMERIC_FEATURES)
    scenario_categorical_features = list(ORIGINAL_CATEGORICAL_FEATURES)
    pca_numeric_features = ["age", "trestbps", "chol"]

    if use_feature_engineering:
        scenario_X, scenario_categorical_features, scenario_numeric_features = add_synthetic_features(
            scenario_X,
            scenario_categorical_features,
            scenario_numeric_features
        )

    X_train, X_test, y_train, y_test = train_test_split(
        scenario_X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    if use_pca:
        X_train, X_test, pca_columns = add_pca_components_to_split(
            X_train,
            X_test,
            pca_numeric_features,
            n_components=3
        )

        scenario_numeric_features = scenario_numeric_features + pca_columns

    preprocessors = {
        'scaled': build_preprocessor(scenario_numeric_features, scenario_categorical_features, scale_numeric=True),
        'unscaled': build_preprocessor(scenario_numeric_features, scenario_categorical_features, scale_numeric=False)
    }
    models = get_models()

    results_df = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        scenario_name,
        models,
        preprocessors
    )
    save_model_results_to_csv(results_df, scenario_name)


def main():
    df = load_data(DATA_PATH_UCI)

    df = drop_unnecessary_columns(df)
    X = df.drop(columns=['target'])
    y = df['target']

    run_scenario(
        X,
        y,
        '1) Baseline (No Feature Engineering, No PCA)',
        use_feature_engineering=False,
        use_pca=False
    )

    run_scenario(
        X,
        y,
        '2) With Feature Engineering',
        use_feature_engineering=True,
        use_pca=False
    )

    run_scenario(
        X,
        y,
        '3) With PCA Components',
        use_feature_engineering=False,
        use_pca=True
    )

    run_scenario(
        X,
        y,
        '4) With Feature Engineering and PCA Components',
        use_feature_engineering=True,
        use_pca=True
    )


if __name__ == '__main__':
    main()
