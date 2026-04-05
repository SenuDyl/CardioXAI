import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Import from your preprocessing module (ensure preprocessing.py is in the same folder)
from data_preprocessing import (
    load_data,
    drop_unnecessary_columns,
    build_preprocessor,
    ORIGINAL_NUMERIC_FEATURES,
    ORIGINAL_CATEGORICAL_FEATURES,
    RANDOM_STATE
)
from feature_engineering import add_synthetic_features
from ensemble_model import SOFT_VOTE_WEIGHTS

OUT_DIR = 'shap_plots'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Define your project palette (Blue for Healthy/Normal, Red for Disease)
PROJECT_PALETTE = {'Normal': '#4C72B0', 'Disease': '#C44E52'}


def _fit_ensemble_components(X_train_df, y_train_series):
    """Train the same three base models used by the ensemble script."""
    base_numeric_features = list(ORIGINAL_NUMERIC_FEATURES)
    base_categorical_features = list(ORIGINAL_CATEGORICAL_FEATURES)

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

    X_train_fe, fe_cat_features, fe_num_features = add_synthetic_features(
        X_train_df,
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
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='auto', probability=True))
    ])

    knn_preprocessor = build_preprocessor(
        base_numeric_features,
        base_categorical_features,
        scale_numeric=True,
        pca_features=['age', 'trestbps', 'chol'],
        n_components=3
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

    rf_pipeline.fit(X_train_df, y_train_series)
    svm_pipeline.fit(X_train_fe, y_train_series)
    knn_pipeline.fit(X_train_df, y_train_series)

    return rf_pipeline, svm_pipeline, knn_pipeline


def _build_ensemble_predict_fn(rf_pipeline, svm_pipeline, knn_pipeline, feature_names):
    """Create a callable that returns weighted ensemble disease probability."""
    base_numeric_features = list(ORIGINAL_NUMERIC_FEATURES)
    base_categorical_features = list(ORIGINAL_CATEGORICAL_FEATURES)

    def predict_fn(X_like):
        if isinstance(X_like, pd.DataFrame):
            X_df = X_like[feature_names].copy()
        else:
            X_df = pd.DataFrame(X_like, columns=feature_names)

        X_df_fe, _, _ = add_synthetic_features(
            X_df,
            base_categorical_features,
            base_numeric_features
        )

        rf_proba = rf_pipeline.predict_proba(X_df)[:, 1]
        svm_proba = svm_pipeline.predict_proba(X_df_fe)[:, 1]
        knn_proba = knn_pipeline.predict_proba(X_df)[:, 1]

        weighted_proba = np.average(
            np.vstack([rf_proba, svm_proba, knn_proba]),
            axis=0,
            weights=[
                SOFT_VOTE_WEIGHTS['rf'],
                SOFT_VOTE_WEIGHTS['svm'],
                SOFT_VOTE_WEIGHTS['knn']
            ]
        )
        return weighted_proba

    return predict_fn


def generate_ensemble_shap_plot(
    X_train_df,
    y_train_series,
    X_test_df,
    max_background=50,
    max_explain=80,
    top_n=15
):
    """
    Explain the weighted ensemble output using model-agnostic SHAP (KernelExplainer).
    Produces a Top-N directional SHAP bar chart for ensemble disease probability.
    """
    print('Training ensemble base models for SHAP analysis...')
    rf_pipeline, svm_pipeline, knn_pipeline = _fit_ensemble_components(
        X_train_df,
        y_train_series
    )

    feature_names = list(X_train_df.columns)
    ensemble_predict_fn = _build_ensemble_predict_fn(
        rf_pipeline,
        svm_pipeline,
        knn_pipeline,
        feature_names
    )

    background_size = min(max_background, len(X_train_df))
    explain_size = min(max_explain, len(X_test_df))
    background_df = shap.sample(
        X_train_df, background_size, random_state=RANDOM_STATE)
    explain_df = shap.sample(X_test_df, explain_size,
                             random_state=RANDOM_STATE)

    print('Calculating SHAP values for weighted ensemble (KernelExplainer)...')
    explainer = shap.KernelExplainer(ensemble_predict_fn, background_df)
    shap_values = explainer.shap_values(explain_df, nsamples='auto')

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # For binary outcomes, SHAP for healthy probability is the sign-inverse of disease probability
    mean_disease = np.mean(shap_values, axis=0)
    mean_healthy = -mean_disease

    def _plot_ensemble_class(mean_values, class_label, file_name):
        df_ensemble = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': mean_values
        })
        df_ensemble['Magnitude'] = df_ensemble['SHAP_Value'].abs()
        df_top = df_ensemble.sort_values(
            by='Magnitude', ascending=False).head(top_n)
        df_plot = df_top.sort_values(by='SHAP_Value', ascending=True)
        values = df_plot['SHAP_Value'].values

        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        cmap = cm.get_cmap('viridis_r')
        bar_colors = cmap(norm(values))

        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.barh(df_plot['Feature'], values, color=bar_colors,
                edgecolor='black', linewidth=0.5)

        ax.set_title(
            f'Ensemble SHAP Explanation: Impact on {class_label} Probability (Top {top_n})',
            fontsize=16,
            fontweight='bold'
        )
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
        ax.set_xlabel(
            f'Mean SHAP Value (Average impact on ensemble {class_label.lower()} probability)',
            fontsize=14
        )
        ax.set_ylabel('Original Input Features', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=11)
        add_shap_value_labels(ax, values, gap_size=0.00001)

        plt.tight_layout()
        file_path = os.path.join(OUT_DIR, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f'Saved plot successfully to: {file_path}')
        plt.show()

    _plot_ensemble_class(
        mean_disease,
        class_label='Disease',
        file_name='shap_ensemble_disease_top15_directional.png'
    )
    _plot_ensemble_class(
        mean_healthy,
        class_label='Healthy',
        file_name='shap_ensemble_healthy_top15_directional.png'
    )


def add_shap_value_labels(ax, values, fmt='{:+.4f}', gap_size=0.00001):
    """Annotate horizontal bars with signed SHAP values (internal function)."""
    for idx, value in enumerate(values):
        # Determine offset and alignment based on sign
        offset = gap_size if value >= 0 else -gap_size
        ha = 'left' if value >= 0 else 'right'
        ax.text(value + offset, idx, fmt.format(value),
                va='center', ha=ha, fontsize=8, fontweight='bold')


def generate_single_class_plot(df_shap_class, class_label):
    """
    Modular function to generate one clean directional SHAP bar chart.
    Applies a 'viridis_r' gradient color theme based on SHAP value.
    """
    # Calculate absolute value for sorting
    df_shap_class['Magnitude'] = df_shap_class['SHAP_Value'].abs()

    # Take only Top 15 most impactful features to eliminate clutter
    df_top_15 = df_shap_class.sort_values(
        by='Magnitude', ascending=False).head(15)

    # Sort again by raw score for standard horizontal presentation (ascending=True for barh)
    df_plot = df_top_15.sort_values(by='SHAP_Value', ascending=True)
    values = df_plot['SHAP_Value'].values

    # --- NEW: GRADIENT COLOR MAPPING LOGIC ---
    # Create a normalization object to scale values to the min/max of the current plot
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    # Use 'viridis_r' which matches your reference image (Yellow for negative, Purple for positive)
    cmap = cm.get_cmap('viridis_r')
    bar_colors = cmap(norm(values))
    # -----------------------------------------

    # Generate a new separate figure
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Standard directional bar chart using the dynamic gradient colors
    ax.barh(df_plot['Feature'], values, color=bar_colors,
            edgecolor='black', linewidth=0.5)

    # Styling
    ax.set_title(f'SHAP Explanation: Impact on `{class_label}` Class Prediction (Top 15)\nDirectional Feature Influence',
                 fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
    ax.set_xlabel(
        'Mean SHAP Value (Average impact on model output logarithm)', fontsize=14)
    ax.set_ylabel(
        'Features (including One-Hot Encoded variables)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Add labels
    add_shap_value_labels(ax, values, gap_size=0.00001)

    plt.tight_layout()

    # Extract just the first word ('normal' or 'disease') to make OS-safe filenames
    safe_filename_prefix = class_label.split(' ')[0].lower()
    file_path = os.path.join(
        OUT_DIR, f'shap_{safe_filename_prefix}_top15_directional.png')

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot successfully to: {file_path}")

    plt.show()


def generate_separated_shap_analysis(fitted_rf_model, X_test_df, y_test_series, preprocessor):
    """Orchestrator function to calculate SHAP and call plotting for both classes."""
    print("Preparing data and extracting features...")
    # 1. Process test data
    X_test_processed = preprocessor.transform(X_test_df)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()

    # 2. Get feature names (Crucial to include the OHE names)
    feature_names = preprocessor.get_feature_names_out()
    X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)

    # 3. Calculate SHAP values
    print("Calculating SHAP TreeExplainer values (this may take a moment)...")
    explainer = shap.TreeExplainer(fitted_rf_model)
    shap_output = explainer.shap_values(X_test_final)

    # 4. Handle diverse SHAP output formats (Class 0=Normal, Class 1=Disease)
    if isinstance(shap_output, list) and len(shap_output) >= 2:
        shap_normal, shap_disease = shap_output[0], shap_output[1]
    elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
        shap_normal, shap_disease = shap_output[:, :, 0], shap_output[:, :, 1]
    elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 2:
        shap_disease = shap_output
        shap_normal = -shap_output
    else:
        raise ValueError('Unsupported SHAP output format.')

    # 5. Create dedicated dataframes for each class (Mean raw average, signed)
    df_normal = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': np.mean(shap_normal, axis=0)
    })

    df_disease = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': np.mean(shap_disease, axis=0)
    })

    # 6. Call plotting functions separately
    print("\nGenerating separated SHAP plots (OHE features included, Top 15 filter applied)...")

    # Plot 1: Normal Class (Blue)
    print("Rendering Normal Class Plot...")
    generate_single_class_plot(df_normal, 'Normal (Healthy)')

    # Plot 2: Disease Class (Red)
    print("Rendering Disease Class Plot...")
    generate_single_class_plot(df_disease, 'Disease (Presence)')


if __name__ == "__main__":
    # --- Integration/Execution Block (Scalable for any project structure) ---
    print("1. Loading and splitting data...")
    df = load_data()
    df = drop_unnecessary_columns(df)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("2. Building Preprocessor and Pipeline (no scaling needed for RF)...")
    preprocessor = build_preprocessor(
        numeric_features=ORIGINAL_NUMERIC_FEATURES,
        categorical_features=ORIGINAL_CATEGORICAL_FEATURES,
        scale_numeric=False
    )

    # Using baseline parameters from project context
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=RANDOM_STATE
        ))
    ])

    print("3. Training Baseline Random Forest Model...")
    pipeline.fit(X_train, y_train)

    print("4. Initiating SHAP analysis orchestrator...")
    generate_separated_shap_analysis(
        fitted_rf_model=pipeline.named_steps['classifier'],
        X_test_df=X_test,
        y_test_series=y_test,
        preprocessor=pipeline.named_steps['preprocessor']
    )

    print("5. Initiating ensemble SHAP analysis...")
    generate_ensemble_shap_plot(
        X_train_df=X_train,
        y_train_series=y_train,
        X_test_df=X_test,
        max_background=40,
        max_explain=40,
        top_n=15
    )
