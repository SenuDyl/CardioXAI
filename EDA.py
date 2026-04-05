import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DATA_PATH_UCI, load_data, drop_unnecessary_columns
from feature_engineering import add_synthetic_features, ORIGINAL_CATEGORICAL_FEATURES, ORIGINAL_NUMERIC_FEATURES

# Load data
df = load_data(DATA_PATH_UCI)

# Drop ID and dataset origin as they have no clinical value
df = drop_unnecessary_columns(df)

# Define output directory for EDA results
EDA_OUTPUT_DIR = "eda_results/"
if not os.path.exists(EDA_OUTPUT_DIR):
    os.makedirs(EDA_OUTPUT_DIR)

# Set visual style for the paper
sns.set_theme(style="whitegrid", palette="muted")
disease_palette = {0: "#4C72B0", 1: "#C44E52"}
disease_label_palette = {'No': "#4C72B0", 'Yes': "#C44E52"}
continuous_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']

# Readable display names for correlation heatmaps
COLUMN_LABELS = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting BP',
    'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'thalch': 'Max Heart Rate',
    'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope',
    'ca': 'Major Vessels',
    'thal': 'Thalassemia',
    'target': 'Heart Disease',
    'hr_competence_ratio': 'HR Competence Ratio',
    'severe_ischemia': 'Severe Ischemia'
}

# Shared display mapping for consistent legends
df_plot = df.copy()
df_plot['Heart Disease'] = df_plot['target'].map({0: 'No', 1: 'Yes'})

# ==========================================
# Phase 1: Histograms
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cols_to_plot = {'age': 'Age',
                'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol'}

titles = ['Age Distribution', 'Resting Blood Pressure', 'Cholesterol']

for i, (col, label) in enumerate(cols_to_plot.items()):
    sns.histplot(data=df_plot, x=col, hue='Heart Disease', kde=True, stat="density",
                 common_norm=False, hue_order=['No', 'Yes'], ax=axes[i],
                 palette=disease_label_palette, alpha=0.5)
    axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
    axes[i].set_xlabel(label, fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)

    # Save each histogram separately for paper-ready figures
    fig_single, ax_single = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df_plot, x=col, hue='Heart Disease', kde=True, stat="density",
                 common_norm=False, hue_order=['No', 'Yes'], ax=ax_single,
                 palette=disease_label_palette, alpha=0.5)
    ax_single.set_title(titles[i], fontsize=14, fontweight='bold')
    ax_single.set_xlabel(label, fontsize=12)
    ax_single.set_ylabel('Density', fontsize=12)
    fig_single.tight_layout()
    # fig_single.savefig(f"{EDA_OUTPUT_DIR}phase1_histogram_{col}.png",
    #                    dpi=300, bbox_inches='tight')
    plt.close(fig_single)

fig.tight_layout()
plt.show()


# ==========================================
# Phase 2: 2D Scatter Plot (Validating hr_competence_ratio)
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot, x='age', y='thalch', hue='Heart Disease',
                hue_order=['No', 'Yes'], palette=disease_label_palette,
                edgecolor='k', s=80, alpha=0.8)

# Draw the Max Heart Rate reference line (y = 220 - age)
x_vals = np.array([df['age'].min(), df['age'].max()])
y_vals = 220 - x_vals
plt.plot(x_vals, y_vals, color='black', linestyle='--',
         linewidth=2, label='Max HR (220 - Age)')

plt.title('Maximum Heart Rate Achieved vs. Age',
          fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Maximum Heart Rate (thalch)', fontsize=14)
plt.legend(title='Heart Disease')
# plt.savefig(f"{EDA_OUTPUT_DIR}phase2_scatter_hr.png",
#             dpi=300, bbox_inches='tight')
plt.show()


# ==========================================
# Phase 3: Grouped Box Plot (Defending severe_ischemia threshold)
# ==========================================
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_plot,
    x='exang',
    y='oldpeak',
    hue='Heart Disease',
    hue_order=['No', 'Yes'],
    palette=disease_label_palette,
    width=0.7,
    gap=0.12)

# Draw the 2.0 ST Depression threshold line
plt.axhline(y=2.0, color='black', linestyle='--',
            linewidth=2, label='Ischemia Threshold (>2.0)')

plt.title('ST Depression (oldpeak) grouped by Exercise-Induced Angina (exang)',
          fontsize=14, fontweight='bold')
plt.xlabel('Exercise Induced Angina (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('ST Depression (oldpeak)', fontsize=12)
plt.legend(title='Heart Disease')
# plt.savefig(f"{EDA_OUTPUT_DIR}phase3_boxplot_ischemia.png",
#             dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Plot 4 & 5: Categorical Distributions
# ==========================================
# Create a fresh figure so these plots are independent from previous phases
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 4: Sex Distribution
sns.countplot(data=df_plot, x='sex', hue='Heart Disease', hue_order=['No', 'Yes'],
              palette=disease_label_palette,
              edgecolor='black', ax=axes[0])
axes[0].set_title('Heart Disease Prevalence by Sex',
                  fontsize=16, fontweight='bold')
axes[0].set_xlabel('Sex', fontsize=14)
axes[0].set_ylabel('Number of Patients', fontsize=14)
axes[0].legend(title='Heart Disease')

# ==========================================
# Plot 5: Chest Pain Type (cp) Distribution
# ==========================================
# Optional: If your dataset uses the long names, we can rotate them so they fit perfectly
sns.countplot(data=df_plot, x='cp', hue='Heart Disease', hue_order=['No', 'Yes'],
              palette=disease_label_palette,
              edgecolor='black', ax=axes[1])
axes[1].set_title('Heart Disease by Chest Pain Type',
                  fontsize=16, fontweight='bold')
axes[1].set_xlabel('Chest Pain Type', fontsize=14)
axes[1].set_ylabel('Number of Patients', fontsize=14)
axes[1].legend(title='Heart Disease')

# Rotate chest pain labels for readability
axes[1].tick_params(axis='x', rotation=15)

fig.tight_layout()
# plt.savefig(f"{EDA_OUTPUT_DIR}phase4_categorical_distributions.png",
#             dpi=300, bbox_inches='tight')
plt.show()


# ==========================================
# Phase 6: Box Plots & Correlation Heatmap (Final EDA Summary)
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} vs Target')

# Hide the unused 6th subplot
axes[5].set_visible(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
# Calculate correlation only on numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
corr_matrix = corr_matrix.rename(index=COLUMN_LABELS, columns=COLUMN_LABELS)

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# Svave the heatmap for paper-ready figure
plt.savefig(f"{EDA_OUTPUT_DIR}phase6_correlation_heatmap.png",
            dpi=300, bbox_inches='tight')
plt.show()

df, _, _ = add_synthetic_features(
    df, ORIGINAL_CATEGORICAL_FEATURES, ORIGINAL_NUMERIC_FEATURES)

# ==========================================
# Phase 7: Final EDA Summary with New Features (Correlation Heatmap)
# ==========================================

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
corr_matrix = corr_matrix.rename(index=COLUMN_LABELS, columns=COLUMN_LABELS)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap with Engineered Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
