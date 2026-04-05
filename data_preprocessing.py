import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

RANDOM_STATE = 42

DATA_PATH_UCI = "data/Heart Disease UCI/processed.cleveland.data"

ALL_COLUMNS = [
    "age",
    "sex",             # 1 = male, 0 = female
    "cp",              # chest pain type (1–4)
    "trestbps",        # resting blood pressure (in mm Hg)
    "chol",            # serum cholesterol (in mg/dl)
    "fbs",             # fasting blood sugar (1 = true, 0 = false)
    "restecg",         # resting ECG (0–2)
    "thalch",          # max heart rate achieved
    "exang",           # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",         # ST depression
    "slope",           # slope of peak exercise ST segment (1–3)
    "ca",              # number of major vessels (0–3) colored by fluoroscopy
    "thal",            # thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    "num"              # disease severity (0–4)
]

# Identify Column Types
ORIGINAL_NUMERIC_FEATURES = [
    'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
ORIGINAL_CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs',
                                 'restecg', 'exang', 'slope', 'thal']

# 1. Standard Numeric pipeline (Impute with KNN, then Scale)
NUMERIC_TRANSFORMER = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# 2. Tree-based Numeric pipeline (Impute with KNN, no scaling)
NUMERIC_TRANSFORMER_TREE = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))
])

# 3. Categorical pipeline (Impute most frequent, One-Hot Encode)
CATEGORICAL_TRANSFORMER = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


def load_data(DATA_PATH_UCI=DATA_PATH_UCI):
    # Load raw UCI data, parsing '?' as NaN
    df = pd.read_csv(DATA_PATH_UCI, names=ALL_COLUMNS, na_values="?")
    return df


def drop_unnecessary_columns(df):
    # Target: 0 = No heart disease, 1 = Heart disease
    df["target"] = (df['num'] > 0).astype(int)

    # Drop the original 'num' column (no 'id' or 'dataset' columns in this raw file)
    df = df.drop(columns=['num'])
    return df

def build_preprocessor(numeric_features, categorical_features, scale_numeric=True, pca_features=None, n_components=3):
    # 1. Standard Numeric Pipeline
    numeric_steps = [('imputer', KNNImputer(n_neighbors=5))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    # 2. Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Kept sparse_output=False for safety
    ])

    # Base transformers
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]

    # 3. Dedicated PCA Pipeline (Only added if we request it)
    if pca_features:
        pca_steps = [
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler_before', StandardScaler()), # Scale before PCA (Required for PCA)
            ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE))
        ]
        # This will calculate the PCA and append the components side-by-side 
        # with the output of the 'num' and 'cat' pipelines!
        if scale_numeric:
            pca_steps.append(('scaler_after', StandardScaler()))
            
        pca_transformer = Pipeline(steps=pca_steps)
        transformers.append(('pca', pca_transformer, pca_features))

    return ColumnTransformer(transformers=transformers)