# -*- coding: utf-8 -*-






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df = pd.read_csv("C:/Users/Alejandro/Desktop/Data/AmrCount/Amr_Count.tsv", sep='\t', index_col=0)


categorical_columns = ['genus', 'species', 'antibiotic']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])


target_column = 'phenotype'
X = df.drop(columns=['measurement_value', target_column])
y = df[target_column]
y = LabelEncoder().fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_columns = df.columns[df.columns.get_loc('measurement_value') + 1:]


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=800))  
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns)
    ],
    remainder='passthrough'  
)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


classifier = RandomForestClassifier(random_state=42)


classifier.fit(X_train_transformed, y_train)


y_pred = classifier.predict(X_test_transformed)


print(classification_report(y_test, y_pred))

