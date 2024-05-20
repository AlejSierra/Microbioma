# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
matrix_file = "C:/Users/Alejandro/Desktop/Data/ResistanceCountStrict.tsv"
data = pd.read_csv(matrix_file, sep='\t', index_col=0)
salmonella_data = data[data['genus'] == 'Klebsiella']

"target"
X = salmonella_data.drop(columns=['phenotype', 'genus']).values
y = salmonella_data['phenotype'].values

"entrenamiento"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
numeric_features = list(range(5, X.shape[1])) 
numeric_transformer = StandardScaler()

"datos no numericos"
non_numeric_features = [0, 1, 2, 4]  
non_numeric_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('non_num', non_numeric_transformer, non_numeric_features)
    ])

"Componentes Principales"
pca = PCA(n_components=5)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('pca', pca),
                           ('classifier', RandomForestClassifier())])

"Ajuste del modelo"
pipeline.fit(X_train, y_train)

"Predicción"
y_pred = pipeline.predict(X_test)

"Evaluación"
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
