import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Cargar dataset
dataset = pd.read_csv("data.csv")

# Eliminar la última columna vacía
dataset = dataset.drop(columns=["Unnamed: 32"])

# Variable dependiente y variables independientes
X_raw = dataset.iloc[:, 2:]  # columnas numéricas
y_raw = dataset.iloc[:, 1]   # columna diagnosis (M o B)

# Codificar etiquetas M/B a 1/0
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Reemplazar NaNs por la media en X
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X_raw)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenar el SVM
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicción y evaluación
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Matriz de confusión:", cm)
print("Precisión:", acc)




