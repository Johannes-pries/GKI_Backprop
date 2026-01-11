import pandas as pd
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# 1. Daten laden
df = pd.read_csv("./iris.csv")
X = df.iloc[:, :-1].values # .values stellt sicher, dass es ein Numpy-Array ist
Y = df.iloc[:, -1].values

# 2. Labels umwandeln (von "setosa" etc. zu 0, 1, 2)
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# 3. Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Skalierung korrigieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # WICHTIG: Nur transform, nicht fit!

# 5. Modell anpassen
model = Sequential()
# Input shape muss der Anzahl der Features entsprechen (bei Iris meist 4)
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=20, activation='relu'))
# 3 Units für 3 Klassen, Softmax für Wahrscheinlichkeiten
model.add(Dense(units=3, activation='sigmoid'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 6. Training
history = model.fit(X_train_scaled, Y_train, epochs=1000, batch_size=32, validation_data=(X_test_scaled, Y_test))

# 7. Evaluation
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print(f"Loss: {loss:.3f}, Accuracy: {accuracy*100:.2f}%")