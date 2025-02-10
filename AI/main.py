import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
import matplotlib.pyplot as plt

# Wymuś użycie CPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Ustawienia
MODEL_PATH = "ann_model.h5"
EPOCHS = 10
BATCH_SIZE = 32

# Funkcja tworząca model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generowanie danych
def generate_data():
    X = np.random.rand(10000, 10)
    y = (X.sum(axis=1) > 5).astype(int)
    return X, y

# Wczytaj lub stwórz nowy model
if os.path.exists(MODEL_PATH):
    print("Wczytywanie istniejącego modelu...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Rekompilacja
else:
    print("Tworzenie nowego modelu...")
    X, _ = generate_data()
    model = create_model(X.shape[1])

# Dane treningowe
X, y = generate_data()

# Trening modelu
print("Trening modelu...")
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

# Zapisz model
print("Zapisywanie modelu...")
model.save(MODEL_PATH)
print("Model zapisany w:", MODEL_PATH)

# Testowanie modelu
X_test, y_test = generate_data()
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Strata: {loss:.4f}, Dokładność: {accuracy:.4f}")

# Wizualizacja wyników na jednym diagramie
fig, ax1 = plt.subplots(figsize=(10, 6))

# Wykres straty
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Strata', color='tab:red')
ax1.plot(history.history['loss'], color='tab:red', label='Strata')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Tworzymy drugi wykres na tej samej planszy dla dokładności
ax2 = ax1.twinx()
ax2.set_ylabel('Dokładność', color='tab:blue')
ax2.plot(history.history['accuracy'], color='tab:blue', label='Dokładność')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Tytuł i wyświetlenie wykresu
fig.tight_layout()
plt.title('Strata i dokładność modelu podczas treningu')
plt.show()
