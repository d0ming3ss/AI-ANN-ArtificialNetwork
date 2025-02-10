# ANN ArtificialNetwork

# Sieć neuronowa w TensorFlow

## Opis
Ten projekt implementuje prostą sieć neuronową w TensorFlow, która klasyfikuje dane na podstawie sumy ich wartości. Model jest trenowany na losowo wygenerowanych danych i zapisuje się po zakończeniu treningu, aby móc być ponownie użytym.

## Wymagania

Aby uruchomić projekt, potrzebujesz:
- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib

Możesz zainstalować wymagane pakiety za pomocą:
```bash
pip install tensorflow numpy matplotlib
```

## Struktura projektu

- `create_model(input_dim)`: Tworzy model sieci neuronowej.
- `generate_data()`: Generuje dane treningowe i testowe.
- Wczytanie istniejącego modelu lub stworzenie nowego.
- Trening modelu na wygenerowanych danych.
- Zapis modelu po zakończeniu treningu.
- Ewaluacja modelu na danych testowych.
- Wizualizacja wyników treningu.

## Uruchomienie
Aby uruchomić projekt, wykonaj polecenie:
```bash
python main.py
```

## Wizualizacja wyników
Podczas treningu modelu zostaną wygenerowane wykresy prezentujące:
- Straty (loss)
- Dokładności (accuracy)

## Autor inż. Dominik Ciura
Projekt został opracowany jako część nauki i eksploracji TensorFlow.

