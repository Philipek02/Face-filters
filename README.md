# Face Filter (OpenCV AR)

Nakładanie prostych filtrów — okulary lub wąsy — na twarz wykrytą w obrazie z kamery.

## Wymagania

```bash
Python ≥ 3.9
pip install opencv-python numpy
```

## Uruchomienie

```bash
python face_filter.py          # domyślna kamera (index 0)
python face_filter.py --cam 1  # inna kamera / plik wideo
```

## Sterowanie w oknie

| Klawisz | Działanie     |
| ------- | ------------- |
| **g**   | załóż okulary |
| **m**   | załóż wąsy    |
| **n**   | usuń filtr    |
| **q**   | zakończ       |

## Dodawanie własnych filtrów

1. Umieść pliki PNG z przezroczystym tłem w katalogu `filters/`.
2. Zaktualizuj słownik `FILTERS` w `face_filter.py`, np.:

```python
FILTERS = {
    "glasses": "filters/my_glasses.png",
    "stash":   "filters/my_mustache.png",
}
```

> **Wskazówka:** Szukaj fraz `"glasses png transparent"`, `"moustache png transparent"` (lub PL odpowiedników) w Google Grafika — zapisuj warianty PNG z alfą.

## Działanie pod maską

* Detekcja twarzy — klasyfikator Haar Cascade `haarcascade_frontalface_default.xml`.
* Filtr PNG skalowany jest do szerokości twarzy (± korekta procentowa) i nakładany z użyciem kanału alfa.
* Kamera / wideo odbierane przez OpenCV (`cv2.VideoCapture`). 

## Rozwiązywanie problemów

* **`IndexError: tuple index out of range`** — plik PNG nie ma kanału alfa.

  * Użyj wersji z przeźroczystym tłem lub przekonwertuj w GIMP/Krita.
* Czarny ekran? Podaj poprawny index kamery (`--cam 0 | 1 | 2 …`) albo ścieżkę do pliku wideo.
