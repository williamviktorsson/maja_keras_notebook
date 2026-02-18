# Design: Bygg din egen Sten-Sax-Påse AI (07_rps_kamera.ipynb)

## Kontext
Fristående notebook (men förutsätter att elever gjort 06_keras.ipynb). Skriven på svenska. Körs i Google Colab med GPU. Ingen tung teorirepetition — eleverna vet vad CNN är.

## Mål
- Elever tränar ett CNN på egna bilder
- Realtidskamera visar procentuella prediktioner live
- Smidig övergång från exempeldata till egna bilder

---

## Struktur (7 sektioner)

### 1. Intro
Markdown-cell: vad vi bygger, vad som krävs (Google-konto, kamera).

### 2. Koppla Google Drive
En cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Förklaring: varför Drive (bilder sparas mellan sessioner).

### 3. Fyll exempelmappar (kör en gång)
- Skapar `My Drive/rps_exempel/rock/`, `paper/`, `scissors/`
- Laddar TF `rock_paper_scissors` dataset
- Sparar ~150 bilder per klass som JPG till Drive
- Cellen har en guard: hoppa över om mappen redan finns och inte är tom

### 4. Välj dataset
```python
DATASET = 'exempel'  # Ändra till 'eget' när du har egna bilder
```
Beroende på värdet pekar `DATA_DIR` på:
- `My Drive/rps_exempel/` — förifylld med TF-bilder
- `My Drive/rps_eget/` — tom, eleverna fyller på själva (rock/, paper/, scissors/)

Instruktion om hur man lägger dit bilder (telefon → Drive-app, eller dra-och-släpp på drive.google.com).

### 5. Ladda och träna
**Preprocessing (samma funktion för träning och live-kamera):**
```python
IMG_SIZE = 150

def forbehandla_bild(img):  # img = numpy RGB-array
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype('float32') / 255.0
```

**Ladda bilder från Drive:**
- Glob alla filer i varje klassmapp
- `PIL.Image.open(...).convert('RGB')` — hanterar JPG, PNG, WebP, BMP
- Applicera `forbehandla_bild()`
- 80/20 train/val split

**CNN-arkitektur** (samma mönster som 06_keras.ipynb):
```
Input(150×150×3)
Conv2D(32, 3×3, relu) → MaxPooling
Conv2D(64, 3×3, relu) → MaxPooling
Conv2D(64, 3×3, relu) → MaxPooling
Flatten → Dense(64, relu) → Dropout(0.3)
Dense(3, softmax)
```

**Träning:**
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15)
```

### 6. Utvärdera
- Träningshistorik (loss + accuracy kurvor)
- Test accuracy
- Confusion matrix med seaborn

### 7. Realtidskamera
**Python-sida:**
```python
def starta_kamera(model, klassnamn=['rock', 'paper', 'scissors']):
    # Registrerar Python-callback som JS kan anropa
    # Injicerar HTML/JS i output-cellen
    ...
```

Inuti: `google.colab.output.register_callback('klassificera', callback_fn)`

Callback tar base64-bild, kör `forbehandla_bild()`, returnerar JSON med procent.

**JS-sida (gömd i hjälpfunktionen):**
- Skapar `<video>` (320×240 webcam feed)
- `setInterval` var 500ms: fångar canvas-frame → `google.colab.kernel.invokeFunction('klassificera', ...)` → uppdaterar HTML-staplar

**Display:**
```
[Video 320×240]

✊ Rock:    ████████████░░░░  74%
✋ Paper:   ██░░░░░░░░░░░░░░  15%
✌ Scissors:█░░░░░░░░░░░░░░░  11%
```
Färg på den starkaste klassen (grön = hög confidence, orange = osäker).

**Anrop från eleven:**
```python
starta_kamera(model)
```

---

## Bildformat
Inga krav — PIL `.convert('RGB')` hanterar JPG, PNG, WebP, BMP, RGBA automatiskt.

## Dataflöde
```
Drive-bild (jpg/png) → PIL.open().convert('RGB') → numpy → forbehandla_bild() → model.predict()
Kamera-frame (JS/b64) → cv2.imdecode() → cv2.cvtColor(BGR→RGB) → forbehandla_bild() → model.predict()
```
Identisk preprocessing på båda vägar — ingen train/inference mismatch.
