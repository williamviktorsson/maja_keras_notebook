# RPS Realtidskamera — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Skapa `07_rps_kamera.ipynb` — en fristående Colab-notebook där elever tränar ett CNN på egna bilder och klassificerar sten-sax-påse i realtid via webcam.

**Architecture:** Sju notebook-sektioner. Träningsdata hämtas från Google Drive (förifyllda exempelmappar från TF-dataset, eller elevernas egna bilder). Preprocessing sker i en delad funktion som används vid både träning och live-inferens. Realtidskameran drivs av JavaScript i Colab output-cellen; JS anropar Python via `google.colab.kernel.invokeFunction`.

**Tech Stack:** TensorFlow/Keras 3, TensorFlow Datasets, PIL, OpenCV (cv2), google.colab.output, IPython.display.HTML

---

### Task 1: Skapa notebook-fil och cell 1 — Intro (markdown)

**Files:**
- Create: `07_rps_kamera.ipynb`

**Step 1: Skapa notebook**

Skapa filen `07_rps_kamera.ipynb` med en enda markdown-cell:

```markdown
# Bygg din egen Sten-Sax-Påse AI 🤖✊✋✌️

Du har lärt dig hur CNN:er fungerar. Nu bygger du ett från grunden — tränat på **dina egna bilder**.

## Vad vi gör
1. Koppla Google Drive och hämta träningsdata
2. Träna ett CNN på bilder av sten, sax och påse
3. Testa modellen live med din webcam

## Vad du behöver
- Ett Google-konto med Drive
- En kamera (inbyggd eller extern)
- Kör allt uppifrån och ned, en cell i taget
```

**Step 2: Verifiera**

Öppna i Colab, kontrollera att markdown-cellen renderas korrekt.

**Step 3: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add rps camera notebook skeleton with intro"
```

---

### Task 2: Cell 2 — Imports

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till kod-cell**

```python
import os
import json
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

from IPython.display import display, HTML
from google.colab import output as colab_output

print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print("Allt importerat! ✓")
```

**Step 2: Verifiera**

Kör cellen i Colab. Förväntat output:
```
TensorFlow: 2.x.x
Keras: 3.x.x
Allt importerat! ✓
```

**Step 3: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add imports cell"
```

---

### Task 3: Cell 3 — Preprocessing-funktion (delad)

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 1: Förbered data

Samma förbehandlingsfunktion används för **träningsbilder** och **live-kamerabilder** — annars tränar vi på en typ av data men testar på en annan.
```

**Step 2: Lägg till kod-cell**

```python
IMG_SIZE = 150  # Alla bilder skalas till 150×150 pixlar

def forbehandla_bild(img):
    """
    Tar en numpy RGB-array (valfri storlek) och returnerar
    en normaliserad float32-array med formen (150, 150, 3).
    Används vid både träning och live-inferens.
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype('float32') / 255.0

print(f"Bildstorlek: {IMG_SIZE}×{IMG_SIZE} pixlar")
```

**Step 3: Verifiera**

```python
# Snabbtest
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = forbehandla_bild(dummy)
assert result.shape == (150, 150, 3), f"Fel shape: {result.shape}"
assert result.max() <= 1.0, "Inte normaliserat till 0-1"
print("forbehandla_bild() fungerar! ✓")
```

**Step 4: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add shared preprocessing function"
```

---

### Task 4: Cell 4 — Google Drive

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 2: Google Drive

Kör cellen nedan. En popup öppnas — logga in med ditt Google-konto och ge tillstånd.

Dina bilder sparas i Drive så att de finns kvar nästa gång du öppnar Colab.
```

**Step 2: Lägg till kod-cell**

```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/My Drive'
print(f"Drive kopplat! ✓")
print(f"Dina filer ligger under: {DRIVE_ROOT}")
```

**Step 3: Verifiera**

Kör cellen. Förväntat output efter inloggning:
```
Drive kopplat! ✓
Dina filer ligger under: /content/drive/My Drive
```

---

### Task 5: Cell 5 — Exempelmappar (fyll från TF-dataset)

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 3: Exempeldata

Kör den här cellen **en gång** för att fylla dina Drive-mappar med exempelbilder från ett färdigt dataset.

Mappstrukturen som skapas:
```
My Drive/
  rps_exempel/
    rock/        ← ~150 bilder
    paper/       ← ~150 bilder
    scissors/    ← ~150 bilder
  rps_eget/
    rock/        ← tom (fyll på med dina egna bilder!)
    paper/       ← tom
    scissors/    ← tom
```
```

**Step 2: Lägg till kod-cell**

```python
EXEMPEL_DIR = f'{DRIVE_ROOT}/rps_exempel'
EGET_DIR    = f'{DRIVE_ROOT}/rps_eget'
KLASSER     = ['rock', 'paper', 'scissors']

def skapa_mappar(base_dir):
    for klass in KLASSER:
        os.makedirs(f'{base_dir}/{klass}', exist_ok=True)

skapa_mappar(EXEMPEL_DIR)
skapa_mappar(EGET_DIR)

def spara_exempelbilder(data_dir, bilder_per_klass=150):
    """Hämtar rock_paper_scissors från TF Datasets och sparar som JPG till Drive."""
    ds = tfds.load('rock_paper_scissors', split='train', as_supervised=True)

    räknare = [0, 0, 0]
    mål     = bilder_per_klass

    for img_tensor, label_tensor in ds:
        i = int(label_tensor.numpy())
        if räknare[i] >= mål:
            continue

        img_np  = img_tensor.numpy().astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        sökväg  = f'{data_dir}/{KLASSER[i]}/{räknare[i]:04d}.jpg'
        img_pil.save(sökväg, quality=90)
        räknare[i] += 1

        if all(r >= mål for r in räknare):
            break

    for i, klass in enumerate(KLASSER):
        print(f"  {klass}: {räknare[i]} bilder sparade")

# Guard: hoppa över om mappen redan är ifylld
rock_dir = Path(f'{EXEMPEL_DIR}/rock')
befintliga = len(list(rock_dir.glob('*.jpg')))

if befintliga >= 50:
    print(f"Exempelmappar redan ifyllda ({befintliga} bilder i rock/). Hoppar över. ✓")
else:
    print("Laddar ner exempelbilder från TF Datasets...")
    print("(Det här tar ~1-2 minuter — körs bara en gång.)\n")
    spara_exempelbilder(EXEMPEL_DIR)
    print(f"\nKlart! Bilderna ligger nu i Drive under rps_exempel/")
```

**Step 3: Verifiera**

Kör cellen. Förväntat output:
```
Laddar ner exempelbilder från TF Datasets...
(Det här tar ~1-2 minuter — körs bara en gång.)

  rock: 150 bilder sparade
  paper: 150 bilder sparade
  scissors: 150 bilder sparade

Klart! Bilderna ligger nu i Drive under rps_exempel/
```

Kör om cellen — ska visa:
```
Exempelmappar redan ifyllda (150 bilder i rock/). Hoppar över. ✓
```

**Step 4: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add drive folder setup and TF dataset download"
```

---

### Task 6: Cell 6 — Välj dataset

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 4: Välj dataset

Ändra `DATASET` nedan för att välja vilka bilder modellen ska tränas på.

| Värde | Bilder |
|-------|--------|
| `'exempel'` | De 150 exempelbilder vi just laddade ner |
| `'eget'` | Dina egna bilder i `rps_eget/` |

**Hur lägger man till egna bilder?**
1. Öppna [drive.google.com](https://drive.google.com)
2. Navigera till `rps_eget/rock/` och ladda upp foton av en knytnäve
3. Gör samma för `paper/` och `scissors/`
4. Minst 30 bilder per klass rekommenderas, ju fler desto bättre
5. JPG, PNG, WebP — alla format fungerar
```

**Step 2: Lägg till kod-cell**

```python
DATASET = 'exempel'  # Ändra till 'eget' när du har egna bilder

if DATASET == 'exempel':
    DATA_DIR = EXEMPEL_DIR
elif DATASET == 'eget':
    DATA_DIR = EGET_DIR
else:
    raise ValueError(f"DATASET måste vara 'exempel' eller 'eget', fick: '{DATASET}'")

print(f"Använder dataset: '{DATASET}'")
print(f"Sökväg: {DATA_DIR}")

# Räkna bilder
for klass in KLASSER:
    klass_path = Path(f'{DATA_DIR}/{klass}')
    filer = list(klass_path.glob('*.jpg')) + list(klass_path.glob('*.jpeg')) + \
            list(klass_path.glob('*.png')) + list(klass_path.glob('*.webp'))
    print(f"  {klass}: {len(filer)} bilder")
```

**Step 3: Verifiera**

Med `DATASET = 'exempel'`:
```
Använder dataset: 'exempel'
Sökväg: /content/drive/My Drive/rps_exempel
  rock: 150 bilder
  paper: 150 bilder
  scissors: 150 bilder
```

---

### Task 7: Cell 7 — Ladda bilder + visa exempel

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 5: Ladda och träna
```

**Step 2: Lägg till kod-cell**

```python
def ladda_bilder(data_dir):
    """
    Laddar alla bilder från data_dir/{rock,paper,scissors}/.
    Returnerar X (numpy-array med bilder) och y (klassetiketter).
    Stödjer JPG, JPEG, PNG, WebP.
    """
    X, y = [], []
    totalt = 0
    fel    = 0

    for i, klass in enumerate(KLASSER):
        klass_dir = Path(f'{data_dir}/{klass}')
        filer = (list(klass_dir.glob('*.jpg'))  +
                 list(klass_dir.glob('*.jpeg')) +
                 list(klass_dir.glob('*.png'))  +
                 list(klass_dir.glob('*.webp')))

        laddade = 0
        for fil in filer:
            try:
                img = Image.open(fil).convert('RGB')   # hanterar RGBA, gråskala etc.
                img = np.array(img)
                img = forbehandla_bild(img)
                X.append(img)
                y.append(i)
                laddade += 1
            except Exception as e:
                print(f"  Varning: kunde inte ladda {fil.name}: {e}")
                fel += 1

        print(f"  {klass}: {laddade} bilder laddade")
        totalt += laddade

    print(f"\nTotalt: {totalt} bilder ({fel} fel)")
    return np.array(X, dtype='float32'), np.array(y, dtype='int32')

print("Laddar bilder från Drive...")
X, y = ladda_bilder(DATA_DIR)
print(f"X.shape: {X.shape}   y.shape: {y.shape}")
```

**Step 3: Lägg till kodcell — visa bilder**

```python
# Visa 12 slumpmässiga bilder
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
idx = np.random.choice(len(X), 12, replace=False)

for ax, i in zip(axes.flatten(), idx):
    ax.imshow(X[i])
    ax.set_title(KLASSER[y[i]])
    ax.axis('off')

plt.suptitle('Exempelbilder från dataset', fontsize=14)
plt.tight_layout()
plt.show()
```

**Step 4: Verifiera**

Output ska visa 12 bilder märkta med rock/paper/scissors, inga felmeddelanden.

**Step 5: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add image loading and preview cells"
```

---

### Task 8: Cell 8 — Dela upp data + bygg modell + träna

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till kod-cell — dela upp**

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Träning:    {len(X_train)} bilder")
print(f"Validering: {len(X_val)} bilder")
```

**Step 2: Lägg till kod-cell — bygg CNN**

```python
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')  # rock, paper, scissors
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

**Step 3: Lägg till kod-cell — träna**

```python
print("Tränar modellen...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    verbose=1
)

_, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValiderings-accuracy: {val_acc*100:.1f}%")
```

**Step 4: Verifiera**

Modellen ska träna 15 epochs utan fel. Förväntat val_accuracy med exempeldata: >85%.

**Step 5: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add train/val split, CNN architecture, training cells"
```

---

### Task 9: Cell 9 — Utvärdera (kurvor + confusion matrix)

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 6: Utvärdera modellen
```

**Step 2: Lägg till kod-cell**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Träningskurvor
axes[0].plot(history.history['loss'],     label='Träning')
axes[0].plot(history.history['val_loss'], label='Validering')
axes[0].set_title('Förlust (Loss)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Träning')
axes[1].plot(history.history['val_accuracy'], label='Validering')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.show()

# Confusion matrix
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
cm     = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=KLASSER, yticklabels=KLASSER)
plt.title('Confusion Matrix (validering)')
plt.ylabel('Sant')
plt.xlabel('Förutsagt')
plt.tight_layout()
plt.show()
```

**Step 3: Verifiera**

Två plottar ska renderas: träningskurvor + confusion matrix utan fel.

**Step 4: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add evaluation plots and confusion matrix"
```

---

### Task 10: Cell 10 — Realtidskamera (hjälpfunktion)

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till markdown-cell**

```markdown
---
## Del 7: Testa med webcam i realtid! 📷

Kör cellen nedan — din kamera startar och modellen gissar vad den ser var 500:e millisekund.

Håll upp handen med sten ✊, påse ✋ eller sax ✌️ framför kameran!
```

**Step 2: Lägg till kod-cell — definiera starta_kamera**

```python
def starta_kamera(model, klassnamn=None):
    """
    Startar en live-kamerafeed i Colab och klassificerar varje frame.
    Visar procentuella sannolikheter uppdaterade var 500ms.

    Args:
        model:      tränad Keras-modell med softmax-output
        klassnamn:  lista med klassnamn, t.ex. ['rock', 'paper', 'scissors']
    """
    if klassnamn is None:
        klassnamn = KLASSER

    # --- Python-callback som JS anropar ---
    def klassificera_callback(img_b64):
        binary    = base64.b64decode(img_b64.split(',')[1])
        img_array = np.frombuffer(binary, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_proc  = forbehandla_bild(img_rgb)
        img_input = np.expand_dims(img_proc, axis=0)
        pred      = model.predict(img_input, verbose=0)[0]
        return {k: float(v) for k, v in zip(klassnamn, pred)}

    colab_output.register_callback('klassificera_bild', klassificera_callback)

    # --- HTML + JS ---
    klass_json  = json.dumps(klassnamn)
    emoji_map   = json.dumps({'rock': '✊', 'paper': '✋', 'scissors': '✌️'})

    html = f"""
<div id="rps-wrap" style="font-family: 'Courier New', monospace; max-width: 400px;">
  <video id="rps-video" width="320" height="240" autoplay playsinline
         style="border: 2px solid #555; border-radius: 6px; display:block;"></video>
  <div id="rps-pred" style="margin-top:10px; font-size:15px; line-height:2.0;">
    Startar kamera...
  </div>
</div>

<script>
(async () => {{
  const video   = document.getElementById('rps-video');
  const predDiv = document.getElementById('rps-pred');
  const klasser = {klass_json};
  const emojis  = {emoji_map};

  // Starta webcam
  const stream  = await navigator.mediaDevices.getUserMedia({{video: true}});
  video.srcObject = stream;
  await video.play();

  const canvas  = document.createElement('canvas');
  canvas.width  = video.videoWidth  || 320;
  canvas.height = video.videoHeight || 240;

  async function uppdatera() {{
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const imgData = canvas.toDataURL('image/jpeg', 0.7);

    let result;
    try {{
      result = await google.colab.kernel.invokeFunction(
        'klassificera_bild', [imgData], {{}}
      );
    }} catch (e) {{
      predDiv.innerHTML = '<span style="color:red;">Fel: ' + e.message + '</span>';
      return;
    }}

    const preds = result.data['application/json'];

    // Hitta bästa klass
    let bästaKlass = klasser[0];
    let bästaVärde = 0;
    for (const k of klasser) {{
      if (preds[k] > bästaVärde) {{ bästaVärde = preds[k]; bästaKlass = k; }}
    }}

    // Bygg HTML-staplar
    let html = '';
    for (const k of klasser) {{
      const pct    = Math.round(preds[k] * 100);
      const filled = Math.round(preds[k] * 20);
      const bar    = '█'.repeat(Math.max(0, filled)) +
                     '░'.repeat(Math.max(0, 20 - filled));
      const isTop  = (k === bästaKlass);
      const color  = isTop
        ? (pct >= 60 ? '#22c55e' : '#f59e0b')
        : '#9ca3af';
      const weight = isTop ? 'bold' : 'normal';
      const emoji  = emojis[k] || '?';
      html += `<div style="color:${{color}};font-weight:${{weight}};">` +
              `${{emoji}} ${{k.padEnd(9)}} ${{bar}} ${{String(pct).padStart(3)}}%</div>`;
    }}
    predDiv.innerHTML = html;
  }}

  setInterval(uppdatera, 500);
}})();
</script>
"""
    display(HTML(html))
    print("Kameran är igång! Håll upp handen framför kameran. ✓")
```

**Step 3: Verifiera**

Kör cellen — ingen output, bara funktionen definieras.

**Step 4: Commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add real-time camera helper function"
```

---

### Task 11: Cell 11 — Starta kameran

**Files:**
- Modify: `07_rps_kamera.ipynb`

**Step 1: Lägg till kod-cell**

```python
starta_kamera(model)
```

**Step 2: Verifiera**

Kör cellen. Webbläsaren ber om kamera-tillstånd. Efter tillstånd:
- Live-videofeed visas (320×240)
- Under videon: tre rader med ✊/✋/✌️, staplar och procentsatser
- Procentsatserna uppdateras var 500ms utan fel i konsolen
- Grön färg på den klass med högst sannolikhet (>60%)
- Orange färg om modellen är osäker (bästa klassen <60%)

**Step 3: Testa edge cases**
- Håll upp sten → rock ska dominera
- Håll upp påse → paper ska dominera
- Håll upp sax → scissors ska dominera
- Tom bakgrund → sannolikheter ska fördelas jämnt

**Step 4: Final commit**

```bash
git add 07_rps_kamera.ipynb
git commit -m "feat: add camera trigger cell — notebook complete"
```

---

## Verifieringschecklista (hela notebook)

- [ ] Alla celler körs utan fel från topp till botten i Colab
- [ ] Guard i cell 5 hoppar korrekt över nedladdning vid omkörning
- [ ] `DATASET = 'eget'` med tomma mappar ger tydligt felmeddelande (FileNotFoundError eller tom X-array)
- [ ] `forbehandla_bild()` används på identiskt sätt i ladda_bilder() och klassificera_callback()
- [ ] Kameran startar och procentsatser uppdateras i realtid
- [ ] PNG med alfa-kanal hanteras (convert('RGB') i ladda_bilder)
