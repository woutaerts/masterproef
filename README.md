# Detectie en Tracking van Lopers via Computer Vision

**Masterproef – KU Leuven, Faculteit Industriële Ingenieurswetenschappen, Campus Groep T Leuven**
**Academiejaar 2025–2026**

> *Vergelijking van klassieke methoden en deep learning-methoden voor videoanalyse van lopers in groep*
> Wout Aerts — Promotor: Prof. dr. ir. P. Vandewalle

---

## Overzicht

Deze repository bevat de volledige Python-codebasis ontwikkeld voor een masterproef die klassieke computer vision vergelijkt met deep learning voor de automatische detectie, tracking en biomechanische analyse van lopers in groep op basis van top-down drone-beelden.

Er worden twee pipelines geïmplementeerd en geëvalueerd:

- **Klassieke pipeline (Dataset A):** HSV-kleursegmentatie gericht op magenta petjes van lopers op een atletiekpiste. Geëvalueerd met Precision, Recall, F1-score, MOTA, MOTP.
- **Deep learning-pipeline (Dataset B & C):** Custom YOLO11-model getraind via transfer learning op Roboflow, gecombineerd met ByteTrack voor stabiele multi-object tracking. Toegepast op dichte marathonbeelden (Matadibrug, Gent). Geëvalueerd via de fragmentation rate.

Naast detectie en tracking extraheert de deep learning-pipeline drie biomechanische parameters rechtstreeks uit het videobeeld:


| Parameter             | Methode                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------ |
| Loopsnelheid          | Euclidische verplaatsing van bounding box-centroïden, omgezet naar m/pixel via kalibratie |
| Ruimtelijke verdeling | Cumulatieve heatmap met homografiegebaseerde cameracompensatie                             |
| Cadans (SPM)          | Bandpass-gefilterd medio-lateraal head sway-signaal, piek-tot-piekdetectie                 |

De cadansmetingen worden gevalideerd aan de hand van IMU-sensordata (Dataset C, J16-experiment in Puurs), met een gemiddelde absolute fout van **7,11 SPM (4,6%)** over 15 proefpersonen.

---

## Repositorystructuur

```
masterproef-running-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── dataset_A/
│   │   ├── ground-truth-generatie.ipynb       ← Manuele centroïde-annotatie (Google Colab)
│   │   ├── tuning.ipynb                       ← Interactieve HSV- en filtertuning
│   │   └── evaluatie.ipynb                    ← Precision / Recall / F1 / MOTA / MOTP
│   │
│   ├── dataset_B/
│   │   ├── tuning.ipynb                       ← Architectuurbenchmark & hyperparametertuning
│   │   ├── extractie.ipynb                    ← JSON-extractietool & brugkalibratie
│   │   ├── snelheidsanalyse.ipynb             ← Snelheidsstatistieken & heatmap
│   │   └── cadansanalyse.ipynb                ← Cadansextractie via head sway
│   │
│   └── dataset_C/
│       ├── extractie.ipynb                    ← Detectie & tracking naar JSON (J16-video)
│       ├── cadansanalyse.ipynb                ← Cadansextractie (J16-dataset)
│       └── vergelijking.ipynb                 ← AI vs. IMU-sensor & Bland-Altmananalyse
│
└── data/
    ├── detecties/
    │   ├── marathon.json                      ← YOLO11-tracks: 30-seconden pelotonsegment
    │   ├── marathon_uitgebreid.json           ← YOLO11-tracks: 5-minuten controlesegment
    │   └── P_J16.json                         ← YOLO11-tracks: J16-experiment Puurs
    ├── ground_truth/
    │   ├── dataset_A/
    │   │   └── ground_truth_atletiekpiste.json ← Manuele centroïdeannotaties (60 frames)
    │   └── dataset_C/
    │       └── P_J16.h5                       ← IMU-sensordata (opvragen via e-mail)
    ├── heatmap/
    │   └── matadibrug.png                     ← Referentieframe voor heatmap-achtergrond 
    └── video/
        ├── Atletiekpiste.mp4                  ← Opvragen via e-mail
        ├── Marathon.mp4                       ← Opvragen via e-mail
        └── P_J16.MP4                          ← Opvragen via e-mail
```

---

## Toegang tot de data

Vanwege privacy- en licentieoverwegingen zijn de videobestanden en ruwe sensordata **niet opgenomen** in deze repository. De vooraf berekende YOLO11-detectiebestanden (JSON) **zijn wel** inbegrepen en laten de meeste analyses toe zonder de originele video's.


| Bestand                                                       | Openbaar beschikbaar | Hoe verkrijgbaar      |
| ------------------------------------------------------------- | -------------------- | --------------------- |
| `data/detecties/marathon.json`                                | Ja                   | Inbegrepen in de repo |
| `data/detecties/marathon_uitgebreid.json`                     | Ja                   | Inbegrepen in de repo |
| `data/detecties/P_J16.json`                                   | Ja                   | Inbegrepen in de repo |
| `data/ground_truth/dataset_A/ground_truth_atletiekpiste.json` | Ja                   | Inbegrepen in de repo |
| `Atletiekpiste.mp4`                                           | Nee                  | Opvragen via e-mail   |
| `Marathon.mp4`                                                | Nee                  | Opvragen via e-mail   |
| `P_J16.MP4`                                                   | Nee                  | Opvragen via e-mail   |
| `P_J16.h5` (IMU-sensordata)                                   | Nee                  | Opvragen via e-mail   |
| Roboflow API-key                                              | Nee                  | Opvragen via e-mail   |

Stuur een e-mail naar **wout.aerts@student.kuleuven.be** om toegang te vragen tot de videobestanden, sensordata of de Roboflow API-key.

> Het videomateriaal is afkomstig uit het doctoraatsonderzoek van dr. Jasper Lottefier (KU Leuven) en doctoraatstudent Senne Strobbe (KU Leuven / UGent). Hun data wordt uitsluitend beschikbaar gesteld voor academische reproduceerbaarheid.

---

## Aan de slag

⚠️ **Belangrijke opmerking over Python-versies:** Zorg ervoor dat je **Python 3.11 of Python 3.12** gebruikt. De dependency `inference` (die gebruikt wordt voor de Roboflow/YOLO-modellen) ondervindt momenteel compatibiliteits- en build-problemen op de recentere Python 3.13 vanwege restricties in onderliggende pakketten (zoals `numpy<2.0` en `scipy<1.13`).

### 1. Repository klonen

```bash
git clone https://github.com/woutaerts/masterproef.git
cd masterproef
```

### 2. Afhankelijkheden installeren

```bash
pip install -r requirements.txt
```

### 3. Databestanden plaatsen

Kopieer de videobestanden naar `data/video/` en het sensorbestand naar `data/ground_truth/dataset_C/` nadat je ze via e-mail hebt ontvangen. De JSON-detectiebestanden staan al in `data/detecties/`.

### 4. Notebooks openen

Alle notebooks in `notebooks/dataset_B/` en `notebooks/dataset_C/` die analyses uitvoeren op basis van JSON-detectiebestanden kunnen **lokaal worden uitgevoerd zonder videobestanden**. Notebooks die video-invoer vereisen (ground truth-generatie, detectie-extractie, architectuurbenchmark) hebben de originele MP4-bestanden en een Roboflow API-key nodig.

---

## Beschrijving van de notebooks

### Dataset A — Klassieke pipeline (Atletiekpiste)


| Notebook                       | Beschrijving                                                                                                                                                              | Vereist video |
|--------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `ground-truth-generatie.ipynb` | Interactieve JavaScript-annotatieinterface (uitsluitend Google Colab). Slaat centroïdcoördinaten op als JSON-bestand.                                                   | Ja            |
| `tuning.ipynb`                 | Interactieve HSV-kleursliders en geometrische filtertuning. Toont het binaire detectiemasker in real time via`ipywidgets`.                                                | Ja            |
| `evaluatie.ipynb`              | Volledige evaluatie op de testset: Precision, Recall, F1-score, MOTA, MOTP en IDSW via`motmetrics`. Valt terug op hardcoded ground truth als er geen JSON wordt gevonden. | Ja            |

**Resultaten (Dataset A, testset — 50 frames, 8 lopers):**


| Metriek   | Waarde      |
| --------- | ----------- |
| Precision | 100,00%     |
| Recall    | 100,00%     |
| F1-score  | 100,00%     |
| MOTA      | 1,00 (100%) |
| MOTP      | 3,89 px     |
| IDSW      | 0           |

---

### Dataset B — Deep learning-pipeline (Marathon, Matadibrug Gent)


| Notebook                           | Beschrijving                                                                                                                                                                                                               | Vereist video / API-key |
|------------------------------------| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `tuning.ipynb`                     | Systematische benchmark van vier architecturen (RF-DETR Nano/Medium/Large, YOLO11, YOLO12, YOLO26) over meerdere Confidence Threshold-waarden en Lost Track Buffer-waarden. Output: getelde lopers vs. ground truth (280). | Ja                      |
| `extractie.ipynb`                  | JSON-extractietool: voert YOLO11 + ByteTrack uit op de marathonvideo en slaat bounding box-centroïden per tracker-ID op. Bevat ook een interactieve brughoekanntatietool voor homografiekalibratie.                       | Ja                      |
| `snelheidsanalyse.ipynb`           | Laadt`marathon.json`, berekent snelheid per loper (kalibratie: 5,8 m / 608 px), genereert een snelheidshistogram en een genormaliseerde heatmap met homografiecompensatie.                                                 | Alleen JSON             |
| `cadansanalyse.ipynb`              | Laadt`marathon.json`, past een 2e-orde Butterworth-bandpassfilter (0,8–2,0 Hz) toe op het Y-centroïdsignaal, detecteert pieken en berekent cadans (SPM) via de piek-tot-piekmethode.                                     | Alleen JSON             |

**Resultaten (Dataset B, 280 ground truth-lopers):**


| Metriek                         | Waarde                   |
| ------------------------------- | ------------------------ |
| Gegenereerde unieke tracker-IDs | 295                      |
| Fragmentation rate (F_rate)     | 5,4%                     |
| Mediane snelheid                | 10,21 km/h (5:52 min/km) |
| Mediane cadans                  | 164 SPM                  |

---

### Dataset C — Biomechanische validatie (J16, Puurs)


| Notebook              | Beschrijving                                                                                                                                                                                                                                   | Vereist video / API-key |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `extractie.ipynb`     | Voert YOLO11 (Puurs fine-tuned model) + ByteTrack uit op`P_J16.MP4` en slaat bounding box-centroïden op als `P_J16.json`.                                                                                                                     | Ja                      |
| `cadansanalyse.ipynb` | Past hetzelfde head sway-cadansalgoritme toe als Dataset B op de J16-trackingdata.                                                                                                                                                             | Alleen JSON             |
| `vergelijking.ipynb`  | Laadt`P_J16.json` en `P_J16.h5`. Koppelt tracker-IDs aan sensor-IDs via de handmatige mappingtabel. Berekent MAE, RMSE en genereert een Bland-Altmangrafiek. Bevat ook ruimtelijke maskering om het effect van 180°-bochten te kwantificeren. | JSON + H5               |

**Resultaten (Dataset C, 15 proefpersonen):**


| Metriek                                  | Waarde   |
| ---------------------------------------- | -------- |
| MAE (AI vs. IMU-sensor)                  | 7,11 SPM |
| Relatieve MAE                            | 4,6%     |
| RMSE                                     | 8,52 SPM |
| Systematische bias                       | +6,9 SPM |
| MAE na ruimtelijke maskering van bochten | ~2,5 SPM |

---

## Technische details

### Kalibratieconstante

Snelheidsberekeningen voor Dataset B gebruiken een enkelvoudige pixel-naar-meterconstante afgeleid van de bekende breedte van de Matadibrug (5,8 m = 608 pixels in 4K):

```
m_px = 5,8 / 608 ≈ 9,54 × 10⁻³ m/px
```

### Bandpassfilterparameters

Cadansextractie gebruikt een 2e-orde Butterworth-bandpassfilter, tweemaal toegepast (voor- en achterwaarts via `filtfilt`) om fasevervorming te vermijden:

```
f_laag = 0,8 Hz  (96 SPM  — fysiologische ondergrens)
f_hoog = 2,0 Hz  (240 SPM — fysiologische bovengrens)
```

### Tracker-ID naar sensor-ID mapping (Dataset C)

De koppeling tussen ByteTrack-IDs en IMU-sensorlabels is hardcoded in `vergelijking.ipynb` op basis van manuele visuele matching. Proefpersonen P_12 en P_66 genereerden elk meerdere tracker-IDs door track fragmentation; hun cadanswaarden worden gewogen gemiddeld op basis van trackduur voor de vergelijking met de sensormeting.

---

## Afhankelijkheden

Installeer alle afhankelijkheden met:

```bash
pip install -r requirements.txt
```

Het Roboflow `inference`-pakket is enkel vereist voor notebooks die live modelinferentie uitvoeren (`dataset_B/tuning.ipynb`, `dataset_B/extractie.ipynb`, `dataset_C/extractie.ipynb`). Alle analysenotebooks die werken op vooraf berekende JSON-bestanden hebben dit pakket niet nodig.

---

## Dankwoord

- Dr. Jasper Lottefier (KU Leuven) — originele atletiekpistdataset en doctoraatsonderzoek naar voetgangersbrug-trillingen door loopacties.
- Doctoraatstudent Senne Strobbe (KU Leuven / UGent) — J16-video en IMU-sensordata van het Puurs-experiment.
- Prof. dr. ir. P. Van den Broeck en dr. Katrien Van Nimmen (KU Leuven) — aanvullende data rond het pelotoneffect.
- Promotor: Prof. dr. ir. Patrick Vandewalle (KU Leuven).
