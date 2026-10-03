# Data Cards — Diabetic Retinopathy Public Datasets

Grade scheme: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
(for Messidor-1: 2 = Moderate+Severe merged, 3 = Proliferative, no separate grade 4).

---

## Data card 1

| Field | Value |
|---|---|
| Dataset name | APTOS 2019 (Blindness Detection) |
| Source URL | https://www.kaggle.com/c/aptos2019-blindness-detection |
| License | Check competition rules on the Kaggle page before redistribution (research/competition use; no explicit open license stated) |
| Size (images) | 3,662 |
| Classes | 5-grade DR: 0 No DR (1,805), 1 Mild (370), 2 Moderate (999), 3 Severe (193), 4 Proliferative (295) |
| Known issues | Grade 0 (No DR) is ~49% of the set, strong imbalance; single source (Aravind Eye Hospital, India); no official patient ID field published — verify before assuming patient-level grouping is possible; label noise reported in prior literature |
| Used for | Train (primary training set, also used for the internal E1 test split) |

---

## Data card 2

| Field | Value |
|---|---|
| Dataset name | Diabetic Retinopathy Resized (EyePACS / Kaggle DR 2015) |
| Source URL | https://www.kaggle.com/c/diabetic-retinopathy-detection |
| License | Check competition rules on the Kaggle page before redistribution |
| Size (images) | 35,126 |
| Classes | 5-grade DR: 0 No DR (25,810), 1 Mild (2,443), 2 Moderate (5,290), 3 Severe (873), 4 Proliferative (710) |
| Known issues | Majority class (No DR) is ~73.5%, severe imbalance; well-documented label noise in prior literature; filenames encode patient ID and eye (e.g. `12345_left.jpeg`, `12345_right.jpeg`) — use this for patient-level splitting; image quality varies (blur, exposure, compression artifacts) |
| Used for | Train (primary training set, combined with APTOS 2019 for multi-dataset / E3 experiments) |

---

## Data card 3

| Field | Value |
|---|---|
| Dataset name | IDRiD (Indian Diabetic Retinopathy Image Dataset) — labeled "Multimodal Datasets: CFP, OCT, UWF" in local folder, actual modality used is CFP |
| Source URL | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |
| License | IEEE DataPort terms — confirm redistribution/derivative-model terms before public deployment |
| Size (images) | 516 |
| Classes | 5-grade DR: 0 No DR (134), 1 Mild (20), 2 Moderate (136), 3 Severe (74), 4 Proliferative (152); also includes DME grading and pixel-level lesion segmentation masks (not used in this DR-grading summary) |
| Known issues | Small dataset, so per-class metrics (especially Mild, n=20) will have wide confidence intervals; label file is `idrid_labels.csv`; single source (India); less imbalanced than APTOS/EyePACS, useful specifically because its distribution differs from the training sets |
| Used for | External test (E2), and as part of combined training in E3 (train on APTOS + IDRiD, test on Messidor-1) |

---

## Data card 4

| Field | Value |
|---|---|
| Dataset name | Messidor-1 |
| Source URL | https://www.adcis.net/en/third-party/messidor/ |
| License | ADCIS third-party research-use terms — registration required, confirm terms before redistribution |
| Size (images) | 1,200 |
| Classes | 4-level scheme (not the standard 0–4): 0 No DR (546), 1 Mild (153), 2 Moderate+Severe merged (247), 3 Proliferative (254). No separate Grade 4 — map carefully before combining with APTOS/EyePACS/IDRiD labels |
| Known issues | Grading scheme differs from the other three datasets (Moderate and Severe are merged into one class), so label mapping must be explicit and documented, not assumed; images captured at different resolutions, with and without pupil dilation, from 3 ophthalmology departments; no patient ID or laterality field confirmed — check source documentation before splitting |
| Used for | External test only (E2/E3 final held-out set — never used in training) |