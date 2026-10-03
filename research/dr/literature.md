# Literature Table: Diabetic Retinopathy Module (SwasthyaAI)

Each row summarises one paper: what it did, what it reported, where it falls short,
and how this project addresses the shortcoming.

**Note on laterality:** laterality (left eye / right eye) is defined at the image
level — each image record carries an explicit field marking which eye it is from.
This field is used to group a patient's two eyes together before any
train/validation/test split, so that both eyes of one patient always fall on the
same side of a split. This is checked for each paper below.

---

## Paper 1

**Kumar G & Shalini K C (2025).** Diabetic Retinopathy Detection using Deep
Learning Techniques. *IJNRD* 10(9).

| | |
|---|---|
| **Dataset** | APTOS 2019 (3,662 images, grades 0–4); DR-Resized (~35k images, likely EyePACS 2015 resized) |
| **Method** | ResNet-50, DenseNet121, EfficientNet-B3 (ImageNet transfer learning); resize 300×300 + ImageNet normalisation; augmentation: rotation, flips, colour jitter, Gaussian blur, affine; sampler + class-weighted loss + label smoothing; AdamW + OneCycleLR |
| **Metrics reported** | Accuracy, weighted precision, recall, F1 (balanced accuracy and macro F1 promised but not reported) |
| **Key result** | APTOS: ResNet-50 86.5% (EffNet-B3 85.5%, DenseNet 81.1%); DR-Resized: EffNet-B3 77.8% (ResNet 73.9%, DenseNet 71.6%) |
| **Limitations found** | Severe DR recall only 26% on APTOS; Mild recall only 15% on DR-Resized (majority class ~71%); overfitting visible in loss curves but denied in text; no cross-dataset test; split method not described (2 eyes per patient, no laterality field used); three imbalance methods stacked with no ablation; minimal preprocessing; Gaussian blur used as augmentation; ordinal grades treated as flat classes (no QWK, CIs, or multiple seeds); text inconsistencies (224 vs 300 px, ResNet described with "compound scaling") |
| **Our fix (SwasthyaAI)** | We will report per-class sensitivity, macro-AUC, QWK, bootstrap CIs, 3+ seeds; we will do a patient-level split using laterality (left/right eye) so both eyes of one patient stay together; we will run an external test on IDRiD and Messidor-2; we will use one imbalance method at a time (ablation), with calibration and ECE reported; we will do a retina crop + Ben Graham + CLAHE at 512 px, and drop blur; we will use an ordinal head (CORAL/CORN) |

---

## Paper 2

**Tang et al. (2026).** A multimodal retinal image dataset for diabetic
retinopathy detection using foundation models (MMRDR). *Scientific Data* 13:639.

| | |
|---|---|
| **Dataset** | MMRDR: 24,460 images total — CFP 11,118 (from OIA-DDR), UWF 10,404, OCT 2,938; 5-grade DR, 7 lesion types, 3-class DME |
| **Method** | Dataset descriptor + benchmark; baselines at 512×512: InternVL3 / HuatuoGPT-Vision (zero-shot and LoRA), ResNet-50, ViT, RETFound, FLAIR, KeepFIT |
| **Metrics reported** | Accuracy and F1 only (DR grade, lesion, DME); no AUC, QWK, per-class results, or CIs |
| **Key result** | CFP DR grade: ResNet-50 0.823, RETFound 0.822; UWF: ResNet-50 0.745 (best); zero-shot LVLMs poor; fine-tuned InternVL3-38B 0.781 (CFP) |
| **Limitations found** | CFP split at image level, not patient level (patient IDs unavailable in source OIA-DDR), so CFP scores are optimistic; CFP is the existing OIA-DDR set, not an independent collection; UWF/OCT are split at patient level and include a laterality field (left/right eye), but CFP does not; modalities are not paired (different patients across CFP/UWF/OCT); single hospital; one OCT slice per eye; labels are mostly single-reader, and inter-rater calibration used only 130 images; aggressive quality filtering removed ~35% of UWF images; baseline comparison is confounded (linear probe vs full fine-tune); the text's claim about foundation-model robustness contradicts Table 3's own AccG numbers |
| **Our fix (SwasthyaAI)** | We will use MMRDR-CFP only as a test-only set, and only if DDR is not already in our training data; we will re-group near-duplicates (pHash) before any re-split; we will add a laterality field to the CFP subset where recoverable so both eyes of a patient stay on the same side of the split; we will use the same training recipe for every backbone; we will build our own quality gate and "ungradable" path; we will report QWK, per-class metrics and CIs |

---

## Paper 3

**Akhtar et al. (2025).** A deep learning based model for diabetic retinopathy
grading (RSG-Net). *Scientific Reports* 15:3763.

| | |
|---|---|
| **Dataset** | Messidor-1 (1,200 images); 4 grades (0–3, Moderate+Severe merged) and a binary task; augmented to 8,304 images (4-class) and 4,800 images (binary) |
| **Method** | Custom CNN from scratch (4 conv layers, 2 pooling layers, dense 128, batch norm, dropout 0.1); crop, 3×3 Gaussian blur, global histogram equalisation, resize to 200×200; augmented copies saved to disk; SGD lr 0.001, 30 epochs |
| **Metrics reported** | Accuracy, sensitivity, specificity, F1, PPV, NPV, AUC, likelihood ratios |
| **Key result** | Test accuracy 99.36% (4-class), 99.37% (binary); AUC ~99.98%; claims to outperform prior work (Table 9) |
| **Limitations found** | Augmentation performed **before** the train/test split — near-copies of one retina end up in both train and test, so the reported accuracy largely measures memorisation, not generalisation (the critical flaw); no patient- or laterality-based split described at all; artificially balanced test set; a very large dense layer relative to ~1,200 real images (training accuracy reaches ~100%); blur, global histogram equalisation and 200 px resolution all damage small lesions; non-standard label scheme; several numbers in the text are inconsistent with the paper's own confusion matrix (10 errors shown vs "5" stated; sensitivity 0.9527 in text vs 0.9941 in Table 8); single small dataset, with no CIs, seeds, or external test |
| **Our fix (SwasthyaAI)** | We will split the original images first, then augment on the fly for the training set only; we will group by patient/laterality where available before splitting; we will keep the natural class distribution in validation/test; we will use a pretrained backbone with global pooling instead of a large dense layer from scratch; we will use CLAHE/Ben Graham at 512 px instead of blur + global HE; we will use the standard 0–4 label scheme; we will use k-fold cross-validation, multiple seeds, and CIs; we will generate all reported numbers directly from saved prediction files |

---

## Research gaps drawn from the three papers

| Common gap | Paper 1 | Paper 2 | Paper 3 |
|---|---|---|---|
| No patient-/laterality-level splitting (both eyes of one patient kept together) | Not described, no laterality field used | CFP split at image level despite a laterality field existing for UWF/OCT | No laterality or patient grouping described at all |
| No external / cross-dataset test | Absent | Internal test split only | Absent |
| Accuracy-centric reporting (no QWK, per-class, CIs) | Yes | Yes | Yes |
| Weak or harmful preprocessing | Resize only; blur augmentation | Fixed 512 px resize | Blur, global HE, 200 px |
| Ordinal structure of grades ignored | Yes | Yes | Yes |
| No image-quality gate | Yes | Manual filtering only | Yes |
| No calibration, uncertainty, or explainability | Yes | Yes | Yes |

---

*Note: figures were checked against each paper's own tables and confusion
matrices. Recall values for Paper 1 and the split ratios for Paper 3 are
calculated from the published matrices, not copied from the text.*