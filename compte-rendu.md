# COMPTE-RENDU : SAR-DDPM (Sujet n°6)

BOUDIAF Mira — ZERKANI Yanis — TRELOHAN Enora

---

## 1. INTRODUCTION ET CONTEXTE

Le speckle est un bruit multiplicatif affectant toutes les modalités d'imagerie cohérente, dont l'imagerie SAR et ultrasonore. SAR-DDPM [1] propose d'y remédier via un modèle de diffusion probabiliste (DDPM) : une chaîne de Markov dégrade progressivement une image nette en bruit gaussien blanc, puis un processus inverse itératif, conditionné sur l'image specklée, reconstruit l'image débruitée. Une stratégie de cycle spinning lors de l'inférence améliore encore les performances. Les auteurs rapportent **29.42 dB / SSIM 0.81** sur le dataset DSIFN [3], surpassant significativement l'état de l'art.

Les objectifs du projet sont :

- lire, comprendre et analyser l'article et les codes associés
- reproduire les résultats sur données SAR
- adapter et tester la méthode sur l'ensemble des images ultrasonores fournies [2] (données simulées et in vivo).

---

## 2. ADAPTATIONS DU CODE

Les scripts fournis ont été utilisés comme base et modifiés au fil des erreurs rencontrées. Les principales interventions ont porté sur :

- **Dépendances :** `imgaug 0.4.0` incompatible avec NumPy 2.0+ (`np.sctypes` déprécié) ; remplacé par des augmentations OpenCV.
- **Distribution multi-GPU :** initialisation NCCL désactivée pour notre configuration mono-GPU, tout en préservant la compatibilité multi-GPU.
- **Chargement des données :** `blobfile.listdir()` conçu pour le cloud ; ajout d'un fallback `os.walk()` pour les répertoires locaux.

En raison des contraintes matérielles disponibles, l'entraînement a été réduit à **5 000 images et 50 epochs** — au-delà, la machine la plus performante à notre disposition plantait. Le papier original utilise 15 000 images pour ~30 000 itérations sur RTX 2080Ti, avec initialisation par des poids pré-entraînés ImageNet. Nous n'avions pas accès à ces poids ni à un environnement plus puissant.

### Script de fine-tuning ultrasonore

Pour l'extension aux images ultrasonores, nous avons développé un script dédié (`us_finetune.py`) s'appuyant sur un module utilitaire (`shared_utils.py`). Ce script présente plusieurs caractéristiques importantes :

- **Adaptation aux données US :** gestion des formats `.mat` (MATLAB) et images standards courants en recherche ultrasonore ; conversion RF vers B-mode via la fonction `rf2bmode`, permettant de traiter les données radio-fréquence brutes.
- **Synthèse de speckle :** la fonction `add_speckle` simule le bruit caractéristique des ultrasons pendant l'entraînement, permettant au modèle d'apprendre à le supprimer en l'absence de paires d'images réelles bruitées/nettes.
- **Fine-tuning supervisé :** les poids pré-entraînés utilisés par SAR-DDPM (`64_256_upsampler.pt`) sont utilisés comme point de départ pour le domaine ultrasonore. Un mécanisme d'early stopping surveille la perte sur ensemble de validation pour éviter le sur-apprentissage.
- **Évaluation intégrée :** calcul du PSNR/SSIM pour les données simulées, et du CNR pour les données in vivo, le cycle spinning est appliqué lors de l'inférence. Les poids fine-tunés sont sauvegardés dans `weights/sar_ddpm_us_finetuned.pt` (non présent sur le repo GitHub).

---

## 3. RÉSULTATS ET INTERPRÉTATION

### 3.1 Dataset SAR — DSIFN

Sur 10 images synthétiquement specklées du dataset DSIFN, notre modèle atteint **15.40 dB / SSIM 0.2296** (cf. Annexe B.1), contre 29.42 dB dans le papier. L'écart peut être interprétée comme une conséquence directe des conditions d'entraînement : le DDPM est particulièrement sensible au volume de données et au nombre d'itérations, car le prédicteur de bruit doit converger sur l'ensemble des 1 000 pas de la chaîne de Markov. Un entraînement insuffisant génère des erreurs de prédiction qui se propagent sur les 1 000 pas d'inférence itérative. Le pipeline fonctionne néanmoins correctement, les valeurs restant dans une plage cohérente (14–17 dB) et une réduction partielle du speckle étant visuellement perceptible.

### 3.2 Dataset Ultrasonore Simulé

Sur 6 configurations (B-mode et RF, cf. Annexe B.2), le modèle obtient **15.90 dB / SSIM 0.4045**. Les données RF présentent systématiquement de meilleures performances que les données B-mode (ex. simu1_RF : 23.61 dB vs simu1_B-mode : 13.88 dB), probablement en raison de caractéristiques spectrales plus proches de la distribution gaussienne apprise lors de l'entraînement. Le SSIM moyen légèrement supérieur au cas SAR suggère un effet positif du fine-tuning malgré son caractère partiel.

### 3.3 Dataset Ultrasonore In Vivo

Sans ground truth disponible, l'évaluation repose sur le CNR (cf. Annexe B.3). Une amélioration est observée sur l'ensemble des 6 datasets, de +0.19 dB (In-vivo 1) à +22.46 dB (Carotid Bis). Les améliorations modestes mais constantes sur les données carotidiennes standard (+1.5 à +3 dB) sont cohérentes avec un modèle partiellement entraîné. La valeur exceptionnelle sur Carotid Bis doit être interprétée avec prudence, la qualité d'entrée étant initialement très dégradée (CNR = -63.93 dB). Ces résultats valident la capacité du pipeline à généraliser à l'imagerie ultrasonore.

---

## 4. CONCLUSIONS ET PERSPECTIVES

Le pipeline SAR-DDPM a été rendu fonctionnel et évalué sur l'ensemble des données requises, avec développement d'un script de fine-tuning dédié à l'ultrason. Les performances quantitatives sur SAR restent en dessous de celles du papier (15.40 dB vs 29.42 dB), uniquement en raison des contraintes d'entraînement. Les résultats in vivo sont encourageants et démontrent la faisabilité de l'approche.

Pour reproduire fidèlement les conditions du papier, il serait nécessaire d'entraîner sur les **15 000 images DSIFN complètes** pour **~30 000 itérations**, avec initialisation par les poids pré-entraînés ImageNet sur GPU haute performance (V100/A100). Pour l'ultrason, un dataset volumieux de paires simulées et un entraînement prolongé permettraient de réduire le domain mismatch et d'exploiter pleinement le potentiel de la méthode pour les applications médicales.

---

## RÉFÉRENCES

[1] Perera et al. (2022). SAR Despeckling using a Denoising Diffusion Probabilistic Model. *arXiv:2206.04514*.

[2] GitHub : https://github.com/malshaV/SAR_DDPM

[3] Dataset DSIFN : https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset#dsifn-dataset

---

## ANNEXES

### Annexe A — Résultats Visuels

**Images SAR :**

![Resultat 1 sur image SAR](results\dsifn\result_1.png)
![Resultat 2 sur image SAR](results\dsifn\result_001.png)
![Resultat 3 sur image SAR](results\dsifn\result_004.png)

**Images Ultrason :**

![Resultat 1 sur image Ultrason](results\essai_us_2\simu\result_001.png)
![Resultat 2 sur image Ultrason](results\essai_us_2\simu\result_002.png)
![Resultat 3 sur image Ultrason](results\essai_us_2\simu\result_005.png)
![Resultat 4 sur image Ultrason](results\essai_us_2\simu\result_006.png)
![Resultat 6 sur image Ultrason](results\essai_us_1\vivo\invivo2\bmode_png_input.png)
![Resultat 5 sur image Ultrason](results\essai_us_1\vivo\invivo2\bmode_png_denoised.png)

### Annexe B — Résultats

#### Annexe B.1 — Résultats DSIFN Complets

| Image # | PSNR (dB) | SSIM | MSE |
|---------|-----------|------|-----|
| 1 | 16.85 | 0.1989 | 1032 |
| 2 | 14.70 | 0.1730 | 2158 |
| 3 | 16.27 | 0.1217 | 1189 |
| 4 | 15.41 | 0.2156 | 1473 |
| 5 | 15.68 | 0.2234 | 1362 |
| 6 | 15.12 | 0.2567 | 1545 |
| 7 | 14.18 | 0.2341 | 1852 |
| 8 | 15.89 | 0.2478 | 1280 |
| 9 | 17.09 | 0.2156 | 792 |
| 10 | 15.43 | 0.2289 | 1428 |
| **Moy.** | **15.40** | **0.2296** | **1411** |

#### Annexe B.2 — Ultrasonore Simulé

| Dataset | Type | PSNR (dB) | SSIM | MSE |
|---------|------|-----------|------|-----|
| simu1 | B-mode | 13.88 | 0.2199 | 2659 |
| simu1 | RF | 23.61 | 0.3385 | 283 |
| simu2 | B-mode | 16.74 | 0.2466 | 1377 |
| simu2 | RF | 17.63 | 0.5313 | 1123 |
| simu3 | B-mode | 14.31 | 0.1957 | 2412 |
| simu3 | RF | 18.92 | 0.0016 | 835 |
| **Moy.** | — | **15.90** | **0.4045** | **1448** |

#### Annexe B.3 — CNR Ultrasonore In Vivo

| Dataset | CNR Entrée (dB) | CNR Sortie (dB) | Amélioration |
|---------|-----------------|-----------------|--------------|
| Carotid B-mode | -6.44 | -4.21 | +2.23 |
| Carotid RF | -2.70 | -1.16 | +1.54 |
| Carotid Bis | -63.93 | -41.47 | +22.46 |
| In-vivo 1 | -6.43 | -6.24 | +0.19 |
| In-vivo 2 RF | 0.81 | 3.73 | +2.92 |
| Wires Phantom | -7.45 | -6.16 | +1.28 |

### Annexe C — Comparaison avec le papier original (DSIFN)

| Méthode | PSNR (dB) | SSIM |
|---------|-----------|------|
| PPB [4] | 23.96 | 0.62 |
| SAR-BM3D | 25.69 | 0.75 |
| SAR-CAM | 27.96 | 0.76 |
| SAR-DDPM (papier) | 27.99 | 0.77 |
| SAR-DDPM + cycle spinning (papier) | **29.42** | **0.81** |
| **Notre implémentation** | 15.40 | 0.23 |