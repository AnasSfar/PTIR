
# Traitement de trajectoires GPS

## Lancer le script

**Prérequis**

```bash
pip install -r requirements.txt
```

**Configuration**

Dans `gps_pipeline.py`, modifie les deux lignes suivantes :

```python
DATA_DIR = r"chemin/vers/le/dossier"   # dossier contenant le CSV
CSV_NAME = "12_3332.csv"               # nom du fichier CSV
```

**Lancement**

```bash
python gps_pipeline.py
```

**Sorties générées** (dans le même dossier que le script) :
- `map_12_3332_segmented_pipeline.html` — carte interactive
- `map_12_3332_segmented_pipeline_refined.csv` — trajectoires raffinées

---

## Objectif

L'objectif de ce travail est de transformer des données GPS brutes en trajectoires exploitables afin d'analyser les déplacements (trajets fréquents, portions communes, etc.).

Les données brutes contiennent beaucoup de bruit : arrêts, erreurs GPS, zones propres à chaque individu…
Elles ne peuvent donc pas être utilisées directement.

---

## Pipeline de traitement

Le traitement se fait en 7 étapes.

### 1. Nettoyage des données

* suppression des valeurs manquantes
* suppression des coordonnées invalides
* filtrage des points hors zone d'étude (Île-de-France)

---

### 2. Pré-sampling léger

Avant toute analyse, on fusionne les points consécutifs qui sont à la fois très proches dans l'espace et dans le temps :

* distance ≤ 3 m
* intervalle ≤ 3 secondes

Ces points sont remplacés par un point moyen (latitude, longitude, dernier timestamp du groupe).

Objectif : supprimer les doublons GPS inutiles (GPS stationnaire avec drift) sans altérer les vrais déplacements ni perturber la détection des staypoints.

---

### 3. Détection des arrêts (staypoints)

On détecte les moments où l'utilisateur reste au même endroit :

* durée ≥ 10 minutes
* dans un rayon d'environ 50 m

Ces arrêts correspondent à des lieux significatifs (domicile, travail, pause longue…).

---

### 4. Segmentation en trajets

Les données sont ensuite découpées en plusieurs trajets.

Principe :

* un arrêt (staypoint) marque la fin d'un trajet
* le trajet suivant commence après cet arrêt

On obtient donc une suite de déplacements indépendants.

---

### 5. Suppression des zones locales (trimming)

Pour chaque trajet, on enlève les points proches :

* du point de départ
* du point d'arrivée

avec un rayon d'environ 100 m.

Objectif :

* supprimer les zones spécifiques à chaque individu
* ne garder que les parties réellement utiles du déplacement

---

### 6. Sous-échantillonnage (sampling)

Les données GPS étant très denses, on réduit le nombre de points :

* 1 point sur 5 conservé

Cela permet :

* d'alléger les calculs
* de rendre les trajectoires plus lisibles
* sans perdre leur forme globale

---

## Sorties

### Carte interactive (HTML)

Une carte interactive est générée à la fin du pipeline. (`map_12_3332_segmented_pipeline.html`)

* trajectoire brute (en fond, très transparente)
* trajets segmentés (couleurs différentes)
* points de départ et d'arrivée
* zones locales supprimées
* lieux uniques regroupés (staypoints fusionnés)

### CSV raffiné

Un fichier CSV est également exporté (`map_12_3332_segmented_pipeline_refined.csv`) contenant les points après trimming, avec une colonne `TRAJECTORY_ID` indiquant à quel trajet appartient chaque point.

---

## Exemple de résultat

Sur un fichier (12_3332.csv) :

* ~175 000 points bruts
* → pré-sampling léger (suppression des doublons GPS)
* → détection des staypoints et segmentation en trajets
* → trimming et sampling de chaque trajet
* → carte interactive + CSV raffiné exportés

---

## Remarque

Le choix des paramètres (3 m / 3 s pour le pré-sampling, 10 min / 50 m pour les staypoints, 100 m de trimming, sampling 1/5) est basé sur une analyse préalable des données et constitue un compromis entre :

* suppression du bruit
* conservation des informations utiles

---
