# Progress Log

## Vacances 
- Choix d'article ASDS (Fast And Scalable Big Data Trajectory)
- Lecture d'article

## 2026-04-20
- Synthèse ASDS en 180 secondes

## 2026-04-23

### Exploration du dataset
- Analyse de la structure du dataset NetMob
- Identification des fichiers principaux :
  - `gps_dataset/` → trajectoires GPS (données principales)
  - `displacements_dataset.csv` → informations contextuelles (O/D, temps, mode)
- Compréhension des colonnes :
  - GPS : LATITUDE, LONGITUDE, TIME
  - déplacements : Origin (O), Destination (D)

---

### Compréhension du problème
- Objectif identifié :
  → clusteriser des trajectoires pour extraire des patterns de mobilité
- Différence clarifiée :
  - clustering → structure des déplacements
  - dataset déplacements → interprétation des clusters
- Lecture et compréhension de l’article :
  - notion de trajectoire
  - distance trajDTW
  - problème de scalabilité

---

### Mise en place du pipeline initial
Implémentation d’un premier pipeline fonctionnel :

1. Chargement des trajectoires GPS
2. Construction des trajectoires (liste de points)
3. Filtrage des trajets trop courts
4. Sampling (réduction du dataset)
5. Calcul des distances avec DTW
6. Clustering avec DBSCAN
7. Visualisation des trajectoires

---

### Implémentation technique

#### Chargement des données
- Lecture des fichiers CSV du dossier `gps_dataset`
- Extraction des colonnes :
  - LATITUDE
  - LONGITUDE
- Transformation en trajectoires exploitables

---

#### Distance entre trajectoires
- Implémentation de DTW (Dynamic Time Warping)
- Utilisation d’une distance euclidienne simple entre points
- Compréhension :
  - permet d’aligner des trajectoires de longueurs différentes

---

#### Clustering
- Utilisation de DBSCAN avec matrice de distance pré-calculée
- Paramètres utilisés :
  - eps = 0.01
  - min_samples = 4
- Identification :
  - clusters
  - bruit (label = -1)

---

### Limitations identifiées
- DTW très coûteux (O(n²))
- nécessité de faire du sampling (200 trajectoires)
- DBSCAN sensible au paramètre `eps`
- distance simplifiée (pas encore trajDTW complet)

---

### Résultat actuel
- Pipeline fonctionnel de clustering de trajectoires
- Visualisation des clusters obtenus
- Sauvegarde des résultats (`clustering_results.csv`)

---

## Prochaines étapes

- Ajustement des paramètres DBSCAN
- Amélioration de la distance (vers trajDTW)
- Analyse des clusters avec O/D (displacements_dataset)
- Ajout de dimension temporelle
- Optimisation performance (Fast-clusiVAT)

---