
# Traitement de trajectoires GPS

## Objectif

L’objectif de ce travail est de transformer des données GPS brutes en trajectoires exploitables afin d’analyser les déplacements (trajets fréquents, portions communes, etc.).

Les données brutes contiennent beaucoup de bruit : arrêts, erreurs GPS, zones propres à chaque individu…
Elles ne peuvent donc pas être utilisées directement.

---

## Pipeline de traitement

Le traitement se fait en plusieurs étapes.

### 1. Nettoyage des données

* suppression des valeurs manquantes
* suppression des coordonnées invalides
* filtrage des points hors zone d’étude (Île-de-France)

---

### 2. Détection des arrêts (staypoints)

On détecte les moments où l’utilisateur reste au même endroit :

* durée ≥ 10 minutes
* dans un rayon d’environ 50 m

Ces arrêts correspondent à des lieux significatifs (domicile, travail, pause longue…).

---

### 3. Segmentation en trajets

Les données sont ensuite découpées en plusieurs trajets.

Principe :

* un arrêt (staypoint) marque la fin d’un trajet
* le trajet suivant commence après cet arrêt

On obtient donc une suite de déplacements indépendants.

---

### 4. Suppression des zones locales (trimming)

Pour chaque trajet, on enlève les points proches :

* du point de départ
* du point d’arrivée

avec un rayon d’environ 100 m.

Objectif :

* supprimer les zones spécifiques à chaque individu
* ne garder que les parties réellement utiles du déplacement

---

### 5. Sous-échantillonnage (sampling)

Les données GPS étant très denses, on réduit le nombre de points :

* 1 point sur 5 conservé

Cela permet :

* d’alléger les calculs
* de rendre les trajectoires plus lisibles
* sans perdre leur forme globale

---

## Exemple de résultat

Sur un fichier (12_3332.csv):

* ~175 000 points bruts
* → 85 trajets détectés
* chaque trajet est ensuite nettoyé, filtré et simplifié

Le résultat est une carte contenant plusieurs trajets distincts, chacun correspondant à un déplacement réel.

---

## Visualisation

Une carte interactive est générée à la fin du pipeline. (map_12_3332_segmented_pipeline.html)

* trajectoire brute (en fond, très transparente)
* trajets segmentés (couleurs différentes)
* points de départ et d’arrivée
* zones locales supprimées

---

## Remarque

Le choix des paramètres (10 min, 50 m, 100 m, sampling) est basé sur une analyse préalable des données et constitue un compromis entre :

* suppression du bruit
* conservation des informations utiles

---