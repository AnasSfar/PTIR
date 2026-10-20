<sub>[🇬🇧 English](README.md) · 🇫🇷 Français</sub>

# PTIR · Analyse de la mobilité humaine à partir de trajectoires GPS

**Quels trajets les habitants de la région parisienne partagent-ils vraiment, et diffèrent-ils selon l'âge, le sexe ou le mode de transport ?**
Ce projet de recherche construit une chaîne complète qui transforme des traces GPS brutes en trajectoires comparables, les regroupe en clusters et en extrait les corridors communs d'une population urbaine.

![Clusters de trajets représentatifs dans la zone focalisée](results/report_assets/map_focused_general_capture.png)

<sub>Clusters représentatifs pour 200 utilisateurs d'une même zone : 4 701 trajets raffinés, 1,8 M de points GPS.</sub>

## Points clés

- **Pipeline de données complet** appliqué aux 2 036 fichiers CSV bruts du jeu de données GPS NetMob25 (Île-de-France) : nettoyage, détection des points d'arrêt, segmentation des déplacements, découpage des extrémités et rééchantillonnage, jusqu'à un fichier de trajectoires raffinées de plusieurs Go
- **Similarité entre trajectoires** en projetant chaque trajet sur une grille spatiale et en comparant les ensembles de cellules avec l'**indice de Jaccard**
- **Choix des paramètres guidé par les données** : une étude de sensibilité sur des seuils de 0,25 à 0,60 pour trouver une valeur qui évite à la fois la fusion excessive et la fragmentation
- **Corridors communs** extraits sous forme de segments pondérés sur une grille plus fine
- **Analyse par profils** : la même méthode appliquée par tranche d'âge, sexe, mode de transport principal, possession d'une voiture et abonnement Navigo
- **Cartes interactives** (Folium / Leaflet) et captures PNG automatiques pour le rapport

**Stack :** Python · pandas · Folium / Leaflet · Matplotlib · Playwright (captures de cartes) · Pillow

## Méthode

![Pipeline](results/report_assets/schema_suite_pipeline.png)

1. **Nettoyer et segmenter** les traces GPS brutes (valeurs aberrantes, points d'arrêt, limites des trajets, découpage des extrémités, échantillonnage).
2. **Projeter** chaque trajet sur une grille spatiale (cellules de 350 m).
3. **Comparer** les trajets avec la similarité de Jaccard de leurs ensembles de cellules.
4. **Regrouper** les trajets similaires et garder une ligne représentative par cluster.
5. **Agréger** des segments pondérés sur une grille de 90 m pour faire ressortir les corridors communs.
6. **Croiser avec les profils** des utilisateurs (âge, sexe, mode, voiture, Navigo).

Les seuils de nettoyage (écarts de temps, distances, vitesses, durées des points d'arrêt) ont été choisis à partir d'une analyse exploratoire de leurs distributions, conservée dans [`analyse&test/`](analyse&test/).

### Choix du seuil de Jaccard

![Sensibilité au seuil de Jaccard](results/jaccard_threshold_study/jaccard_threshold_sensitivity.png)

- **Analyse générale :** seuils testés de `0,25` à `0,60`, avec au moins `5` cellules communes. **`0,42`** a été retenu comme compromis : il limite la fusion excessive tout en produisant des clusters collectifs interprétables.
- **Analyse focalisée et par profils :** seuil abaissé à **`0,35`**. Sur un sous-échantillon, et plus encore après filtrage par profil, il y a moins de trajets comparables, et `0,42` fragmentait trop les résultats.

L'analyse par profils utilise un échantillon focalisé de `200` utilisateurs et affiche jusqu'à `6` clusters représentatifs par profil.

## Résultats

| Corridors communs (segments pondérés) | Répartition des profils dans l'échantillon |
|---|---|
| ![Segments pondérés](results/report_assets/map_focused_segments_capture.png) | ![Profils](results/report_assets/visualisation_profils_utilisateurs.png) |

Clusters représentatifs par profil :

| Cyclistes | Abonnés Navigo | 35–44 ans |
|---|---|---|
| ![Vélo](results/report_assets/profile_maps/mode_bike.png) | ![Navigo](results/report_assets/profile_maps/navigo_yes.png) | ![35-44 ans](results/report_assets/profile_maps/age_35_44.png) |

Toutes les figures du rapport sont dans [`results/report_assets/`](results/report_assets/). Les cartes HTML interactives sont générées dans `results/focused_clustering_output/` ; la plupart ne sont pas versionnées car elles peuvent devenir volumineuses.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| [`src/`](src/) | Scripts Python du pipeline et des analyses |
| [`sample_data/`](sample_data/) | Petit sous-jeu de données pour relancer une démonstration |
| [`results/report_assets/`](results/report_assets/) | Figures finales et captures de cartes utilisées dans le rapport |
| [`results/focused_clustering_output/`](results/focused_clustering_output/) | CSV de synthèse de l'analyse focalisée |
| [`results/jaccard_threshold_study/`](results/jaccard_threshold_study/) | Graphiques et CSV de l'étude du seuil |
| [`analyse&test/`](analyse&test/) | Analyse exploratoire derrière les seuils de nettoyage |
| [`docs/references/`](docs/references/) | Articles de référence |

### Scripts principaux

| Script | Rôle |
|---|---|
| `src/gps_pipeline.py` | Pipeline initial : nettoyage, points d'arrêt, segmentation, découpage, échantillonnage |
| `src/trajectory_clustering.py` | Clustering général des trajectoires par cellules et similarité de Jaccard |
| `src/analyze_jaccard_threshold.py` | Étude de sensibilité du seuil de Jaccard (0,25 à 0,60) |
| `src/focused_clustering.py` | Analyse focalisée sur les utilisateurs d'une zone : clusters, segments et profils |
| `src/aggregate_common_paths.py` | Agrégation des segments pondérés et des corridors communs |
| `src/capture_existing_maps.py` | Captures PNG des cartes générales |
| `src/capture_profile_map_images.py` | Captures PNG des cartes par profils |
| `src/visualize_profile_summary.py` | Graphique de répartition des profils |
| `src/create_sample_data.py` | Construit le sous-jeu de démonstration à partir des sorties locales |

## Lancer le projet

```bash
pip install -r requirements.txt
```

**Démonstration avec les données incluses dans le dépôt :**

```bash
python src/focused_clustering.py --target-users 200 --reuse-selection --reuse-focused-points \
  --focused-output sample_data/focused_demo_refined_trajectories.csv \
  --selected-users-output sample_data/selected_same_zone_users.csv \
  --individuals-input sample_data/individuals_dataset.csv \
  --displacements-input sample_data/displacements_dataset.csv \
  --jaccard-threshold 0.35 --profile-map-top-clusters 6
```

**Analyse complète** (nécessite les gros fichiers locaux) :

```bash
python src/focused_clustering.py --target-users 200 --jaccard-threshold 0.35 --profile-map-top-clusters 6
```

**Régénérer les figures :**

```bash
python src/capture_existing_maps.py         # captures des cartes générales
python src/capture_profile_map_images.py    # captures des cartes par profils
python src/visualize_profile_summary.py     # graphique de répartition des profils
```

### À propos des données

Le jeu de données complet n'est pas versionné : `all_refined_trajectories.csv` pèse plusieurs Go, les CSV intermédiaires atteignent des dizaines à des centaines de Mo, et les fichiers sources (`NetMob25CleanedData/`) restent en local. Pour que le projet reste reproductible, [`sample_data/`](sample_data/) contient :

| Fichier | Description |
|---|---|
| `focused_demo_refined_trajectories.csv` | Points GPS focalisés et sous-échantillonnés pour 200 utilisateurs |
| `selected_same_zone_users.csv` | Les 200 utilisateurs de la zone focalisée |
| `individuals_dataset.csv` | Attributs individuels utilisés pour les profils |
| `displacements_dataset.csv` | Modes de transport dominants utilisés pour les profils |

Ce sous-jeu ne remplace pas le jeu complet, mais il suffit pour relancer le clustering et l'analyse par profils.

## Références

- Li et al., *A Trajectory Collaboration Based Map Matching Approach for Low-Sampling-Rate GPS Trajectories*, Sensors, 2020. [mdpi.com/1424-8220/20/7/2057](https://www.mdpi.com/1424-8220/20/7/2057)
- Ville de Paris, *Les Rives de Seine s'offrent à vous*. [paris.fr](https://www.paris.fr/pages/les-rives-de-seine-s-offrent-a-vous-21328)

---

Projet de recherche (PTIR) à l'INSA Lyon · **Anas Sfar** · [LinkedIn](https://www.linkedin.com/in/anas-sfar-331071370/)
