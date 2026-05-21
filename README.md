# PTIR - Analyse de trajectoires GPS

Ce depot contient le code et les sorties principales de mon projet PTIR sur l'analyse de mobilite humaine a partir de trajectoires GPS.

Le projet construit une chaine complete : partir de donnees GPS brutes, les nettoyer, segmenter les deplacements, puis comparer les trajectoires pour faire apparaitre des axes communs et des differences selon les profils utilisateurs. La pipeline a ete appliquee sur les 2036 fichiers CSV du dataset GPS pour produire un fichier de trajectoires nettoyees, `all_refined_trajectories.csv`, utilise ensuite pour les analyses collectives.

## Ce que le projet fait

1. Nettoyage et segmentation des trajectoires GPS.
2. Projection des trajectoires sur une grille spatiale.
3. Clustering par similarite de Jaccard.
4. Etude de sensibilite du seuil de Jaccard.
5. Extraction de segments ponderes et de corridors communs.
6. Analyse exploratoire par profils utilisateurs.

## Fichiers principaux

| Fichier | Role |
|---|---|
| `src/gps_pipeline.py` | Pipeline initiale : nettoyage, staypoints, segmentation, trimming, sampling. |
| `src/trajectory_clustering.py` | Clustering general des trajectoires par cellules et similarite de Jaccard. |
| `src/analyze_jaccard_threshold.py` | Test de plusieurs seuils de Jaccard entre 0.25 et 0.60. |
| `src/focused_clustering.py` | Analyse focalisee sur des utilisateurs d'une meme zone, clusters, segments et profils. |
| `src/aggregate_common_paths.py` | Agregation des segments ponderes et corridors communs. |
| `src/capture_existing_maps.py` | Genere les captures PNG des cartes generales pour le rapport. |
| `src/capture_profile_map_images.py` | Genere les captures PNG des cartes par profils. |
| `src/visualize_profile_summary.py` | Genere le graphique de repartition des profils utilisateurs. |
| `src/create_sample_data.py` | Cree un petit jeu de donnees de demonstration a partir des sorties locales. |

## Organisation du depot

| Dossier | Contenu |
|---|---|
| `src/` | Scripts Python du projet. |
| `sample_data/` | Sous-jeu de donnees fourni pour relancer une demonstration. |
| `results/report_assets/` | Images finales et captures utilisees dans le rapport. |
| `results/report_assets/profile_maps/` | Captures des cartes par profils. |
| `results/focused_clustering_output/` | Petits CSV de synthese et sorties focalisees utiles. |
| `results/jaccard_threshold_study/` | Graphiques et CSV de l'etude du seuil Jaccard. |
| `docs/references/` | Documents et articles de reference. |

## Resultats a regarder en priorite

Les images utiles pour le rapport sont dans `results/report_assets/`.

| Sortie | Description |
|---|---|
| `results/report_assets/map_focused_general_capture.png` | Carte des clusters generaux sur l'echantillon focalise. |
| `results/report_assets/map_focused_segments_capture.png` | Carte des segments ponderes / corridors communs. |
| `results/report_assets/visualisation_profils_utilisateurs.png` | Repartition des profils dans l'echantillon de 200 utilisateurs. |
| `results/report_assets/profile_maps/age_35_44.png` | Clusters representatifs pour la tranche d'age 35-44 ans. |
| `results/report_assets/profile_maps/mode_bike.png` | Clusters representatifs pour le profil BIKE. |
| `results/report_assets/profile_maps/navigo_yes.png` | Clusters representatifs pour les utilisateurs avec abonnement Navigo. |

Les cartes HTML interactives sont generees dans `results/focused_clustering_output/`, mais elles ne sont pas toutes versionnees car certaines peuvent devenir volumineuses. Les captures PNG sont privilegiees pour la remise.

## Parametres retenus

### Jaccard general

Pour l'analyse generale, j'ai teste plusieurs seuils entre `0.25` et `0.60`, en gardant une contrainte minimale de `5` cellules communes. Le seuil `0.42` a ete retenu comme compromis : il limite la fusion excessive tout en conservant des clusters collectifs interpretables.

### Analyse focalisee et profils

Pour les cartes focalisees et l'analyse par profils, le seuil a ete abaisse a `0.35`. La raison est simple : sur un sous-echantillon, et encore plus apres filtrage par profil, le nombre de trajectoires comparables diminue. Un seuil a `0.42` fragmentait trop les resultats.

L'analyse par profils utilise un echantillon focalise de `200` utilisateurs et affiche jusqu'a `6` clusters representatifs par profil.

## Donnees fournies pour relancer une demonstration

Le dataset complet n'est pas versionne car il est trop lourd. Pour que le projet reste relancable, le depot contient un sous-jeu dans `sample_data/` :

| Fichier | Description |
|---|---|
| `sample_data/focused_demo_refined_trajectories.csv` | Points GPS focalises et sous-echantillonnes pour 200 utilisateurs. |
| `sample_data/selected_same_zone_users.csv` | Liste des 200 utilisateurs de la zone focalisee. |
| `sample_data/individuals_dataset.csv` | Informations individuelles utiles pour les profils. |
| `sample_data/displacements_dataset.csv` | Modes de deplacement dominants pour les profils. |

Ce sous-jeu ne remplace pas le dataset complet, mais il permet au correcteur de relancer la partie clustering / profils sans avoir les 2036 fichiers CSV originaux.

## Donnees et fichiers lourds

Les gros fichiers de donnees ne sont pas destines a etre commit :

- `all_refined_trajectories.csv` fait plusieurs Go ;
- les CSV intermediaires complets peuvent depasser plusieurs dizaines ou centaines de Mo ;
- les donnees sources `NetMob25CleanedData/` restent locales.

Le depot conserve surtout le code, les petits fichiers de synthese, les figures et les captures utiles pour comprendre les resultats.

## Lancer les scripts

Installer les dependances :

```bash
pip install -r requirements.txt
```

Relancer une demonstration avec les donnees incluses dans le depot :

```bash
python src/focused_clustering.py --target-users 200 --reuse-selection --reuse-focused-points --focused-output sample_data/focused_demo_refined_trajectories.csv --selected-users-output sample_data/selected_same_zone_users.csv --individuals-input sample_data/individuals_dataset.csv --displacements-input sample_data/displacements_dataset.csv --jaccard-threshold 0.35 --profile-map-top-clusters 6
```

Relancer l'analyse complete si les gros fichiers locaux sont disponibles :

```bash
python src/focused_clustering.py --target-users 200 --jaccard-threshold 0.35 --profile-map-top-clusters 6
```

Regenerer les captures des cartes generales :

```bash
python src/capture_existing_maps.py
```

Regenerer les captures des cartes par profils :

```bash
python src/capture_profile_map_images.py
```

Regenerer le graphique de repartition des profils :

```bash
python src/visualize_profile_summary.py
```

## References

- Li et al., "A Trajectory Collaboration Based Map Matching Approach for Low-Sampling-Rate GPS Trajectories", Sensors, 2020.  
  https://www.mdpi.com/1424-8220/20/7/2057

- Ville de Paris, "Les Rives de Seine s'offrent a vous".  
  https://www.paris.fr/pages/les-rives-de-seine-s-offrent-a-vous-21328
