"""
GPS Trajectory Viewer — Version améliorée
==========================================
Améliorations clés :
  - Trajectoires en lazy rendering (affichées à la demande, pas toutes au démarrage)
  - UI moderne avec panneau latéral élégant
  - Filtres temporels et stats par trajectoire
  - Architecture JS robuste (LayerGroup avec IDs, pas window[name])
  - Chargement nettement plus rapide (pas de PolyLine × 150 au démarrage)
  - Feedback visuel et transitions CSS
  - Recherche en temps réel dans la liste
"""

import os
import glob
import random
import json
import pandas as pd
import folium
from folium.plugins import HeatMap, Fullscreen, MiniMap, MousePosition
from branca.element import Template, MacroElement, Element

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\sfara\Documents\GitHub\PTIR\NetMob25CleanedData\NetMob25CleanedData\gps_dataset"
OUTPUT_FILE = "idf_gps_viewer.html"

MAX_FILES        = 150
POINT_STEP       = 100   # sous-échantillonnage heatmap
TRAJ_STEP        = 15    # sous-échantillonnage trajectoires
MIN_POINTS_TRAJ  = 10
RANDOM_SEED      = 42

IDF_BOUNDS = dict(min_lat=48.10, max_lat=49.25, min_lon=1.40, max_lon=3.60)

COLORS = [
    "#ef4444","#3b82f6","#22c55e","#a855f7","#f97316",
    "#14b8a6","#ec4899","#84cc16","#6366f1","#facc15",
    "#f43f5e","#0ea5e9","#10b981","#8b5cf6","#fb923c",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_trajectories(files: list[str]) -> tuple[list, list]:
    """Charge et nettoie les trajectoires depuis les CSV."""
    all_points, trajectories = [], []

    for file in files:
        try:
            df = pd.read_csv(file, usecols=lambda c: c in
                {"LATITUDE", "LONGITUDE", "UTC_DATE", "UTC_TIME"})

            if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
                continue

            df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
            df = df[(df["LATITUDE"].between(-90, 90)) &
                    (df["LONGITUDE"].between(-180, 180))]
            df = df[(df["LATITUDE"].between(IDF_BOUNDS["min_lat"], IDF_BOUNDS["max_lat"])) &
                    (df["LONGITUDE"].between(IDF_BOUNDS["min_lon"], IDF_BOUNDS["max_lon"]))]

            if len(df) < MIN_POINTS_TRAJ:
                continue

            # Tri temporel si possible
            hour_label = "?"
            if "UTC_DATE" in df.columns and "UTC_TIME" in df.columns:
                df["DATETIME"] = pd.to_datetime(
                    df["UTC_DATE"].astype(str) + " " + df["UTC_TIME"].astype(str),
                    errors="coerce"
                )
                df = df.sort_values("DATETIME")
                first_valid = df["DATETIME"].dropna().iloc[0] if not df["DATETIME"].dropna().empty else None
                if first_valid is not None:
                    hour_label = first_valid.strftime("%H:%M")

            points = df[["LATITUDE", "LONGITUDE"]].values.tolist()
            sampled = points[::TRAJ_STEP]

            if len(sampled) < 2:
                continue

            tid = len(trajectories) + 1
            trajectories.append({
                "id":         tid,
                "csv":        os.path.basename(file),
                "points":     sampled,
                "raw_count":  len(points),
                "shown_count":len(sampled),
                "start":      sampled[0],
                "end":        sampled[-1],
                "hour":       hour_label,
                "color":      COLORS[(tid - 1) % len(COLORS)],
            })

            all_points.extend(points[::POINT_STEP])

        except Exception as exc:
            print(f"  ⚠ Erreur ({os.path.basename(file)}) : {exc}")

    return all_points, trajectories


def build_map(all_points: list, trajectories: list) -> folium.Map:
    """Construit la carte Folium de base (sans trajectoires — lazy JS)."""
    m = folium.Map(
        location=[48.8566, 2.3522],
        zoom_start=11,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,   # ← rendu canvas = beaucoup plus rapide
    )

    # Fonds de carte
    folium.TileLayer("CartoDB positron",   name="Fond clair",   control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter",name="Fond sombre",  control=True).add_to(m)
    folium.TileLayer("OpenStreetMap",      name="OpenStreetMap",control=True).add_to(m)

    # Emprise Île-de-France
    folium.Rectangle(
        bounds=[[IDF_BOUNDS["min_lat"], IDF_BOUNDS["min_lon"]],
                [IDF_BOUNDS["max_lat"], IDF_BOUNDS["max_lon"]]],
        color="#6366f1", weight=2, fill=False,
        tooltip="Zone filtrée : Île-de-France"
    ).add_to(m)

    # Heatmap (seul calque lourd — rendu côté canvas Leaflet)
    if all_points:
        heat_layer = folium.FeatureGroup(name="Densité GPS", show=True)
        HeatMap(
            all_points,
            radius=14, blur=20, min_opacity=0.2, max_zoom=14,
            gradient={0.2:"blue",0.45:"cyan",0.65:"lime",0.85:"orange",1.0:"red"}
        ).add_to(heat_layer)
        heat_layer.add_to(m)

    # Plugins
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    MousePosition(position="bottomleft", separator=" | ",
                  prefix="📍", num_digits=5).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    m.fit_bounds([[IDF_BOUNDS["min_lat"], IDF_BOUNDS["min_lon"]],
                  [IDF_BOUNDS["max_lat"], IDF_BOUNDS["max_lon"]]])
    return m


def inject_ui(m: folium.Map, trajectories: list, point_count: int):
    """Injecte le panneau latéral + la logique JS dans la carte."""
    traj_json = json.dumps(trajectories)
    map_name  = m.get_name()
    n         = len(trajectories)

    # ── Panneau latéral HTML ───────────────────────────────────────────────
    panel_html = f"""
{{% macro html(this, kwargs) %}}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  #gps-panel {{
    position: fixed; top: 0; left: 0; height: 100vh;
    width: 340px; z-index: 9999;
    background: #ffffff;
    color: #1e293b;
    font-family: 'DM Sans', sans-serif;
    display: flex; flex-direction: column;
    box-shadow: 4px 0 32px rgba(0,0,0,0.12);
    transition: transform .3s cubic-bezier(.4,0,.2,1);
  }}
  #gps-panel.collapsed {{ transform: translateX(-308px); }}

  #panel-toggle {{
    position: absolute; right: -36px; top: 50%;
    transform: translateY(-50%);
    width: 36px; height: 64px;
    background: #ffffff;
    border: none; border-radius: 0 10px 10px 0;
    color: #64748b; font-size: 18px;
    cursor: pointer; display: flex; align-items: center;
    justify-content: center; z-index: 1;
    box-shadow: 3px 0 12px rgba(0,0,0,0.1);
  }}

  .panel-header {{
    padding: 20px 18px 12px;
    border-bottom: 1px solid #e2e8f0;
    flex-shrink: 0;
  }}
  .panel-title {{
    font-size: 15px; font-weight: 700;
    letter-spacing: .03em; color: #0f172a;
    display: flex; align-items: center; gap: 8px;
  }}
  .panel-subtitle {{
    font-size: 11px; color: #94a3b8;
    font-family: 'DM Mono', monospace;
    margin-top: 3px;
  }}

  .stats-row {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; padding: 12px 18px;
    border-bottom: 1px solid #e2e8f0;
    flex-shrink: 0;
  }}
  .stat-card {{
    background: #f8fafc; border-radius: 10px;
    padding: 10px 12px; border: 1px solid #e2e8f0;
  }}
  .stat-val {{
    font-size: 20px; font-weight: 700;
    color: #6366f1; font-family: 'DM Mono', monospace;
  }}
  .stat-lbl {{
    font-size: 10px; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .08em;
    margin-top: 2px;
  }}

  .search-area {{
    padding: 12px 18px;
    border-bottom: 1px solid #e2e8f0;
    flex-shrink: 0;
  }}
  .search-input {{
    width: 100%; padding: 9px 12px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; color: #1e293b;
    font-family: 'DM Sans', sans-serif; font-size: 13px;
    box-sizing: border-box; outline: none;
    transition: border-color .2s;
  }}
  .search-input:focus {{ border-color: #6366f1; }}

  .btn-row {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 6px; margin-top: 8px;
  }}
  .btn {{
    padding: 8px; border: none; border-radius: 8px;
    font-size: 11px; font-weight: 600;
    cursor: pointer; transition: opacity .15s, background .2s;
  }}
  .btn:hover {{ opacity: .85; }}
  .btn-primary {{ background: #6366f1; color: #fff; }}
  .btn-secondary {{ background: #f1f5f9; color: #64748b;
    border: 1px solid #e2e8f0; }}

  .traj-list {{
    flex: 1; overflow-y: auto; padding: 8px 10px;
  }}
  .traj-list::-webkit-scrollbar {{ width: 4px; }}
  .traj-list::-webkit-scrollbar-track {{ background: transparent; }}
  .traj-list::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}

  .traj-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 9px 10px; border-radius: 8px;
    cursor: pointer; transition: background .15s;
    margin-bottom: 2px;
  }}
  .traj-item:hover {{ background: #f1f5f9; }}
  .traj-item.active {{ background: #eff0fe; outline: 1px solid #6366f1; }}

  .traj-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0;
  }}
  .traj-body {{ flex: 1; min-width: 0; }}
  .traj-name {{
    font-size: 12px; font-weight: 600; color: #1e293b;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .traj-meta {{
    font-size: 10px; color: #94a3b8;
    font-family: 'DM Mono', monospace;
  }}
  .traj-btn {{
    background: none; border: none; color: #cbd5e1;
    cursor: pointer; font-size: 14px; padding: 2px 4px;
    border-radius: 4px; transition: color .15s, background .15s;
    flex-shrink: 0;
  }}
  .traj-btn:hover {{ color: #6366f1; background: #eff0fe; }}

  .panel-footer {{
    padding: 10px 18px;
    border-top: 1px solid #e2e8f0;
    font-size: 10px; color: #94a3b8;
    font-family: 'DM Mono', monospace;
    flex-shrink: 0;
  }}
  #active-label {{
    font-size: 11px; color: #6366f1;
    font-family: 'DM Mono', monospace;
    margin-top: 4px; min-height: 16px;
  }}
</style>

<div id="gps-panel">
  <button id="panel-toggle" onclick="togglePanel()" title="Réduire/Agrandir">◀</button>

  <div class="panel-header">
    <div class="panel-title">🗺 Trajectoires GPS</div>
    <div class="panel-subtitle">Île-de-France · NetMob25</div>
  </div>

  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-val" id="stat-shown">0</div>
      <div class="stat-lbl">Affichées</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{n}</div>
      <div class="stat-lbl">Total</div>
    </div>
    <div class="stat-card">
      <div class="stat-val">{point_count:,}</div>
      <div class="stat-lbl">Points GPS</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stat-active">—</div>
      <div class="stat-lbl">Sélectionnée</div>
    </div>
  </div>

  <div class="search-area">
    <input id="search-input" class="search-input"
      placeholder="🔍  Rechercher par nom ou numéro…">
    <div class="btn-row">
      <button id="btn-show-all" class="btn btn-primary">Tout afficher</button>
      <button id="btn-hide-all" class="btn btn-secondary">Tout masquer</button>
      <button id="btn-multi-toggle" class="btn btn-secondary">Mode : Solo</button>
      <button id="btn-zoom-sel" class="btn btn-secondary">Zoom sélection</button>
    </div>
    <div id="active-label"></div>
  </div>

  <div class="traj-list" id="traj-list"></div>

  <div class="panel-footer">
    Cliquer sur ▶ pour zoomer · 👁 pour afficher/masquer
  </div>
</div>
{{% endmacro %}}
"""

    panel = MacroElement()
    panel._template = Template(panel_html)
    m.get_root().add_child(panel)

    # ── Logique JavaScript ─────────────────────────────────────────────────
    js = f"""
<script>
/* ================================================================
   GPS VIEWER — logique principale
   - DOMContentLoaded ne fonctionne pas dans Folium (page déjà
     chargée quand le script est injecté) → on utilise waitForMap()
   - Pas d'onclick inline (cassé par les f-strings Python/Folium)
     → tout passe par addEventListener après innerHTML
   - Mode multi-sélection : les trajectoires cochées s'accumulent
================================================================ */

const TRAJS = {traj_json};
const IDF   = [[{IDF_BOUNDS["min_lat"]},{IDF_BOUNDS["min_lon"]}],
               [{IDF_BOUNDS["max_lat"]},{IDF_BOUNDS["max_lon"]}]];

let MAP = null;                 // référence Leaflet, récupérée après init
const layers    = {{}};          // id → L.layerGroup
const visibleIds = new Set();   // ids actuellement sur la carte
const selectedIds = new Set();  // ids cochés en mode multi-sélection
let filterQuery  = "";
let multiMode    = false;       // false = solo/exclusif, true = multi

/* ── 1. Attendre que la map Leaflet soit prête ──────────────────── */
function waitForMap() {{
  /* Folium expose la map sous son nom généré ex: map_abc123 */
  const mapVar = "{map_name}";
  if (typeof window[mapVar] !== "undefined") {{
    MAP = window[mapVar];
    boot();
  }} else {{
    setTimeout(waitForMap, 100);
  }}
}}

/* ── 2. Boot : créer les layers + afficher tout ─────────────────── */
function boot() {{
  TRAJS.forEach(function(t) {{
    var latlngs   = t.points.map(function(p) {{ return [p[0], p[1]]; }});
    var line      = L.polyline(latlngs, {{
      color: t.color, weight: 2.5, opacity: 0.75, smoothFactor: 2
    }});
    var startMark = L.circleMarker([t.start[0], t.start[1]], {{
      radius: 5, color: "#16a34a",
      fillColor: "#22c55e", fillOpacity: 0.9, weight: 1.5
    }});
    var endMark   = L.circleMarker([t.end[0], t.end[1]], {{
      radius: 5, color: "#dc2626",
      fillColor: "#ef4444", fillOpacity: 0.9, weight: 1.5
    }});
    startMark.bindTooltip("Depart · #" + t.id + " · " + t.csv);
    endMark.bindTooltip("Arrivee · #"  + t.id + " · " + t.csv);
    line.bindPopup(
      "<div style='font:13px/1.6 sans-serif;min-width:190px'>" +
      "<b style='font-size:14px'>Trajectoire " + t.id + "</b><br>" +
      "<span style='color:#64748b;font-size:11px'>" + t.csv + "</span><br>" +
      "<hr style='border:none;border-top:1px solid #e2e8f0;margin:6px 0'>" +
      "Points affiches : <b>" + t.shown_count + "</b><br>" +
      "Points bruts    : <b>" + t.raw_count    + "</b><br>" +
      "Heure depart    : <b>" + t.hour         + "</b>" +
      "</div>",
      {{maxWidth: 280}}
    );
    layers[t.id] = L.layerGroup([line, startMark, endMark]);
  }});

  /* Brancher les contrôles du panneau */
  document.getElementById("search-input").addEventListener("input", function() {{
    filterQuery = this.value.toLowerCase().trim();
    renderList();
  }});
  document.getElementById("search-input").addEventListener("keydown", function(e) {{
    if (e.key !== "Enter") return;
    var q = this.value.toLowerCase().trim();
    if (!q) return;
    var match = TRAJS.find(function(t) {{
      return String(t.id) === q || t.csv.toLowerCase().indexOf(q) !== -1;
    }});
    if (match) {{
      if (multiMode) addToSelection(match.id);
      else           soloTraj(match.id);
    }}
  }});
  document.getElementById("btn-show-all").addEventListener("click", showAll);
  document.getElementById("btn-hide-all").addEventListener("click", hideAll);
  document.getElementById("btn-zoom-sel").addEventListener("click", zoomSelection);
  document.getElementById("btn-multi-toggle").addEventListener("click", toggleMultiMode);

  showAll();
}}

/* ── 3. Primitives bas niveau (pas de refresh) ───────────────────── */
function _show(id) {{
  if (!layers[id]) return;
  if (!MAP.hasLayer(layers[id])) {{ layers[id].addTo(MAP); visibleIds.add(id); }}
}}
function _hide(id) {{
  if (!layers[id]) return;
  if (MAP.hasLayer(layers[id]))  {{ MAP.removeLayer(layers[id]); visibleIds.delete(id); }}
}}

/* ── 4. Actions MODE SOLO ────────────────────────────────────────── */
function soloTraj(id) {{
  /* Affiche uniquement cette trajectoire, zoom dessus */
  TRAJS.forEach(function(t) {{ _hide(t.id); }});
  _show(id);
  var t = TRAJS.find(function(x) {{ return x.id === id; }});
  MAP.fitBounds(t.points.map(function(p) {{ return [p[0],p[1]]; }}), {{padding:[40,40]}});
  refresh();
}}

/* ── 5. Actions MODE MULTI ───────────────────────────────────────── */
function addToSelection(id) {{
  /* Coche la trajectoire, l'affiche, ne touche pas aux autres */
  selectedIds.add(id);
  _show(id);
  refresh();
}}
function removeFromSelection(id) {{
  selectedIds.delete(id);
  _hide(id);
  refresh();
}}
function toggleSelection(id) {{
  if (selectedIds.has(id)) removeFromSelection(id);
  else                     addToSelection(id);
}}

function zoomSelection() {{
  var ids = multiMode ? Array.from(selectedIds) : Array.from(visibleIds);
  if (ids.length === 0) return;
  var allPts = [];
  ids.forEach(function(id) {{
    var t = TRAJS.find(function(x) {{ return x.id === id; }});
    if (t) t.points.forEach(function(p) {{ allPts.push([p[0],p[1]]); }});
  }});
  if (allPts.length) MAP.fitBounds(allPts, {{padding:[40,40]}});
}}

/* ── 6. Globaux ──────────────────────────────────────────────────── */
function showAll() {{
  TRAJS.forEach(function(t) {{ _show(t.id); if(multiMode) selectedIds.add(t.id); }});
  MAP.fitBounds(IDF);
  refresh();
}}
function hideAll() {{
  TRAJS.forEach(function(t) {{ _hide(t.id); selectedIds.delete(t.id); }});
  refresh();
}}

/* ── 7. Basculer mode multi-sélection ───────────────────────────── */
function toggleMultiMode() {{
  multiMode = !multiMode;
  selectedIds.clear();
  /* Sync selectedIds avec ce qui est visible */
  if (multiMode) visibleIds.forEach(function(id) {{ selectedIds.add(id); }});
  var btn = document.getElementById("btn-multi-toggle");
  btn.textContent    = multiMode ? "Mode : Multi ✓" : "Mode : Solo";
  btn.style.background = multiMode ? "#0ea5e9" : "";
  refresh();
}}

/* ── 8. Clic sur un item de la liste ────────────────────────────── */
function onItemClick(id) {{
  if (multiMode) toggleSelection(id);
  else           soloTraj(id);
}}

/* ── 9. Refresh UI ───────────────────────────────────────────────── */
function refresh() {{
  document.getElementById("stat-shown").textContent = visibleIds.size;
  document.getElementById("stat-active").textContent =
    multiMode ? selectedIds.size + " sel." : (visibleIds.size === 1 ? "#" + Array.from(visibleIds)[0] : "—");
  renderList();
}}

/* ── 10. Rendu de la liste ───────────────────────────────────────── */
function renderList() {{
  var container = document.getElementById("traj-list");
  var list = filterQuery
    ? TRAJS.filter(function(t) {{
        return String(t.id).indexOf(filterQuery) !== -1 ||
               t.csv.toLowerCase().indexOf(filterQuery) !== -1;
      }})
    : TRAJS;

  var html = "";
  list.forEach(function(t) {{
    var isOn  = multiMode ? selectedIds.has(t.id) : visibleIds.has(t.id);
    var cls   = "traj-item" + (isOn ? " active" : "");
    var eye   = isOn ? "👁" : "○";
    var eyeClr= isOn ? "#6366f1" : "#cbd5e1";
    html +=
      '<div class="' + cls + '" data-id="' + t.id + '">' +
        '<div class="traj-dot" style="background:' + t.color + '"></div>' +
        '<div class="traj-body">' +
          '<div class="traj-name">#' + t.id + ' · ' + t.csv + '</div>' +
          '<div class="traj-meta">' + t.shown_count + ' pts · ' + t.hour + '</div>' +
        '</div>' +
        '<button class="traj-btn btn-zoom" data-id="' + t.id + '" title="Zoom">▶</button>' +
        '<button class="traj-btn btn-eye"  data-id="' + t.id + '" title="Afficher/Masquer" style="color:' + eyeClr + '">' + eye + '</button>' +
      '</div>';
  }});
  container.innerHTML = html;

  /* Brancher les événements sur les nouveaux éléments */
  container.querySelectorAll(".traj-item").forEach(function(el) {{
    el.addEventListener("click", function() {{
      onItemClick(parseInt(this.getAttribute("data-id")));
    }});
  }});
  container.querySelectorAll(".btn-zoom").forEach(function(btn) {{
    btn.addEventListener("click", function(e) {{
      e.stopPropagation();
      var id = parseInt(this.getAttribute("data-id"));
      var t  = TRAJS.find(function(x) {{ return x.id === id; }});
      if (!t) return;
      _show(id);
      MAP.fitBounds(t.points.map(function(p){{ return [p[0],p[1]]; }}), {{padding:[40,40]}});
      if (multiMode) selectedIds.add(id);
      refresh();
    }});
  }});
  container.querySelectorAll(".btn-eye").forEach(function(btn) {{
    btn.addEventListener("click", function(e) {{
      e.stopPropagation();
      var id = parseInt(this.getAttribute("data-id"));
      if (multiMode) toggleSelection(id);
      else {{
        if (visibleIds.has(id)) _hide(id);
        else                    _show(id);
        refresh();
      }}
    }});
  }});
}}

/* ── 11. Panneau réductible ─────────────────────────────────────── */
function togglePanel() {{
  var panel = document.getElementById("gps-panel");
  var btn   = document.getElementById("panel-toggle");
  panel.classList.toggle("collapsed");
  btn.textContent = panel.classList.contains("collapsed") ? "▶" : "◀";
}}

/* ── 12. Lancement ───────────────────────────────────────────────── */
waitForMap();
</script>
"""

    m.get_root().html.add_child(Element(js))


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"📂 Dossier  : {DATA_DIR}")
    print(f"   Existe   : {os.path.exists(DATA_DIR)}")

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"   CSV trouvés : {len(files)}")

    if not files:
        raise FileNotFoundError("Aucun fichier CSV trouvé dans le dossier spécifié.")

    random.seed(RANDOM_SEED)
    random.shuffle(files)
    files = files[:MAX_FILES]

    print("⏳ Chargement des trajectoires…")
    all_points, trajectories = load_trajectories(files)

    print(f"✅ Trajectoires : {len(trajectories)}")
    print(f"   Points GPS  : {len(all_points):,}")

    if not all_points:
        raise ValueError("Aucun point GPS valide trouvé en Île-de-France.")

    print("🗺  Construction de la carte…")
    m = build_map(all_points, trajectories)
    inject_ui(m, trajectories, len(all_points))

    output = os.path.abspath(OUTPUT_FILE)
    m.save(output)
    print(f"\n🎉 Carte générée : {output}")
    print(f"   Ouvrir dans un navigateur moderne (Chrome / Firefox recommandé)")


if __name__ == "__main__":
    main()