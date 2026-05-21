from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


BASE = Path(__file__).resolve().parents[1]
HTML = BASE / "results" / "focused_clustering_output" / "map_focused_profile_clusters.html"
OUT = BASE / "results" / "report_assets" / "profile_maps"

CAPTURES = [
    ("sex_man", "SEX", "Man", "Profil sexe : hommes"),
    ("sex_woman", "SEX", "Woman", "Profil sexe : femmes"),
    ("mode_walking", "DOMINANT_MODE", "WALKING", "Mode dominant : marche"),
    ("mode_subway", "DOMINANT_MODE", "SUBWAY", "Mode dominant : metro"),
    ("mode_bike", "DOMINANT_MODE", "BIKE", "Mode dominant : velo"),
    ("age_35_44", "AGE_GROUP", "35-44", "Age : 35-44 ans"),
    ("navigo_yes", "NAVIGO_SUB", "Yes", "Abonnement Navigo : oui"),
    ("no_car", "NB_CAR_GROUP", "0 car", "Foyer sans voiture"),
]


async def wait_for_map_ready(page) -> None:
    await page.goto(HTML.resolve().as_uri(), wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:
        pass
    await page.wait_for_selector("#profile-criterion", timeout=30000)
    await page.wait_for_selector("#profile-value", timeout=30000)
    await page.wait_for_timeout(2500)


async def select_profile(page, criterion: str, value: str) -> dict:
    return await page.evaluate(
        """
        async ({ criterion, value }) => {
          function changeSelect(id, selectedValue) {
            const select = document.getElementById(id);
            select.value = selectedValue;
            select.dispatchEvent(new Event("change", { bubbles: true }));
          }

          changeSelect("profile-criterion", criterion);
          await new Promise(resolve => setTimeout(resolve, 250));
          changeSelect("profile-value", value);
          await new Promise(resolve => setTimeout(resolve, 450));

          const maps = Object.values(window).filter(
            item => item && item.setView && item.eachLayer && item.getBounds
          );
          for (const map of maps) {
            map.setView([48.856, 2.365], 13);
          }

          const baseLabels = Array.from(document.querySelectorAll('.leaflet-control-layers-base label'));
          const lightBase = baseLabels.find(el => el.innerText && el.innerText.includes("Fond clair"));
          if (lightBase) {
            const input = lightBase.querySelector("input");
            if (input && !input.checked) input.click();
          }

          return {
            lines: document.getElementById("profile-line-count")?.textContent || "0",
            trips: document.getElementById("profile-trip-count")?.textContent || "0",
            users: document.getElementById("profile-user-count")?.textContent || "0",
          };
        }
        """,
        {"criterion": criterion, "value": value},
    )


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 850}, device_scale_factor=1)
        await wait_for_map_ready(page)

        summary_rows = ["file,criterion,value,title,lines,trips,max_users_in_displayed_cluster"]
        for filename, criterion, value, title in CAPTURES:
            metrics = await select_profile(page, criterion, value)
            await page.wait_for_timeout(1200)
            output = OUT / f"{filename}.png"
            await page.screenshot(path=str(output), full_page=False)
            summary_rows.append(
                f"{output.name},{criterion},{value},{title},{metrics['lines']},{metrics['trips']},{metrics['users']}"
            )
            print(output)

        (OUT / "profile_map_captures_summary.csv").write_text(
            "\n".join(summary_rows) + "\n",
            encoding="utf-8",
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
