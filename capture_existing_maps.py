from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


BASE = Path(__file__).resolve().parent
OUT = BASE / "report_assets_suite"
OUT.mkdir(exist_ok=True)


async def capture(
    page,
    html_path: Path,
    output: Path,
    center: tuple[float, float],
    zoom: int,
    enable_layer_text: str | None = None,
) -> None:
    await page.set_viewport_size({"width": 1280, "height": 760})
    await page.goto(html_path.resolve().as_uri(), wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_load_state("networkidle", timeout=25000)
    except Exception:
        pass
    await page.wait_for_timeout(3500)
    await page.evaluate(
        """
        ({ center, zoom, enableLayerText }) => {
          const maps = Object.values(window).filter(
            value => value && value.setView && value.eachLayer && value.getZoom
          );
          for (const map of maps) {
            map.setView(center, zoom);
          }
          if (enableLayerText) {
            const labels = Array.from(document.querySelectorAll('.leaflet-control-layers-overlays label'));
            const label = labels.find(el => el.innerText && el.innerText.includes(enableLayerText));
            if (label) {
              const input = label.querySelector('input');
              if (input && !input.checked) input.click();
            }
          }
        }
        """,
        {"center": list(center), "zoom": zoom, "enableLayerText": enable_layer_text},
    )
    await page.wait_for_timeout(1500)
    await page.screenshot(path=str(output), full_page=False)


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await capture(
            page,
            BASE / "focused_clustering_output" / "map_focused_general_clusters.html",
            OUT / "map_focused_general_capture.png",
            (48.855, 2.37),
            12,
        )
        await capture(
            page,
            BASE / "focused_clustering_output" / "map_focused_general_clusters.html",
            OUT / "map_focused_segments_capture.png",
            (48.855, 2.37),
            12,
            "Segments ponderes - tous profils",
        )
        await capture(
            page,
            BASE / "focused_clustering_output" / "map_focused_profile_clusters.html",
            OUT / "map_focused_profile_capture.png",
            (48.855, 2.37),
            12,
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
