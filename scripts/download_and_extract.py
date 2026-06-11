"""
Download mod jars from Modrinth and extract 16x16 item textures.

This script:
1. Looks up each mod on Modrinth by slug
2. Downloads the jar for a target game version
3. Extracts textures/item/*.png (16x16 RGBA)
4. Applies the same filtering as the reference project
5. Outputs npy arrays ready for training

Usage:
    uv run scripts/download_and_extract.py [--game-version 1.20.1] [--out-dir data/extracted]
"""
import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import numpy as np
from PIL import Image

MODRINTH_API = "https://api.modrinth.com/v2"
USER_AGENT = "text2px/1.0 (github.com/LangQi99/text2px)"

MOD_SLUGS = {
    "adastra": "ad-astra",
    "advancedae": "advanced-ae",
    "ae2fabric": "ae2",
    "ae2neoforge": "ae2",
    "aetherredux": "aether-redux",
    "alexscave": "alexs-caves",
    "alexsmobs": "alexs-mobs",
    "alexsmobsdelight": "alexs-mobs-delight",
    "anvilcraft": "anvilcraft",
    "aoa3": "advent-of-ascension-nether-update",
    "apotheosis": "apotheosis",
    "aquaculture": "aquaculture",
    "aquamirae": "aquamirae",
    "arsnoveau": "ars-nouveau",
    "atifacts": "artifacts",
    "atomspheric": "atmospheric",
    "barbequesdelight": "barbecues-delight",
    "biomancy": "biomancy",
    "botania": "botania",
    "cobblemon": "cobblemon",
    "compositematerial": None,
    "confluenceotherworld": None,
    "create": "create",
    "cusine": "cuisine-delight",
    "eeeabsmobs": None,
    "endersdelight": "enders-delight",
    "enigmaticaddons": None,
    "enigmaticlegacy": "enigmatic-legacy",
    "environmental": "environmental",
    "extrabotany": "extrabotany",
    "extradelight": "extra-delight",
    "farmersdelight": "farmers-delight",
    "forbiddenarcanaus": "forbidden-arcanus",
    "goety": None,
    "goetydelight": None,
    "gtceu": "gregtechceu-modern",
    "iceandfire": "ice-and-fire-dragons",
    "immersiveengineering": "immersive-engineering",
    "infernalexpansion": "infernal-expansion",
    "ironspellbook": "irons-spells-n-spellbooks",
    "justdirethings": "just-dire-things",
    "kaleidoscopeend": None,
    "kaleidoscopenether": None,
    "latiaocraft": None,
    "leadersdelight": None,
    "legendarymonsters": None,
    "lenderscataclysm": "the-cataclysm",
    "malum": "malum",
    "manametalmod": None,
    "mekanism": "mekanism",
    "minecraft": None,
    "morecolorful": None,
    "mowziesmobs": "mowzies-mobs",
    "naturalist": "naturalist",
    "oceanicdelight": None,
    "oritech": "oritech",
    "pixelmon": None,
    "powergrid": None,
    "projectE": "projecte",
    "quark": "quark",
    "rankine": "project-rankine",
    "refinedstorage": "refined-storage",
    "relics": "relics-mod",
    "reliquary": "reliquary",
    "simplysword": "simply-swords",
    "smc": None,
    "spartan": None,
    "supplemmentaries": "supplementaries",
    "terramity": None,
    "thermal": "thermal-foundation",
    "theundergarden": "the-undergarden",
    "touhoulittlemaid": "touhou-little-maid",
    "twilightdelight": None,
    "twilightforest": "the-twilight-forest",
}


def modrinth_get(path, params=None):
    url = f"{MODRINTH_API}{path}"
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get_jar_url(slug, game_version):
    params = {"game_versions": json.dumps([game_version]), "limit": "5"}
    versions = modrinth_get(f"/project/{slug}/version", params)
    for v in versions:
        for f in v["files"]:
            if f["filename"].endswith(".jar") and f.get("primary", True):
                return f["url"], f["filename"]
    if versions and versions[0]["files"]:
        f = versions[0]["files"][0]
        return f["url"], f["filename"]
    return None, None


def download_jar(url):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as r:
        return io.BytesIO(r.read())


def remain_english(s):
    s = s.lower().replace("_", " ")
    return "".join(re.findall(r"[a-z ]", s))


def is_grayscale(img):
    if img.mode == "L":
        return True
    rgb = img.convert("RGB")
    for r, g, b in rgb.getdata():
        if r != g or g != b:
            return False
    return True


def extract_textures(jar_bytes, mod_name):
    results = []
    try:
        zf = zipfile.ZipFile(jar_bytes)
    except zipfile.BadZipFile:
        return results

    for name in zf.namelist():
        if not name.endswith(".png"):
            continue
        if "/textures/item/" not in name and "/textures/items/" not in name:
            continue
        try:
            data = zf.read(name)
            img = Image.open(io.BytesIO(data))
        except Exception:
            continue

        if img.size[0] != 16:
            continue
        if img.mode not in ("RGBA", "P"):
            continue

        img_rgba = img.convert("RGBA")

        if img_rgba.size[1] == 16:
            frames = [img_rgba]
        elif img_rgba.size[1] % 16 == 0:
            frames = []
            for i in range(img_rgba.size[1] // 16):
                frames.append(img_rgba.crop((0, i * 16, 16, (i + 1) * 16)))
        else:
            continue

        basename = Path(name).stem
        for idx, frame in enumerate(frames):
            if is_grayscale(frame):
                continue
            arr = np.array(frame, dtype=np.uint8)
            if arr[:, :, 3].sum() == 0:
                continue
            label = remain_english(basename)
            if not label.strip():
                continue
            if idx > 0:
                label = f"{label} {idx}"
            results.append((label, arr))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-version", default="1.20.1")
    parser.add_argument("--out-dir", default="data/extracted")
    parser.add_argument("--mods", nargs="*", default=None, help="Subset of mods to download")
    parser.add_argument("--include-vanilla", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    all_labels = []
    download_log = {}

    if args.include_vanilla:
        print("  Downloading Minecraft vanilla client...")
        try:
            manifest = json.loads(urlopen(
                Request("https://launchermeta.mojang.com/mc/game/version_manifest.json",
                        headers={"User-Agent": USER_AGENT}), timeout=15).read())
            version_url = None
            for v in manifest["versions"]:
                if v["id"] == args.game_version:
                    version_url = v["url"]
                    break
            if not version_url:
                for v in manifest["versions"]:
                    if v["type"] == "release":
                        version_url = v["url"]
                        break
            version_meta = json.loads(urlopen(
                Request(version_url, headers={"User-Agent": USER_AGENT}), timeout=15).read())
            client_url = version_meta["downloads"]["client"]["url"]
            jar_bytes = download_jar(client_url)
            textures = extract_textures(jar_bytes, "minecraft")
            print(f"    -> {len(textures)} textures extracted")
            for label, arr in textures:
                all_labels.append(label)
                all_images.append(arr)
            download_log["minecraft"] = {"url": client_url, "count": len(textures)}
        except Exception as e:
            print(f"  ERROR downloading vanilla: {e}")

    mods_to_process = args.mods if args.mods else [k for k in MOD_SLUGS.keys() if k != "minecraft"]

    for mod_name in mods_to_process:
        slug = MOD_SLUGS.get(mod_name)
        if slug is None:
            print(f"  SKIP {mod_name} (no Modrinth slug)")
            continue

        try:
            url, filename = get_jar_url(slug, args.game_version)
            if not url:
                print(f"  SKIP {mod_name}/{slug} (no jar found for {args.game_version})")
                continue

            print(f"  Downloading {mod_name} ({slug}): {filename}...")
            jar_bytes = download_jar(url)
            textures = extract_textures(jar_bytes, mod_name)
            print(f"    -> {len(textures)} textures extracted")

            for label, arr in textures:
                all_labels.append(label)
                all_images.append(arr)

            download_log[mod_name] = {"slug": slug, "url": url, "filename": filename, "count": len(textures)}

        except Exception as e:
            print(f"  ERROR {mod_name}/{slug}: {e}")
            download_log[mod_name] = {"slug": slug, "error": str(e)}

    if not all_images:
        print("No textures extracted!")
        return

    img_array = np.stack(all_images)
    label_array = np.array(all_labels)
    print(f"\nTotal: {len(all_images)} textures from {len(download_log)} mods")
    print(f"Saving to {out_dir}/")

    np.save(out_dir / f"img16data-{len(all_images)}n.npy", img_array)
    np.save(out_dir / f"img16index-{len(all_images)}n.npy", label_array)

    with open(out_dir / "download_log.json", "w") as f:
        json.dump(download_log, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
