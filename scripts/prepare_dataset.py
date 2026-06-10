"""
Prepare Minecraft item texture dataset.
Downloads or copies Minecraft item textures and creates labels.json.
Expects a Minecraft .jar or extracted assets folder.

Usage:
  python prepare_dataset.py <minecraft_jar_or_assets_dir> [output_dir]
"""
import os
import sys
import json
import zipfile
import shutil
from pathlib import Path


ITEM_DESCRIPTIONS = {
    "acacia_boat": "acacia wood boat",
    "amethyst_shard": "purple amethyst shard crystal",
    "apple": "red apple fruit",
    "armor_stand": "wooden armor stand",
    "arrow": "pointed arrow projectile",
    "axe": "iron axe tool",
    "baked_potato": "baked potato food",
    "bamboo": "green bamboo stick",
    "beef": "raw beef meat",
    "beetroot": "red beetroot vegetable",
    "beetroot_seeds": "beetroot seeds",
    "birch_boat": "birch wood boat",
    "blaze_powder": "orange blaze powder",
    "blaze_rod": "yellow blaze rod",
    "bone": "white bone",
    "bone_meal": "white bone meal powder",
    "book": "brown leather book",
    "bow": "wooden bow weapon",
    "bowl": "wooden bowl",
    "bread": "brown bread loaf food",
    "brick": "red clay brick",
    "bucket": "iron bucket",
    "cake": "birthday cake with frosting",
    "carrot": "orange carrot vegetable",
    "charcoal": "black charcoal piece",
    "chest_minecart": "minecart with chest",
    "chicken": "raw chicken meat",
    "clay_ball": "gray clay ball",
    "clock": "golden clock timepiece",
    "coal": "black coal piece",
    "cocoa_beans": "brown cocoa beans",
    "cod": "raw cod fish",
    "compass": "iron compass navigation",
    "cooked_beef": "cooked steak meat food",
    "cooked_chicken": "cooked chicken meat food",
    "cooked_cod": "cooked cod fish food",
    "cooked_mutton": "cooked mutton meat food",
    "cooked_porkchop": "cooked porkchop meat food",
    "cooked_rabbit": "cooked rabbit meat food",
    "cooked_salmon": "cooked salmon fish food",
    "cookie": "chocolate chip cookie",
    "copper_ingot": "orange copper ingot",
    "crossbow": "wooden crossbow weapon",
    "dark_oak_boat": "dark oak wood boat",
    "diamond": "blue diamond gem",
    "diamond_axe": "diamond axe tool",
    "diamond_boots": "diamond boots armor",
    "diamond_chestplate": "diamond chestplate armor",
    "diamond_helmet": "diamond helmet armor",
    "diamond_hoe": "diamond hoe tool",
    "diamond_horse_armor": "diamond horse armor",
    "diamond_leggings": "diamond leggings armor",
    "diamond_pickaxe": "diamond pickaxe tool",
    "diamond_shovel": "diamond shovel tool",
    "diamond_sword": "blue diamond sword weapon",
    "dragon_breath": "purple dragon breath bottle",
    "dried_kelp": "dark green dried kelp food",
    "egg": "white egg",
    "emerald": "green emerald gem",
    "enchanted_book": "glowing enchanted book",
    "ender_eye": "green ender eye orb",
    "ender_pearl": "green ender pearl orb",
    "experience_bottle": "green experience bottle",
    "feather": "white feather",
    "fermented_spider_eye": "red fermented spider eye",
    "fire_charge": "orange fire charge ball",
    "firework_rocket": "firework rocket",
    "fishing_rod": "wooden fishing rod",
    "flint": "gray flint stone",
    "flint_and_steel": "flint and steel lighter",
    "ghast_tear": "white ghast tear drop",
    "glass_bottle": "empty glass bottle",
    "glistering_melon_slice": "golden melon slice",
    "glowstone_dust": "yellow glowstone dust",
    "gold_ingot": "shiny gold ingot bar",
    "gold_nugget": "small gold nugget",
    "golden_apple": "golden apple magical food",
    "golden_axe": "golden axe tool",
    "golden_boots": "golden boots armor",
    "golden_carrot": "golden carrot food",
    "golden_chestplate": "golden chestplate armor",
    "golden_helmet": "golden helmet armor",
    "golden_hoe": "golden hoe tool",
    "golden_horse_armor": "golden horse armor",
    "golden_leggings": "golden leggings armor",
    "golden_pickaxe": "golden pickaxe tool",
    "golden_shovel": "golden shovel tool",
    "golden_sword": "golden sword weapon",
    "gunpowder": "gray gunpowder",
    "heart_of_the_sea": "blue heart of the sea crystal",
    "honeycomb": "yellow honeycomb",
    "ink_sac": "black ink sac",
    "iron_axe": "iron axe tool",
    "iron_boots": "iron boots armor",
    "iron_chestplate": "iron chestplate armor",
    "iron_helmet": "iron helmet armor",
    "iron_hoe": "iron hoe tool",
    "iron_horse_armor": "iron horse armor",
    "iron_ingot": "silver iron ingot bar",
    "iron_leggings": "iron leggings armor",
    "iron_nugget": "small iron nugget",
    "iron_pickaxe": "iron pickaxe tool",
    "iron_shovel": "iron shovel tool",
    "iron_sword": "iron sword weapon",
    "item_frame": "wooden item frame",
    "jungle_boat": "jungle wood boat",
    "kelp": "green kelp seaweed",
    "lapis_lazuli": "blue lapis lazuli gem",
    "lava_bucket": "bucket filled with lava",
    "lead": "rope lead leash",
    "leather": "brown leather piece",
    "leather_boots": "leather boots armor",
    "leather_chestplate": "leather chestplate armor",
    "leather_helmet": "leather helmet armor",
    "leather_leggings": "leather leggings armor",
    "magma_cream": "orange magma cream ball",
    "map": "empty map paper",
    "melon_seeds": "melon seeds",
    "melon_slice": "red melon slice food",
    "milk_bucket": "bucket filled with milk",
    "minecart": "iron minecart vehicle",
    "mushroom_stew": "mushroom stew bowl food",
    "mutton": "raw mutton meat",
    "name_tag": "paper name tag",
    "nautilus_shell": "pink nautilus shell",
    "nether_star": "white nether star",
    "netherite_axe": "netherite axe tool",
    "netherite_boots": "netherite boots armor",
    "netherite_chestplate": "netherite chestplate armor",
    "netherite_helmet": "netherite helmet armor",
    "netherite_hoe": "netherite hoe tool",
    "netherite_ingot": "dark netherite ingot bar",
    "netherite_leggings": "netherite leggings armor",
    "netherite_pickaxe": "netherite pickaxe tool",
    "netherite_scrap": "dark netherite scrap piece",
    "netherite_shovel": "netherite shovel tool",
    "netherite_sword": "dark netherite sword weapon",
    "oak_boat": "oak wood boat",
    "painting": "framed painting art",
    "paper": "white paper sheet",
    "phantom_membrane": "gray phantom membrane wing",
    "poisonous_potato": "green poisonous potato",
    "porkchop": "raw porkchop meat",
    "potato": "brown potato vegetable",
    "potion": "glass potion bottle",
    "prismarine_crystals": "blue prismarine crystals",
    "prismarine_shard": "blue prismarine shard",
    "pufferfish": "yellow pufferfish",
    "pumpkin_pie": "pumpkin pie food",
    "pumpkin_seeds": "pumpkin seeds",
    "quartz": "white nether quartz crystal",
    "rabbit": "raw rabbit meat",
    "rabbit_foot": "brown rabbit foot",
    "rabbit_hide": "white rabbit hide",
    "rabbit_stew": "rabbit stew bowl food",
    "raw_copper": "orange raw copper ore",
    "raw_gold": "yellow raw gold ore",
    "raw_iron": "beige raw iron ore",
    "redstone": "red redstone dust",
    "rotten_flesh": "brown rotten flesh",
    "saddle": "brown leather saddle",
    "salmon": "raw salmon fish",
    "scute": "green turtle scute",
    "shears": "iron shears tool",
    "shield": "wooden shield defense",
    "slime_ball": "green slime ball",
    "snowball": "white snowball",
    "spider_eye": "red spider eye",
    "spruce_boat": "spruce wood boat",
    "spyglass": "copper spyglass telescope",
    "stick": "brown wooden stick",
    "stone_axe": "stone axe tool",
    "stone_hoe": "stone hoe tool",
    "stone_pickaxe": "stone pickaxe tool",
    "stone_shovel": "stone shovel tool",
    "stone_sword": "stone sword weapon",
    "string": "white string thread",
    "sugar": "white sugar powder",
    "sugar_cane": "green sugar cane",
    "sweet_berries": "red sweet berries",
    "totem_of_undying": "golden totem of undying",
    "trident": "blue trident weapon",
    "tropical_fish": "orange tropical fish",
    "turtle_helmet": "green turtle shell helmet",
    "water_bucket": "bucket filled with water",
    "wheat": "golden wheat grain",
    "wheat_seeds": "wheat seeds",
    "wooden_axe": "wooden axe tool",
    "wooden_hoe": "wooden hoe tool",
    "wooden_pickaxe": "wooden pickaxe tool",
    "wooden_shovel": "wooden shovel tool",
    "wooden_sword": "wooden sword weapon",
    "writable_book": "book and quill writable",
    "written_book": "written signed book",
}


def extract_from_jar(jar_path, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    found = {}
    with zipfile.ZipFile(jar_path, 'r') as jar:
        for name in jar.namelist():
            if 'assets/minecraft/textures/item/' in name and name.endswith('.png'):
                base = os.path.basename(name)
                item_name = base.replace('.png', '')
                if item_name in ITEM_DESCRIPTIONS:
                    jar.extract(name, '/tmp/mc_extract')
                    src = os.path.join('/tmp/mc_extract', name)
                    dst = os.path.join(images_dir, base)
                    shutil.copy2(src, dst)
                    found[base] = ITEM_DESCRIPTIONS[item_name]

    return found


def extract_from_dir(assets_dir, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    textures_dir = None
    for root, dirs, files in os.walk(assets_dir):
        if root.endswith('textures/item') or root.endswith('textures/items'):
            textures_dir = root
            break

    if textures_dir is None:
        candidates = [
            os.path.join(assets_dir, 'assets', 'minecraft', 'textures', 'item'),
            os.path.join(assets_dir, 'assets', 'minecraft', 'textures', 'items'),
            assets_dir,
        ]
        for c in candidates:
            if os.path.isdir(c):
                textures_dir = c
                break

    if textures_dir is None:
        print(f"Error: Could not find textures/item directory in {assets_dir}")
        sys.exit(1)

    found = {}
    for fname in os.listdir(textures_dir):
        if fname.endswith('.png'):
            item_name = fname.replace('.png', '')
            if item_name in ITEM_DESCRIPTIONS:
                src = os.path.join(textures_dir, fname)
                dst = os.path.join(images_dir, fname)
                shutil.copy2(src, dst)
                found[fname] = ITEM_DESCRIPTIONS[item_name]

    return found


def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_dataset.py <minecraft_jar_or_assets_dir> [output_dir]")
        print("\nProvide either:")
        print("  - Path to minecraft .jar file")
        print("  - Path to extracted assets directory containing textures/item/*.png")
        sys.exit(1)

    source = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/minecraft_items"

    if source.endswith('.jar') or source.endswith('.zip'):
        print(f"Extracting from JAR: {source}")
        found = extract_from_jar(source, output_dir)
    else:
        print(f"Extracting from directory: {source}")
        found = extract_from_dir(source, output_dir)

    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(found, f, indent=2, ensure_ascii=False)

    print(f"\nDataset prepared: {len(found)} items")
    print(f"Images: {os.path.join(output_dir, 'images')}")
    print(f"Labels: {labels_path}")

    if len(found) == 0:
        print("\nWARNING: No items found! Make sure you're pointing to the correct path.")
        print("For a Minecraft .jar, item textures are at: assets/minecraft/textures/item/")


if __name__ == "__main__":
    main()
