"""
Generate synthetic 16x16 pixel art training data.
Creates simple but recognizable pixel art items with descriptions.
This serves as our training dataset when Minecraft assets aren't available.
"""
import os
import json
import numpy as np
from PIL import Image


def create_image(pixels, palette):
    img = np.zeros((16, 16, 4), dtype=np.uint8)
    for y in range(16):
        for x in range(16):
            color_idx = pixels[y][x] if y < len(pixels) and x < len(pixels[y]) else 0
            if color_idx == 0:
                img[y, x] = [0, 0, 0, 0]
            else:
                img[y, x] = palette[color_idx]
    return Image.fromarray(img, mode='RGBA')


ITEMS = {}

# --- SWORDS ---
def make_sword(blade_color, handle_color, guard_color):
    p = [[0]*16 for _ in range(16)]
    # blade diagonal
    for i in range(8):
        r, c = 2+i, 13-i
        if 0 <= r < 16 and 0 <= c < 16:
            p[r][c] = 1
            if c-1 >= 0: p[r][c-1] = 2
    # guard
    p[10][5] = 3; p[10][6] = 3; p[10][4] = 3
    # handle
    for i in range(3):
        p[11+i][4-i] = 4
    palette = {1: blade_color, 2: [*blade_color[:3], 180], 3: guard_color, 4: handle_color}
    return p, palette

sword_variants = {
    "diamond_sword": ([100, 220, 255, 255], [139, 90, 43, 255], [200, 200, 50, 255], "blue diamond sword weapon"),
    "iron_sword": ([200, 200, 200, 255], [139, 90, 43, 255], [160, 160, 160, 255], "gray iron sword weapon"),
    "golden_sword": ([255, 215, 0, 255], [139, 90, 43, 255], [200, 200, 50, 255], "golden sword weapon"),
    "wooden_sword": ([160, 120, 60, 255], [100, 70, 30, 255], [120, 90, 40, 255], "brown wooden sword weapon"),
    "stone_sword": ([128, 128, 128, 255], [139, 90, 43, 255], [100, 100, 100, 255], "gray stone sword weapon"),
    "netherite_sword": ([60, 50, 50, 255], [139, 90, 43, 255], [80, 60, 60, 255], "dark netherite sword weapon"),
}

for name, (blade, handle, guard, desc) in sword_variants.items():
    px, pal = make_sword(blade, handle, guard)
    ITEMS[name] = (px, pal, desc)

# --- PICKAXES ---
def make_pickaxe(head_color, stick_color):
    p = [[0]*16 for _ in range(16)]
    # head
    for c in range(4, 12):
        p[3][c] = 1
        p[4][c] = 1
    p[5][4] = 1; p[5][5] = 1; p[5][10] = 1; p[5][11] = 1
    # stick
    for i in range(8):
        p[5+i][7+i//2 if i < 4 else 8] = 2
        if i >= 4:
            p[5+i][8] = 2
    palette = {1: head_color, 2: stick_color}
    return p, palette

pick_variants = {
    "diamond_pickaxe": ([100, 220, 255, 255], [139, 90, 43, 255], "blue diamond pickaxe tool"),
    "iron_pickaxe": ([200, 200, 200, 255], [139, 90, 43, 255], "gray iron pickaxe mining tool"),
    "golden_pickaxe": ([255, 215, 0, 255], [139, 90, 43, 255], "golden pickaxe tool"),
    "wooden_pickaxe": ([160, 120, 60, 255], [100, 70, 30, 255], "brown wooden pickaxe tool"),
    "stone_pickaxe": ([128, 128, 128, 255], [139, 90, 43, 255], "gray stone pickaxe tool"),
    "netherite_pickaxe": ([60, 50, 50, 255], [139, 90, 43, 255], "dark netherite pickaxe tool"),
}

for name, (head, stick, desc) in pick_variants.items():
    px, pal = make_pickaxe(head, stick)
    ITEMS[name] = (px, pal, desc)

# --- GEMS/INGOTS ---
def make_gem(color1, color2):
    p = [[0]*16 for _ in range(16)]
    # diamond shape
    coords = [
        (4,7),(4,8),
        (5,6),(5,7),(5,8),(5,9),
        (6,5),(6,6),(6,7),(6,8),(6,9),(6,10),
        (7,5),(7,6),(7,7),(7,8),(7,9),(7,10),
        (8,5),(8,6),(8,7),(8,8),(8,9),(8,10),
        (9,6),(9,7),(9,8),(9,9),
        (10,7),(10,8),
    ]
    for r,c in coords:
        p[r][c] = 1
    # highlight
    p[5][7] = 2; p[6][6] = 2; p[6][7] = 2
    palette = {1: color1, 2: color2}
    return p, palette

ITEMS["diamond"] = (*make_gem([80, 200, 240, 255], [150, 240, 255, 255]), "blue diamond gem crystal")
ITEMS["emerald"] = (*make_gem([50, 200, 80, 255], [100, 255, 130, 255]), "green emerald gem crystal")

def make_ingot(color1, color2):
    p = [[0]*16 for _ in range(16)]
    for r in range(6, 11):
        for c in range(4, 12):
            p[r][c] = 1
    for c in range(5, 11):
        p[5][c] = 2
    p[6][4] = 2; p[6][5] = 2
    palette = {1: color1, 2: color2}
    return p, palette

ITEMS["gold_ingot"] = (*make_ingot([218, 165, 32, 255], [255, 215, 0, 255]), "shiny gold ingot bar")
ITEMS["iron_ingot"] = (*make_ingot([180, 180, 180, 255], [220, 220, 220, 255]), "silver iron ingot bar")
ITEMS["copper_ingot"] = (*make_ingot([180, 100, 50, 255], [220, 130, 70, 255]), "orange copper ingot bar")
ITEMS["netherite_ingot"] = (*make_ingot([50, 40, 40, 255], [80, 60, 60, 255]), "dark netherite ingot bar")

# --- FOOD ---
def make_apple(color1, stem_color):
    p = [[0]*16 for _ in range(16)]
    # stem
    p[3][8] = 2; p[4][8] = 2
    # body
    for r in range(5, 12):
        for c in range(5, 11):
            dist = ((r-8)**2 + (c-8)**2)
            if dist < 12:
                p[r][c] = 1
    palette = {1: color1, 2: stem_color}
    return p, palette

ITEMS["apple"] = (*make_apple([220, 30, 30, 255], [80, 50, 20, 255]), "red apple fruit food")
ITEMS["golden_apple"] = (*make_apple([255, 200, 0, 255], [80, 50, 20, 255]), "golden apple magical food")

def make_round_food(color1, color2):
    p = [[0]*16 for _ in range(16)]
    for r in range(5, 12):
        for c in range(5, 11):
            if ((r-8)**2 + (c-8)**2) < 11:
                p[r][c] = 1
    p[6][7] = 2; p[6][8] = 2; p[7][6] = 2
    palette = {1: color1, 2: color2}
    return p, palette

ITEMS["ender_pearl"] = (*make_round_food([20, 80, 80, 255], [50, 200, 180, 255]), "green ender pearl orb")
ITEMS["slime_ball"] = (*make_round_food([80, 200, 80, 255], [120, 240, 120, 255]), "green slime ball")
ITEMS["snowball"] = (*make_round_food([230, 240, 255, 255], [255, 255, 255, 255]), "white snowball")
ITEMS["egg"] = (*make_round_food([240, 230, 210, 255], [255, 250, 240, 255]), "white egg oval")
ITEMS["fire_charge"] = (*make_round_food([200, 80, 0, 255], [255, 150, 0, 255]), "orange fire charge ball")

# --- POTIONS ---
def make_potion(liquid_color):
    p = [[0]*16 for _ in range(16)]
    # neck
    for r in range(3, 6):
        p[r][7] = 3; p[r][8] = 3
    # body
    for r in range(6, 13):
        for c in range(5, 11):
            if ((r-9)**2 + (c-8)**2) < 14:
                p[r][c] = 1
    # liquid
    for r in range(8, 12):
        for c in range(6, 10):
            if p[r][c] == 1:
                p[r][c] = 2
    palette = {1: [200, 200, 220, 200], 2: liquid_color, 3: [180, 180, 180, 255]}
    return p, palette

ITEMS["potion_healing"] = (*make_potion([220, 50, 50, 255]), "red healing potion bottle")
ITEMS["potion_speed"] = (*make_potion([100, 180, 255, 255]), "blue speed potion bottle")
ITEMS["potion_poison"] = (*make_potion([80, 180, 30, 255]), "green poison potion bottle")
ITEMS["potion_strength"] = (*make_potion([150, 30, 30, 255]), "dark red strength potion bottle")
ITEMS["experience_bottle"] = (*make_potion([100, 255, 100, 255]), "green experience bottle")

# --- ARROWS/STICKS ---
def make_arrow():
    p = [[0]*16 for _ in range(16)]
    # shaft
    for i in range(10):
        r, c = 3+i, 12-i
        if 0<=r<16 and 0<=c<16:
            p[r][c] = 1
    # head
    p[2][13] = 2; p[3][12] = 2; p[2][12] = 2; p[3][13] = 2
    # feather
    p[12][3] = 3; p[13][2] = 3; p[11][3] = 3
    palette = {1: [139, 90, 43, 255], 2: [160, 160, 160, 255], 3: [220, 220, 220, 255]}
    return p, palette

ITEMS["arrow"] = (*make_arrow(), "pointed arrow projectile")

def make_stick():
    p = [[0]*16 for _ in range(16)]
    for i in range(10):
        p[3+i][6+i//2] = 1
    palette = {1: [139, 90, 43, 255]}
    return p, palette

ITEMS["stick"] = (*make_stick(), "brown wooden stick")

# --- ARMOR ---
def make_helmet(color1, color2):
    p = [[0]*16 for _ in range(16)]
    for r in range(4, 9):
        for c in range(4, 12):
            p[r][c] = 1
    for c in range(5, 11):
        p[3][c] = 1
    # visor
    for c in range(5, 11):
        p[9][c] = 1
    p[10][5] = 1; p[10][10] = 1
    # highlight
    p[4][5] = 2; p[5][5] = 2
    palette = {1: color1, 2: color2}
    return p, palette

helmet_variants = {
    "diamond_helmet": ([80, 200, 240, 255], [150, 240, 255, 255], "blue diamond helmet armor"),
    "iron_helmet": ([180, 180, 180, 255], [220, 220, 220, 255], "gray iron helmet armor"),
    "golden_helmet": ([218, 165, 32, 255], [255, 215, 0, 255], "golden helmet armor"),
    "leather_helmet": ([139, 90, 43, 255], [170, 120, 60, 255], "brown leather helmet armor"),
    "netherite_helmet": ([50, 40, 40, 255], [80, 60, 60, 255], "dark netherite helmet armor"),
}

for name, (c1, c2, desc) in helmet_variants.items():
    px, pal = make_helmet(c1, c2)
    ITEMS[name] = (px, pal, desc)

# --- CHESTPLATE ---
def make_chestplate(color1, color2):
    p = [[0]*16 for _ in range(16)]
    # shoulders
    for c in range(3, 13):
        p[3][c] = 1; p[4][c] = 1
    # body
    for r in range(5, 13):
        for c in range(5, 11):
            p[r][c] = 1
    # arms
    for r in range(5, 9):
        p[r][3] = 1; p[r][4] = 1; p[r][11] = 1; p[r][12] = 1
    p[5][5] = 2; p[5][6] = 2
    palette = {1: color1, 2: color2}
    return p, palette

chest_variants = {
    "diamond_chestplate": ([80, 200, 240, 255], [150, 240, 255, 255], "blue diamond chestplate armor"),
    "iron_chestplate": ([180, 180, 180, 255], [220, 220, 220, 255], "gray iron chestplate armor"),
    "golden_chestplate": ([218, 165, 32, 255], [255, 215, 0, 255], "golden chestplate armor"),
    "netherite_chestplate": ([50, 40, 40, 255], [80, 60, 60, 255], "dark netherite chestplate armor"),
}

for name, (c1, c2, desc) in chest_variants.items():
    px, pal = make_chestplate(c1, c2)
    ITEMS[name] = (px, pal, desc)

# --- BOOTS ---
def make_boots(color1, color2):
    p = [[0]*16 for _ in range(16)]
    # left boot
    for r in range(7, 13):
        p[r][4] = 1; p[r][5] = 1
    for c in range(3, 7):
        p[12][c] = 1; p[13][c] = 1
    # right boot
    for r in range(7, 13):
        p[r][10] = 1; p[r][11] = 1
    for c in range(9, 13):
        p[12][c] = 1; p[13][c] = 1
    p[7][4] = 2; p[7][10] = 2
    palette = {1: color1, 2: color2}
    return p, palette

boot_variants = {
    "diamond_boots": ([80, 200, 240, 255], [150, 240, 255, 255], "blue diamond boots armor"),
    "iron_boots": ([180, 180, 180, 255], [220, 220, 220, 255], "gray iron boots armor"),
    "golden_boots": ([218, 165, 32, 255], [255, 215, 0, 255], "golden boots armor"),
    "leather_boots": ([139, 90, 43, 255], [170, 120, 60, 255], "brown leather boots armor"),
    "netherite_boots": ([50, 40, 40, 255], [80, 60, 60, 255], "dark netherite boots armor"),
}

for name, (c1, c2, desc) in boot_variants.items():
    px, pal = make_boots(c1, c2)
    ITEMS[name] = (px, pal, desc)

# --- BUCKETS ---
def make_bucket(content_color=None):
    p = [[0]*16 for _ in range(16)]
    # bucket body
    for r in range(6, 13):
        p[r][5] = 1; p[r][10] = 1
    for c in range(5, 11):
        p[12][c] = 1
    # handle
    p[5][6] = 1; p[4][7] = 1; p[4][8] = 1; p[5][9] = 1
    # content
    if content_color:
        for r in range(7, 12):
            for c in range(6, 10):
                p[r][c] = 2
    palette = {1: [180, 180, 180, 255]}
    if content_color:
        palette[2] = content_color
    return p, palette

ITEMS["bucket"] = (*make_bucket(), "empty iron bucket")
ITEMS["water_bucket"] = (*make_bucket([50, 100, 200, 255]), "bucket filled with water")
ITEMS["lava_bucket"] = (*make_bucket([220, 100, 0, 255]), "bucket filled with lava")
ITEMS["milk_bucket"] = (*make_bucket([250, 250, 250, 255]), "bucket filled with milk")

# --- BOW ---
def make_bow():
    p = [[0]*16 for _ in range(16)]
    # curve
    coords = [(3,10),(4,11),(5,12),(6,12),(7,12),(8,12),(9,12),(10,12),(11,11),(12,10)]
    for r,c in coords:
        p[r][c] = 1
    # string
    for r in range(3, 13):
        p[r][9] = 2
    palette = {1: [139, 90, 43, 255], 2: [200, 200, 200, 255]}
    return p, palette

ITEMS["bow"] = (*make_bow(), "wooden bow weapon curved")

# --- BOOKS ---
def make_book(cover_color):
    p = [[0]*16 for _ in range(16)]
    for r in range(4, 13):
        for c in range(4, 12):
            p[r][c] = 1
    # pages
    for r in range(5, 12):
        for c in range(5, 11):
            p[r][c] = 2
    # spine
    for r in range(4, 13):
        p[r][4] = 3
    palette = {1: cover_color, 2: [240, 230, 200, 255], 3: [*[max(0,c-40) for c in cover_color[:3]], 255]}
    return p, palette

ITEMS["book"] = (*make_book([139, 90, 43, 255]), "brown leather book")
ITEMS["enchanted_book"] = (*make_book([100, 50, 150, 255]), "purple glowing enchanted book")
ITEMS["writable_book"] = (*make_book([60, 40, 20, 255]), "dark book and quill writable")

# --- MISC ---
def make_bone():
    p = [[0]*16 for _ in range(16)]
    # shaft
    for i in range(8):
        p[4+i][6+i//2] = 1
    # ends
    p[3][5] = 1; p[3][6] = 1; p[4][5] = 1
    p[12][10] = 1; p[12][11] = 1; p[11][11] = 1
    palette = {1: [240, 235, 220, 255]}
    return p, palette

ITEMS["bone"] = (*make_bone(), "white bone item")

def make_feather():
    p = [[0]*16 for _ in range(16)]
    for i in range(10):
        p[3+i][5+i//2] = 1
    for i in range(6):
        p[4+i][6+i//2] = 2
    palette = {1: [220, 220, 230, 255], 2: [180, 180, 200, 255]}
    return p, palette

ITEMS["feather"] = (*make_feather(), "white feather quill")

def make_coal():
    p = [[0]*16 for _ in range(16)]
    for r in range(5, 12):
        for c in range(5, 11):
            if ((r-8)**2 + (c-8)**2) < 12:
                p[r][c] = 1
    p[6][7] = 2; p[7][6] = 2
    palette = {1: [30, 30, 30, 255], 2: [60, 60, 60, 255]}
    return p, palette

ITEMS["coal"] = (*make_coal(), "black coal piece mineral")
ITEMS["charcoal"] = (*make_coal(), "dark charcoal piece")

def make_redstone():
    p = [[0]*16 for _ in range(16)]
    # scattered dust pattern
    spots = [(5,6),(6,8),(7,5),(7,9),(8,7),(9,6),(9,10),(10,8),(11,7),(6,10),(8,4),(10,5)]
    for r,c in spots:
        p[r][c] = 1
    palette = {1: [200, 0, 0, 255]}
    return p, palette

ITEMS["redstone"] = (*make_redstone(), "red redstone dust powder")
ITEMS["glowstone_dust"] = (make_redstone()[0], {1: [255, 200, 50, 255]}, "yellow glowstone dust powder")
ITEMS["gunpowder"] = (make_redstone()[0], {1: [100, 100, 100, 255]}, "gray gunpowder dust")
ITEMS["sugar"] = (make_redstone()[0], {1: [240, 240, 240, 255]}, "white sugar powder")

def make_pearl():
    p = [[0]*16 for _ in range(16)]
    for r in range(5, 12):
        for c in range(5, 11):
            dist = ((r-8)**2 + (c-8)**2)
            if dist < 12:
                p[r][c] = 1
    p[6][6] = 2; p[6][7] = 2; p[7][6] = 2
    palette = {1: [20, 80, 80, 255], 2: [50, 200, 180, 255]}
    return p, palette

ITEMS["ender_pearl_v2"] = (*make_pearl(), "dark green ender pearl teleport orb")

# --- FOOD items ---
def make_bread():
    p = [[0]*16 for _ in range(16)]
    for r in range(7, 11):
        for c in range(3, 13):
            p[r][c] = 1
    for c in range(4, 12):
        p[6][c] = 2
    palette = {1: [180, 120, 50, 255], 2: [200, 150, 70, 255]}
    return p, palette

ITEMS["bread"] = (*make_bread(), "brown bread loaf food")

def make_carrot():
    p = [[0]*16 for _ in range(16)]
    for i in range(9):
        p[6+i//2][4+i] = 1
        if i < 7: p[7+i//2][4+i] = 1
    p[5][11] = 2; p[4][12] = 2; p[5][12] = 2
    palette = {1: [230, 130, 30, 255], 2: [50, 180, 50, 255]}
    return p, palette

ITEMS["carrot"] = (*make_carrot(), "orange carrot vegetable food")

def make_fish():
    p = [[0]*16 for _ in range(16)]
    for r in range(6, 11):
        for c in range(4, 12):
            p[r][c] = 1
    # tail
    p[7][12] = 1; p[8][12] = 1; p[6][13] = 1; p[10][13] = 1
    # eye
    p[7][5] = 2
    palette = {1: [150, 120, 80, 255], 2: [30, 30, 30, 255]}
    return p, palette

ITEMS["cod"] = (*make_fish(), "raw cod fish")
ITEMS["salmon"] = (make_fish()[0], {1: [200, 80, 60, 255], 2: [30, 30, 30, 255]}, "raw salmon fish orange")
ITEMS["cooked_cod"] = (make_fish()[0], {1: [180, 150, 100, 255], 2: [30, 30, 30, 255]}, "cooked cod fish food")

def make_steak():
    p = [[0]*16 for _ in range(16)]
    for r in range(5, 12):
        for c in range(4, 12):
            if ((r-8)**2 + (c-8)**2) < 15:
                p[r][c] = 1
    for r in range(6, 10):
        for c in range(5, 9):
            p[r][c] = 2
    palette = {1: [120, 50, 30, 255], 2: [180, 60, 40, 255]}
    return p, palette

ITEMS["cooked_beef"] = (*make_steak(), "cooked steak meat food brown")
ITEMS["beef"] = (make_steak()[0], {1: [180, 40, 40, 255], 2: [220, 80, 80, 255]}, "raw beef meat red")

# --- TOOLS ---
def make_shovel(head_color, stick_color):
    p = [[0]*16 for _ in range(16)]
    # head
    for r in range(3, 7):
        for c in range(6, 10):
            p[r][c] = 1
    p[3][7] = 1; p[3][8] = 1
    # stick
    for r in range(7, 14):
        p[r][8] = 2
    palette = {1: head_color, 2: stick_color}
    return p, palette

shovel_variants = {
    "diamond_shovel": ([80, 200, 240, 255], [139, 90, 43, 255], "blue diamond shovel tool"),
    "iron_shovel": ([180, 180, 180, 255], [139, 90, 43, 255], "gray iron shovel tool"),
    "golden_shovel": ([218, 165, 32, 255], [139, 90, 43, 255], "golden shovel tool"),
    "stone_shovel": ([128, 128, 128, 255], [139, 90, 43, 255], "gray stone shovel tool"),
    "wooden_shovel": ([160, 120, 60, 255], [100, 70, 30, 255], "brown wooden shovel tool"),
}

for name, (head, stick, desc) in shovel_variants.items():
    px, pal = make_shovel(head, stick)
    ITEMS[name] = (px, pal, desc)

# --- AXE ---
def make_axe(head_color, stick_color):
    p = [[0]*16 for _ in range(16)]
    # head
    for r in range(3, 8):
        for c in range(8, 12):
            p[r][c] = 1
    p[4][12] = 1; p[5][12] = 1; p[6][12] = 1
    # stick
    for i in range(9):
        p[4+i][7-i//2] = 2
    palette = {1: head_color, 2: stick_color}
    return p, palette

axe_variants = {
    "diamond_axe": ([80, 200, 240, 255], [139, 90, 43, 255], "blue diamond axe tool"),
    "iron_axe": ([180, 180, 180, 255], [139, 90, 43, 255], "gray iron axe tool"),
    "golden_axe": ([218, 165, 32, 255], [139, 90, 43, 255], "golden axe tool"),
    "stone_axe": ([128, 128, 128, 255], [139, 90, 43, 255], "gray stone axe tool"),
    "wooden_axe": ([160, 120, 60, 255], [100, 70, 30, 255], "brown wooden axe tool"),
}

for name, (head, stick, desc) in axe_variants.items():
    px, pal = make_axe(head, stick)
    ITEMS[name] = (px, pal, desc)

# --- STAR / SPECIAL ---
def make_star():
    p = [[0]*16 for _ in range(16)]
    center_r, center_c = 8, 8
    # star points
    points = [(4,8),(5,7),(5,9),(6,6),(6,7),(6,8),(6,9),(6,10),
              (7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),
              (8,6),(8,7),(8,8),(8,9),(8,10),
              (9,5),(9,6),(9,7),(9,8),(9,9),(9,10),(9,11),
              (10,6),(10,7),(10,8),(10,9),(10,10),
              (11,7),(11,8),(11,9),(12,8)]
    for r,c in points:
        p[r][c] = 1
    p[7][8] = 2; p[8][8] = 2; p[8][7] = 2
    palette = {1: [240, 240, 220, 255], 2: [255, 255, 255, 255]}
    return p, palette

ITEMS["nether_star"] = (*make_star(), "white nether star glowing")

def make_blaze_rod():
    p = [[0]*16 for _ in range(16)]
    for i in range(12):
        p[2+i][7] = 1; p[2+i][8] = 1
    p[4][6] = 2; p[6][9] = 2; p[9][6] = 2; p[11][9] = 2
    palette = {1: [220, 180, 50, 255], 2: [255, 220, 80, 255]}
    return p, palette

ITEMS["blaze_rod"] = (*make_blaze_rod(), "yellow blaze rod glowing")

def make_trident():
    p = [[0]*16 for _ in range(16)]
    # prongs
    p[2][6] = 1; p[2][8] = 1; p[2][10] = 1
    p[3][6] = 1; p[3][8] = 1; p[3][10] = 1
    p[4][6] = 1; p[4][7] = 1; p[4][8] = 1; p[4][9] = 1; p[4][10] = 1
    # shaft
    for r in range(5, 14):
        p[r][8] = 1
    palette = {1: [60, 140, 180, 255]}
    return p, palette

ITEMS["trident"] = (*make_trident(), "blue trident weapon aquatic")

# --- ADDITIONAL VARIETY ---
def make_heart():
    p = [[0]*16 for _ in range(16)]
    coords = [(5,5),(5,6),(5,9),(5,10),(6,4),(6,5),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),
              (7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),
              (8,5),(8,6),(8,7),(8,8),(8,9),(8,10),
              (9,6),(9,7),(9,8),(9,9),(10,7),(10,8)]
    for r,c in coords:
        p[r][c] = 1
    palette = {1: [220, 30, 30, 255]}
    return p, palette

ITEMS["totem_of_undying"] = (make_heart()[0], {1: [218, 165, 32, 255]}, "golden totem of undying artifact")

# Add more simple color variants for training diversity
simple_items = {
    "lapis_lazuli": ([30, 50, 200, 255], "blue lapis lazuli gem"),
    "prismarine_shard": ([70, 180, 170, 255], "blue prismarine shard crystal"),
    "amethyst_shard": ([150, 80, 200, 255], "purple amethyst shard crystal"),
}

for name, (color, desc) in simple_items.items():
    px, pal = make_gem(color, [min(255, c+50) for c in color[:3]] + [255])
    ITEMS[name] = (px, pal, desc)


def main():
    output_dir = "data/minecraft_items"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels = {}
    for name, (pixels, palette, description) in ITEMS.items():
        img = create_image(pixels, palette)
        filename = f"{name}.png"
        img.save(os.path.join(images_dir, filename))
        labels[filename] = description

    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Generated {len(labels)} items in {output_dir}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
