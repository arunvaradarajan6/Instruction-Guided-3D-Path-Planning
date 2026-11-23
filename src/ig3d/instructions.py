"""Instruction corpus.

v1 (the paper): 128 sentences over 8 classes, 16 each.
v2: v1 plus 32 extra training phrasings per class (384 total).
HARD_TEST: 80 out-of-register sentences, never trained on.
"""
from __future__ import annotations

import os

import numpy as np

TIER = {
    "open": 1, "tight": 1, "low": 1, "high": 1, "shortest": 1,
    "level": 2,
    "climb_open": 3, "descend_open": 3,
}
CLASSES = tuple(TIER)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

OPEN = [
    "Aim for wider paths.", "Prefer the roomier route.", "Stay in open space.",
    "Give obstacles a wide berth.", "Keep plenty of clearance on all sides.",
    "Choose the widest opening you can find.", "Avoid tight squeezes.",
    "Fly through the most open part of the volume.", "Steer clear of narrow gaps.",
    "Keep a safe margin from every surface.", "Take the least confined route.",
    "Maximise clearance along the way.", "Do not cut it close to anything.",
    "Use the large openings, not the small ones.", "Leave a lot of space around you.",
    "Prefer open volume over cramped shortcuts.",
]

TIGHT = [
    "Mind small gaps.", "Prefer the narrow passage.", "Squeeze through the tight openings.",
    "Take the constricted route.", "Use the small hole rather than the big one.",
    "Thread through the tightest gap.", "Keep clearance to a minimum.",
    "Go where the space is tightest.", "Slip through the narrow slot.",
    "Choose the smaller of the two openings.", "Fly close to the surfaces.",
    "Take the cramped way through.", "Minimise clearance along the way.",
    "Avoid the wide open parts.", "Pick the pinch point.", "Keep the route tight.",
]

LOW = [
    "Stay low.", "Fly close to the floor.", "Keep a low altitude throughout.",
    "Remain near ground level.", "Do not fly high.", "Hug the deck.",
    "Keep well below the ceiling.", "Operate at low level.",
    "Choose the lowest workable height.", "Stay in the bottom of the volume.",
    "Keep your cruising height small.", "Fly a low profile.",
    "Stick to the lower levels.", "Stay nearer the floor than the ceiling.",
    "Do not rise into the upper space.", "The lower the route, the better.",
]

HIGH = [
    "Stay high.", "Fly well above the clutter.", "Keep a high altitude throughout.",
    "Remain near the top of the volume.", "Do not fly low.", "Hug the ceiling.",
    "Keep well above the floor.", "Operate at high level.",
    "Choose the highest workable height.", "Stay in the upper part of the space.",
    "Keep your cruising height large.", "Fly a high profile.",
    "Stick to the upper levels.", "Stay nearer the ceiling than the floor.",
    "Do not drop into the lower space.", "The higher the route, the better.",
]

SHORTEST = [
    "Take the shortest path.", "Fly straight to the goal.", "Minimise travel distance.",
    "Do not detour.", "Take the most direct route.", "Distance is all that matters.",
    "Get there as quickly as you can.", "Optimise for length only.",
    "Reach the goal in the fewest metres.", "Do not go out of your way.",
    "Cut straight across.", "Be direct.", "Waste no distance.",
    "Ignore clearance and height, just be short.", "Keep the trip as short as possible.",
    "Route directly to the destination.",
]

LEVEL = [
    "Keep it level.", "Minimise vertical manoeuvring.",
    "Avoid changing altitude if you can.", "Hold a steady height.",
    "Going up and down costs more than going across.",
    "Prefer horizontal travel to vertical travel.",
    "Keep the vertical profile as flat as you can.",
    "Do not bob up and down.", "Fly a flat trajectory.",
    "Height changes are expensive, so avoid them.",
    "Stay on one level as much as possible.",
    "Trade sideways distance for a flatter route.",
    "Avoid altitude changes even if the route gets longer.",
    "Keep the climb and descent totals small.",
    "A level route is worth a detour.", "Do not waste energy going vertical.",
]

CLIMB_OPEN = [
    "Do your climbing in the open, not in the clutter.",
    "Only gain height where there is room around you.",
    "Do not ascend near obstacles.",
    "Climb in clear air, never inside a gap.",
    "Gain your altitude before you enter the tight parts.",
    "Rising close to a surface is dangerous, so rise in the open.",
    "Never climb while squeezing through something.",
    "If you must go up, do it where there is space.",
    "Avoid upward motion in confined areas.",
    "Get your height in the open volume, then transit.",
    "Climbing near walls is not acceptable.",
    "Ascend only where clearance is generous.",
    "Do not gain altitude inside a narrow passage.",
    "Pick somewhere roomy to make your climb.",
    "Upward legs belong in open space.",
    "Keep climbs away from the clutter.",
]

DESCEND_OPEN = [
    "Do your descending in the open, not through the gaps.",
    "Only lose height where there is room around you.",
    "Do not descend near obstacles.",
    "Drop down in clear air, never inside a gap.",
    "Lose your altitude after you leave the tight parts.",
    "Sinking close to a surface is dangerous, so sink in the open.",
    "Never descend while squeezing through something.",
    "If you must go down, do it where there is space.",
    "Avoid downward motion in confined areas.",
    "Transit first, then give up your height in the open volume.",
    "Descending near walls is not acceptable.",
    "Descend only where clearance is generous.",
    "Do not lose altitude inside a narrow passage.",
    "Pick somewhere roomy to make your descent.",
    "Downward legs belong in open space.",
    "Keep descents away from the clutter.",
]

BY_CLASS = {
    "open": OPEN, "tight": TIGHT, "low": LOW, "high": HIGH, "shortest": SHORTEST,
    "level": LEVEL, "climb_open": CLIMB_OPEN, "descend_open": DESCEND_OPEN,
}

# ---------------------------------------------------------------------------------------
# Corpus v2: 32 extra TRAINING phrasings per class (256 more sentences).
#
# They are appended after the paper's 128 sentences, so indices 0..127 keep their meaning
# and every checkpoint trained on v1 still evaluates correctly. They deliberately lean on
# the things frozen encoders get wrong in v1: negation ("don't ... high"), the opposite
# pole named and rejected ("not the big gap"), indirect goals ("save battery on climbs"),
# and operational/pilot vocabulary.
# ---------------------------------------------------------------------------------------

EXTRA = {
    "open": [
        "Go through the biggest gap.", "Pick the spacious way through.",
        "Stay away from walls and pillars.", "Keep your distance from everything.",
        "Never pass close to an obstacle.", "Choose the big opening, not the small one.",
        "Don't squeeze through anything narrow.", "I want lots of room around the drone.",
        "Fly where there is the most free space.", "Wide gaps only, please.",
        "Avoid anything that looks like a bottleneck.", "Route through the largest hole in each wall.",
        "Stay centred in the open air.", "Don't get near the edges of the openings.",
        "Prefer generous clearance even if it takes longer.", "A roomy route is better than a short one.",
        "Keep a buffer between you and every surface.", "Ignore the small gaps and use the large ones.",
        "Don't brush past any structure.", "Pick the passage with the most elbow room.",
        "Clearance first, everything else second.", "Stay well clear of the clutter.",
        "The more space around you the better.", "Avoid flying near any solid object.",
        "Keep it roomy.", "Choose the uncluttered path.",
        "Do not thread any needles.", "Take the airy route through the structure.",
        "Fly down the middle of the free space.", "Don't take the cramped shortcut; take the wide way.",
        "Give every wall a generous margin.", "Stay where nothing is close to you.",
    ],
    "tight": [
        "Go through the smallest gap.", "Pick the narrow way through.",
        "Stay close to the walls.", "Hug the obstacles.",
        "Squeeze through rather than go around.", "Choose the small opening, not the big one.",
        "Don't use the wide openings.", "I want the drone to fly through the tight spots.",
        "Fly where the free space is smallest.", "Narrow gaps only, please.",
        "Go for the bottleneck.", "Route through the smallest hole in each wall.",
        "Skim along the surfaces.", "Thread the needle.",
        "Prefer tight passages even if they are uncomfortable.", "A snug route is better than a roomy one.",
        "Keep close to structure the whole way.", "Ignore the large gaps and use the small ones.",
        "Brush past the obstacles.", "Pick the passage with the least room.",
        "Tightness first, everything else second.", "Stay in among the clutter.",
        "The less space around you the better.", "Fly near the solid objects.",
        "Keep it snug.", "Choose the cluttered path.",
        "Avoid open areas; stay in the confined parts.", "Take the cramped route through the structure.",
        "Fly right next to the edges of the gaps.", "Don't take the wide way; squeeze through.",
        "Give the walls as little margin as possible.", "Stay where things are close to you.",
    ],
    "low": [
        "Keep low.", "Fly near the ground.",
        "Don't climb high.", "Stay down.",
        "Keep altitude to a minimum.", "Fly as low as you safely can.",
        "Avoid the upper part of the space.", "Low altitude only.",
        "Skim just above the floor.", "Stay under the high zone.",
        "Keep your height down the whole way.", "Remain in the lower half.",
        "Don't go anywhere near the ceiling.", "Ground level is where you should be.",
        "Choose the bottom route, not the top one.", "Fly beneath everything you can.",
        "Prefer lower altitudes over higher ones.", "Keep the drone close to the deck.",
        "Avoid climbing up into the upper space.", "Stay below the treetops and rooflines where possible.",
        "Do not fly high, keep it low.", "The ceiling is off limits; stay near the floor.",
        "Keep a low ceiling on your altitude.", "Go under, not over.",
        "Fly the lowest route available.", "Hold a low altitude.",
        "Keep down near the bottom.", "Stay out of the upper levels.",
        "Lower is better.", "Cruise low.",
        "Never rise far above the floor.", "Pick the route that stays nearest the ground.",
    ],
    "high": [
        "Keep high.", "Fly near the ceiling.",
        "Don't fly low.", "Stay up.",
        "Keep altitude at a maximum.", "Fly as high as you safely can.",
        "Avoid the lower part of the space.", "High altitude only.",
        "Skim just below the ceiling.", "Stay above the low zone.",
        "Keep your height up the whole way.", "Remain in the upper half.",
        "Don't go anywhere near the floor.", "The top of the space is where you should be.",
        "Choose the top route, not the bottom one.", "Fly over everything you can.",
        "Prefer higher altitudes over lower ones.", "Keep the drone up near the top.",
        "Avoid dropping down into the lower space.", "Stay above the treetops and rooflines where possible.",
        "Do not fly low, keep it high.", "The floor is off limits; stay near the ceiling.",
        "Keep a high floor on your altitude.", "Go over, not under.",
        "Fly the highest route available.", "Hold a high altitude.",
        "Keep up near the top.", "Stay out of the lower levels.",
        "Higher is better.", "Cruise high.",
        "Never sink far below the ceiling.", "Pick the route that stays nearest the ceiling.",
    ],
    "shortest": [
        "Shortest route, please.", "Go straight there.",
        "Take the quickest line to the goal.", "No detours.",
        "Make the trip as short as you can.", "Just get to the destination directly.",
        "Fly as the crow flies.", "Minimise path length.",
        "Pick the most efficient route by distance.", "Don't wander.",
        "Go the direct way.", "Head straight for the target.",
        "Take the beeline.", "The shorter the path, the better.",
        "Don't take the scenic route.", "Choose the route with the least distance.",
        "Distance is the only thing I care about.", "Cover as little ground as possible.",
        "Fly the most direct line available.", "Keep the path short and direct.",
        "Straight line where you can.", "Don't add extra distance for anything.",
        "Fastest way there by length.", "Get from start to goal with minimal travel.",
        "Avoid any unnecessary distance.", "Use the most direct path to the goal.",
        "Take whatever way is shortest.", "Shortcut wherever possible.",
        "Go direct.", "Point-to-point, no extra metres.",
        "Ignore other preferences and minimise distance.", "Route as straight as the obstacles allow.",
    ],
    "level": [
        "Keep a constant altitude.", "Don't change height.",
        "Fly flat.", "Stay at the same altitude the whole way.",
        "Avoid climbing and descending.", "Maintain your altitude.",
        "Minimise vertical movement.", "Go around obstacles rather than over or under them.",
        "Hold altitude and steer sideways instead.", "A longer flat route beats a shorter hilly one.",
        "Climbing and descending are both expensive.", "Keep the altitude profile flat.",
        "Avoid any up-and-down motion.", "Don't go up and down unless you have to.",
        "Stay on the same horizontal plane.", "Keep vertical changes to a minimum.",
        "Prefer detours sideways over changes in height.", "Level flight, please.",
        "Fly at one height.", "Keep it flat.",
        "No unnecessary altitude changes.", "Vertical motion costs extra; avoid it.",
        "Steer left or right instead of up or down.", "Keep the total climb and descent low.",
        "Hold a level path even if it is longer.", "Don't hop over things; go around them.",
        "Don't duck under things; go around them.", "Altitude changes should be rare.",
        "Avoid both climbing and diving.", "Maintain height throughout the flight.",
        "Keep a flat profile from start to finish.", "Save energy by not changing altitude.",
    ],
    "climb_open": [
        "Climb only in the open.", "When you go up, make sure there is space around you.",
        "Gain altitude away from the walls.", "Do your ascending in clear space.",
        "Never climb near an obstacle.", "Climbing is fine, but only where it is roomy.",
        "Don't gain height in a tight spot.", "Rise in open air, not next to structure.",
        "Ascents must happen with plenty of clearance.", "If you need height, get it in the open.",
        "Avoid climbing through narrow gaps.", "Climb before entering the cramped parts.",
        "Upward movement only in open areas.", "Don't go up while close to anything.",
        "Climbs belong in clear air.", "Keep your climbs well away from walls.",
        "Ascend where nothing is near you.", "Climbing near surfaces is risky; climb in the open.",
        "Gain height in spacious areas only.", "No climbing in confined spaces.",
        "Make your altitude gains in the open.", "When climbing, stay clear of obstacles.",
        "Go up only when there is room on all sides.", "Don't ascend inside a gap.",
        "Pick an open area for gaining height.", "Climb out in the clear.",
        "Save the climbing for open space.", "Any climb should be done far from structure.",
        "Never rise while squeezing through.", "Ascend in open volume, then go through the gaps.",
        "Height gains must be made in open air.", "Do the upward legs where it is roomy.",
    ],
    "descend_open": [
        "Descend only in the open.", "When you go down, make sure there is space around you.",
        "Lose altitude away from the walls.", "Do your descending in clear space.",
        "Never descend near an obstacle.", "Descending is fine, but only where it is roomy.",
        "Don't lose height in a tight spot.", "Sink in open air, not next to structure.",
        "Descents must happen with plenty of clearance.", "If you need to drop, do it in the open.",
        "Avoid descending through narrow gaps.", "Descend after leaving the cramped parts.",
        "Downward movement only in open areas.", "Don't go down while close to anything.",
        "Descents belong in clear air.", "Keep your descents well away from walls.",
        "Lose height where nothing is near you.", "Descending near surfaces is risky; descend in the open.",
        "Drop height in spacious areas only.", "No descending in confined spaces.",
        "Make your altitude losses in the open.", "When descending, stay clear of obstacles.",
        "Go down only when there is room on all sides.", "Don't descend inside a gap.",
        "Pick an open area for losing height.", "Come down out in the clear.",
        "Save the descending for open space.", "Any descent should be done far from structure.",
        "Never sink while squeezing through.", "Go through the gaps first, then descend in open volume.",
        "Height losses must be made in open air.", "Do the downward legs where it is roomy.",
    ],
}

# ---------------------------------------------------------------------------------------
# Hard test set: 10 per class, NEVER used for training under any corpus. Written in a
# different register from both v1 and v2: situational context, rationale, double negation,
# and sentences that mention the opposite pole. Used by `e3 --hard` and `e1 --sentences hard`.
# ---------------------------------------------------------------------------------------

HARD_TEST = {
    "open": [
        "The prop guards are fragile, so give yourself room everywhere.",
        "I'd rather take the long way than risk clipping something.",
        "The narrow window is not an option today.",
        "Pretend the drone is twice as wide as it really is.",
        "Our GPS is drifting, so stay far from anything solid.",
        "Squeezing through gaps is how we lost the last drone.",
        "Imagine every wall is electrified and keep away.",
        "If two openings are available, the bigger one wins.",
        "Small holes are for small drones, and ours is not small.",
        "Stay in the middle of whatever space you are in.",
    ],
    "tight": [
        "We're testing the obstacle sensors, so stay near the walls.",
        "The big opening is blocked by a crane, use the little one.",
        "Pretend the open areas are no-fly zones.",
        "Take the gap a cautious pilot would avoid.",
        "Fly like you are inspecting the surfaces up close.",
        "The wide route is not what I want.",
        "Stick close to structure for the inspection pass.",
        "If two openings are available, the smaller one wins.",
        "Open space is boring; go where it's cramped.",
        "Don't avoid the narrow passages.",
    ],
    "low": [
        "There's a helicopter overhead, so stay down.",
        "Airspace above ten metres is restricted today.",
        "We want ground-level footage on this run.",
        "Don't fly any higher than you need to.",
        "The upper half of the space is not where you should be.",
        "Act like the ceiling is much lower than it is.",
        "Stay beneath the canopy the whole time.",
        "Altitude is the enemy on this flight.",
        "Going high is the wrong answer here.",
        "Keep under the power lines.",
    ],
    "high": [
        "Pedestrians are below, so stay up and away from them.",
        "Stay above the crowd.",
        "We want aerial overview footage on this run.",
        "Don't fly any lower than you need to.",
        "The lower half of the space is not where you should be.",
        "Act like the floor is lava.",
        "Stay over the canopy the whole time.",
        "Altitude is your friend on this flight.",
        "Going low is the wrong answer here.",
        "Keep over the rooftops.",
    ],
    "shortest": [
        "The battery is almost dead, get there by the shortest way.",
        "Time is money, so no wandering.",
        "Whatever the obstacles, minimise how far you fly.",
        "Draw the shortest line you can and follow it.",
        "Don't worry about clearance or height, just distance.",
        "Every extra metre counts against you.",
        "We just need the package delivered by the most direct route.",
        "A longer path is never acceptable today.",
        "Don't detour for comfort.",
        "Go the way a string pulled tight would go.",
    ],
    "level": [
        "The camera gimbal is broken, so don't change altitude.",
        "Our altitude sensor is unreliable; hold whatever height you start at.",
        "Climbing drains the battery, and so does diving, so avoid both.",
        "Walk around obstacles like a ground robot would.",
        "Don't trade a short climb for a short cut.",
        "The passenger gets airsick if the drone goes up and down.",
        "Fly as if there were a glass ceiling and a glass floor at your current height.",
        "Neither ascending nor descending is welcome on this run.",
        "Altitude changes are the most expensive thing you can do.",
        "I don't care how far sideways you go, as long as the height stays the same.",
    ],
    "climb_open": [
        "The rotor wash bounces off walls when climbing, so climb away from them.",
        "Climbing in a gap once snapped a propeller; climb somewhere open.",
        "Get high first in the open, then go through the gaps.",
        "Ascending next to structure is banned, ascending in open air is fine.",
        "Climbing is allowed, but not when you are boxed in.",
        "Treat climbs near walls as dangerous.",
        "If there is no room around you, don't go up.",
        "Do the climbing part of the route somewhere with space.",
        "A climb inside a narrow opening is the one thing I don't want.",
        "Going up is fine as long as nothing is close by.",
    ],
    "descend_open": [
        "The rotor wash bounces off walls when descending, so descend away from them.",
        "Descending in a gap once snapped a propeller; descend somewhere open.",
        "Go through the gaps first, then come down once you're in the open.",
        "Descending next to structure is banned, descending in open air is fine.",
        "Descending is allowed, but not when you are boxed in.",
        "Treat descents near walls as dangerous.",
        "If there is no room around you, don't go down.",
        "Do the descending part of the route somewhere with space.",
        "A descent inside a narrow opening is the one thing I don't want.",
        "Going down is fine as long as nothing is close by.",
    ],
}

# ---------------------------------------------------------------------------------------
# Corpus selection. Default "v1" reproduces the paper exactly. Set IG3D_CORPUS=v2 to train
# and evaluate with the extra phrasings. The choice is stored in every checkpoint.
# ---------------------------------------------------------------------------------------

CORPUS = os.environ.get("IG3D_CORPUS", "v1").lower()
if CORPUS not in ("v1", "v2"):
    raise ValueError("IG3D_CORPUS must be v1 or v2, got {!r}".format(CORPUS))

N_CORE = 16 * len(CLASSES)

ALL_SENTENCES: list[str] = []
ALL_LABELS_LIST: list[int] = []
for _c in CLASSES:
    for _s in BY_CLASS[_c]:
        ALL_SENTENCES.append(_s)
        ALL_LABELS_LIST.append(CLASS_TO_IDX[_c])
if CORPUS == "v2":
    for _c in CLASSES:
        for _s in EXTRA[_c]:
            ALL_SENTENCES.append(_s)
            ALL_LABELS_LIST.append(CLASS_TO_IDX[_c])
ALL_LABELS = np.asarray(ALL_LABELS_LIST)

HARD_SENTENCES: list[str] = [s for c in CLASSES for s in HARD_TEST[c]]
HARD_LABELS = np.asarray([CLASS_TO_IDX[c] for c in CLASSES for _ in HARD_TEST[c]])

assert all(len(v) == 16 for v in BY_CLASS.values()), \
    {k: len(v) for k, v in BY_CLASS.items()}
assert all(len(v) == 32 for v in EXTRA.values()), {k: len(v) for k, v in EXTRA.items()}
assert all(len(v) == 10 for v in HARD_TEST.values()), {k: len(v) for k, v in HARD_TEST.items()}
_everything = [s for c in CLASSES for s in BY_CLASS[c] + EXTRA[c] + HARD_TEST[c]]
_norm = [s.lower().strip(" .") for s in _everything]
assert len(set(_norm)) == len(_norm), sorted(
    {s for s in _norm if _norm.count(s) > 1})
assert len(ALL_SENTENCES) == (N_CORE if CORPUS == "v1" else N_CORE + 32 * len(CLASSES))


def split_indices(n_heldout_per_class: int = 4, seed: int = 0):
    """Per-class train/held-out split over indices into ALL_SENTENCES.

    The held-out set is always drawn from the paper's 128 sentences with the paper's RNG,
    so it is identical under v1 and v2. Under v2 every extra phrasing joins TRAINING, which
    makes "does more training wording help on the same unseen wording?" a clean comparison.
    """
    rng = np.random.default_rng(seed)
    core = ALL_LABELS[:N_CORE]
    train, held = [], []
    for ci in range(len(CLASSES)):
        idx = rng.permutation(np.flatnonzero(core == ci))
        held.extend(idx[:n_heldout_per_class].tolist())
        train.extend(idx[n_heldout_per_class:].tolist())
    train.extend(range(N_CORE, len(ALL_SENTENCES)))
    return np.sort(np.asarray(train)), np.sort(np.asarray(held))


def classes_in_tier(tier: int) -> tuple[str, ...]:
    return tuple(c for c in CLASSES if TIER[c] == tier)
