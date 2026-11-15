"""Region definitions on the 14x14 ViT patch grid. SK_1 (south_korea) and SK_2
(south_korea_wide) are two distinct, explicitly named South Korea regions."""
import torch

from .config import device


def get_bbox_from_indices(indices_list, grid_size=14, patch_size=16):
    rows = [idx // grid_size for idx in indices_list]
    cols = [idx % grid_size for idx in indices_list]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    x, y = min_col * patch_size, min_row * patch_size
    w = (max_col - min_col + 1) * patch_size
    h = (max_row - min_row + 1) * patch_size
    return (x, y, w, h)


# --- SK_1: 3x3-patch-block South Korea crop, +14 offset ---
# Default South Korea region: training-time preview and MSE/retrieval evaluation.
SK_1_BASE = [62, 63, 64, 76, 77, 78, 90, 91, 92, 104, 105, 106]
SK_1_OFFSET = 14
SK_1_INDICES = [i + SK_1_OFFSET for i in SK_1_BASE]

# --- SK_2: 4x4-patch-block South Korea crop, +14 offset ---
# The +14 offset centers the block on Korea like SK_1 (checked by overlaying on
# the temporal-mean coastline image); a +41 offset would place it mostly over
# open ocean south of the peninsula.
SK_2_BASE = [62, 63, 64, 65, 76, 77, 78, 79, 90, 91, 92, 93, 104, 105, 106, 107]
SK_2_OFFSET = 14
SK_2_INDICES = [i + SK_2_OFFSET for i in SK_2_BASE]

LOW_LEFT_BASE = [126, 127, 128, 129, 130, 140, 141, 142, 143, 144,
                  154, 155, 156, 157, 158, 168, 169, 170, 171, 172,
                  182, 183, 184, 185, 186]
LOW_LEFT_OFFSET = 4
LOW_LEFT_INDICES = [i + LOW_LEFT_OFFSET for i in LOW_LEFT_BASE]

TOTAL_INDICES = list(range(196))

# Boxes are computed from the offset indices, i.e. the same patches the
# embeddings are indexed with. Computing them from the un-offset BASE lists
# would draw/crop a region shifted off the patches actually searched.
REGIONS = {
    "south_korea":       {"indices": SK_1_INDICES,   "box": get_bbox_from_indices(SK_1_INDICES),   "color": "red"},
    "south_korea_wide":  {"indices": SK_2_INDICES,   "box": get_bbox_from_indices(SK_2_INDICES),   "color": "orange"},
    "low_left":          {"indices": LOW_LEFT_INDICES, "box": get_bbox_from_indices(LOW_LEFT_INDICES), "color": "cyan"},
    "total_patches":     {"indices": TOTAL_INDICES,  "box": (0, 0, 224, 224), "color": "white"},
}


def region_tensor(region_name, device=device):
    return torch.tensor(REGIONS[region_name]["indices"], device=device, dtype=torch.long)


REGIONS_TO_EVAL = ["south_korea", "south_korea_wide", "low_left", "total_patches"]


if __name__ == "__main__":
    for name, r in REGIONS.items():
        print(f"{name:18s} | {len(r['indices']):3d} patches | box={r['box']}")
