"""Package the curated HU_D03 MJCF into the mjlab robot folder.

Copies referenced STL meshes into xmls/assets/, flattens mesh paths, sets
meshdir="assets", and strips the standalone <keyframe> (mjlab generates its own
keyframe from EntityCfg.InitialStateCfg).
"""

import pathlib
import re
import shutil

ROOT = pathlib.Path.home() / "ml" / "humanoid-description" / "HU_D03_description"
SRC_XML = ROOT / "hu_d03.xml"
PKG = (
    pathlib.Path.home()
    / "ml/mjlab/src/mjlab/asset_zoo/robots/limx_hu_d03/xmls"
)
ASSETS = PKG / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

text = SRC_XML.read_text()

# Copy every referenced mesh into assets/ (flattened to basename).
files = sorted(set(re.findall(r'file="([^"]+\.STL)"', text, flags=re.I)))
for f in files:
    src = ROOT / f
    assert src.exists(), f"missing mesh: {src}"
    shutil.copy(src, ASSETS / pathlib.Path(f).name)

# Rewrite mesh paths to basenames and meshdir to "assets".
text = re.sub(r'file="[^"]*?([^"/]+\.STL)"', r'file="\1"', text, flags=re.I)
text = re.sub(r'meshdir="[^"]*"', 'meshdir="assets"', text)

# Drop the standalone keyframe block (mjlab generates its own from init_state).
text = re.sub(r"\s*<keyframe>.*?</keyframe>", "", text, flags=re.S)

# Drop the test-only ground plane (the scene/terrain provides the ground).
text = re.sub(r'\s*<geom\b[^>]*\bname="floor"[^>]*/>', "", text)

(PKG / "hu_d03.xml").write_text(text)
print(f"[ok] wrote {PKG / 'hu_d03.xml'}")
print(f"[ok] copied {len(files)} meshes into {ASSETS}")
