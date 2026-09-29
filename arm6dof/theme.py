"""Design tokens for the arm6dof GUI.

Ported from the user's `design-scheme` project — the **Night register** of
"La Pagina Barbara" (see design-scheme/STYLE.md + web/style.css). Black ground,
steel-grey italic Cormorant Garamond serif, hairline rules, square corners,
bracket corners on boxes; no fills, gradients or shadows.

Status colours borrow the design scheme's *Scan* register signals (acid lime /
scan red), which read as HUD state — appropriate for a machine-vision tool.

This module is the single source of styling truth: change a colour here and the
whole GUI follows.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- palette (Night register) -------------------------------------------
INK = "#000000"          # ground
CHAR = "#1D1D1B"         # charcoal panel ground
GREY = "#848482"         # BARBARA grey — primary type/line
FG_DIM = "#5C5C5A"       # secondary type
FG_HI = "#B9B9B6"        # hover / emphasis
HAIR = "#2A2A28"         # hairline rules

# ---- status (Scan register signals) --------------------------------------
LIME = "#C8F53C"         # PASS — acid lime accent
RED = "#FF2B1C"          # FAIL — scan red

# ---- geometry ------------------------------------------------------------
U = 8                    # 8 px grid base
STROKE = 1               # hairline; corners always 0 radius

# ---- typography ----------------------------------------------------------
# Cormorant Garamond Italic ships in the design-scheme repo; register it with
# matplotlib and reference the family by name in Tk (falls back to Georgia).
_FONT_DIR = (Path(os.environ.get("USERPROFILE", Path.home()))
             / "OneDrive - amazon.com" / "Desktop" / "1 - projects"
             / "design-scheme" / "web" / "fonts")
CORMORANT_TTF = _FONT_DIR / "CormorantGaramond-Italic.ttf"

SERIF = ("Cormorant Garamond", "Georgia", "serif")
MONO = ("Consolas", "Courier New", "monospace")   # data/HUD register


def register_matplotlib_font() -> str:
    """Register Cormorant Garamond with matplotlib; return the family to use.

    Falls back to a serif already present if the ttf isn't found, so the GUI
    still runs on a machine without the design-scheme repo checked out.
    """
    try:
        from matplotlib import font_manager
        if CORMORANT_TTF.exists():
            font_manager.fontManager.addfont(str(CORMORANT_TTF))
            return font_manager.FontProperties(
                fname=str(CORMORANT_TTF)).get_name()
    except Exception:
        pass
    return "Georgia"
