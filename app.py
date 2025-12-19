# app.py
import io
import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths (NEW STRUCTURE)
# =========================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PIC_DIR  = APP_DIR / "pictures"

PRICES_FILE = DATA_DIR / "paper_prices.csv"
AVAIL_FILE  = DATA_DIR / "paper_availability.csv"

BOX_FILES = {
    ("EXT", "CM"): DATA_DIR / "box_ext_cm.csv",
    ("EXT", "IN"): DATA_DIR / "box_ext_in.csv",
    ("INT", "CM"): DATA_DIR / "box_int_cm.csv",
    ("INT", "IN"): DATA_DIR / "box_int_in.csv",
}

LOGO_PATH = PIC_DIR / "logo.png"

# =========================
# Ratios / Settings
# =========================
FLUTE_RATIOS = {"C": 1.45, "B": 1.35, "E": 1.26}
FLUTE_HEIGHT = {"E": 0.22, "B": 0.35, "C": 0.50}
LOSS_PRESETS = [5.0, 8.5, 10.0]

# =========================
# Safe formula evaluator (for box csv expressions)
# =========================
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub, ast.UAdd, ast.Load, ast.Name, ast.Mod
)

def safe_eval(expr: str, names: dict) -> float:
    """
    Evaluate a simple math expression safely.
    Allowed: numbers, + - * / ** ( ) and variable names in `names`.
    """
    if expr is None:
        raise ValueError("Empty expression")
    expr = str(expr).strip()
    if expr.startswith("="):
        expr = expr[1:].strip()

    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsafe expression part: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"Unknown variable: {node.id}")

    return float(eval(compile(tree, "<safe_eval>", "eval"), {"__builtins__": {}}, names))

# =========================
# Data loaders
# =========================
@st.cache_data
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE)
    df.columns = [c.strip() for c in df.columns]
    df["PaperType"] = df["PaperType"].astype(str).str.strip()
    df["AvgPricePerKg"] = pd.to_numeric(df["AvgPricePerKg"], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_availability() -> pd.DataFrame:
    df = pd.read_csv(AVAIL_FILE)
    df.columns = [c.strip() for c in df.columns]
    df["PaperType"] = df["PaperType"].astype(str).str.strip()
    df["GSM"] = pd.to_numeric(df["GSM"], errors="coerce")
    df = df.dropna(subset=["GSM"])
    df["GSM"] = df["GSM"].astype(int)
    return df

def paper_to_gsms(avail_df: pd.DataFrame) -> dict:
    return {p: sorted(g["GSM"].unique().tolist()) for p, g in avail_df.groupby("PaperType")}

@st.cache_data
def load_box_table(size_type: str, unit: str) -> pd.DataFrame:
    """
    Loads one of the 4 box CSVs.
    Must contain at least: [BoxType/Description, SheetLenExpr, SheetWidExpr]
    """
    path = BOX_FILES[(size_type, unit)]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    col_upper = {c: c.upper() for c in df.columns}

    # Find desc
    desc_col = None
    for c in df.columns:
        if col_upper[c] in ["DESCRIPTION", "DESC", "BOXTYPE", "BOX TYPE"]:
            desc_col = c
            break
    if desc_col is None:
        desc_col = df.columns[0]

    # Assume next 2 columns = len and width expressions
    if len(df.columns) < 3:
        raise ValueError(f"{path.name} must have at least 3 columns: desc, len_expr, wid_expr")

    if df.columns[0] == desc_col:
        len_col, wid_col = df.columns[1], df.columns[2]
    else:
        candidates = [c for c in df.columns if c != desc_col]
        if len(candidates) < 2:
            raise ValueError(f"{path.name} must have at least 3 columns: desc, len_expr, wid_expr")
        len_col, wid_col = candidates[0], candidates[1]

    out = df[[desc_col, len_col, wid_col]].copy()
    out.columns = ["BoxType", "SheetLenExpr", "SheetWidExpr"]

    out["BoxType"] = out["BoxType"].astype(str).str.strip()
    out["SheetLenExpr"] = out["SheetLenExpr"].astype(str).str.strip()
    out["SheetWidExpr"] = out["SheetWidExpr"].astype(str).str.strip()
    out = out.dropna(subset=["BoxType"]).reset_index(drop=True)
    return out

# =========================
# Diagram
# =========================
def draw_diagram(layer_names, layer_papers, layer_gsms, flute_types):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x0, x1 = 0.08, 0.92
    y = 0.12
    liner_thk = 0.10
    gap = 0.05

    def rect(y0, h, txt):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, h, fill=False, linewidth=2))
        ax.text((x0 + x1) / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=10)

    def flute(y0, h, txt, ft):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, h, fill=False, linewidth=2))
        xs = np.linspace(x0 + 0.01, x1 - 0.01, 800)
        amp = h * 0.28
        cycles = 18 if ft == "E" else 14 if ft == "B" else 10
        ys = y0 + h / 2 + amp * np.sin(2 * np.pi * cycles * (xs - x0) / (x1 - x0))
        ax.plot(xs, ys, linewidth=1.5)
        ax.text((x0 + x1) / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=10)

    fi = 0
    for name, p, gsm in zip(layer_names, layer_papers, layer_gsms):
        label = f"{name}: {p} | {gsm} GSM"
        if "Fluting" in name:
            ft = flute_types[fi]
            h = FLUTE_HEIGHT.get(ft, 0.35)
            flute(y, h, f"{label} ({ft})", ft)
            y += h + gap
            fi += 1
        else:
            rect(y, liner_thk, label)
            y += liner_thk + gap

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(1.0, y + 0.05))
    ax.axis("off")
    return fig

# =========================
# UI
# =========================
st.set_page_config(page_title="Paper Combination Cost", layout="wide")

# Centered header + logo
if LOGO_PATH.exists():
    colA, colB, colC = st.columns([1, 1, 1])
    with colB:
        st.image(str(LOGO_PATH), width=120)
        st.markdown("<h1 style='text-align:center; margin-top:0;'>Paper Combination Cost</h1>", unsafe_allow_html=True)
else:
    st.title("Paper Combination Cost")

prices_df = load_prices()
avail_df = load_availability()

papers = sorted(prices_df["PaperType"].unique().tolist())
price_map = dict(zip(prices_df["PaperType"], prices_df["AvgPricePerKg"]))
p2g = paper_to_gsms(avail_df)

# ✅ ONLY CM and TL for fluting paper dropdowns
flute_papers = [p for p in papers if str(p).strip().upper() in ["CM", "TL"]]
if not flute_papers:
    flute_papers = papers  # fallback if CM/TL not found

left, right = st.columns([1, 1.45])

# -------------------------
# LEFT PANEL
# -------------------------
with left:
    with st.expander("Board Setup", expanded=True):
        ply = st.radio("Ply", [3, 5], horizontal=True)

        loss_mode = st.radio("Normal Loss", ["Preset", "Custom"], horizontal=True)
        if loss_mode == "Preset":
            loss_pct = st.selectbox("Loss (%)", LOSS_PRESETS, index=1)
        else:
            loss_pct = st.number_input("Normal Loss (%)", min_value=0.0, step=0.5, value=10.0)

        loss_factor = 1 + (loss_pct / 100.0)
        st.caption(f"Loss factor = {loss_factor:.3f}")

    # ✅ Streamlined Layer Selection (row layout)
    with st.expander("Combination", expanded=True):

        # Row 1: Top Liner
        r1c1, r1c2 = st.columns([1.6, 1.0])
        with r1c1:
            top_paper = st.selectbox("Top Liner (Paper)", papers, key="top_paper")
        with r1c2:
            top_gsm = st.selectbox("Top Liner (GSM)", p2g.get(top_paper, [0]), key="top_gsm")

        # Row 2: Fluting 1  ✅ (CM/TL only)
        r2c1, r2c2, r2c3 = st.columns([1.6, 1.0, 0.9])
        with r2c1:
            flut1_paper = st.selectbox("Fluting 1 (Paper)", flute_papers, key="flut1_paper")
        with r2c2:
            flut1_gsm = st.selectbox("Fluting 1 (GSM)", p2g.get(flut1_paper, [0]), key="flut1_gsm")
        with r2c3:
            flut1_type = st.selectbox("Flute", ["C", "B", "E"], key="flut1_type")

        # Defaults
        mid_paper = mid_gsm = flut2_paper = flut2_gsm = flut2_type = None

        if ply == 5:
            st.divider()

            # Row 3: Mid Liner
            r3c1, r3c2 = st.columns([1.6, 1.0])
            with r3c1:
                mid_paper = st.selectbox("Mid Liner (Paper)", papers, key="mid_paper")
            with r3c2:
                mid_gsm = st.selectbox("Mid Liner (GSM)", p2g.get(mid_paper, [0]), key="mid_gsm")

            # Row 4: Fluting 2 ✅ (CM/TL only)
            r4c1, r4c2, r4c3 = st.columns([1.6, 1.0, 0.9])
            with r4c1:
                flut2_paper = st.selectbox("Fluting 2 (Paper)", flute_papers, key="flut2_paper")
            with r4c2:
                flut2_gsm = st.selectbox("Fluting 2 (GSM)", p2g.get(flut2_paper, [0]), key="flut2_gsm")
            with r4c3:
                flut2_type = st.selectbox("Flute", ["C", "B", "E"], key="flut2_type")

        st.divider()

        # Row 5: Bottom Liner
        r5c1, r5c2 = st.columns([1.6, 1.0])
        with r5c1:
            bot_paper = st.selectbox("Bottom Liner (Paper)", papers, key="bot_paper")
        with r5c2:
            bot_gsm = st.selectbox("Bottom Liner (GSM)", p2g.get(bot_paper, [0]), key="bot_gsm")

    # ✅ Box Type → Board Size (row layout like Le/Br/He)
    with st.expander("Box Measurements", expanded=True):

        r1c1, r1c2, r1c3 = st.columns([1.2, 1.0, 1.6])
        with r1c1:
            size_type = st.selectbox("Dimensions", ["EXT", "INT"], index=0)
        with r1c2:
            unit = st.selectbox("Unit", ["CM", "IN"], index=0)

        try:
            box_df = load_box_table(size_type, unit)
            box_list = box_df["BoxType"].dropna().astype(str).tolist()
            if len(box_list) == 0:
                raise ValueError("No BoxType rows found in the selected box csv.")
        except Exception as e:
            st.error(f"Box CSV issue: {e}")
            box_df = None
            box_list = []

        with r1c3:
            if box_df is None or len(box_list) == 0:
                box_type = None
                st.selectbox("Box Type", ["(no box types)"], disabled=True)
                sheet_len_expr = None
                sheet_wid_expr = None
            else:
                box_type = st.selectbox("Box Type", box_list, index=0)
                row = box_df.loc[box_df["BoxType"] == box_type].iloc[0]
                sheet_len_expr = row["SheetLenExpr"]
                sheet_wid_expr = row["SheetWidExpr"]

        cA, cB, cC = st.columns(3)
        if unit == "CM":
            Le = cA.number_input("Le (cm)", min_value=0.0, value=20.0, step=0.5)
            Br = cB.number_input("Br (cm)", min_value=0.0, value=15.0, step=0.5)
            He = cC.number_input("He (cm)", min_value=0.0, value=10.0, step=0.5)
        else:
            Le = cA.number_input("Le (inch)", min_value=0.0, value=8.0, step=0.25)
            Br = cB.number_input("Br (inch)", min_value=0.0, value=6.0, step=0.25)
            He = cC.number_input("He (inch)", min_value=0.0, value=4.0, step=0.25)

# -------------------------
# Build layer lists
# -------------------------
if ply == 3:
    layer_names = ["Top Liner", "Fluting 1", "Bottom Liner"]
    layer_papers = [top_paper, flut1_paper, bot_paper]
    base_gsms = [float(top_gsm), float(flut1_gsm), float(bot_gsm)]
    flute_types = [flut1_type]
else:
    layer_names = ["Top Liner", "Fluting 1", "Mid Liner", "Fluting 2", "Bottom Liner"]
    layer_papers = [top_paper, flut1_paper, mid_paper, flut2_paper, bot_paper]
    base_gsms = [float(top_gsm), float(flut1_gsm), float(mid_gsm), float(flut2_gsm), float(bot_gsm)]
    flute_types = [flut1_type, flut2_type]

# Effective GSM
effective_gsms = []
fi = 0
for name, gsm in zip(layer_names, base_gsms):
    if "Fluting" in name:
        ft = flute_types[fi]
        effective_gsms.append(gsm * FLUTE_RATIOS.get(ft, 1.0))
        fi += 1
    else:
        effective_gsms.append(gsm)

total_gsm = float(sum(effective_gsms))
total_gsm_loss = float(sum(g * loss_factor for g in effective_gsms))

# Mix average price/kg (before loss)
material_cost_index = float(sum(g * price_map.get(p, 0.0) for g, p in zip(effective_gsms, layer_papers)))
mix_avg_price_per_kg = (material_cost_index / total_gsm) if total_gsm > 0 else 0.0

# Board size calculation
board_len_m = board_wid_m = area_m2 = weight_kg = cost_per_box = 0.0
eval_error = None

if sheet_len_expr and sheet_wid_expr:
    try:
        names = {"Le": float(Le), "Br": float(Br), "He": float(He)}
        sheet_len = safe_eval(sheet_len_expr, names)
        sheet_wid = safe_eval(sheet_wid_expr, names)

        if unit == "CM":
            board_len_m = sheet_len / 100.0
            board_wid_m = sheet_wid / 100.0
        else:
            board_len_m = sheet_len * 0.0254
            board_wid_m = sheet_wid * 0.0254

        area_m2 = board_len_m * board_wid_m
        weight_kg = (total_gsm_loss * area_m2) / 1000.0
        cost_per_box = weight_kg * mix_avg_price_per_kg

    except Exception as e:
        eval_error = str(e)

# -------------------------
# RIGHT PANEL
# -------------------------
with right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total GSM (before N/L)", f"{total_gsm:.2f}")
    m2.metric("Total GSM (with N/L)", f"{total_gsm_loss:.2f}")
    m3.metric("Mix Avg Price / KG", f"{mix_avg_price_per_kg:.2f}")
    m4.metric("Cost per Box / Board (paper)", f"{cost_per_box:.2f}")

    st.markdown("## Board Measurements")
    if eval_error:
        st.error(f"Could not calculate board size: {eval_error}")
        st.info("Check your box CSV expressions and variables (Le, Br, He).")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Board Width (m)", f"{board_wid_m:.3f}")
        b2.metric("Board Length (m)", f"{board_len_m:.3f}")
        b3.metric("Area (m²)", f"{area_m2:.3f}")
        b4.metric("Weight per Box / Board (kg)", f"{weight_kg:.4f}")

    st.markdown("## Diagram")
    fig = draw_diagram(layer_names, layer_papers, [int(x) for x in base_gsms], flute_types)
    st.pyplot(fig, clear_figure=True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download Diagram (PNG)", buf.getvalue(), "board_diagram.png", "image/png")

    with st.expander("Details (optional)", expanded=False):
        details_df = pd.DataFrame({
            "Layer": layer_names,
            "Paper": layer_papers,
            "Base GSM": base_gsms,
            "Effective GSM": np.round(effective_gsms, 3),
            "Price/KG": [price_map.get(p, 0.0) for p in layer_papers],
        })
        st.dataframe(details_df, use_container_width=True)

        if sheet_len_expr and sheet_wid_expr:
            st.markdown("### Box formula used")
            st.write(f"**Box Type:** {box_type}  ({size_type} / {unit})")
            st.code(f"Sheet Length Expr: {sheet_len_expr}\nSheet Width  Expr: {sheet_wid_expr}")
