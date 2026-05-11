import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# -----------------------------------
# LOAD DATA
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

path_df = pd.read_csv(BASE_DIR / "data" / "path_coefficients.csv")
r2_df = pd.read_csv(BASE_DIR / "data" / "r2_scores.csv")

# -----------------------------------
# SIGNIFICANCE FUNCTION
# -----------------------------------

def sig_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# -----------------------------------
# FIGURE SETUP
# -----------------------------------

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

fig.patch.set_facecolor('#0d1b2a')
ax.set_facecolor('#0d1b2a')

# -----------------------------------
# COLORS (VIBRANT)
# -----------------------------------

C_EXP = "#00b4d8"
C_REAL = "#4cc9f0"
C_ADAPT = "#2ecc71"
C_GAP = "#e63946"
C_SAT = "#f72585"
C_GROW = "#f77f00"

C_ARROW = "#a8dadc"

# -----------------------------------
# BOX FUNCTION
# -----------------------------------

def box(x, y, label, color):
    rect = mpatches.FancyBboxPatch(
        (x-1, y-0.4), 2, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5
    )
    ax.add_patch(rect)

    ax.text(x, y, label,
            ha='center', va='center',
            color='white',
            fontsize=10,
            fontweight='bold')

# -----------------------------------
# ARROW FUNCTION
# -----------------------------------

def arrow(src, tgt, beta, p, rad=0.0):

    x1, y1 = src
    x2, y2 = tgt

    ax.annotate('',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->',
            color=C_ARROW,
            lw=2,
            connectionstyle=f'arc3,rad={rad}'
        )
    )

    mx = (x1 + x2)/2
    my = (y1 + y2)/2

    star = sig_star(p)

    ax.text(mx, my,
        f"β={beta:.2f}{star}",
        color='white',
        fontsize=8,
        ha='center',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='#1b263b',
            edgecolor='white'
        )
    )

# -----------------------------------
# NODE POSITIONS
# -----------------------------------

pos = {
"Expectation": (2,10),
"Reality": (2,7),
"Adaptability": (2,4),

"Gap": (6,7),

"Satisfaction": (10,7),
"Growth": (16,7)
}

# -----------------------------------
# DRAW NODES
# -----------------------------------

box(*pos["Expectation"], "Expectation", C_EXP)
box(*pos["Reality"], "Reality", C_REAL)
box(*pos["Adaptability"], "Adaptability", C_ADAPT)
box(*pos["Gap"], "Gap", C_GAP)

box(*pos["Satisfaction"], "Satisfaction", C_SAT)
box(*pos["Growth"], "Growth", C_GROW)

# -----------------------------------
# DRAW PATHS (AUTO FROM CSV)
# -----------------------------------

for _, row in path_df.iterrows():

    path = row["Path"]
    beta = row["Beta"]
    p = row["p_value"]

    src, tgt = path.split(" → ")

    if src in pos and tgt in pos:
        arrow(pos[src], pos[tgt], beta, p, rad=0.2)

# -----------------------------------
# R² VALUES
# -----------------------------------

r2_dict = dict(zip(r2_df["Construct"], r2_df["R2"]))

ax.text(10,5.5,
    f"R² = {round(r2_dict.get('Satisfaction',0),3)}",
    color="white", ha="center", fontsize=11)

ax.text(16,5.5,
    f"R² = {round(r2_dict.get('Growth',0),3)}",
    color="white", ha="center", fontsize=11)

# -----------------------------------
# TITLE + LEGEND
# -----------------------------------

plt.title("SEM Path Diagram (Behavioral Transition)",
          fontsize=16,
          color='white',
          fontweight='bold')

legend = [
    mpatches.Patch(color=C_ARROW, label="Structural Paths")
]

ax.legend(handles=legend,
          loc='lower left',
          facecolor='#1b263b',
          labelcolor='white')

# -----------------------------------
# SAVE
# -----------------------------------

plt.savefig(BASE_DIR / "data" / "sem_model_pro.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

print("✅ Professional SEM Diagram Generated")