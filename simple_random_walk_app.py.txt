import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ---- Page configuration ----
st.set_page_config(page_title="Gambler's Fortune", layout="wide")

st.title("Gambler's Fortune Simulation")

st.markdown(
    "Use the sliders to choose the parameters, then click "
    "**Run simulation** to see the evolution of the gambler's fortune."
)

# ---- Sliders ----
p_selected = st.slider(
    "Probability of HEADS (p)",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
    format="%.2f",
)

S_0 = st.slider(
    "Initial fortune (S₀)",
    min_value=1,
    max_value=100,
    value=50,
    step=1,
)

M = st.slider(
    "Number of tosses (M)",
    min_value=5,
    max_value=100,
    value=50,
    step=5,
)

# Fixed probabilities for comparison
p_list = [p_selected, 0.25, 0.75]
p_labels = [
    f"p = {p_selected:.2f} (selected)",
    "p = 0.25",
    "p = 0.75",
]
colors = ["tab:blue", "tab:orange", "tab:green"]


def simulate_gambler(S0, M, p):
    """Simulate the gambler's fortune for M tosses with probability p."""
    gameHT = np.random.binomial(1, p, M)  # 0/1
    game_money = gameHT * 2 - 1           # -1/+1
    fortune = np.zeros(M + 1)
    fortune[0] = S0
    for i in range(M):
        fortune[i + 1] = fortune[i] + game_money[i]
    return gameHT, fortune


# Placeholders for animation and text
plot_placeholder = st.empty()
text_placeholder = st.empty()

if st.button("Run simulation"):
    # ---- Simulate all three processes once ----
    fortunes = []
    for p in p_list:
        _, fortune = simulate_gambler(S_0, M, p)
        fortunes.append(fortune)

    # Time axis (0..M)
    tosses = np.arange(0, M + 1)

    # ---- Animation loop ----
    for step in range(1, M + 1):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

        for ax, fortune, label, color in zip(axes, fortunes, p_labels, colors):
            # Plot fortune path up to current step
            ax.plot(
                tosses[: step + 1],
                fortune[: step + 1],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=4,
                color=color,
                label=label,
            )
            ax.set_title(label)
            ax.set_xlabel("Number of tosses")
            ax.grid(True)

        axes[0].set_ylabel("Gambler's fortune")
        fig.suptitle("Evolution of the gambler's fortune", fontsize=14)
        fig.tight_layout()

        plot_placeholder.pyplot(fig)
        time.sleep(0.05)  # controls animation speed

    # ---- Text under the plots (after animation finishes) ----
    final_values = [fortune[-1] for fortune in fortunes]

    text_placeholder.markdown(
        f"""
        **Simulation parameters**  

        - **S₀** = {S_0} – Initial state (starting fortune)  
        - **M** = {M} – Number of tosses in the game  

        **Final fortune values:**  
        - For {p_labels[0]}: **{final_values[0]:.0f}**  
        - For {p_labels[1]}: **{final_values[1]:.0f}**  
        - For {p_labels[2]}: **{final_values[2]:.0f}**
        """
    )