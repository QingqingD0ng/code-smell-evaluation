import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
# The data from your table.
# Columns correspond to models, rows correspond to methods.
performance_data = np.array(
    [
        [14.86, 27.70, 29.05],  # Zero Shot
        [18.92, 25.68, 29.05],  # Quality Focused
        [10.81, 24.32, 33.11],  # Persona Based
        [11.49, 31.76, 31.08],  # CoT
        [4.73, 20.95, 21.62],  # RCI
    ]
)

# Define the category labels
model_names = ["Phi-3-mini", "Phi-4", "Qwen2.5-Cooder"]
method_names = ["Zero Shot", "Quality Focused", "Persona Based", "CoT", "RCI"]

# --- Chart Generation ---
x = np.arange(len(model_names))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 7))

# Loop through the data to plot each method's bars for all models
for i, method_data in enumerate(performance_data):
    offset = width * multiplier
    rects = ax.bar(x + offset, method_data, width, label=method_names[i])
    ax.bar_label(rects, padding=3, fmt="%.2f%%", fontsize=8)
    multiplier += 1

# --- Customization and Styling ---
# Add a descriptive title and axis labels
ax.set_ylabel("Performance Score (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12)

# Set the x-axis tick labels to be the model names, centered
ax.set_xticks(x + width * (len(method_names) - 1) / 2)
ax.set_xticklabels(model_names)

# Add a legend to identify the methods
ax.legend(loc="upper left", ncols=1)

# Set y-axis limits to give space for labels on top of bars
ax.set_ylim(0, max(performance_data.flatten()) * 1.2)

# Improve readability
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#DDDDDD")

ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)

fig.tight_layout()

# --- Display the Chart ---
plt.show()

print("Generated Grouped Bar Chart using Python.")
