import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DEFINE YOUR LABELS AND DATA ---
# Define the models, datasets, and techniques
models = ["Phi-3", "Phi-4", "Qwen2.5"]
datasets = ["CoderEval", "BigCodeBench"]
techniques = [
    "Zero Shot",
    "Quality Focused",
    "Persona Based",
    "Chain-of-Thought",
    "RCI",
]

# Corrected Data Values
values = [
    # Phi-3-mini CoderEval (5 values)
    1.70,
    1.41,
    1.65,
    1.62,
    1.88,
    # Phi-3-mini BigCodeBench (5 values)
    2.34,
    2.51,
    2.72,
    2.36,
    2.917,
    # Phi-4 CoderEval (5 values)
    1.18,
    1.10,
    1.18,
    1.18,
    1.52,
    # Phi-4 BigCodeBench (5 values)
    1.93,
    1.99,
    2.02,
    2.07,
    2.47,
    # Qwen2.5-Coder CoderEval (5 values)
    1.74,
    1.00,
    1.52,
    1.09,
    1.61,
    # Qwen2.5-Coder BigCodeBench (5 values)
    1.97,
    1.92,
    2.03,
    1.83,
    2.71,
]

# Create the data structure for the DataFrame
data_list = []
value_index = 0
for model in models:
    for dataset in datasets:
        for technique in techniques:
            row_label = f"{model} {dataset}"
            data_list.append(
                {
                    "model_dataset": row_label,
                    "technique": technique,
                    "mean_total_smells": values[value_index],
                }
            )
            value_index += 1

df = pd.DataFrame(data_list)

# --- 2. PIVOT AND ORGANIZE DATA ---
# Pivot the DataFrame to create a matrix
heatmap_data = df.pivot(
    index="model_dataset", columns="technique", values="mean_total_smells"
)

# Ensure the rows and columns are in the desired order
row_order = [f"{m} {d}" for m in models for d in datasets]
heatmap_data = heatmap_data.reindex(index=row_order, columns=techniques)

# --- 3. CREATE THE PLOT ---
plt.figure(figsize=(10, 9))  # Increased size for better readability with more rows
ax = sns.heatmap(
    heatmap_data,
    annot=True,  # Show numbers in the cells
    fmt=".2f",  # Format numbers to 2 decimal places
    cmap="plasma",  # A vibrant, presentation-friendly colormap
    linewidths=1.0,  # Add lines between cells
    linecolor="white",
    cbar=False,  # Color bar removed here
    annot_kws={"size": 14},  # Font size for the numbers
)

# Remove the x and y labels
ax.set_xlabel("")
ax.set_ylabel("")

plt.xticks(
    fontsize=14, rotation=45, ha="right", fontweight="bold"
)  # Rotate x-labels for better fit
plt.yticks(fontsize=14, rotation=0, fontweight="bold")

plt.tight_layout()  # Adjust plot to prevent labels from overlapping

# --- 5. SAVE AND SHOW THE PLOT ---
plt.savefig("updated_heatmap.png", dpi=300)
plt.show()
