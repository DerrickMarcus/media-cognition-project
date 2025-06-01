import matplotlib.pyplot as plt
import pandas as pd

# Create the DataFrame with the provided data
data = {
    "Experiment": [
        "Exp 1",
        "Exp 2",
        "Exp 3",
        "Exp 4",
        "Exp 5",
        "Exp 6",
        "Exp 7",
        "Exp 8",
    ],
    "T2I_top1": [29.63, 31.56, 26.74, 8.87, 30.99, 31.59, 44.49, 49.23],
    "T2I_top5": [54.84, 57.02, 51.11, 22.27, 56.50, 59.12, 76.50, 80.65],
    "T2I_top10": [64.16, 66.78, 61.52, 30.75, 65.52, 69.15, 83.91, 87.64],
    "I2T_top1": [30.00, 33.19, 27.09, 9.07, 31.51, 32.40, 46.56, 49.48],
    "I2T_top5": [53.63, 57.02, 51.26, 23.04, 56.62, 59.24, 76.57, 80.82],
    "I2T_top10": [62.83, 65.72, 60.68, 31.31, 65.69, 68.76, 84.28, 87.77],
}

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))

# Plot Text to Image (T2I) Recall
plt.subplot(1, 2, 1)
x = range(len(df))
width = 0.25
plt.bar([i - width for i in x], df["T2I_top1"], width=width, label="Top-1")
plt.bar(x, df["T2I_top5"], width=width, label="Top-5")
plt.bar([i + width for i in x], df["T2I_top10"], width=width, label="Top-10")
plt.xticks(x, df["Experiment"], rotation=45, ha="right", fontsize=8)
plt.title("Text → Image Recall")
plt.ylabel("Recall (%)")
plt.legend()
plt.tight_layout()

# Plot Image to Text (I2T) Recall
plt.subplot(1, 2, 2)
plt.bar([i - width for i in x], df["I2T_top1"], width=width, label="Top-1")
plt.bar(x, df["I2T_top5"], width=width, label="Top-5")
plt.bar([i + width for i in x], df["I2T_top10"], width=width, label="Top-10")
plt.xticks(x, df["Experiment"], rotation=45, ha="right", fontsize=8)
plt.title("Image → Text Recall")
plt.ylabel("Recall (%)")
plt.legend()
plt.tight_layout()
plt.savefig("exp/images/result.png", dpi=300)
plt.show()
