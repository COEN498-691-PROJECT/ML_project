import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================
# 1. Folder path
folder_path = r"C:\Users\mahsa\Downloads\EmotiBit_PARSED (1)\EmotiBit_PARSED"

# Find all Excel and CSV files
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
files = excel_files + csv_files

if not files:
    print("No Excel or CSV files found.")
    exit()

# Create a folder to save results
output_folder = os.path.join(folder_path, "EDA_Results")
os.makedirs(output_folder, exist_ok=True)

# =======================================
# 2. Loop through each file (per participant)
for file in files:
    participant_name = os.path.splitext(os.path.basename(file))[0]
    print(f"\nProcessing {participant_name} ...")

    # Read file
    ext = os.path.splitext(file)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(file)
    elif ext == ".csv":
        df = pd.read_csv(file)
    else:
        print(f"Skipping {file} (unknown format)")
        continue

    # 3. Data cleaning
    df = df.drop_duplicates()
    print(f"   Shape after removing duplicates: {df.shape}")

    # 4. Select numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("   No numeric columns found, skipping file.")
        continue

    # 5. Descriptive statistics
    summary = numeric_df.describe()
    summary_file = os.path.join(output_folder, f"{participant_name}_summary.csv")
    summary.to_csv(summary_file)
    print(f"   Summary saved: {summary_file}")

    # 6. Histogram for numeric columns
    sample_cols = numeric_df.columns[:8]  # only first 8 columns for clarity
    numeric_df[sample_cols].hist(figsize=(12, 10), bins=20)
    plt.suptitle(f"Distribution of Numeric Columns - {participant_name}", fontsize=14)
    plt.tight_layout()
    hist_path = os.path.join(output_folder, f"{participant_name}_hist.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"   Histogram saved: {hist_path}")

    # 7. Correlation Heatmap
    corr = numeric_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Heatmap - {participant_name}", fontsize=14)
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, f"{participant_name}_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"   Heatmap saved: {heatmap_path}")

print("\nEDA completed for all participants! Results saved in:", output_folder)
