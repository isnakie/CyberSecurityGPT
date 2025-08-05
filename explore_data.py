import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load data
df = pd.read_csv(r"data/processed/Larger_Group_Labels.csv")

# Basic info
print("First few rows:\n", df.head())
print("\nClass distribution:\n", df['label'].value_counts())

# Plot label distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title("Label Distribution")
plt.tight_layout()
plt.show()

