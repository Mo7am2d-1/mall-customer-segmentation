"""Mall customer segmentation with K-Means clustering.

Dataset: Mall_Customers.csv (200 customers: age, annual income, spending score).
We use the elbow method to pick a cluster count, then cluster customers on
(annual income, spending score) and visualize the segments.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the data
df = pd.read_csv("Mall_Customers.csv")
print(f"Loaded {len(df)} customers, columns: {list(df.columns)}")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 2. Elbow method: try k = 1..10 and look at within-cluster variance (inertia)
inertias = []
ks = range(1, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ks, inertias, marker="o")
axes[0].set_xlabel("number of clusters k")
axes[0].set_ylabel("inertia (within-cluster sum of squares)")
axes[0].set_title("Elbow method")

# 3. The elbow is around k=5, so fit the final model with 5 clusters
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

for c in range(K):
    pts = df[df["cluster"] == c]
    axes[1].scatter(pts["Annual Income (k$)"], pts["Spending Score (1-100)"],
                    s=40, label=f"cluster {c} (n={len(pts)})")
centers = kmeans.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], s=200, c="black", marker="X", label="centers")
axes[1].set_xlabel("Annual Income (k$)")
axes[1].set_ylabel("Spending Score (1-100)")
axes[1].set_title(f"Customer segments (k={K})")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results.svg", dpi=120)
print("Saved results.svg\n")

# 4. Describe each segment
print(df.groupby("cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1))
