# -*- coding: utf-8 -*-
# ======================================================================
# Análisis de clustering con K-Means para el escenario de 15 gimnasios
# Requisitos: pandas, numpy, matplotlib, seaborn, scikit-learn
# Uso:
# 1) Coloca este archivo en la misma carpeta que 'escenario_gimnasios_15.csv'
# 2) Ejecuta: python analisis_gimnasios_kmeans.py
# 3) Se guardarán imágenes 
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------------------------------------------
# 1) Cargar datos
# -------------------------------------------------------------
df = pd.read_csv("escenario_gimnasios_15.csv", encoding="utf-8-sig")
print("Forma (filas, columnas):", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores faltantes por columna:\n", df.isna().sum())

# -------------------------------------------------------------
# 2) Estadística descriptiva
# -------------------------------------------------------------
desc = df.select_dtypes("number").describe()
print("\nDescripción numérica:\n", desc)

var1 = "Ingresos_Mensuales"
var2 = "Precio_Membresia"

print("\n=== Indicadores de tendencia central ===")
print(f"{var1} -> media: {df[var1].mean(skipna=True):.2f}, "
      f"mediana: {df[var1].median(skipna=True):.2f}, "
      f"moda: {df[var1].mode(dropna=True).iloc[0] if not df[var1].mode(dropna=True).empty else None}")
print(f"{var2} -> media: {df[var2].mean(skipna=True):.2f}, "
      f"mediana: {df[var2].median(skipna=True):.2f}, "
      f"moda: {df[var2].mode(dropna=True).iloc[0] if not df[var2].mode(dropna=True).empty else None}")

# -------------------------------------------------------------
# 3) Visualización inicial
# -------------------------------------------------------------
plt.figure()
df[var1].plot(kind="box", title=f"Boxplot - {var1}")
plt.tight_layout()
plt.savefig("boxplot_ingresos.png", dpi=120)
plt.show()

plt.figure()
df[var1].plot(kind="hist", bins=10, title=f"Histograma - {var1}")
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("hist_ingresos.png", dpi=120)
plt.show()

plt.figure()
df[var2].plot(kind="box", title=f"Boxplot - {var2}")
plt.tight_layout()
plt.savefig("boxplot_precio.png", dpi=120)
plt.show()

# Heatmap de correlaciones (modificado con texto más pequeño y escala verde)
num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap="Greens",
    annot_kws={"size": 8},
    cbar=True
)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title("Heatmap de correlaciones (variables numéricas)", fontsize=12)
plt.tight_layout()
plt.savefig("heatmap_correlaciones.png", dpi=120)
plt.show()

# -------------------------------------------------------------
# 4) Preparación para clustering
# -------------------------------------------------------------
features_all = df.select_dtypes("number").copy()
features_all = features_all.fillna(features_all.mean(numeric_only=True))

# Eliminar variables altamente correlacionadas
high_corr_threshold = 0.90
corr_matrix = features_all.corr()
to_drop = set()
cols = corr_matrix.columns.tolist()

for i, c1 in enumerate(cols):
    for c2 in cols[i + 1:]:
        r = corr_matrix.loc[c1, c2]
        if pd.notna(r) and abs(r) >= high_corr_threshold:
            to_drop.add(c2)

X = features_all.drop(columns=list(to_drop)) if to_drop else features_all.copy()

print("\n=== Selección de variables para clustering ===")
print("Columns usadas:", list(X.columns))
if to_drop:
    print("Eliminadas por alta correlación (|r|>=0.90):", list(to_drop))

# -------------------------------------------------------------
# 5) Escalamiento y selección de k
# -------------------------------------------------------------
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

k_values = range(2, 7)
inertias = []
sil_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    sil = silhouette_score(Xs, labels)
    sil_scores.append(sil)

plt.figure()
plt.plot(list(k_values), inertias, marker="o")
plt.xticks(list(k_values))
plt.title("Método del codo (inertia vs k)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=120)
plt.show()

plt.figure()
plt.plot(list(k_values), sil_scores, marker="o")
plt.xticks(list(k_values))
plt.title("Índice de Silueta vs k")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig("kmeans_silhouette.png", dpi=120)
plt.show()

best_k = int(k_values[np.argmax(sil_scores)])
print(f"\nSugerencia automática de k: {best_k}")
print("Inertias por k:", dict(zip(k_values, inertias)))
print("Siluetas por k:", dict(zip(k_values, sil_scores)))

# -------------------------------------------------------------
# 6) Entrenamiento K-Means
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(Xs)

centers_std = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_std)
centers_df = pd.DataFrame(centers_orig, columns=X.columns)
centers_df.index = [f"Centro_{i}" for i in range(best_k)]

print("\n=== Centros de K-Means (escala original) ===")
print(centers_df.round(2))

# Distancias entre centros
def pairwise_distances(M):
    diffs = M[:, None, :] - M[None, :, :]
    d = np.sqrt((diffs**2).sum(axis=2))
    return d

dist_centers = pairwise_distances(centers_orig)
dist_df = pd.DataFrame(dist_centers, index=centers_df.index, columns=centers_df.index)
print("\n=== Distancias entre centros (escala original) ===")
print(dist_df.round(2))




