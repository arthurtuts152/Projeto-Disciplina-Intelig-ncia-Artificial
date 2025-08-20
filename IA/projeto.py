import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import time

caminho_arquivo = 'base-IA-final.csv'  
df = pd.read_csv(caminho_arquivo, encoding='latin-1')

colunas_para_analise = [
    'idhm_educa',
    'idhm_renda',
    'qtd_pes_pob',
    'qtd_pes_baixa_renda',
]

df_clean = df[colunas_para_analise].dropna()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_clean[colunas_para_analise])
numeric_columns = colunas_para_analise

k_values = range(6, 15)  
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.show()

inertias = []
k_values = range(2, 15)

for k in k_values:
    kmeans = KMeans(n_clusters=k).fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_values, inertias, marker='o')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

print("\n" + "="*40)
print("EXECUTANDO COM K Ideal")
print("="*40)

start_time = time.time()
kmeans = KMeans(n_clusters=12, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
tempo_kmeans = time.time() - start_time
silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)

print(f"K-Means concluído !")
print(f"Silhouette Score: {silhouette_kmeans:.3f}")
print(f"Inércia: {kmeans.inertia_:.2f}")  
print(f"Tempo de execução: {tempo_kmeans:.3f} segundos")

print("\n" + "="*40)
print("TESTANDO PARÂMETROS DBSCAN")
print("="*40)

eps_values = [0.3, 0.5, 0.7, 1.1]
min_samples_values = [3, 5, 6, 10, 20]

print(f"{'EPS':<6} {'MIN_SAMPLES':<12} {'CLUSTERS':<10} {'RUÍDO':<8} {'SILHOUETTE':<12}")
print("-" * 60)

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan_test = DBSCAN(eps=eps, min_samples=min_samples)
        labels_test = dbscan_test.fit_predict(X_scaled)
        n_clusters = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        n_noise = list(labels_test).count(-1)
        if n_clusters > 1:
            mask = labels_test != -1
            silhouette = silhouette_score(X_scaled[mask], labels_test[mask]) if np.sum(mask) > 1 else 0
        else:
            silhouette = 0
        print(f"{eps:<6} {min_samples:<12} {n_clusters:<10} {n_noise:<8} {silhouette:<12}")


print("\n" + "="*40)
print("EXECUTANDO DBSCAN")
print("="*40)

start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_scaled)
tempo_dbscan = time.time() - start_time

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

if n_clusters_dbscan > 1:
    mask = labels_dbscan != -1
    if np.sum(mask) > 1:
        silhouette_dbscan = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    else:
        silhouette_dbscan = 0
else:
    silhouette_dbscan = 0

print(f"DBSCAN concluído!")
print(f"Clusters encontrados: {n_clusters_dbscan}")
print(f"Pontos de ruído: {n_noise}")
print(f"Silhouette Score: {silhouette_dbscan:.3f}")
print(f"Tempo de execução: {tempo_dbscan:.3f} segundos")


print("\n" + "="*50)
print("COMPARAÇÃO DOS ALGORITMOS")
print("="*50)

print(f"K-MEANS:")
print(f"  Silhouette Score: {silhouette_kmeans:.3f}")
print(f"  Inércia: {kmeans.inertia_:.2f}")  
print(f"  Tempo: {tempo_kmeans:.3f}s")

print(f"\nDBSCAN:")
print(f"  Clusters: {n_clusters_dbscan}")
print(f"  Silhouette Score: {silhouette_dbscan:.3f}")
print(f"  Tempo: {tempo_dbscan:.3f}s")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

algorithms = ['K-Means', 'DBSCAN']
scores = [silhouette_kmeans, silhouette_dbscan]
axes[0].bar(algorithms, scores, color=['skyblue', 'lightcoral'])
axes[0].set_title('Silhouette Score')
axes[0].set_ylabel('Score')

times = [tempo_kmeans, tempo_dbscan]
axes[1].bar(algorithms, times, color=['lightgreen', 'orange'])
axes[1].set_title('Tempo de Execução')
axes[1].set_ylabel('Segundos')

plt.tight_layout()
plt.show()


print("\n" + "="*40)
print("ANÁLISE DOS CLUSTERS K-MEANS")
print("="*40)

df_resultado = df.copy()
df_resultado['cluster_kmeans'] = labels_kmeans
df_resultado['cluster_dbscan'] = labels_dbscan

cluster_scores = {}
for cluster_id in sorted(set(labels_kmeans)):
    cluster_data = df_resultado[df_resultado['cluster_kmeans'] == cluster_id]
    print(f"\nCluster {cluster_id}: {len(cluster_data)} municípios")
    
    for col in numeric_columns[:3]:
        print(f"  {col} (média): {cluster_data[col].mean():.2f}")
    
    score_medio = (
    (1 - cluster_data['idhm_educa'].mean()) * 0.3 +  
    (1 - cluster_data['idhm_renda'].mean()) * 0.3 +  
    (cluster_data['qtd_pes_pob'].mean() / 10000) * 0.2 +  
    (cluster_data['qtd_pes_baixa_renda'].mean() / 10000) * 0.2  
)
    cluster_scores[cluster_id] = score_medio

cluster_prioritario = max(cluster_scores, key=cluster_scores.get)

print("\n" + "="*40)
print("TOP 10 MUNICÍPIOS PRIORITÁRIOS")
print("="*40)

clusters_prioritarios = [5, 0, 9]
ordem_prioridade = {5: 0, 0: 1, 9: 2}  

df_prioritarios = df_resultado[df_resultado['cluster_kmeans'].isin(clusters_prioritarios)].copy()
df_prioritarios['score_individual'] = (
    (1 - df_prioritarios['idhm_educa']) * 0.3 +  
    (1 - df_prioritarios['idhm_renda']) * 0.3 +
    (df_prioritarios['qtd_pes_pob'] / 10000) * 0.2 +  
    (df_prioritarios['qtd_pes_baixa_renda'] / 10000) * 0.2
)

df_prioritarios['prioridade_cluster'] = df_prioritarios['cluster_kmeans'].map(ordem_prioridade)
top10_prioritarios = df_prioritarios.sort_values(['prioridade_cluster', 'score_individual'], ascending=[True, False]).head(10)

print("\nTop 10 municípios prioritários:")
print("-" * 85)
print(f"{'Pos':<5}{'Cluster':<10}{'Código IBGE':<15}{'Score':<20}")
print("-" * 85)

for i, (_, row) in enumerate(top10_prioritarios.iterrows(), 1):
    print(f"{i:<5}{int(row['cluster_kmeans']):<10}{int(row['codigo_ibge']):<15}{row['score_individual']:,.2f}")

print("\n" + "="*40)
print("ANÁLISE DOS CLUSTERS DBSCAN")
print("="*40)

labels_dbscan_adj = labels_dbscan.copy()
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

cluster_scores_dbscan = {}
for cluster_id in sorted(set(labels_dbscan_adj)):
    cluster_mask = labels_dbscan_adj == cluster_id
    cluster_data = df_resultado[cluster_mask]
    
    if cluster_id == -1:
        print(f"\nRuído (Cluster -1): {len(cluster_data)} municípios")
    else:
        print(f"\nCluster {cluster_id}: {len(cluster_data)} municípios")
    
    for col in numeric_columns:
        print(f"  {col} (média): {cluster_data[col].mean():.2f}")
    
    if len(cluster_data) > 0:
        score_medio = (
            (1 - cluster_data['idhm_educa'].mean()) * 0.3 +
            (1 - cluster_data['idhm_renda'].mean()) * 0.3 +
            (cluster_data['qtd_pes_pob'].mean() / 10000) * 0.2 +
            (cluster_data['qtd_pes_baixa_renda'].mean() / 10000) * 0.2
        )
        cluster_scores_dbscan[cluster_id] = score_medio
        print(f"  Score de prioridade: {score_medio:.4f}")

clusters_validos = [c for c in cluster_scores_dbscan.keys() if c != -1]
if clusters_validos:
    cluster_prioritario_dbscan = max(clusters_validos, key=lambda x: cluster_scores_dbscan[x])
    print(f"\nCluster DBSCAN mais prioritário: {cluster_prioritario_dbscan} (Score: {cluster_scores_dbscan[cluster_prioritario_dbscan]:.4f})")
else:
    print("\nNenhum cluster válido encontrado no DBSCAN (apenas ruído)")

print("\n" + "="*40)
print("TOP 10 MUNICÍPIOS PRIORITÁRIOS (DBSCAN)")
print("="*40)

clusters_prioritarios_dbscan = [c for c in sorted(cluster_scores_dbscan.keys()) if c != -1]

if clusters_prioritarios_dbscan:
    clusters_ordenados = sorted(clusters_prioritarios_dbscan, key=lambda x: cluster_scores_dbscan[x], reverse=True)
    
    df_prioritarios_dbscan = df_resultado[df_resultado['cluster_dbscan'].isin(clusters_prioritarios_dbscan)].copy()

    df_prioritarios_dbscan['score_individual'] = (
        (1 - df_prioritarios_dbscan['idhm_educa']) * 0.3 +
        (1 - df_prioritarios_dbscan['idhm_renda']) * 0.3 +
        (df_prioritarios_dbscan['qtd_pes_pob'] / 10000) * 0.2 +
        (df_prioritarios_dbscan['qtd_pes_baixa_renda'] / 10000) * 0.2
    )
    
    ordem_prioridade_dbscan = {c: i for i, c in enumerate(clusters_ordenados)}
    df_prioritarios_dbscan['prioridade_cluster'] = df_prioritarios_dbscan['cluster_dbscan'].map(ordem_prioridade_dbscan)

    top10_prioritarios_dbscan = df_prioritarios_dbscan.sort_values(
        ['prioridade_cluster', 'score_individual'],
        ascending=[True, False]
    ).head(10)
    
    print("\nTop 10 municípios prioritários:")
    print("-" * 85)
    print(f"{'Pos':<5}{'Cluster':<10}{'Código IBGE':<15}{'Score':<20}")
    print("-" * 85)
    
    for i, (_, row) in enumerate(top10_prioritarios_dbscan.iterrows(), 1):
        print(f"{i:<5}{int(row['cluster_dbscan']):<10}{int(row['codigo_ibge']):<15}{row['score_individual']:,.2f}")


