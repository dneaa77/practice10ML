from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT/'data', ROOT/'results'
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

def run():
    rng = np.random.default_rng(42)
    features = ['average_check', 'monthly_purchases', 'monthly_visits']
    centers = [(350, 3, 5), (950, 12, 18), (1800, 5, 9), (650, 23, 30)]
    df = pd.DataFrame(np.vstack([rng.normal(c, [85, 1, 1.5], (100, 3)) for c in centers]), columns=features).round(2).clip(lower=0)
    df.to_csv(DATA/'customers.csv', index=False)
    df = pd.read_csv(DATA/'customers.csv')
    X = StandardScaler().fit_transform(df[features])
    def model(k, seed=42):
        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit(X)
    def profiles(labels, filename):
        result = df.assign(cluster=labels).groupby('cluster').agg({**{f:'mean' for f in features}}).round(2)
        result['count'] = pd.Series(labels).value_counts()
        result.to_csv(OUT/filename)
        return result
    def scatter(x, y, labels, title, filename):
        fig, ax = plt.subplots(); s = ax.scatter(x, y, c=labels, cmap='tab10', s=16); ax.set(title=title, xlabel='Feature 1', ylabel='Feature 2'); fig.colorbar(s, ax=ax, label='Cluster'); fig.tight_layout(); fig.savefig(OUT/filename, dpi=140); plt.close(fig)
    a = model(3)
    pa = profiles(a.labels_, 'A_profiles.csv')
    scatter(df.average_check, df.monthly_purchases, a.labels_, 'A: average check vs monthly purchases', 'A_clusters.png')
    scores = []
    for k in range(2,11):
        m = model(k); scores.append((k,m.inertia_,silhouette_score(X,m.labels_)))
    scores = pd.DataFrame(scores, columns=['k','inertia','silhouette']); scores.to_csv(OUT/'B_metrics.csv', index=False)
    for col in ['inertia','silhouette']:
        fig, ax = plt.subplots(); ax.plot(scores.k,scores[col],marker='o'); ax.set(xlabel='k',ylabel=col,title=col.capitalize()); ax.set_xticks(scores.k); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT/f'B_{col}.png',dpi=140); plt.close(fig)
    k = int(scores.loc[scores.silhouette.idxmax(),'k']); b = model(k); pb = profiles(b.labels_, 'B_profiles.csv')
    names = {i: ('Частые покупки' if r.monthly_purchases>18 else 'Премиальные' if r.average_check>1400 else 'Активные' if r.monthly_purchases>8 else 'Редкие покупки') for i,r in pb.iterrows()}
    pb['segment'] = pd.Series(names); pb.to_csv(OUT/'B_profiles.csv')
    ari = adjusted_rand_score(b.labels_,model(k,7).labels_)
    scatter(df.average_check,df.monthly_purchases,b.labels_,'B: selected clusters','B_clusters.png')
    moon, truth = make_moons(n_samples=500,noise=.045,random_state=42)
    pd.DataFrame({'x':moon[:,0],'y':moon[:,1],'true_label':truth}).to_csv(DATA/'moons.csv',index=False)
    Z = StandardScaler().fit_transform(moon)
    algorithms = {'KMeans':KMeans(n_clusters=2,n_init=20,random_state=42),'DBSCAN':DBSCAN(eps=.23,min_samples=5),'Agglomerative':AgglomerativeClustering(n_clusters=2,linkage='single')}
    rows=[]
    for name, alg in algorithms.items():
        labels=alg.fit_predict(Z); mask=labels!=-1; count=len(set(labels)-{-1}); sil=silhouette_score(Z[mask],labels[mask]) if 1<count<mask.sum() else np.nan
        rows.append((name,count,int((~mask).sum()),sil,adjusted_rand_score(truth,labels)))
        scatter(moon[:,0],moon[:,1],labels,f'C: {name}',f'C_{name}.png')
    c=pd.DataFrame(rows,columns=['algorithm','clusters','noise','silhouette_without_noise','ARI']); c.to_csv(OUT/'C_metrics.csv',index=False)
    scatter(moon[:,0],moon[:,1],truth,'C: original two moons','C_data.png')
    report=f'A profiles:\n{pa.to_string()}\n\nB metrics:\n{scores.round(4).to_string(index=False)}\n\nSelected k: {k}\n{pb.to_string()}\nStability ARI: {ari:.4f}\n\nC metrics:\n{c.round(4).to_string(index=False)}\n'
    (OUT/'summary.txt').write_text(report,encoding='utf-8'); print(report)

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
