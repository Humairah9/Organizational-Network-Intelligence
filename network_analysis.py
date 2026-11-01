# %% [markdown]
# # Organizational Network Intelligence
# ### Who talks to whom? Network analysis of the SNAP `email-Eu-core` dataset
#
# This notebook explores internal email communication at a large European research institution:
# who the key communicators are, how the organisation clusters, how much it talks across
# departments, how quickly information can travel, who bridges groups, and who behaves unusually.
#
# **Data:** [SNAP `email-Eu-core`](https://snap.stanford.edu/data/email-Eu-core.html) and
# `email-Eu-core-department-labels` (downloaded automatically on first run).
#
# **A note on what an "edge" is.** In this dataset a directed edge `u -> v` means that person `u`
# sent person `v` *at least one* email. The 25,571 edges are therefore unique sender-receiver
# pairs, not a count of emails. Throughout, "emails" really means "email links".

# %% Setup
import gzip
import shutil
import urllib.request
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms.community import louvain_communities, modularity
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="notebook")

SEED = 42
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "images"
TAB_DIR = ROOT / "outputs" / "tables"
for d in (DATA_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

SNAP_BASE = "https://snap.stanford.edu/data/"
FILES = {
    "edges": "email-Eu-core.txt",
    "labels": "email-Eu-core-department-labels.txt",
}
RESULTS = {}  # key numbers collected here are written to outputs/results_summary.md


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def save_table(df, name, index=False):
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)


# %% [markdown]
# ## 1. Load the data
# Files are cached in `data/` so the download only happens once.

# %% Load
def fetch(name):
    """Return the path to a plain-text SNAP file, downloading and unzipping it if needed."""
    txt = DATA_DIR / name
    if txt.exists():
        return txt
    gz = DATA_DIR / f"{name}.gz"
    if not gz.exists():
        print(f"Downloading {name}.gz from SNAP ...")
        urllib.request.urlretrieve(SNAP_BASE + f"{name}.gz", gz)
    with gzip.open(gz, "rb") as src, open(txt, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return txt


edges = pd.read_csv(fetch(FILES["edges"]), sep=r"\s+", header=None, names=["Sender", "Receiver"])
labels = pd.read_csv(fetch(FILES["labels"]), sep=r"\s+", header=None, names=["Node", "Department"])
dept_of = labels.set_index("Node")["Department"].to_dict()

print(f"Edge list:   {len(edges):,} rows, missing values: {int(edges.isna().sum().sum())}")
print(f"Labels:      {len(labels):,} employees in {labels['Department'].nunique()} departments, "
      f"missing values: {int(labels.isna().sum().sum())}")
print(f"Duplicate sender-receiver rows: {int(edges.duplicated().sum())}")
edges.head()

# %% [markdown]
# ## 2. Build the network and describe it
# Self-loops (people emailing themselves) are kept in the raw graph `G_raw` but removed for the
# analysis graph `G`, because they say nothing about how people communicate with each other.

# %% Build graph
G_raw = nx.from_pandas_edgelist(edges, "Sender", "Receiver", create_using=nx.DiGraph())
nx.set_node_attributes(G_raw, dept_of, "Department")
n_selfloops = nx.number_of_selfloops(G_raw)

G = G_raw.copy()
G.remove_edges_from(list(nx.selfloop_edges(G)))
U = G.to_undirected()

lscc_nodes = max(nx.strongly_connected_components(G), key=len)
overview = {
    "Employees (nodes)": G.number_of_nodes(),
    "Departments": labels["Department"].nunique(),
    "Directed links incl. self-loops": G_raw.number_of_edges(),
    "Self-loops": n_selfloops,
    "Directed links used in analysis": G.number_of_edges(),
    "Density (directed)": round(nx.density(G), 4),
    "Reciprocity (share of links that are returned)": round(nx.reciprocity(G), 3),
    "Weakly connected components": nx.number_weakly_connected_components(G),
    "Largest strongly connected component": len(lscc_nodes),
    "Average clustering coefficient (undirected)": round(nx.average_clustering(U), 3),
}
RESULTS["overview"] = overview
pd.Series(overview, name="value").to_frame()

# %% [markdown]
# ## 3. Who are the key communicators? (In-degree vs out-degree)
# *In-degree* = how many different people emailed this person. *Out-degree* = how many different
# people this person emailed. This is computed on the **full** network (not a random sample).

# %% Q1
deg = pd.DataFrame({
    "Node": list(G.nodes()),
    "In-Degree": [G.in_degree(n) for n in G.nodes()],
    "Out-Degree": [G.out_degree(n) for n in G.nodes()],
}).assign(Department=lambda d: d["Node"].map(dept_of))
deg["Total"] = deg["In-Degree"] + deg["Out-Degree"]

top_in = deg.nlargest(10, "In-Degree")
top_out = deg.nlargest(10, "Out-Degree")
save_table(deg, "degree_table")

rho, _ = spearmanr(deg["In-Degree"], deg["Out-Degree"])
RESULTS["q1"] = {
    "top_in": [list(r) for r in top_in[["Node", "In-Degree", "Department"]].itertuples(index=False)],
    "top_out": [list(r) for r in top_out[["Node", "Out-Degree", "Department"]].itertuples(index=False)],
    "overlap": sorted(set(top_in["Node"]) & set(top_out["Node"])),
    "spearman_in_out": round(float(rho), 3),
    "median_in": float(deg["In-Degree"].median()),
    "median_out": float(deg["Out-Degree"].median()),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.barplot(data=top_in, x="In-Degree", y=top_in["Node"].astype(str), color="#e4572e", ax=axes[0])
axes[0].set(title="Top 10 by in-degree (most senders reaching them)", ylabel="Employee ID")
sns.barplot(data=top_out, x="Out-Degree", y=top_out["Node"].astype(str), color="#2a9d8f", ax=axes[1])
axes[1].set(title="Top 10 by out-degree (most recipients reached)", ylabel="Employee ID")
save_fig(fig, "01_top_communicators")

# Visualise how the top communicators connect to each other
core = set(top_in["Node"]) | set(top_out["Node"])
H = G.subgraph(core)
pos = nx.spring_layout(H, seed=SEED, k=1.2)
role = {n: ("Top in & out" if n in set(top_in["Node"]) & set(top_out["Node"])
            else "Top in-degree" if n in set(top_in["Node"]) else "Top out-degree") for n in H}
palette = {"Top in-degree": "#e4572e", "Top out-degree": "#2a9d8f", "Top in & out": "#f4a259"}
fig, ax = plt.subplots(figsize=(8, 6.5))
nx.draw_networkx_edges(H, pos, alpha=0.25, arrowsize=8, edge_color="gray", ax=ax)
for r, c in palette.items():
    nodes = [n for n in H if role[n] == r]
    if nodes:
        nx.draw_networkx_nodes(H, pos, nodelist=nodes, node_color=c, node_size=420, label=r, ax=ax)
nx.draw_networkx_labels(H, pos, font_size=8, font_weight="bold", ax=ax)
ax.set_title("How the top communicators connect to each other")
ax.legend(loc="best")
ax.axis("off")
save_fig(fig, "01b_top_communicator_subgraph")
top_in.merge(top_out, on=["Node", "Department"], how="outer", suffixes=(" (in table)", " (out table)"))

# %% [markdown]
# ## 4. Are there communities, and do they match departments?
# Louvain community detection runs on the undirected version of the **whole** network. To see
# whether the communities are just the org chart, they are compared with the department labels
# using normalised mutual information (NMI) and the adjusted Rand index (ARI).

# %% Q2
communities = louvain_communities(U, seed=SEED)
communities = sorted(communities, key=len, reverse=True)
comm_of = {n: i for i, c in enumerate(communities) for n in c}
Q = modularity(U, communities)

node_df = pd.DataFrame({"Node": list(U.nodes())})
node_df["Community"] = node_df["Node"].map(comm_of)
node_df["Department"] = node_df["Node"].map(dept_of)
nmi = normalized_mutual_info_score(node_df["Department"], node_df["Community"])
ari = adjusted_rand_score(node_df["Department"], node_df["Community"])

comm_summary = (
    node_df.groupby("Community")
    .agg(Size=("Node", "size"),
         Dominant_Department=("Department", lambda s: s.value_counts().index[0]),
         Dominant_Share=("Department", lambda s: s.value_counts(normalize=True).iloc[0]))
    .reset_index()
)
cross_comm_edges = sum(comm_of[u] != comm_of[v] for u, v in U.edges()) / U.number_of_edges()
save_table(comm_summary, "communities")
RESULTS["q2"] = {
    "n_communities": len(communities),
    "modularity": round(Q, 3),
    "nmi_vs_departments": round(nmi, 3),
    "ari_vs_departments": round(ari, 3),
    "share_edges_between_communities": round(cross_comm_edges, 3),
    "largest_sizes": [len(c) for c in communities[:10]],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1]})
pos_all = nx.spring_layout(U, seed=SEED, iterations=60)
cmap = plt.get_cmap("tab10")
colors = [cmap(comm_of[n]) if comm_of[n] < 10 else "#cccccc" for n in U.nodes()]
nx.draw_networkx_edges(U, pos_all, alpha=0.03, width=0.4, ax=axes[0])
nx.draw_networkx_nodes(U, pos_all, node_size=14, node_color=colors, ax=axes[0])
axes[0].set_title(f"Louvain communities (10 largest coloured) | modularity = {Q:.2f}")
axes[0].axis("off")
top_c = comm_summary.head(12)
sns.barplot(data=top_c, x="Community", y="Size", color="#577590", ax=axes[1])
for i, r in enumerate(top_c.itertuples()):
    axes[1].text(i, r.Size, f"d{r.Dominant_Department}\n{r.Dominant_Share:.0%}", ha="center", va="bottom", fontsize=7)
axes[1].set(title="Community size (label: dominant department & its share)")
save_fig(fig, "02_communities")
comm_summary.head(10)

# %% [markdown]
# ## 5. Within departments or across them?
# Each link is classed as intra- or inter-department. Because the departments differ a lot in
# size, the observed split is compared with what *random* mixing would give: if links were
# placed between random pairs of people, the intra-department share would be
# `sum(n_d * (n_d - 1)) / (N * (N - 1))`.

# %% Q3
e = edges.assign(S=edges["Sender"].map(dept_of), R=edges["Receiver"].map(dept_of))
e["Intra"] = e["S"] == e["R"]
e_clean = e[e["Sender"] != e["Receiver"]]

intra_all = e["Intra"].mean()
intra_clean = e_clean["Intra"].mean()
sizes = labels["Department"].value_counts()
N = len(labels)
expected = float((sizes * (sizes - 1)).sum() / (N * (N - 1)))

per_dept = (e_clean.groupby("S")["Intra"].agg(Links="size", Intra_Share="mean").reset_index()
            .rename(columns={"S": "Department"}))
save_table(per_dept, "intra_share_by_department")
RESULTS["q3"] = {
    "intra_share_incl_selfloops": round(float(intra_all), 4),
    "intra_links_incl_selfloops": int(e["Intra"].sum()),
    "inter_links_incl_selfloops": int((~e["Intra"]).sum()),
    "intra_share_excl_selfloops": round(float(intra_clean), 4),
    "expected_intra_share_random": round(expected, 4),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].pie([intra_clean, 1 - intra_clean], labels=["Intra-department", "Inter-department"],
            autopct="%1.1f%%", startangle=140, colors=["#1f77b4", "#ff7f0e"])
axes[0].set_title("Share of links (self-loops excluded)")
sns.barplot(x=["Observed", "Expected under\nrandom mixing"], y=[intra_clean, expected],
            hue=["Observed", "Expected under\nrandom mixing"], palette=["#1f77b4", "#bbbbbb"], legend=False, ax=axes[1])
axes[1].set(title="Intra-department share vs chance", ylabel="Share of links that stay in-department")
for i, v in enumerate([intra_clean, expected]):
    axes[1].text(i, v, f"{v:.1%}", ha="center", va="bottom")
save_fig(fig, "03_intra_vs_inter")

# %% [markdown]
# ## 6. Which departments send and receive the most?

# %% Q4
sent = e_clean.groupby("S").size().rename("Sent")
recv = e_clean.groupby("R").size().rename("Received")
dept_flow = pd.concat([sent, recv], axis=1).fillna(0).astype(int)
dept_flow.index.name = "Department"
dept_flow["Net (Sent - Received)"] = dept_flow["Sent"] - dept_flow["Received"]
dept_flow["Headcount"] = sizes.reindex(dept_flow.index).fillna(0).astype(int)
dept_flow["Sent per person"] = (dept_flow["Sent"] / dept_flow["Headcount"]).round(1)
dept_flow = dept_flow.sort_values("Sent", ascending=False).reset_index()
save_table(dept_flow, "department_flow")
RESULTS["q4"] = {"top_sent": [list(r) for r in dept_flow.head(5)[["Department", "Sent", "Received", "Headcount"]].itertuples(index=False)]}

top10 = dept_flow.head(10).melt(id_vars="Department", value_vars=["Sent", "Received"], var_name="Direction", value_name="Links")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
sns.barplot(data=top10, x="Department", y="Links", hue="Direction", order=dept_flow.head(10)["Department"],
            palette=["#3498db", "#e74c3c"], ax=axes[0])
axes[0].set_title("Top 10 departments by outgoing links")
sns.scatterplot(data=dept_flow, x="Headcount", y="Sent", ax=axes[1], s=60, color="#577590")
for r in dept_flow.head(6).itertuples():
    axes[1].annotate(f"d{r.Department}", (r.Headcount, r.Sent), xytext=(4, 4), textcoords="offset points")
axes[1].set_title("Outgoing links scale with headcount")
save_fig(fig, "04_department_flow")
dept_flow.head(10)

# %% [markdown]
# ## 7. How efficiently does information flow?
# Shortest paths are only defined between people who can actually reach each other, so they are
# computed inside the largest strongly connected component (LSCC) (everyone in it can reach
# everyone else by following the direction of emails).

# %% Q5
G_scc = G.subgraph(lscc_nodes)
dist_counts = Counter()
for _, lengths in nx.all_pairs_shortest_path_length(G_scc):
    dist_counts.update(d for d in lengths.values() if d > 0)  # d == 0 is a node to itself
total_pairs = sum(dist_counts.values())
aspl = sum(d * c for d, c in dist_counts.items()) / total_pairs
diameter = max(dist_counts)
RESULTS["q5"] = {
    "lscc_nodes": len(lscc_nodes),
    "lscc_share_of_network": round(len(lscc_nodes) / G.number_of_nodes(), 3),
    "avg_shortest_path": round(aspl, 2),
    "diameter": diameter,
    "share_within_3_hops": round(sum(c for d, c in dist_counts.items() if d <= 3) / total_pairs, 3),
}

dist_df = pd.DataFrame({"Path length": list(dist_counts), "Pairs": list(dist_counts.values())}).sort_values("Path length")
dist_df["Share"] = dist_df["Pairs"] / total_pairs
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(data=dist_df, x="Path length", y="Share", color="royalblue", ax=ax)
ax.axvline(aspl - 1, color="crimson", ls="--", label=f"Average = {aspl:.2f}")  # categorical x starts at length 1
ax.set(title="Shortest-path lengths between employees (LSCC)", ylabel="Share of ordered pairs")
ax.legend()
save_fig(fig, "05_shortest_paths")
dist_df

# %% [markdown]
# ## 8. Can we predict who will become a key communicator?
# The dataset has no timestamps, so "becoming" a key communicator cannot be observed directly. A
# fair stand-in is a **link-holdout** experiment:
#
# 1. Randomly split all links into an *observed* set (70%) and a *future* set (30%).
# 2. Compute every employee's features from the observed links only.
# 3. Label an employee a "future key communicator" if they are in the top 10% by in-degree in the
#    future links.
#
# Unlike a model whose target (e.g. PageRank) is computed from the very same features, this keeps
# the answer out of the inputs. Models are compared with cross-validation against a simple
# baseline.

# %% Q6
rng = np.random.default_rng(SEED)
E = np.array(list(G.edges()))
observed_mask = rng.random(len(E)) < 0.70
G_obs = nx.DiGraph(); G_obs.add_nodes_from(G.nodes()); G_obs.add_edges_from(map(tuple, E[observed_mask]))
G_fut = nx.DiGraph(); G_fut.add_nodes_from(G.nodes()); G_fut.add_edges_from(map(tuple, E[~observed_mask]))

nodes = list(G.nodes())
pr = nx.pagerank(G_obs)
bt = nx.betweenness_centrality(G_obs)
cl = nx.clustering(G_obs.to_undirected())
feat = pd.DataFrame({
    "In-Degree": [G_obs.in_degree(n) for n in nodes],
    "Out-Degree": [G_obs.out_degree(n) for n in nodes],
    "Betweenness": [bt[n] for n in nodes],
    "PageRank": [pr[n] for n in nodes],
    "Clustering": [cl[n] for n in nodes],
}, index=nodes)
dept_dummies = pd.get_dummies(pd.Series({n: dept_of[n] for n in nodes}), prefix="dept").astype(int)

future_in = pd.Series({n: G_fut.in_degree(n) for n in nodes})
y = (future_in >= future_in.quantile(0.90)).astype(int)
print(f"Positive class: {y.sum()} of {len(y)} employees ({y.mean():.1%})")

X_net = feat
X_all = feat.join(dept_dummies)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rf = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=SEED, n_jobs=-1)
candidates = {
    "Baseline: logistic on in-degree only": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), feat[["In-Degree"]]),
    "Random forest: department only": (rf(), dept_dummies),
    "Random forest: network features": (rf(), X_net),
    "Random forest: network + department": (rf(), X_all),
}
rows = []
for name, (model, X_) in candidates.items():
    s = cross_validate(model, X_, y, cv=cv, scoring=["roc_auc", "average_precision"])
    rows.append({"Model": name, "ROC AUC": s["test_roc_auc"].mean(), "ROC AUC std": s["test_roc_auc"].std(),
                 "Avg precision": s["test_average_precision"].mean()})
model_df = pd.DataFrame(rows).round(3)
save_table(model_df, "prediction_model_comparison")

# Grouped permutation importance on a held-out test set
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=SEED)
final = rf().fit(X_tr, y_tr)
base_auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
groups = {c: [c] for c in feat.columns}
groups["Department (all one-hot columns)"] = list(dept_dummies.columns)
imp = []
for g, cols in groups.items():
    drops = []
    for _ in range(20):
        Xp = X_te.copy()
        Xp[cols] = Xp[cols].sample(frac=1, random_state=int(rng.integers(1e9))).values
        drops.append(base_auc - roc_auc_score(y_te, final.predict_proba(Xp)[:, 1]))
    imp.append({"Feature": g, "AUC drop when shuffled": np.mean(drops)})
imp_df = pd.DataFrame(imp).sort_values("AUC drop when shuffled", ascending=False)
save_table(imp_df, "prediction_feature_importance")
RESULTS["q6"] = {"models": model_df.to_dict("records"), "importance": imp_df.round(4).to_dict("records"),
                 "positive_rate": round(float(y.mean()), 3)}

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=imp_df, x="AUC drop when shuffled", y="Feature", color="#6a4c93", ax=ax)
ax.set_title("What drives the prediction? (permutation importance, held-out set)")
save_fig(fig, "06_prediction_importance")
model_df

# %% [markdown]
# ## 9. Who are the bridges?
# Betweenness centrality measures how often a person sits on the shortest path between two others.
# Alongside it, each person's *cross-department share* (the fraction of their contacts who work
# in a different department) shows whether they connect departments, not just people.

# %% Q7
bet = nx.betweenness_centrality(G)
contacts = {n: set(U.neighbors(n)) for n in U.nodes()}
bridge_df = pd.DataFrame({
    "Node": list(bet),
    "Betweenness": list(bet.values()),
}).assign(
    Department=lambda d: d["Node"].map(dept_of),
    Degree=lambda d: d["Node"].map(lambda n: len(contacts[n])),
    Cross_Dept_Share=lambda d: d["Node"].map(lambda n: np.mean([dept_of[m] != dept_of[n] for m in contacts[n]]) if contacts[n] else np.nan),
    Departments_Reached=lambda d: d["Node"].map(lambda n: len({dept_of[m] for m in contacts[n]})),
).sort_values("Betweenness", ascending=False)
save_table(bridge_df, "bridge_table")

top_bridges = bridge_df.head(10)
rho_b, _ = spearmanr(bridge_df["Betweenness"], bridge_df["Degree"])
RESULTS["q7"] = {
    "top_bridges": [list(r) for r in top_bridges[["Node", "Betweenness", "Department", "Degree", "Cross_Dept_Share"]].round(4).itertuples(index=False)],
    "spearman_betweenness_degree": round(float(rho_b), 3),
    "articulation_points": len(list(nx.articulation_points(U))),
    "overlap_with_top_degree": len(set(top_bridges["Node"]) & set(deg.nlargest(10, "Total")["Node"])),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.barplot(data=top_bridges, x="Betweenness", y=top_bridges["Node"].astype(str), hue="Cross_Dept_Share",
            palette="rocket_r", dodge=False, legend=False, ax=axes[0])
axes[0].set(title="Top 10 bridges by betweenness", ylabel="Employee ID")
sc = axes[1].scatter(bridge_df["Degree"], bridge_df["Betweenness"], c=bridge_df["Cross_Dept_Share"], cmap="viridis", s=14, alpha=0.8)
plt.colorbar(sc, ax=axes[1], label="Cross-department share of contacts")
axes[1].set(title=f"Betweenness vs number of contacts (Spearman = {rho_b:.2f})", xlabel="Contacts (undirected degree)", ylabel="Betweenness")
save_fig(fig, "07_bridges")
top_bridges.round(4)

# %% [markdown]
# ## 10. Unusual email behaviour
# Isolation Forest scores each employee on several behavioural features at once: how many
# people they email, how many email them, how often links are reciprocated, how many departments
# they reach, and what share of their contacts is outside their own department. The top 5% most
# isolated profiles are flagged (the contamination rate is a choice, not a finding).
#
# Anomalies in a single static snapshot are *unusual*, not *malicious*. They are leads to look
# at, and an unusually busy coordinator is far more likely than a security problem.

# %% Q8
und = {n: set(U.neighbors(n)) for n in U.nodes()}
beh = pd.DataFrame(index=list(G.nodes()))
beh["In"] = [G.in_degree(n) for n in beh.index]
beh["Out"] = [G.out_degree(n) for n in beh.index]
beh["Mutual_Share"] = [sum(G.has_edge(n, m) and G.has_edge(m, n) for m in und[n]) / len(und[n]) if und[n] else 0 for n in beh.index]
beh["Depts_Reached"] = [len({dept_of[m] for m in und[n]}) for n in beh.index]
beh["Cross_Dept_Share"] = [np.mean([dept_of[m] != dept_of[n] for m in und[n]]) if und[n] else 0 for n in beh.index]
beh["Log_In"] = np.log1p(beh["In"]); beh["Log_Out"] = np.log1p(beh["Out"])
cols = ["Log_In", "Log_Out", "Mutual_Share", "Depts_Reached", "Cross_Dept_Share"]

iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=SEED)
beh["Flag"] = iso.fit_predict(StandardScaler().fit_transform(beh[cols]))
beh["Anomaly_Score"] = -iso.score_samples(StandardScaler().fit_transform(beh[cols]))
beh["Department"] = beh.index.map(dept_of)
median_total = (beh["In"] + beh["Out"]).median()
beh["Profile"] = np.where(beh["Flag"] == -1,
                          np.where((beh["In"] + beh["Out"]) > median_total, "Unusually active/broad", "Unusually quiet/narrow"), "Typical")
flagged = beh[beh["Flag"] == -1].sort_values("Anomaly_Score", ascending=False)
beh.index.name = "Node"
save_table(beh.reset_index(), "anomaly_table")
RESULTS["q8"] = {
    "n_flagged": int(len(flagged)),
    "profile_counts": flagged["Profile"].value_counts().to_dict(),
    "top_flagged": [list(r) for r in flagged.head(10).reset_index().rename(columns={"index": "Node"})[["Node", "In", "Out", "Department", "Profile"]].itertuples(index=False)],
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=beh.reset_index(), x="In", y="Out", hue="Profile", size="Anomaly_Score", sizes=(15, 120),
                palette={"Typical": "#bbbbbb", "Unusually active/broad": "#d62828", "Unusually quiet/narrow": "#1d3557"}, alpha=0.8, ax=axes[0])
axes[0].set(title="In- vs out-degree with flagged profiles", xscale="symlog", yscale="symlog")
sns.histplot(beh["Anomaly_Score"], bins=40, color="#577590", ax=axes[1])
axes[1].axvline(flagged["Anomaly_Score"].min(), color="crimson", ls="--", label="Flag threshold (top 5%)")
axes[1].set(title="Anomaly score distribution"); axes[1].legend()
save_fig(fig, "08_anomalies")
flagged.head(10)[["In", "Out", "Mutual_Share", "Depts_Reached", "Cross_Dept_Share", "Department", "Profile", "Anomaly_Score"]].round(3)

# %% [markdown]
# ## 11. Write the results summary
# Collects the headline numbers into `outputs/results_summary.md` so the README can be updated
# from real output.

# %% Summary
def fmt_rows(rows):
    return "\n".join("- " + ", ".join(str(x) for x in r) for r in rows)

q = RESULTS
lines = [
    "# Results summary (auto-generated by network_analysis.py)", "",
    "## Network overview",
    *[f"- **{k}:** {v}" for k, v in q["overview"].items()], "",
    "## Q1. Key communicators",
    f"- Median in-degree {q['q1']['median_in']:.0f}, median out-degree {q['q1']['median_out']:.0f}; "
    f"in/out Spearman correlation {q['q1']['spearman_in_out']}",
    f"- Nodes in both top-10 lists: {q['q1']['overlap'] or 'none'}",
    "- Top 10 in-degree (node, in-degree, department):", fmt_rows(q["q1"]["top_in"]),
    "- Top 10 out-degree (node, out-degree, department):", fmt_rows(q["q1"]["top_out"]), "",
    "## Q2. Communities",
    f"- {q['q2']['n_communities']} Louvain communities, modularity {q['q2']['modularity']}",
    f"- Agreement with departments: NMI {q['q2']['nmi_vs_departments']}, ARI {q['q2']['ari_vs_departments']}",
    f"- Share of links between different communities: {q['q2']['share_edges_between_communities']:.1%}",
    f"- Ten largest community sizes: {q['q2']['largest_sizes']}", "",
    "## Q3. Intra vs inter-department",
    f"- Including self-loops: {q['q3']['intra_links_incl_selfloops']:,} intra / {q['q3']['inter_links_incl_selfloops']:,} inter "
    f"({q['q3']['intra_share_incl_selfloops']:.2%} intra)",
    f"- Excluding self-loops: {q['q3']['intra_share_excl_selfloops']:.2%} intra",
    f"- Expected intra share if links were random: {q['q3']['expected_intra_share_random']:.2%}", "",
    "## Q4. Departments by outgoing links (department, sent, received, headcount)",
    fmt_rows(q["q4"]["top_sent"]), "",
    "## Q5. Information flow",
    f"- Largest strongly connected component: {q['q5']['lscc_nodes']} people "
    f"({q['q5']['lscc_share_of_network']:.1%} of the network)",
    f"- Average shortest path {q['q5']['avg_shortest_path']}, diameter {q['q5']['diameter']}, "
    f"{q['q5']['share_within_3_hops']:.1%} of pairs within 3 hops", "",
    "## Q6. Prediction (link-holdout experiment)",
    f"- Positive class rate {q['q6']['positive_rate']:.1%}",
    *[f"- {m['Model']}: ROC AUC {m['ROC AUC']} (+/- {m['ROC AUC std']}), avg precision {m['Avg precision']}" for m in q["q6"]["models"]],
    "- Permutation importance (AUC drop): " + ", ".join(f"{r['Feature']} {r['AUC drop when shuffled']}" for r in q["q6"]["importance"]), "",
    "## Q7. Bridges (node, betweenness, department, contacts, cross-dept share)",
    fmt_rows(q["q7"]["top_bridges"]),
    f"- Spearman(betweenness, contacts) = {q['q7']['spearman_betweenness_degree']}; "
    f"articulation points: {q['q7']['articulation_points']}; "
    f"top-10 bridges that are also top-10 by total degree: {q['q7']['overlap_with_top_degree']}", "",
    "## Q8. Anomalies",
    f"- {q['q8']['n_flagged']} flagged: {q['q8']['profile_counts']}",
    "- Most unusual (node, in, out, department, profile):", fmt_rows(q["q8"]["top_flagged"]),
]
(ROOT / "outputs" / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
