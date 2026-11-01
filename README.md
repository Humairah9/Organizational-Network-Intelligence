# Organizational Network Intelligence

Using Python, I analysed the internal email network of a large European research institution (1,005 employees, 25,571 directed email links) to find out who the key communicators are, how the organisation clusters, and where information flows or gets stuck. I built the network with NetworkX, measured it with centrality, community-detection, shortest-path, machine-learning and anomaly-detection methods, and tested whether the org chart matches how people actually communicate. The main finding is that **[one-sentence headline finding, e.g. who really connects the organisation, filled in after running the notebook]**.

## Table of Contents

- [Dashboard Preview](#dashboard-preview)
- [Introduction](#introduction)
- [Data Overview and Dataset Structure](#data-overview-and-dataset-structure)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Analysis and Findings](#analysis-and-findings)
- [Conclusion](#conclusion)
- [Recommendations](#recommendations)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Tools and Repository Structure](#tools-and-repository-structure)

## Dashboard Preview

![Communities in the email network](outputs/figures/02_communities.png)

*[One-line caption of the key takeaway from this chart.]*

## Introduction

An organisation needs to know who its people rely on to get things done. Without that view, key communicators and the people who connect departments go unnoticed, and silos form without anyone seeing them.

This project analyses the SNAP `email-Eu-core` network using Python to answer practical questions about communication. The analysis aims to:

- Identify the key communicators by emails received and sent.
- Detect communities and test whether they match departments.
- Measure how much communication stays within departments and how much crosses them.
- Measure how efficiently information travels through the network.
- Test whether future key communicators can be predicted.
- Find the people who bridge groups and the people with unusual email behaviour.

## Data Overview and Dataset Structure

The data comes from [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) and has two files. The network has **1,005 employees** and **25,571 directed links** across **42 departments**, with a density of 0.0253, so only about 2.5% of all possible connections are used.

A link `u -> v` means *u sent v at least one email*, so links are unique sender-receiver pairs, not a count of emails.

| File | Column | Description |
|---|---|---|
| email-Eu-core | Sender | ID of the employee who sent at least one email |
| email-Eu-core | Receiver | ID of the employee who received it |
| email-Eu-core-department-labels | Node | Employee ID |
| email-Eu-core-department-labels | Department | Department the employee belongs to (0 to 41) |

## Data Cleaning and Preparation

| Issue | Treatment |
|---|---|
| Missing values | None found in either file |
| Duplicate sender-receiver rows | [number] found [and removed / none] |
| Self-loops (people emailing themselves) | [number] found; removed from network metrics, but intra/inter-department shares are reported both with and without them |
| Data types | Checked, all integer IDs, no changes needed |

All network metrics below use the cleaned graph of [N] links.

## Analysis and Findings

### 1. Key Communicators

*In-degree* is the number of different people who emailed someone; *out-degree* is the number of different people they emailed. Both are measured on the full network.

| Rank | Top in-degree (employee, dept, value) | Top out-degree (employee, dept, value) |
|---|---|---|
| 1 | [ ] | [ ] |
| 2 | [ ] | [ ] |
| 3 | [ ] | [ ] |

[Interpretation: do the same people lead both lists? How far above the median do they sit?]

![Top communicators](outputs/figures/01_top_communicators.png)

![Top communicator subgraph](outputs/figures/01b_top_communicator_subgraph.png)

### 2. Communities

Louvain detection found **[N] communities** with a modularity of **[value]**. Agreement with the department labels is NMI **[value]** and ARI **[value]**, and **[share]%** of links run between communities.

[Interpretation: do communities mirror departments or cut across them?]

![Communities](outputs/figures/02_communities.png)

### 3. Within Departments or Across Them

| Measure | Share of links |
|---|---|
| Intra-department (including self-loops) | [ ]% |
| Intra-department (excluding self-loops) | [ ]% |
| Expected if links were random | [ ]% |

[Interpretation: people stay in-department more or less than chance, and what that implies.]

![Intra vs inter](outputs/figures/03_intra_vs_inter.png)

### 4. Department Activity

| Department | Headcount | Sent | Received |
|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] |
| [ ] | [ ] | [ ] | [ ] |
| [ ] | [ ] | [ ] | [ ] |

[Interpretation: which departments send more than they receive, and whether activity just follows headcount.]

![Department flow](outputs/figures/04_department_flow.png)

### 5. Information Flow

The largest strongly connected component holds **[N] people ([share]% of the network)**. The average shortest path is **[value]** steps, the diameter is **[value]**, and **[share]%** of pairs can reach each other within 3 hops.

![Shortest paths](outputs/figures/05_shortest_paths.png)

### 6. Predicting Key Communicators

There are no timestamps, so I used a link-holdout test: features come from 70% of links and the target is top-10% in-degree in the other 30%.

| Model | ROC AUC | Average precision |
|---|---|---|
| Baseline: logistic on in-degree only | [ ] | [ ] |
| Random forest: department only | [ ] | [ ] |
| Random forest: network features | [ ] | [ ] |
| Random forest: network + department | [ ] | [ ] |

[Interpretation: does the forest beat the simple baseline, and does department add anything?]

![Feature importance](outputs/figures/06_prediction_importance.png)

### 7. Bridges

| Employee | Department | Betweenness | Contacts | Cross-department share |
|---|---|---|---|---|
| [ ] | [ ] | [ ] | [ ] | [ ]% |
| [ ] | [ ] | [ ] | [ ] | [ ]% |
| [ ] | [ ] | [ ] | [ ] | [ ]% |

Betweenness and number of contacts have a Spearman correlation of **[value]**, and the network has **[N]** articulation points.

![Bridges](outputs/figures/07_bridges.png)

### 8. Unusual Email Behaviour

Isolation Forest flagged **[N]** employees (the top 5% most isolated profiles): **[n]** unusually active or broad and **[n]** unusually quiet or narrow.

![Anomalies](outputs/figures/08_anomalies.png)

## Conclusion

[One sentence on the overall picture.]

- **Key communicators:** [finding]
- **Communities:** [finding]
- **Departments:** [finding]
- **Information flow:** [finding]
- **Bridges and anomalies:** [finding]

Overall, **[headline sentence]**.

## Recommendations

1. **Use key communicators for announcements and knowledge sharing.** [Name the roles or IDs and why.]
2. **Support the bridges.** [Bridges carry cross-department traffic, so protect against over-reliance on them.]
3. **Strengthen weakly connected departments.** [Name them from section 3 or 4.]
4. **Look into flagged profiles.** Check the most unusual ones for a business explanation first, not a security one.
5. **Add timestamps to the data.** Prediction and anomaly detection would be far stronger with dates and message volumes.

## Assumptions and Limitations

- The data is a public research dataset, not a live company system.
- It is a single snapshot with no timestamps, so changes over time cannot be analysed.
- Links record at least one email, so message volume and content are unknown.
- Department is one flat label, with no seniority or role information.
- Anomalies are statistical outliers, not evidence of wrongdoing.
- The prediction test is a proxy for "becoming" a key communicator, since true future behaviour is not observed.

## Tools and Repository Structure

**Tools:** Python, Pandas, NetworkX, scikit-learn, Matplotlib, Seaborn, SciPy, Jupyter.

```
.
├── README.md
├── requirements.txt
├── Organizational_Network_Intelligence.ipynb   # analysis notebook
├── network_analysis.py                         # same analysis as a script
├── data/                                       # downloaded on first run (git-ignored)
└── outputs/
    ├── figures/                                # chart images used in this README
    ├── tables/                                 # CSV outputs
    └── results_summary.md                      # headline numbers, auto-generated
```

## Author

Aisha Hamza | [LinkedIn](https://www.linkedin.com/in/aisha-hamza-a21188187/) | [Email](mailto:hamzaaisha798@gmail.com)
