# Organizational-Network-Intelligence
A project exploring internal communication dynamics through network analysis and machine learning, aimed at uncovering hidden influencers, optimizing collaboration, and strengthening organizational insight.

# Table of Content 
- [Introduction](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#introduction)
- [Research Questions](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#research-questions)
- [Objectives](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#objectives)
- [Data Used](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#dataset-used)
- [Summary Statistics](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#summary-statistics)
- [Methods Applied in the Analysis](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#methods-applied-in-the-analysis)
- [Network Analysis Techniques](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#network-analysis-techniques)
- [Key Findings and Analysis](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#key-findings-and-insights)
- [Visualizations](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#visualizations)
- [Conclusion](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#conclusion)
- [Recommendations](https://github.com/Humairah9/Organizational-Network-Intelligence/blob/main/README.md#recommendations)
  
![images (21)](https://github.com/user-attachments/assets/03df04a3-72a8-41ad-9e6e-d5d3741f39dd)



# Introduction
## Study Overview and Context
This report presents an in-depth analysis of organizational email communication using network analysis techniques. The dataset under study consists of email exchanges among employees, with an additional dataset mapping individuals to their respective departments. Understanding communication patterns within an organization is crucial for optimizing collaboration, identifying key communicators, and detecting potential inefficiencies.

# Research Questions
This study explores eight key questions:
- Which individuals have the highest in-degree and out-degree in the network? What does this reveal about key communicators?
- Are there distinct clusters or communities in the email network? How do these groups interact with one another?
- Do employees communicate more within their department or across departments?
- Which departments have the highest number of outgoing emails? How does this compare to their incoming emails?
- What is the average shortest path between any two individuals? How efficiently does information flow through the network?
- Can we predict which individuals are likely to become key communicators based on their email activity and department affiliation?
- Which individuals serve as "bridges" between different groups? How critical are they for communication across departments?
- Are there individuals who exhibit unusual email activity? Could these anomalies indicate urgent business situations or potential security threats?

# Objectives
The primary objectives of this network analysis are:
- To identify key communicators within the organization based on email exchange patterns.
- To analyze the structure of communication networks and detect potential communities within the organization.
- To assess the balance between intra-departmental and inter-departmental communication.
- To evaluate communication efficiency and information flow across the network.
- To use predictive modeling to identify employees likely to become key communicators.
- To provide actionable recommendations for improving organizational communication and collaboration.
- To detect any anomalies in email communication that may indicate security concerns or business-critical issues.

# Dataset Used
The study utilizes two primary datasets. The first dataset i.e the  Email Communication Dataset, records email exchanges between employees, allowing the creation of a directed network where nodes represent employees and edges represent emails. This helps identify key communicators, bottlenecks, and overall email traffic patterns.
The second dataset i.e the Department Label Dataset, maps employees to departments, enabling the analysis of intra- and inter-department communication. It helps identify departmental collaboration levels, key information hubs, and isolated departments within the network.

# Summary Statistics:
The output provides key statistics about the email communication network. It consists of 1,005 employees (nodes) and 25,571 email interactions (edges). The network density is 0.0253, meaning that only about 2.53% of all possible connections between employees are utilized, indicating a sparse network where most employees do not communicate directly with every other employee.

# Methods Applied in the Analysis
Data Preprocessing and Exploration
- From the analysis, it was found out that the first dataset contains 25,571 email interactions, with two columns: "Sender" and "Receiver," both represented as integer IDs. The second dataset consists of 1,005 employees, with each row mapping an employee (referred to as "Node") to a specific department. This dataset has two columns: "Node" and "Department," both stored as integers.  Both datasets have no missing values, ensuring completeness for network analysis.
- The analysis of email communication across departments reveals significant variations in interaction levels. Certain departments exhibit higher engagement, both in terms of emails sent and received, indicating more active communication networks.
- In terms of network structure, individual nodes demonstrate a clustering coefficient of 1.0, suggesting that their immediate connections are strongly interconnected, forming tightly knit groups. At the department level, clustering coefficients vary, with some departments, like Department 33, displaying a perfect clustering coefficient of 1.0, indicating highly interconnected internal communication. Others show lower values, suggesting weaker intra-departmental connectivity.

# Network Analysis Techniques
- Degree Centrality Analysis: is a fundamental measure in network analysis that helps identify key communicators based on the number of direct connections they have in the network. It was used to identified key communicators based on in-degree (emails received) and out-degree (emails sent).
- Community Detection: is a crucial network analysis technique used to uncover groups of nodes (employees) that interact more frequently with each other than with the rest of the network.  The Louvain method was used to detect clusters in the email network.
- Departmental Communication Analysis: analysis distinguishes between internal and cross-department interactions, ensuring efficient information flow and collaboration. It was used to differentiated intra-department and inter-department communications.
- Shortest Path Analysis: it measures how quickly information spreads in the network. A shorter path means efficient communication, while longer paths indicate delays or bottlenecks. It was used to measure the efficiency of information flow within the network.
- Predictive Modeling: uses a Random Forest classifier to identify key communicators based on network features, helping optimize information flow and leadership recognition. Random Forest classifier was applied to predict key communicators based on network features.

# Key Findings and Insights
- The [graph](https://private-user-images.githubusercontent.com/187115319/443115767-beceae09-15da-431d-8174-fc9d1e026106.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMjc4NzUsIm5iZiI6MTc0NzEyNzU3NSwicGF0aCI6Ii8xODcxMTUzMTkvNDQzMTE1NzY3LWJlY2VhZTA5LTE1ZGEtNDMxZC04MTc0LWZjOWQxZTAyNjEwNi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUxM1QwOTEyNTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hMDE5NDJmMWY3ODE3NjQ5NGJlODFmZmJiMjAxOTNjYjUyNjlkN2E3NDc2ZGUzYjI4NjNmYjliZWRjODUxMGQ4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.exK3oPDTqrlIhud1qiLfz97UwhyKWgQOYF024-pWWkk) highlights key communicators in an email network based on in-degree (emails received) and out-degree (emails sent):
     - **Red nodes** (e.g., 42, 338) have the **highest in-degree**, indicating they **receive the most emails** and likely act as **central contacts or decision-makers**.
     - **Green nodes** (e.g., 21, 303, 411, 424, 498) have the **highest out-degree**, meaning they **send the most emails**, suggesting roles like **team leads or coordinators**.
     - Their **central network positions** show they are critical to **information flow**, while the presence of **clusters and bottlenecks** reveals how communication is structured and possibly constrained within the organization.
- The [graph](https://private-user-images.githubusercontent.com/187115319/443115963-c8f78904-864e-4b6f-8da5-9a69a6730c76.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMjgyNDAsIm5iZiI6MTc0NzEyNzk0MCwicGF0aCI6Ii8xODcxMTUzMTkvNDQzMTE1OTYzLWM4Zjc4OTA0LTg2NGUtNGI2Zi04ZGE1LTlhNjlhNjczMGM3Ni5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUxM1QwOTE5MDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03OTdhOWJjMjc3ZGZmNzJmMWVjYzA5M2EwZjJhNWU4MzgxNmM0YjBkMzIxNDI5ZjI3MjUyYmY0OTJiZWY4ODNkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.txbOWVhwmSZFi_DOokg_AB1yoBrjblZ4TWhiE71qguo) applies the Louvain Method to detect communities in the email network, with each color representing a distinct group of closely connected employees.
   - A central dense cluster indicates a highly connected core group, likely from the same department or key communicators across teams.
   - Several isolated clusters surround the core, suggesting departmental divisions or specialized teams with limited external communication.
   - The separation between clusters highlights potential silos and limited cross-department collaboration.
- The [pie chart](https://private-user-images.githubusercontent.com/187115319/443117736-42d96b3f-95da-497f-91f4-3464264064ab.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMjgzNjQsIm5iZiI6MTc0NzEyODA2NCwicGF0aCI6Ii8xODcxMTUzMTkvNDQzMTE3NzM2LTQyZDk2YjNmLTk1ZGEtNDk3Zi05MWY0LTM0NjQyNjQwNjRhYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUxM1QwOTIxMDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kMzQ3NTI5ZTdmNTA1MjA4NDI0MDQyOWM4NGQ1YzkxOWIyZTNiZmZiZGFiYTAyNGIwYTkzZmMyOGU3ZWIzMjdkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.5aJwncyZKSyUtWF2jG3YBGoR1rWIGHaq_QfasNfU--0) shows that 63.7% of emails are inter-departmental and 36.3% are intra-departmental, indicating more communication occurs across departments than within them. This suggests strong cross-team collaboration, though it may also point to a need for improved internal communication within departments.
- The [chart](https://private-user-images.githubusercontent.com/187115319/443117902-5e3d1218-c966-4f6d-98fb-8ddf61dd1d9d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMjg0NTIsIm5iZiI6MTc0NzEyODE1MiwicGF0aCI6Ii8xODcxMTUzMTkvNDQzMTE3OTAyLTVlM2QxMjE4LWM5NjYtNGY2ZC05OGZiLThkZGY2MWRkMWQ5ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUxM1QwOTIyMzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xNjlmYjEyMjFlMzQ2Yzk1YmEwOTI5MjA2MTIzYTIwNTY3YTVlMWVkYmQ3OGUwMzQ1YTFiMDg5MjMyZjVlZTE0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.W__AzIIrClQDZX4gBeKZDX6h57FroKQqgUJUS8S6RHU) highlights the top 10 departments by email activity, with department 4 leading in outgoing emails, followed by department 36. Both are central to organizational communication, though department 36 sends more emails than it receives, while department 14 receives slightly more. Departments like 21, 7, and 10 show balanced communication, whereas departments 13 and 0 have the lowest activity. Overall, high outgoing email volumes tend to align with high incoming ones, indicating key communicators within the organization.
- The [histogram](https://private-user-images.githubusercontent.com/187115319/443118253-78157da6-f54e-4266-9ae2-4fb32c406e39.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDcxMjg1NzksIm5iZiI6MTc0NzEyODI3OSwicGF0aCI6Ii8xODcxMTUzMTkvNDQzMTE4MjUzLTc4MTU3ZGE2LWY1NGUtNDI2Ni05YWUyLTRmYjMyYzQwNmUzOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNTEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDUxM1QwOTI0MzlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT03NWU0OTA3NTE1OWI3NGI2YWI4ZDkxYjM4YmUzMTliOGJhNjYzNGM1MWQwZDM3MTkyYjg5MTZkNTViMzEzNGQ2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.NstiZLbhhUGSp_m8-nzZRYcMK3dPHBSNgYBUF7Esi4c) shows that most communication paths in the email network are 2 or 3 steps long, indicating efficient information flow and strong overall connectivity. This suggests that employees can typically reach one another with minimal intermediaries. However, the existence of longer paths points to some isolated individuals who require more steps to communicate, potentially slowing information exchange in those cases.
- Overall, the graph confirms that email activity (especially receiving emails) is a strong predictor of influential communicators, while department affiliation has a much weaker influence.

# Visualizations

![image](https://github.com/user-attachments/assets/beceae09-15da-431d-8174-fc9d1e026106)

![image](https://github.com/user-attachments/assets/c8f78904-864e-4b6f-8da5-9a69a6730c76)

![download (2)](https://github.com/user-attachments/assets/e45728bb-1bf9-4b79-b970-0a92023a32dd)

![download (3)](https://github.com/user-attachments/assets/e40309c9-0754-4d18-a085-7fa6165f3271)

![image](https://github.com/user-attachments/assets/42d96b3f-95da-497f-91f4-3464264064ab)

![image](https://github.com/user-attachments/assets/5e3d1218-c966-4f6d-98fb-8ddf61dd1d9d)

![image](https://github.com/user-attachments/assets/78157da6-f54e-4266-9ae2-4fb32c406e39)

![image](https://github.com/user-attachments/assets/df39ae09-e394-46d6-b855-3a57356e9054)

# Conclusion 
Summary of Insights
- The network analysis identified key communicators and their roles within the organization.
- Community detection revealed natural department-based clusters with some bridging individuals.
- Inter-departmental communication was dominant, highlighting strong cross-functional collaboration.
- Predictive modeling successfully identified potential future key communicators, with degree centrality being a strong indicator.

# Recommendations
- Leverage key communicators for organizational announcements and knowledge-sharing.
- Strengthen weakly connected departments by fostering more intra-departmental engagement.
- Monitor anomalies in email activity to detect potential security threats or urgent business needs.
- Enhance predictive modeling by incorporating temporal patterns in email interactions.
