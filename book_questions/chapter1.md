### 1. How Would You Define Machine Learning?

Machine learning is the process of enabling machines to improve at a given task by learning from data rather than relying on explicitly programmed rules.

There are various types of machine learning systems, such as **supervised vs. unsupervised learning**, **batch vs. online learning**, and **instance-based vs. model-based learning**. In an ML project, data is collected into a **training set** and fed into a learning algorithm.

- **Model-based learning** involves adjusting parameters within a model to fit the training data, allowing it to make accurate predictions on both the training set and new data.
- **Instance-based learning** memorizes training examples and generalizes to new data by comparing it to known instances using a similarity measure.

A machine learning system's performance depends on the **quality and quantity of training data**. If the dataset is too small, unrepresentative, noisy, or contains irrelevant features, the model may struggle ("garbage in, garbage out"). Additionally, the model must strike a balance between **simplicity** (to avoid underfitting) and **complexity** (to avoid overfitting).

---

### 2. What Are Four Key Applications Where Machine Learning Excels?

Machine learning is particularly effective in the following scenarios:

1. **Automating Complex Rule-Based Systems** – When traditional solutions require extensive fine-tuning or long lists of rules, ML models can simplify and outperform hardcoded approaches.
2. **Solving Problems with No Clear Traditional Solution** – For complex tasks where conventional methods fail, ML techniques can uncover effective solutions.
3. **Adapting to Changing Environments** – ML models can be retrained on new data, keeping them up to date in dynamic settings.
4. **Extracting Insights from Large Datasets** – Machine learning helps uncover patterns and relationships in vast amounts of data that would be difficult for humans to analyze manually.

---

### 3. What Is a Labeled Training Set?

A **labeled training set** is a dataset where each data point includes both input features and the correct corresponding output (label), which the machine learning model is trained to predict.

For example:

- In a **spam filter**, the labeled training set consists of emails, each marked as either "spam" or "ham" (not spam).
- In a **car price prediction model**, the labeled training set includes cars with attributes like mileage, age, and brand, along with their actual selling prices.

Labeled training sets are fundamental to **supervised learning**, where models learn by comparing their predictions to the actual labels and adjusting accordingly.

---

### 4. What Are the Two Most Common Supervised Learning Tasks?

The two most common supervised learning tasks are **classification** and **regression**.

- **Classification** involves assigning data points to predefined categories. A classic example is a **spam filter**, which is trained on emails labeled as either "spam" or "ham" (not spam) and learns to classify new emails accordingly.
- **Regression** involves predicting a continuous numerical value. For instance, a model that estimates a **car's price** based on features like mileage, age, and brand is performing a regression task. The system learns by analyzing many examples where both the features and the target values (prices) are provided.

Interestingly, some models can be used for both tasks. For example, **logistic regression**, despite its name, is commonly used for classification because it outputs probabilities (e.g., a 20% chance that an email is spam).

---

### 5. Can You Name Four Common Unsupervised Learning Tasks?

In **unsupervised learning**, the training data is **unlabeled**, meaning the system must identify patterns on its own without explicit guidance. Some of the most common unsupervised learning tasks include:

#### 1. **Clustering**

Clustering algorithms identify **groups of similar data points** within a dataset. For example, if you analyze visitor data from your blog, a clustering algorithm might find patterns such as:

- 40% of visitors are **teenagers who love comic books** and visit after school.
- 20% are **adults who enjoy sci-fi** and visit on weekends.

If you use a **hierarchical clustering algorithm**, it might further divide these groups into subgroups, helping you better tailor content to different audiences.

#### 2. **Dimensionality Reduction**

This technique simplifies data by reducing the number of features while preserving as much meaningful information as possible. One common approach is **feature extraction**, where correlated features are merged.  
For instance, in a dataset where a car’s **mileage** and **age** are highly correlated, they might be combined into a single "wear and tear" feature.

Using dimensionality reduction before training a machine learning model has several benefits:

- Reduces memory and storage requirements.
- Speeds up training.
- Sometimes improves performance by eliminating redundant information.

#### 3. **Anomaly Detection & Novelty Detection**

- **Anomaly detection** identifies **unusual or rare occurrences** in data. This is useful for:
  - Detecting fraudulent credit card transactions.
  - Identifying manufacturing defects.
  - Removing outliers from a dataset before training a model.
- **Novelty detection** is similar but focuses on identifying **new instances** that are different from everything seen in the training set. This requires a **clean** training set, meaning it should not contain any instances that you want the algorithm to flag as novel.

  For example, if your training set contains thousands of dog images and **1% of them are Chihuahuas**, a novelty detection algorithm should **not** classify new Chihuahua images as novel, since they were already present in the training data. However, an **anomaly detection algorithm** might still flag Chihuahuas as anomalies if they are rare or significantly different from the other dogs in the dataset (_no offense to Chihuahuas!_).

#### 4. **Association Rule Learning**

This method finds **relationships between features** in large datasets. A well-known example is **market basket analysis**, used in retail. For instance:

- A supermarket may discover that customers who buy **barbecue sauce and potato chips** often **also buy steak**.
- This insight can help businesses optimize product placement and marketing strategies.

Unsupervised learning techniques are valuable for discovering hidden patterns, simplifying complex data, and making predictions without needing labeled examples.

---
