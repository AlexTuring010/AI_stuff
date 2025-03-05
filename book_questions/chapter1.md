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

### 6. What Type of Algorithm Would You Use to Allow a Robot to Walk in Various Unknown Terrains?

To enable a robot to walk in unknown terrains, I would use **reinforcement learning**. This type of algorithm allows the robot, known as the **agent**, to learn through trial and error. The robot can observe its environment, choose actions (like stepping in different directions), and receive feedback in the form of **rewards** (for success) or **penalties** (for failure).

Through this process, the robot learns a strategy, known as a **policy**, that helps it maximize rewards over time. Essentially, the policy guides the robot in deciding the best actions to take based on its current situation, even in unfamiliar environments. For example, many robots use reinforcement learning to improve their walking ability by continuously adjusting their movements to adapt to new terrains.

---

### 7. What Type of Algorithm Would You Use to Segment Your Customers into Multiple Groups?

To segment customers into multiple groups, you would use a **clustering algorithm**, which falls under the category of **unsupervised learning**. In clustering, the algorithm groups similar customers together based on shared characteristics, such as purchasing behavior, demographics, or preferences, without requiring pre-labeled data.

For example, if you have data on customer age, income, and buying habits, a clustering algorithm could identify natural groupings within this data—like one group of high-income, frequent buyers and another of budget-conscious, occasional buyers. This segmentation helps you target different customer groups with tailored marketing strategies, improving customer engagement and satisfaction.

---

### 8. Would You Frame the Problem of Spam Detection as a Supervised Learning Problem or an Unsupervised Learning Problem?

Spam detection is typically framed as a **supervised learning** problem because it uses a **labeled training set** with emails marked as "spam" or "ham." The model learns from these labeled examples to classify new emails.

While **unsupervised learning** techniques, like clustering, could group similar emails together, **supervised learning** is generally more effective for spam detection since the goal is to classify emails into specific categories.

---

### 9. What is an Online Learning System?

An **online learning system** is a type of machine learning system that is trained incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, allowing the system to learn about new data on the fly, as it arrives. This approach is useful for systems that need to adapt to changes extremely rapidly, such as detecting new patterns in the stock market, or for situations where computing resources are limited, such as training a model on a mobile device.

### 10. What is Out-of-Core Learning?

**Out-of-core learning** refers to training machine learning models on datasets that are too large to fit into a single machine's main memory. The algorithm loads a part of the data, runs a training step on that data, and repeats the process until it has processed all the data. This approach is typically done offline and is useful for handling huge datasets efficiently.

### Additional Context

In **batch learning**, the system is trained using all available data at once, which can be time-consuming and resource-intensive. This is typically done offline, and the model is then deployed without further learning. However, the model's performance may degrade over time due to changes in the data, a phenomenon known as model rot or data drift. To address this, the model needs to be retrained regularly on up-to-date data.

In contrast, **online learning** allows the system to learn incrementally, adapting to new data as it arrives. This is particularly useful for applications that require rapid adaptation to changing data or have limited computing resources. However, online learning systems must carefully manage the learning rate to balance between quickly adapting to new data and retaining knowledge of old data.

**Out-of-core learning** is another approach for handling large datasets that cannot fit into memory. It processes the data in chunks, making it possible to train models on massive datasets efficiently.

One challenge with online learning is the risk of performance decline if bad data is fed into the system. This can happen due to bugs or malicious attempts to manipulate the system. To mitigate this risk, it is important to monitor the system closely and be prepared to switch off learning or revert to a previous state if performance drops. Anomaly detection algorithms can also help identify and react to abnormal data inputs.

---

### 11. What type of algorithm relies on similarity measure to make predictions?

The type of algorithm that relies on a similarity measure to make predictions is called instance-based learning. In instance-based learning, the system learns the examples by heart and then generalizes to new cases by using a similarity measure to compare them to the learned examples (or a subset of them).

For example, a spam filter using instance-based learning would flag emails that are very similar to known spam emails by measuring the similarity between the new email and the known spam emails. A basic similarity measure could be counting the number of words they have in common.

A specific instance-based learning algorithm is the k-nearest neighbors (k-NN) regression, where the prediction for a new instance is made based on the average of the target values of the k-nearest neighbors in the training set. For example, if you want to predict the life satisfaction for Cyprus, you would find the countries with the closest GDP per capita and average their life satisfaction values to make the prediction.

---

### 12. What is the difference between a model parameter and a model hyperparameter?

**Model parameters** are the internal variables of the model that are learned from the training data. These parameters define the model's behavior and are adjusted during the training process to minimize the error and improve the model's performance. For example, in a linear regression model, the parameters are the coefficients (e.g., `w0` and `w1`) that define the linear relationship between the input features and the target variable.

**Model hyperparameters**, on the other hand, are external configurations set before the training process begins. They control the learning process and the structure of the model but are not learned from the training data. Hyperparameters must be set prior to training and remain constant during the training process. Examples of hyperparameters include the learning rate, the number of hidden layers in a neural network, and the regularization strength.

In summary:

- **Model parameters** are learned from the training data and define the model's behavior.
- **Model hyperparameters** are set before training and control the learning process and model structure.

---

### 13. What do model-based algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?

**Model-based algorithms** search for the optimal parameters that define the model, which best fit the training data. These parameters are adjusted during the training process to minimize the error between the model's predictions and the actual outcomes in the training data.

The most common strategy they use to succeed is to define a performance measure (such as a cost function or utility function) that quantifies how well the model is performing. The goal is to minimize the cost function (or maximize the utility function) by adjusting the model parameters.

**How they make predictions:**

1. **Model Selection:** Choose the type of model and fully specify its architecture. For example, selecting a linear regression model with one input and one output.
2. **Training the Model:** Use a learning algorithm to find the optimal parameters that minimize the cost function. For example, in linear regression, the algorithm adjusts the coefficients (e.g., `w0` and `w1`) to best fit the training data.
3. **Making Predictions:** Once the model is trained, it can make predictions on new data by applying the learned parameters to the input features. For example, in a linear regression model, the prediction is made using the equation `life_satisfaction = w0 + w1 * GDP_per_capita`.

In summary, model-based algorithms search for the optimal parameters that minimize the cost function, use this strategy to succeed, and make predictions by applying the learned parameters to new input data.

---

### 14. Can you name four of the main challenges in machine learning?

1. **Insufficient Quantity of Training Data**: Machine learning algorithms require a large amount of data to perform well. Simple problems may need thousands of examples, while complex problems like image or speech recognition may need millions.

2. **Nonrepresentative Training Data**: The training data must be representative of the new cases the model will generalize to. Nonrepresentative data can lead to inaccurate predictions, especially if the sample is too small or the sampling method is flawed.

3. **Poor-Quality Data**: Errors, outliers, and noise in the training data can make it difficult for the system to detect underlying patterns, reducing its performance. Cleaning up the data is often necessary for better results.

4. **Irrelevant Features**: The training data must contain relevant features and not too many irrelevant ones. Feature engineering, which includes feature selection, feature extraction, and creating new features, is crucial for the success of a machine learning project.

---

### 15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?

If your model performs great on the training data but generalizes poorly to new instances, it is experiencing **overfitting**. Overfitting occurs when the model is too complex relative to the amount and noisiness of the training data, causing it to detect patterns in the noise that do not generalize to new data.

Three possible solutions to overfitting are:

1. **Simplify the Model**: Select a model with fewer parameters (e.g., a linear model instead of a high-degree polynomial model), reduce the number of attributes in the training data, or apply constraints to the model.

2. **Gather More Training Data**: Increasing the quantity of training data can help the model learn more general patterns and reduce the impact of noise.

3. **Reduce Noise in the Training Data**: Clean the training data by fixing data errors and removing outliers to ensure that the model learns from accurate and relevant information.

---

### 16. What is a test set, and why would you want to use it?

A **test set** is a subset of the data that is used to evaluate the performance of a trained model on new, unseen instances. It provides an estimate of the model's generalization error, which indicates how well the model will perform on new data. Using a test set helps ensure that the model is not overfitting the training data and can generalize well to new cases. This is crucial for understanding how the model will perform in real-world scenarios.

**TIP:** It is common to use 80% of the data for training and hold out 20% for testing. However, for very large datasets, holding out even a small percentage (e.g., 1%) can provide a sufficiently large test set to estimate the generalization error accurately.

---

### 17. What is the purpose of a validation set?

A **validation set** is a subset of the training data used to evaluate and compare different models or hyperparameters during the model selection process. It helps in selecting the best model and tuning hyperparameters without overfitting to the test set. After selecting the best model, it is trained on the full training set, and its performance is evaluated on the test set. The validation set ensures that the model selection and hyperparameter tuning are done in a way that generalizes well to new data.

---

### 18. What is the train-dev set, when do you need it, and how do you use it?

A **train-dev set** is a subset of the training data that is used to evaluate the model during training to detect overfitting. It is useful when the training data is not perfectly representative of the production data.

For example, suppose you want to create a mobile app to take pictures of flowers and automatically determine their species. You can easily download millions of pictures of flowers from the web, but they won't be perfectly representative of the pictures that will actually be taken using the app on a mobile device. Perhaps you only have 1000 representative pictures (i.e., actually taken with the app).

In this case, the most important rule is that both the validation set and the test set must be as representative as possible of the data you expect to use in production. You can shuffle the 1000 representative pictures and put half in the validation set and half in the test set. After training your model on the web pictures, if you observe that the performance of the model on the validation set is disappointing, you will not know whether this is because your model has overfit the training set, or whether this is just due to the mismatch between the web pictures and the mobile app pictures.

One solution is to hold out some of the training pictures (from the web) in yet another set called the train-dev set. After the model is trained (on the training set, not on the train-dev set), you can evaluate it on the train-dev set. If the model performs poorly, then it must have overfit the training set, so you should try to simplify or regularize the model, get more training data, and clean up the training data. But if it performs well on the train-dev set, then you can evaluate the model on the validation set. If it performs poorly, then the problem must be coming from the data mismatch. You can try to tackle this problem by preprocessing the web images to make them look more like the pictures that will be taken by the mobile app, and then retraining the model. Once you have a model that performs well on both the train-dev set and the validation set, you can evaluate it one last time on the test set to know how well it is likely to perform in production.

---

### 19. What can go wrong if you tune hyperparameters using the test set?

If you tune hyperparameters using the test set, you risk overfitting the model to the test set. This means the model may perform well on the test set but poorly on new, unseen data. The test set should only be used for the final evaluation of the model's performance, not for tuning hyperparameters or selecting models.

A common solution to this problem is called **holdout validation**. In holdout validation, you split the training data into a reduced training set and a validation set. You train multiple models with various hyperparameters on the reduced training set and select the model that performs best on the validation set. After this process, you train the best model on the full training set (including the validation set) and evaluate it on the test set to get an estimate of the generalization error.

However, if the validation set is too small, the model evaluations will be imprecise, and you may select a suboptimal model. If the validation set is too large, the remaining training set will be much smaller, which is not ideal for training the final model. To address this, you can use repeated cross-validation, where each model is evaluated on multiple small validation sets, and the evaluations are averaged to get a more accurate measure of performance. The drawback is that this approach increases the training time.
