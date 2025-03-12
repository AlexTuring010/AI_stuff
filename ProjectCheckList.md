# Machine Learning Project Checklist

## 1. Frame the Problem and Look at the Big Picture

- Define the objective in business terms.
- Specify how the solution will be used.
- Identify current solutions/workarounds.
- Decide on the problem framing (supervised/unsupervised, etc.).
- Define performance measures aligned with business objectives.
- Determine minimum performance needed to achieve objectives.
- Explore comparable problems and reuse applicable tools.
- Assess availability of human expertise.
- Consider manual solutions and verify assumptions.

## 2. Get the Data

- List required data and quantities.
- Identify and document data sources.
- Estimate storage requirements.
- Address legal obligations and obtain necessary authorizations.
- Obtain access permissions.
- Set up a workspace with sufficient storage.
- Acquire the data.
- Convert data into a manipulable format without altering it.
- Ensure protection of sensitive information.
- Assess data size and type (e.g., time series, geographical).
- Set aside a test set to avoid data snooping.

## 3. Explore the Data

- Create a manageable sample for exploration.
- Utilize Jupyter notebooks for documentation.
- Study each attribute's characteristics and distribution.
- Identify target attributes for supervised learning.
- Visualize data and analyze attribute correlations.
- Evaluate manual problem-solving methods.
- Consider potential data transformations and additional useful data.
- Document findings from exploration.

## 4. Prepare the Data

- Work on copies of the data to maintain integrity.
- Develop functions for data cleaning and transformation.
- Address outliers and missing values.
- Optional: Perform feature selection and engineering.
- Standardize or normalize features as necessary.

## 5. Shortlist Promising Models

- Train initial models from diverse categories.
- Evaluate model performance using cross-validation.
- Analyze significant variables and error types.
- Conduct iterative feature selection and engineering.
- Shortlist top performing models.

## 6. Fine-Tune the System

- Optimize hyperparameters via cross-validation.
- Consider ensemble methods for model combination.
- Measure final model performance on a test set.

## 7. Present Your Solution

- Document the project process and outcomes.
- Prepare a comprehensive presentation highlighting key points.
- Justify how the solution meets business objectives.
- Include insights, assumptions, and limitations.
- Use visualizations to enhance understanding.

## 8. Launch, Monitor, and Maintain

- Integrate solution into production environment.
- Implement monitoring for live performance and data quality.
- Automate regular model retraining on fresh data.
