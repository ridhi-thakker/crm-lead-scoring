# Predictive Analytics for CRM Lead Conversion: A Data-Driven Lead Scoring Framework

## 1. Abstract
This study presents a data-driven framework for predicting whether a CRM lead is likely to convert into a customer. The project combines data preprocessing, feature engineering, model evaluation, threshold analysis, and deployment into a practical system that supports lead prioritization. The primary goal is to provide a structured and reliable approach for improving sales decision-making using historical lead data.

The work is designed to be suitable for academic submission while also demonstrating how predictive analytics can support business operations in real CRM environments.

---

## 2. Introduction
Customer Relationship Management (CRM) systems store large volumes of information about customer interactions, engagement behaviour, and purchase intent. In many organizations, sales teams still rely on manual judgment or simple rules to decide which leads deserve immediate attention. As a result, high-potential leads may be delayed or overlooked, reducing overall conversion efficiency.

This project addresses that challenge by proposing a predictive analytics solution that estimates the likelihood of conversion for each lead. The objective is not only to improve prediction quality, but also to support better decision-making in lead follow-up and prioritization.

---

## 3. Problem Statement
The core problem is that CRM lead data often contains mixed-quality information and many leads do not have equal conversion potential. Without a systematic method of ranking leads, sales teams may spend time on low-value prospects while missing opportunities that require faster action.

A predictive lead scoring model helps solve this problem by identifying leads with stronger conversion likelihood and enabling a more efficient allocation of sales resources.

---

## 4. Objectives of the Study
The objectives of this research are:
1. To analyze the relationship between CRM variables and conversion outcomes.
2. To preprocess and clean the dataset for machine learning use.
3. To create meaningful features that reflect customer engagement and lead quality.
4. To compare multiple classification models for predicting conversion probability.
5. To select the most suitable model for deployment.
6. To build an interactive application that demonstrates the scoring process.

---

## 5. Literature Review / Theoretical Background
Lead scoring is a widely used concept in customer analytics and sales optimization. In practice, organizations rely on both structured data and behavioural indicators to estimate the likelihood that a prospect will convert. Predictive models can support this process by identifying patterns that are not easily visible through manual observation.

The theoretical basis of this project lies in supervised classification, where historical lead data is used to train models that predict future conversion outcomes. The project also uses threshold-based decision-making to translate prediction probabilities into practical business actions.

---

## 6. Dataset Description
The dataset used in this study is the CRM lead dataset stored in [data/Leads.csv](data/Leads.csv). It contains 9,240 leads and 37 columns, covering customer attributes related to lead source, engagement, activity history, and conversion outcome.

The target variable is binary, with 61.46% of the records classified as non-converted and 38.54% as converted. The dataset also contains missing values in several fields, especially in lead quality, profile-related attributes, and tag information, which makes preprocessing an important part of the workflow.

### 6.1 Exploratory Data Analysis Summary
The exploratory analysis shows that engagement-related variables are strongly associated with conversion potential. In particular, the relationship between website activity and conversion is clear: leads with higher interaction levels show a higher conversion rate than those with low engagement. The analysis also indicates that lead source, activity history, and website behavior provide useful signals for distinguishing between likely and unlikely converters.

These observations justify the use of both statistical preprocessing and modelling techniques to translate raw CRM data into decision-support outputs.

---

## 7. Research Methodology
This project follows a structured analytical workflow consisting of data cleaning, feature engineering, model training, evaluation, and deployment.

### 7.1 Data Preprocessing
The preprocessing stage begins with removing irrelevant columns, handling missing values, and standardizing inconsistent labels. For categorical fields, missing values were filled using the most common category, while numerical fields were imputed using median values where appropriate.

A stratified 80/20 train-test split was applied to preserve the original class distribution in both subsets. After the split, the training data was balanced using SMOTE so that the minority class could be represented more effectively during model training. This ensures that the model is trained on a more balanced dataset while the test set remains untouched for unbiased evaluation.

### 7.1.1 Class Imbalance Handling (SMOTE)
The dataset is imbalanced, with the converted class forming a smaller portion of the full sample. To address this issue, SMOTE was applied only to the training set after the split. The training set expanded from the original split configuration to 9,086 samples after balancing, while the test set remained at 1,848 samples.

This approach is important because it prevents information leakage from the test set and allows performance evaluation to reflect the model's behavior on unseen data.

### 7.2 Feature Engineering
Five engineered features were created to capture lead quality and engagement behaviour more explicitly:
- Engagement_Level: a categorical representation of website engagement based on time spent on the website.
- High_Interaction: a binary indicator showing whether the lead has made at least five visits.
- Lead_Source_Group: a grouped version of the lead source to reduce category noise and improve interpretability.
- Is_Webinar_Lead: a binary feature identifying whether the lead originated from a webinar channel.
- Engagement_Score: a composite score combining website time, visit count, and page view activity.

These features were introduced because raw CRM variables alone do not always reveal behavioural intensity or intent clearly. The engineered features help represent patterns that are likely to influence conversion likelihood.

### 7.3 Model Development
Different supervised learning algorithms were evaluated, including Logistic Regression, Random Forest, XGBoost, and SVM. These models were selected to represent four distinct modeling perspectives: a linear baseline, an ensemble-based tree model, a boosting method, and a kernel-based classifier. This range allows the study to compare both interpretability and predictive strength.

### 7.4 Evaluation Strategy
The models were assessed using standard classification metrics such as accuracy, precision, recall, F1 score, and AUC-ROC. These metrics were used to compare predictive performance and determine the most suitable model for deployment.

---

## 8. Results and Analysis
The experimental results show that Logistic Regression produced the best overall balance of performance among the tested models. The selected model achieved an AUC-ROC of 0.9511, an F1 score of 0.8565, and an accuracy of 0.8902 on the test set.

For comparison, the SVM model performed very closely with an AUC-ROC of 0.9486, an F1 score of 0.8612, and an accuracy of 0.8950. Random Forest and XGBoost showed weaker performance in this dataset. These results indicate that Logistic Regression is effective for distinguishing between converted and non-converted leads while maintaining strong interpretability.

### 8.1 Feature Importance (SHAP)
To understand which variables influenced the model most strongly, SHAP analysis was performed on the Logistic Regression model. The analysis identified several important predictors, including engagement-related and source-related features. The most influential variables included tags, SMS-related activity, chat-based lead source indicators, and other engagement signals.

This result supports the earlier exploratory findings that behavioural activity and source context are meaningful predictors of conversion outcomes.

---

## 9. Discussion
The findings suggest that the proposed framework can be used effectively for CRM lead prioritization. The selected model not only performs well statistically, but also offers a practical way to support sales teams in identifying leads with higher conversion potential.

At the same time, model performance should be interpreted in the context of the dataset and business environment. Real-world CRM systems may involve additional factors such as customer behaviour over time, campaign history, and external market conditions. These elements can further improve predictive accuracy if incorporated into future versions of the system.

---

## 10. Salesforce Demo Simulation Design
To make the proposed solution understandable in a real business setting, the project can be demonstrated using a free Salesforce Developer Edition org. In this setup, the Streamlit application acts as the scoring engine, while Salesforce is used as the display and workflow environment where leads are visually routed and prioritized.

The workflow is designed as follows:
1. The model predicts a conversion probability for each lead using the Streamlit application.
2. The predicted scores are exported into a CSV file.
3. The CSV is imported into the Salesforce Lead object using Data Import Wizard or Data Loader.
4. A custom field named `Lead_Score__c` is used to store the predicted score for each record.
5. A Flow Builder automation evaluates the score and assigns routing actions such as Immediate Follow-up, Review Required, or Low Priority.

This simulation is especially useful for presentation purposes because it shows how the analytical output can be connected to a real CRM environment. It also directly addresses the common question of how the model output would actually work inside Salesforce.

---

## 11. Deployment and Demo Workflow
The Streamlit application is sufficient for demonstrating the prediction logic because it allows users to see the score generation process in an interactive format. However, for the final presentation, the Salesforce org provides a stronger visual proof of how the data would appear and be managed in a real customer-facing workflow.

A practical demo flow would be:
- Use the app to generate lead scores from the dataset.
- Export the results into CSV format.
- Import the CSV into Salesforce so that the Lead records contain the calculated `Lead_Score__c` values.
- Open the Salesforce Lead list view to show how records are prioritized.
- Use Flow Builder to simulate threshold-based routing and demonstrate the business action that follows.

This setup does not require a fully integrated enterprise system. Instead, it provides a realistic and convincing prototype that combines analytical prediction with CRM workflow simulation, which is ideal for academic demonstration and stakeholder discussion.

---

## 12. Business Value
This project demonstrates how machine learning can support CRM decision-making by improving lead prioritization and reducing reliance on manual judgment. It helps sales teams focus on leads with higher conversion potential, improves operational efficiency, and strengthens the connection between analytics and business actions.

The main business benefit is that the system can help identify promising leads earlier, allowing teams to respond faster and use their time more effectively. In addition, the Salesforce demo setup shows how the model output can be translated into a practical, visible workflow that is easy to understand for both technical and non-technical audiences.

---

## 13. Tools and Technical Environment
The project uses Python and several supporting libraries for data preparation, visualization, model training, and deployment. The main dependencies are listed in [streamlit-app/requirements.txt](streamlit-app/requirements.txt).

---

## 14. Limitations and Future Scope
Although the project provides a strong working framework, it is important to acknowledge its limitations. The dataset may not fully represent all real-world conditions, and further validation may be required before large-scale deployment.

Future improvements may include live CRM integration, more advanced feature engineering, model tuning, and richer automation support. The long-term vision is to expand this prototype into a more complete intelligent CRM workflow.

---

## 15. Conclusion
This study presents a complete and practical approach to CRM lead scoring using predictive analytics. It combines data understanding, model evaluation, and deployment into a workflow that is suitable for both academic submission and real-world business use.

Overall, the results show that the proposed system can effectively support lead prioritization and decision-making, while also providing a clear foundation for future extension and improvement.

