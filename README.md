# Job Recommendation System

## LinkedIn Job Listings Analysis Using Machine Learning  

---

## 1. Introduction
The job market is rapidly transforming due to technological advancements, remote work trends, and evolving industry demands. LinkedIn, as a leading professional networking platform, offers an extensive repository of job postings, providing opportunities for data-driven insights.  

This project analyzes LinkedIn job listings to identify high-demand roles, in-demand skills, and salary benchmarks. It also implements a recommendation system to match job seekers with roles aligned to their profiles. The use of ML, NLP, and explainable AI ensures scalable, accurate, and interpretable insights.

---

## 2. Relevant Work
- Traditional ML models like **Logistic Regression** and **Decision Trees** are widely used for job classification.  
- **Neural network embeddings** improve personalized job recommendations.  
- **Explainable AI (XAI)** techniques like SHAP and LIME enhance model interpretability.  
- This project combines traditional ML, neural embeddings, and XAI to address limitations of prior studies, ensuring robust evaluation across multiple metrics.

---

## 3. Methodology
### 3.1 Data Preprocessing
- Cleaned missing values and outliers  
- Applied feature engineering:
  - TF-IDF for text data  
  - One-hot encoding for categorical variables  
  - Normalization for numerical fields  

### 3.2 Model Development
- **Classification:** Logistic Regression, Random Forest, KNN for job types  
- **Regression:** Neural Networks, Gradient Boosting for salary prediction  
- **Recommendation:** Neural embeddings for personalized job suggestions  

### 3.3 Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score (classification), MSE (regression), Precision/Recall (recommendation)  
- SHAP visualizations used for explainability  

---

## 4. Models
| Model | Use Case | Key Notes |
|-------|---------|-----------|
| Logistic Regression | Job type classification | Interpretable baseline |
| Naive Bayes | Text classification | Simple but affected by class imbalance |
| K-Nearest Neighbors (KNN) | Personalized recommendations | High accuracy for job recommendations |
| Random Forest | Classification | Robust feature importance analysis |
| Gradient Boosting | Salary prediction | Captures complex interactions |
| Neural Networks | Recommendations | Outperforms traditional models using embeddings |

Hyperparameter optimization was applied for all models.

---

## 5. Data
- **Size:** 124,000 LinkedIn job postings  
- **Fields:**
  - Job Titles & Descriptions  
  - Work Types (remote, hybrid, onsite)  
  - Salaries (min, max, median)  
  - Company Info (industry, HQ, employee count)  
  - Skills (technical & soft skills)

### 5.1 Data Cleaning
- Categorical missing values → "Unknown"  
- Salary missing values → median imputation  

### 5.2 Data Enrichment
- Merged external datasets for skills and industries to enhance predictive power  

### 5.3 Data Visualization
- Job titles, locations, work types, and experience levels analyzed  
- Correlation heatmaps used for numerical feature relationships  

---

## 6. Feature Engineering
- **Textual data:** TF-IDF vectorization for job descriptions and skills  
- **Categorical encoding:** One-hot encoding for work type and experience level  
- **Numerical normalization:** Min-Max scaling for salary fields  

---

## 7. Training and Testing
- **Data split:** 80% training / 20% testing with stratified sampling  
- **Cross-validation:** 10-fold  
- **Hyperparameter tuning:** Optuna  
- **Oversampling:** Handled class imbalance  
- **Ensemble methods:** Combined multiple models for stability  

---

## 8. Results
### 8.1 Logistic Regression
- Accuracy: 65.26% | AUC: 71.85% | F1-score: 55.32%  
- Moderate performance; captures text-based patterns  

### 8.2 Naive Bayes
- Accuracy: 68.99% | AUC: 68.00% | F1-score: 12.20%  
- Poor recall due to class imbalance  

### 8.3 KNN
- 10-fold CV Accuracy: 97.09% | Test Accuracy: 94.97% | F1-score: 92.12%  
- High performance for personalized recommendations  

### 8.4 Random Forest
- Test Accuracy: 92.09% | AUC: 97.41% | F1-score: 87.85%  
- Robust and interpretable  

### 8.5 Gradient Boosting
- Test Accuracy: 91.67% | AUC: 96.76% | F1-score: 87.24%  
- Strong predictive power  

### 8.6 Job Recommendations
- Neural embedding-based system provides personalized job suggestions  
- SHAP plots highlight skills and location influencing recommendations  

### 8.7 Explainable AI
- SHAP visualizations provide transparency and trust  
- Helps stakeholders validate and interpret recommendations  

---

## 9. Discussion
- KNN and Random Forest excel at classification; Neural Networks for recommendations  
- Challenges: class imbalance, high computational costs  
- Feature engineering and XAI improved accuracy and trust  

---

## 10. Conclusion
- ML and NLP successfully analyzed LinkedIn job listings  
- High-performing models for classification, regression, and recommendation  
- SHAP visualizations enhanced interpretability  
- Future work: longitudinal data, real-time recommendation optimization  

---

## 11. Folder Structure

Job-Recommendation-System/

─ cleaned_text.csv

─ cleaned_text_with_job_info.csv

─ postings.csv

─ companies/          # Company-related datasets

─ jobs/               # Job-related datasets

─ mappings/           # Mappings for skills, industries.

─ ML4GEN.ipynb        # Main project notebook

─ NLP.ipynb           # NLP analysis notebook
