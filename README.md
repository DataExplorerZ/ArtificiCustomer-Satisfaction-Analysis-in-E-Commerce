# Artificial Intelligence Fundamentals: Customer Satisfaction Analysis in E-Commerce

## Overview
This repository contains the implementation of **Artificial Intelligence Fundamentals**, a project aimed at analyzing customer satisfaction in e-commerce using data from an imaginary platform named Shopzilla. The study employs machine learning and deep learning techniques to gain actionable insights into customer behavior and improve service strategies.

---

## **Project Objectives**

1. **Comprehensive Data Analysis**: Identify key features and patterns correlating with customer satisfaction levels.
2. **Model Development**: Build and train machine learning models, including a Decision Tree classifier and a Convolutional Neural Network (CNN), to predict customer satisfaction scores.
3. **Visualization and Insights**: Use data visualization techniques to uncover trends and provide actionable recommendations for enhancing customer service strategies.

---

## **Dataset Description**
The dataset simulates one month of customer interactions on the Shopzilla platform and includes features such as interaction channels, product categories, agent names, customer ratings, and Customer Satisfaction (CSAT) scores.

**Dataset Attributes:**
- **Unique ID**: Identifier for each customer interaction.
- **Channel Name**: Communication channel used (e.g., inbound calls, emails).
- **Category/Sub-category**: Nature of interaction (e.g., product query).
- **Customer Remarks**: Textual feedback from customers.
- **Agent and Manager Details**: Information about service personnel.
- **CSAT Score**: Customer satisfaction rating provided post-interaction.

**Dataset Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ddosad/ecommerce-customer-service-satisfaction/data)

---

## **Implementation Steps**

### 1. **Data Preprocessing**
- Handle missing values using techniques such as row/column removal and imputation.
- Encode categorical features and normalize numerical features.
- Prepare data for training and testing by splitting datasets appropriately.

### 2. **Exploratory Data Analysis (EDA)**
- Visualize trends using box plots, pie charts, histograms, and sunburst charts.
- Analyze factors affecting customer satisfaction, such as agent tenure and interaction channels.

### 3. **Machine Learning Models**
- **Decision Tree Classifier**: Used for feature-based classification of customer satisfaction levels.
- **Convolutional Neural Network (CNN)**: Applied for regression tasks to predict CSAT scores with greater accuracy.

### 4. **Model Evaluation**
- Use metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **mean squared error (MSE)** to assess performance.
- Plot learning curves and model loss trends for detailed analysis.

---

## **Key Features**

- **End-to-End AI Implementation**: From data preprocessing to model deployment.
- **Interactive Visualizations**: Insights through Plotly-based visualizations.
- **Model Optimization**: Regularization, hyperparameter tuning, and dropout layers to prevent overfitting.

---

## **Technologies Used**

- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Plotly, TensorFlow, Keras, Scikit-learn
- **Tools**: Google Colab, Jupyter Notebook, GitHub

---

## **Results**

1. **Decision Tree Classifier**: Achieved 65% accuracy in classifying CSAT scores. High precision and recall for specific classes.
2. **CNN Regression Model**: Showed consistent improvement across epochs with actionable predictions.
3. **Data Visualizations**: Identified key trends and bottlenecks affecting customer satisfaction.

---

## **Ethical Considerations**
- **Data Privacy**: Ensured anonymization of customer data.
- **Fairness**: Regularly audited models for biases to maintain objectivity.

---

## **Future Work**

- Incorporate **Natural Language Processing (NLP)** techniques for analyzing textual customer feedback.
- Explore real-time AI-based customer service solutions.
- Enhance model accuracy by using advanced architectures like Transformers.

---


## **Contact**

**Zeeshan Ali**  
Email: zeeshanali22pch@gmail.com  
