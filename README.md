# 🏠 HouseVision AI  
### Intelligent House Price Prediction Web App  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green?style=for-the-badge&logo=scikit-learn">
  <img src="https://img.shields.io/badge/Visualization-Plotly-purple?style=for-the-badge&logo=plotly">
</p>

---

## 🚀 Overview

**HouseVision AI** is a Machine Learning powered web application that predicts house prices based on key property features such as:

- 🛏 Bedrooms  
- 🛁 Bathrooms  
- 📐 Living Area  
- 🏢 Floors  
- ⭐ Grade  
- 🧱 Basement Area  
- 🏚 Condition  

The project demonstrates an end-to-end ML pipeline from data preprocessing to deployment using Streamlit.

---

## 🎯 Features

✅ Real-time price prediction  
✅ Interactive analytics dashboard  
✅ Plotly-based visualizations  
✅ Prediction history tracking  
✅ Clean and responsive UI  
✅ Deploy-ready structure  

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Preprocessing  
2. Feature Scaling  
3. Regression Model Training  
4. Log Transformation on Target  
5. Model Serialization using Joblib  
6. Web Deployment using Streamlit  

Prediction transformation:

```python
prediction = np.expm1(prediction_log)
