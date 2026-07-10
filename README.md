# 🏥 Insurance Premium Category Predictor

[[Streamlit App]](https://share.streamlit.io/)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application built using **Streamlit** that predicts the risk category of an individual's insurance premium based on demographic, lifestyle, and financial health metrics.

---

## 📌 Features
* **Intelligent Preprocessing:** Dynamically computes BMI, maps standardized tier groups for Indian cities, and classifies structured lifestyle risk directly from minimal user input.
* **Granular Confidence Mapping:** Provides class-wise probability breakdown alongside the main classification result using the integrated predictive model.
* **Modern Web Interface:** Fast, responsive, and completely interactive web UI powered by Streamlit.

---

## 🏗️ Project Architecture & Workflow
The application eliminates the need for an external API layer by tightly coupling the backend feature engineering modules with the presentation tier for seamless runtime inference on standard cloud environments.

1. **User Input Interface:** Captures age, height, weight, income, habits, city, and occupation.
2. **Feature Engineering Engine:** * $BMI = \frac{\text{Weight (kg)}}{\text{Height (m)}^2}$
   * Rule-based clustering for `Age Group`, `City Tier`, and `Lifestyle Risk`.
3. **Inference Pipeline:** Feeds structured input arrays into the underlying pre-trained machine learning model pipeline.
4. **Result Dashboard:** Renders predicted classification boundaries with real-time confidence scores.

---

## 📁 Repository Structure
```text
├── app.py                  # Main Streamlit web application & feature mapping logic
├── predict.py              # Core machine learning inference and probability mapping script
├── requirements.txt        # Production-ready Python package dependencies
└── model.pkl           # Pre-trained core serialization/classification model file
```
## Output Image
<img width="597" height="591" alt="image" src="https://github.com/user-attachments/assets/1b1e93d7-1d41-4d77-9cd7-e90afdcb7978" />
<img width="506" height="225" alt="image" src="https://github.com/user-attachments/assets/7429ad87-0558-4be8-bbea-98a41495581d" />

