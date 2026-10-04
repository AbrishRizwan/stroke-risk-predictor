# 🩺 Stroke Risk Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stroke-risk-predictor-joqchpdvjmvvyszpeztkwr.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning web application that screens patient vitals and flags elevated stroke risk for early clinical triage.

🔗 **Live Application:** https://stroke-risk-predictor-joqchpdvjmvvyszpeztkwr.streamlit.app

![App Preview](app-preview.png)
📌 Problem & Motivation
Stroke is a leading cause of mortality and long-term disability worldwide. While routine clinical indicators (age, average glucose levels, BMI, hypertension) contain early predictive signals, medical datasets suffer from extreme class imbalance (~4.87% positive prevalence).

Standard out-of-the-box classification models default to maximizing overall accuracy, leading to severe false negative rates (missed stroke patients). This project optimizes clinical sensitivity by calibrating decision thresholds to ensure high-risk cases are flagged early for screening.
