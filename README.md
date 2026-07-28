# 2W1C: Machine-Learning-Based Bicycle Safety Routing for Berlin

## Project Overview

**2W1C** is a machine-learning-based bicycle safety routing project for Berlin.

The goal is to help cyclists choose not only the fastest route, but also a safer route. Many navigation tools optimise for distance or travel time. However, the shortest route may pass through road segments with a higher historical accident risk.

This project combines:

- Berlin bicycle accident data from Unfallatlas
- OpenStreetMap road and bicycle infrastructure data
- Time-based features such as hour, weekday, rush hour, and season
- Machine learning to predict road-segment accident risk
- Route optimisation to compare the fastest route with an ML-based safer route

The final output is an interactive map showing a fastest route and a safer route, together with a simple explanation of the risk difference.

---

## Problem Statement

Cyclists in Berlin need safer route recommendations because the shortest or fastest route may pass through high-risk road segments.

This project asks:

> Can we predict bicycle accident risk for Berlin road segments using historical accident data, road infrastructure, and time features, and then use that prediction to recommend a safer cycling route?

---

## Why This Project Matters

Cycling is an important part of urban mobility, but road safety is still a major concern. Historical accident data can show where accidents happened in the past, but a useful safety-routing system should go further.

Instead of only showing past accident hotspots, this project builds a machine-learning model that predicts relative risk for road segments. The predicted risk is then used as part of a routing engine.

This makes the project more than a map dashboard. It becomes a data science and AI system for safety-aware route recommendation.

---

## Main Idea

The project workflow is:

```text
Unfallatlas bicycle accident data
+
OpenStreetMap road network
+
Time features
↓
Map accidents to road segments
↓
Create positive and negative training samples
↓
Train ML accident-risk model
↓
Predict risk for each road segment
↓
Use predicted risk as routing cost
↓
Compare fastest route vs ML-safest route
↓
Show result in Streamlit