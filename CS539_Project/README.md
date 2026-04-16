# Data Analysis Agent - CS 539 Machine Learning Project

## Team Members

- Hongchao Hu
- Niyati Gohil

## Problem Statement

We want to accelerate the EDA (Exploratory Data Analysis) process for data scientists. This project explores how an LLM-based agent can invoke tools to perform pre-summarization and visualization of tabular datasets.

## Simplified Project Design

The project now uses a minimal flow:

1. Load CSV into a pandas DataFrame.
2. Send user request + schema to Gemini.
3. Execute Gemini-generated matplotlib/seaborn code.
4. Save generated figures and return summary.

Removed complexity:

- No separate inspection/statistics tool modules.
- No large notebook bootstrap/setup workflow.

Primary notebook:

- `Data_Analysis_Agent.ipynb`: simplified 3-cell workflow.
