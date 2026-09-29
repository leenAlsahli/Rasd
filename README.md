<p align="center">
  <img src="App/static/logo-lockup.png" alt="Rasd Project Logo" width="300"/>
</p>

# Rasd - AI Review Analysis & Classification System

Rasd is an intelligent Natural Language Processing (NLP) application designed to analyze and classify customer reviews in Arabic (Saudi dialect) using fine-tuned deep learning models. The system automatically detects Customer Intent and Sentiment to support decision-making and streamline customer support workflows.

---

## Key Features

- Intent Classification: Identifies the primary intent behind a review (Inquiry, Complaint, Suggestion, Praise, etc.).
- Sentiment Analysis: Evaluates customer satisfaction levels (Positive, Negative, Neutral).
- High-Performance Backend: A lightweight API built to execute inference and handle requests quickly.
- Secure Local Execution: Runs entirely within the user's local environment without sending data to external servers, ensuring full confidentiality for sensitive review data.
- Batch Processing: Supports analyzing hundreds of customer reviews in a single request.

---

## Data Development & Model Training Journey

This project extends beyond building the software application; it encompasses a complete end-to-end journey of dataset creation, preprocessing, and model experimentation:

1. Dataset Creation & Annotation: 
   Directly wrote, collected, and formatted a custom dataset of customer reviews in Arabic (Saudi dialect) from multiple sources, manually labeling and categorizing each review for intent and sentiment analysis.

2. Text Preprocessing & Cleaning: 
   Cleaned Arabic text, removed noise and special characters, and normalized linguistic variants to prepare the text for machine learning pipelines.

3. Model Experimentation & Comparative Analysis: 
   Evaluated multiple algorithms to achieve optimal performance:
   - Traditional Algorithms (Logistic Regression, SVM): Achieved 65%-70% accuracy. Fast and lightweight, but struggled with context, sarcasm, and regional colloquialisms.
   - General Language Models (AraBERT): Reached 78%-82% accuracy. Performed well on Modern Standard Arabic, but faced challenges with informal Saudi dialect phrasing.
   - Domain-Specific Model (MARBERT): Achieved over 90% accuracy following fine-tuning. Selected as the primary model due to its superior understanding of the Saudi dialect, subtle complaints, and humor, while maintaining fast inference speed.

4. Fine-Tuning & Adaptation: 
   Fine-tuned the specialized Arabic Transformer model on the custom dataset to maximize classification and prediction accuracy.

---

## Tech Stack

- Programming Language: Python
- Backend Framework: FastAPI / Uvicorn
- NLP & Machine Learning: PyTorch, Hugging Face Transformers, MARBERT
- Data Handling: Pandas, Pathlib
- Frontend: HTML5, CSS3, JavaScript

---

## Project Structure

```text
rr/
├── App/                  # Application core logic, static assets, and UI
│   ├── app.py            # Main application entry point
│   ├── static/           # Images, logo, and static assets
│   └── templates/        # Frontend templates
├── Data/                 # Datasets and analytical files
├── notebooks/            # Jupyter notebooks for training and pipelines
├── requirements.txt      # Project dependencies
├── run.sh                # Execution script
└── README.md             # Project documentation
