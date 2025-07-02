# gen-ai-college-feedback-app
sirivarsha1318/gen-ai-project
# 🎓 College Feedback Classifier

An AI-powered Streamlit web app that classifies student feedback into categories like **Academics**, **Facilities**, and **Administration** using rule-based logic and IBM WatsonX Foundation Models.

##  Features

- Classify **single or multiple** feedback entries
- **Batch processing** from CSV uploads
- Categorizes detailed subtopics (e.g., Library → Study Spaces)
- Uses **WatsonX** or **Enhanced Rule-Based** fallback
- Interactive **dashboard** with charts and insights

##  Tech Stack

- Python
- Streamlit
- IBM WatsonX.ai
- Plotly
- Regex-based NLP
- CSV support

##  Setup

```bash
pip install -r requirements.txt
streamlit run app.py
