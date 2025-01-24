# COMP548DL-01-f-2024-Final-Project

# Big Data Consumer Sentiment Analysis

## Project Overview
This project analyzes consumer sentiment on Twitter and retail transaction data to uncover patterns in spending habits and purchasing power during inflationary periods. The primary focus is on using MongoDB for data management, Python for data analysis, and visualizations to provide actionable insights.

## Key Features
1. **Sentiment Analysis**:
   - Conducted on 1.6 million tweets using the TextBlob library to classify tweets into positive, negative, or neutral sentiments.
2. **Retail Transactions**:
   - Analyzed 525,000+ transactions to examine purchasing trends.
3. **Combined Insights**:
   - Attempted to merge sentiment data with transactional data using common dates, though no overlap was found.
4. **Keyword Trends**:
   - Identified and visualized trends in spending and purchasing power-related tweets over time.


## Setup Instructions
### Prerequisites
- Python 3.x
- MongoDB
- Libraries: `pandas`, `nltk`, `matplotlib`, `textblob`, `sklearn`, etc.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/big-data-consumer-sentiment.git
   cd big-data-consumer-sentiment
