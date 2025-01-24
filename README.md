# Consumer Sentiment and Retail Transaction Analysis

## Project Overview
This project analyzes consumer sentiment on Twitter alongside retail transaction data to uncover patterns in spending habits. It demonstrates the ingestion, processing, and analysis of big data using Python and MongoDB.

The main focus includes:
1. **Sentiment Analysis**:
   - Performed on 1.6 million tweets to classify sentiments as positive, negative, or neutral using the TextBlob library.
2. **Retail Transactions**:
   - Explores over 525,000 retail transactions to identify trends in consumer spending patterns.
3. **Date-Based Linking**:
   - Attempts to combine Twitter sentiment data with retail transactions using common dates.

## Features
- **Data Ingestion**:
  - Both datasets (tweets and transactions) are ingested into MongoDB.
- **Sentiment Processing**:
  - Tweets are cleaned and analyzed for sentiment.
- **Trends Over Time**:
  - Trends in sentiment and keyword mentions related to spending habits are visualized.
- **Retail Insights**:
  - Transaction data is explored for quantity and country-based trends.

## How to Run
### Prerequisites
- Python 3.x
- MongoDB
- Libraries: `pandas`, `nltk`, `matplotlib`, `textblob`, etc.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/consumer-sentiment-retail-analysis.git
   cd consumer-sentiment-retail-analysis
