import os
import sys
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentiment_analyzer():
    """Test the SentimentAnalyzer class"""
    try:
        from src.utils.sentiment_analyzer import SentimentAnalyzer
        
        logger.info("Initializing SentimentAnalyzer...")
        analyzer = SentimentAnalyzer()
        
        # Test with a popular stock ticker
        ticker = "AAPL"
        logger.info(f"Testing sentiment analysis for {ticker}...")
        
        # Get sentiment features
        features = analyzer.create_sentiment_features(ticker)
        
        if features is not None and not features.empty:
            logger.info("Successfully retrieved sentiment features:")
            print("\nSentiment Features:")
            print(features)
            return True
        else:
            logger.error("Failed to retrieve sentiment features")
            return False
            
    except Exception as e:
        logger.error(f"Error testing SentimentAnalyzer: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting SentimentAnalyzer test...")
    success = test_sentiment_analyzer()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")
        sys.exit(1)
