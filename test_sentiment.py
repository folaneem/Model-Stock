import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_section(title, char='='):
    """Helper function to print section headers"""
    print(f"\n{char * 80}\n{title.upper():^80}\n{char * 80}")

def test_news_api(analyzer, ticker):
    """Test NewsAPI functionality"""
    print_section(f"Testing NewsAPI for {ticker}")
    try:
        print(f"Fetching news for {ticker}...")
        news = analyzer.get_company_news(ticker, days=7)
        if news is None or news.empty:
            print("No news articles found.")
            return False
        
        print(f"\nFound {len(news)} news articles:")
        print(news[['title', 'date']].head())
        return True
    except Exception as e:
        logger.error(f"Error testing NewsAPI: {e}")
        return False

def test_reddit_sentiment(analyzer, ticker):
    """Test Reddit sentiment analysis"""
    print_section(f"Testing Reddit Sentiment for {ticker}")
    try:
        if not hasattr(analyzer, 'analyze_reddit_sentiment'):
            print("Reddit sentiment analysis not available in this version.")
            return False
            
        print(f"Analyzing Reddit sentiment for {ticker}...")
        sentiment = analyzer.analyze_reddit_sentiment(ticker, limit_per_subreddit=5)
        
        if not sentiment:
            print("No Reddit sentiment data available.")
            return False
            
        print("\nReddit Sentiment Analysis:")
        for key, value in sentiment.items():
            print(f"{key:>25}: {value:.4f}")
        return True
    except Exception as e:
        logger.error(f"Error testing Reddit sentiment: {e}")
        return False

def test_sentiment_features(analyzer, ticker, days=30):
    """Test the create_sentiment_features method with detailed output"""
    print_section(f"Testing Sentiment Features for {ticker}")
    try:
        print(f"Creating sentiment features for {ticker} over {days} days...")
        features = analyzer.create_sentiment_features(ticker, days=days)
        
        if features is None or features.empty:
            print("No sentiment features were returned.")
            return False
            
        print("\nSample of sentiment features:")
        print(features.head())
        
        # Check for variation in key metrics
        print("\nFeature Statistics:")
        numeric_cols = features.select_dtypes(include=['number']).columns
        for col in ['sentiment_score', 'Price_Momentum', 'Volume_Change', 'Volatility']:
            if col in numeric_cols:
                print(f"{col}: min={features[col].min():.4f}, max={features[col].max():.4f}, "
                      f"mean={features[col].mean():.4f}, std={features[col].std():.4f}")
        
        # Check if we have any variation in the data
        has_variation = any(features[col].std() > 0.01 for col in numeric_cols)
        
        if not has_variation:
            print("\nWARNING: Sentiment features show no variation. This may indicate fallback values are being used.")
        else:
            print("\nSUCCESS: Sentiment features show variation across time.")
            
        return has_variation
        
    except Exception as e:
        logger.error(f"Error testing sentiment features: {e}", exc_info=True)
        return False

def main():
    print_section("Starting Sentiment Analyzer Test")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Try both possible import paths
        try:
            from src.utils.sentiment_analyzer import SentimentAnalyzer
        except ImportError:
            from utils.sentiment_analyzer import SentimentAnalyzer
        
        # Initialize analyzer
        print("\nInitializing SentimentAnalyzer...")
        analyzer = SentimentAnalyzer()
        print("Analyzer initialized successfully.")
        
        # Test with a ticker
        ticker = 'AAPL'  # You can change this to test different stocks
        
        # Test NewsAPI
        news_success = test_news_api(analyzer, ticker)
        
        # Test Reddit Sentiment
        reddit_success = test_reddit_sentiment(analyzer, ticker)
        
        # Test Sentiment Features
        features_success = test_sentiment_features(analyzer, ticker)
        
        # Print summary
        print_section("Test Summary", '=')
        print(f"NewsAPI Test: {'SUCCESS' if news_success else 'FAILED'}")
        print(f"Reddit Test:  {'SUCCESS' if reddit_success else 'FAILED'}")
        print(f"Features Test: {'SUCCESS' if features_success else 'FAILED'}")
        
        if not (news_success or reddit_success):
            print("\nWARNING: No data sources available. Check your API keys and internet connection.")
        
        if not features_success:
            print("\nWARNING: Sentiment features test failed. Please check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
