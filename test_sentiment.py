import logging
import sys
from datetime import datetime

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

def main():
    print_section("Starting Sentiment Analyzer Test")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        from src.utils.sentiment_analyzer import SentimentAnalyzer
        
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
        
        # Test combined features
        print_section(f"Testing Combined Features for {ticker}")
        try:
            print("Generating combined sentiment features...")
            features = analyzer.create_sentiment_features(ticker)
            
            print("\nCombined Sentiment Features:")
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    print(f"{key:>30}: {value:.4f}")
                else:
                    print(f"{key:>30}: {value}")
                    
        except Exception as e:
            logger.error(f"Error testing combined features: {e}")
        
        # Print summary
        print_section("Test Summary", '=')
        print(f"NewsAPI Test: {'SUCCESS' if news_success else 'FAILED'}")
        print(f"Reddit Test:  {'SUCCESS' if reddit_success else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Fatal error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
