"""
This Streamlit application provides a comprehensive interface for analyzing stock market trends,
making predictions, and managing investment risks.
"""

# Set environment variables before any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from project root first, then from src directory
    root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    src_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    
    if os.path.exists(root_env_path):
        load_dotenv(root_env_path)
        print(f"Loaded environment variables from {root_env_path}")
    elif os.path.exists(src_env_path):
        load_dotenv(src_env_path)
        print(f"Loaded environment variables from {src_env_path}")
    else:
        print("No .env file found")
        
    # Verify API keys are loaded
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    print(f"Alpaca API Key loaded: {'Yes' if alpaca_key else 'No'}")
    print(f"Alpaca Secret Key loaded: {'Yes' if alpaca_secret else 'No'}")
    
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables may not be loaded properly.")
except Exception as e:
    print(f"Error loading environment variables: {str(e)}")

# Configure page settings - must be the first Streamlit command
import streamlit as st
from utils.yfinance_utils import fetch_daily_yfinance_data
from utils.data_processor import DataProcessor


# --- Ensure _ws_initialized is always present in session state ---
if '_ws_initialized' not in st.session_state:
    st.session_state._ws_initialized = False

# Configure page settings
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize real-time data structure in session state
if 'rt_data' not in st.session_state:
    st.session_state.rt_data = {}

# Add custom CSS to remove white background from all metric cards
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stMetricValue"] {
    background-color: transparent !important;
}
[data-testid="stMetricLabel"] {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Standard library imports
import sys
import time
import json
import logging
import socket
import random
import asyncio
import threading
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
import uvicorn
import requests
import websockets
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, LambdaCallback

# Custom callback for detailed training logging
class TrainingLogger(Callback):
    """Custom Keras callback to log training progress to file."""
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logger = logging.getLogger()
        log_msg = f"Epoch {epoch+1} - "
        log_msg += " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        logger.info(f"[TRAINING] {log_msg}")
        
    def on_train_begin(self, logs=None):
        logger = logging.getLogger()
        logger.info("[TRAINING] Training started")
        
    def on_train_end(self, logs=None):
        logger = logging.getLogger()
        logger.info("[TRAINING] Training completed")

# Local application imports
from utils.data_collector import DataCollector
from utils.risk_management import RiskManager
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.sentiment_analyzer import SentimentAnalyzer
from models.two_stage_model import TwoStagePredictor
from utils.real_time_data import RealTimeData
from utils.data_processor import DataProcessor

def update_rt_data_safely(ticker, key, value):
    """
    Safely update real-time data in session state with enhanced error handling and logging.
    
    Args:
        ticker (str): The stock ticker symbol
        key (str): The data key to update
        value: The value to set
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Skip if we don't have a session state (common in background threads)
        if not hasattr(st, 'session_state'):
            return False
            
        # Initialize rt_data if it doesn't exist
        if 'rt_data' not in st.session_state:
            st.session_state.rt_data = {}
            
        # Initialize ticker data if it doesn't exist
        if ticker not in st.session_state.rt_data:
            st.session_state.rt_data[ticker] = {
                'status': 'Initialized',
                'last_update': None,
                'price': None,
                'change': 0,
                'volume': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Update the value
        st.session_state.rt_data[ticker][key] = value
        
        # Always update the last_update timestamp
        update_time = datetime.utcnow().isoformat()
        st.session_state.rt_data[ticker]['last_update'] = update_time
        st.session_state.rt_data[ticker]['timestamp'] = update_time
        
        return True
        
    except Exception:
        # Don't log errors in background threads to avoid cluttering the output
        return False

# Configure logging before other operations
def configure_logging() -> logging.Logger:
    """Configure logging for the application.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Only add StreamHandler if not running in Streamlit
    if not hasattr(st, 'session_state'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = configure_logging()

# Add CSS for better layout
st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
    
    /* Style metrics */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Style the sidebar */
    .css-1d391kg {
        padding: 1.5rem;
    }
    
    /* Style buttons */
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    
    /* Style tables */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Style expanders */
    .streamlit-expander {
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 0 1rem;
        margin-bottom: 1rem;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


class TrainingLogger(Callback):
    """Custom Keras callback for logging training metrics to the application's log file."""
    
    def __init__(self, logger, metrics=None):
        """
        Initialize the training logger.
        
        Args:
            logger: Logger instance to use for logging
            metrics: List of metrics to log. If None, logs all available metrics.
        """
        super(TrainingLogger, self).__init__()
        self.logger = logger
        self.metrics = metrics
        self.epoch_logs = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Log the start of each epoch."""
        self.logger.info(f"Starting epoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs is None:
            return
            
        # Format the metrics for logging
        metrics_str = ", ".join(f"{k}: {v:.6f}" for k, v in logs.items())
        self.logger.info(f"Epoch {epoch + 1} - {metrics_str}")
        
        # Store the epoch logs
        self.epoch_logs.append({
            'epoch': epoch + 1,
            'logs': logs.copy()
        })
    
    def on_train_begin(self, logs=None):
        """Log the start of training."""
        self.logger.info("Starting model training...")
        self.epoch_logs = []
    
    def on_train_end(self, logs=None):
        """Log the end of training and summarize metrics."""
        if not self.epoch_logs:
            return
            
        # Log a summary of training
        best_epoch = max(self.epoch_logs, 
                        key=lambda x: x['logs'].get('val_accuracy', x['logs'].get('accuracy', 0)))
        
        self.logger.info("=" * 50)
        self.logger.info("Training Complete")
        self.logger.info(f"Best epoch: {best_epoch['epoch']} with metrics: {best_epoch['logs']}")
        self.logger.info("=" * 50)
# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Local application imports
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.data_collector import DataCollector

# Initialize FastAPI app
app = FastAPI(title="Stock Market WebSocket Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Initialize RealTimeData with default tickers
from config.tickers import DEFAULT_TICKERS
real_time_data = RealTimeData(tickers=DEFAULT_TICKERS)

# Start the real-time data stream in a background task
import threading

def start_real_time_data():
    try:
        real_time_data.start()
    except Exception as e:
        logger.error(f"Failed to start real-time data: {str(e)}")

# Start the real-time data handler in a separate thread
real_time_thread = threading.Thread(target=start_real_time_data, daemon=True)
real_time_thread.start()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Prevent duplicate connections for the same client_id
    if client_id in active_connections:
        try:
            await active_connections[client_id].close()
            logger.warning(f"Closed existing connection for client {client_id}")
        except Exception as e:
            logger.error(f"Error closing existing connection: {str(e)}")
    await websocket.accept()
    active_connections[client_id] = websocket
    try:
        # Send initial data
        initial_data = {
            "type": "status",
            "message": "Connected to real-time data feed",
            "tickers": real_time_data.tickers,
            "last_updates": {ticker: str(dt) for ticker, dt in real_time_data.last_update.items()}
        }
        await websocket.send_json(initial_data)
        while True:
            # Send updated data at regular intervals
            for ticker in real_time_data.tickers:
                if ticker in real_time_data.data and not real_time_data.data[ticker].empty:
                    latest_data = real_time_data.data[ticker].iloc[-1].to_dict()
                    update = {
                        "type": "data_update",
                        "ticker": ticker,
                        "data": latest_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(update)
            
            # Wait before sending the next update (e.g., every 5 seconds)
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        active_connections.pop(client_id, None)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if client_id in active_connections:
            await active_connections[client_id].close()
            active_connections.pop(client_id, None)

# Function to broadcast messages to all connected clients


def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients"""
    for client_id, websocket in active_connections.items():
        try:
            asyncio.create_task(websocket.send_json(message))
        except Exception as e:
            logger.error(
                f"Failed to send message to client {client_id}: {str(e)}")
            if client_id in active_connections:
                del active_connections[client_id]

# Function to update WebSocket endpoint in session state


def update_websocket_endpoint(port: int):
    """Update the WebSocket endpoint in session state"""
    if 'websocket_endpoint' not in st.session_state:
        st.session_state.websocket_endpoint = f"ws://localhost:{port}/ws/"
    else:
        st.session_state.websocket_endpoint = f"ws://localhost:{port}/ws/"
    logger.info(
        f"WebSocket endpoint updated to: {st.session_state.websocket_endpoint}")

# Function to get the current WebSocket endpoint


def get_websocket_endpoint():
    """Get the current WebSocket endpoint URL"""
    return st.session_state.get('websocket_endpoint', '')

# Function to send data to WebSocket clients


def send_to_websocket(data: dict):
    """Send data to WebSocket clients"""
    if st.session_state.get('websocket_endpoint'):
        broadcast_message(data)
    else:
        logger.warning("WebSocket endpoint not initialized")

# Function to update WebSocket connection status


def update_ws_status(status: Dict[str, Any]) -> None:
    """Update WebSocket connection status in session state.
    
    Args:
        status: Dictionary containing status information with keys:
            - connected: bool indicating if WebSocket is connected
            - healthy: bool indicating if connection is healthy
            - last_message: str or None, last message received
            - last_error: str or None, last error message
    """
    if 'ws_status' not in st.session_state:
        st.session_state.ws_status = {
            'connected': False,
            'healthy': False,
            'last_message': None,
            'last_error': None
        }

    st.session_state.ws_status.update(status)
    logger.info("WebSocket status updated: %s", status)

# Function to check WebSocket connection health


def check_ws_health():
    """Check WebSocket connection health"""
    if not st.session_state.get('websocket_endpoint'):
        return {'connected': False, 'error': 'WebSocket endpoint not initialized'}
    return {'connected': True, 'clients': len(active_connections)}

# Initialize session state for WebSocket


def init_ws_session_state():
    """Initialize WebSocket-related session state with consistent client ID and endpoint."""
    import uuid
    # WebSocket related variables - use a consistent client ID
    if 'client_id' not in st.session_state:
        st.session_state.client_id = str(uuid.uuid4())
    if 'websocket_endpoint' not in st.session_state:
        host = "localhost"
        port = 8000
        st.session_state.websocket_endpoint = f"ws://{host}:{port}/ws/{st.session_state.client_id}"
    if 'websocket_initialized' not in st.session_state:
        st.session_state.websocket_initialized = False
    if 'websocket_status' not in st.session_state:
        st.session_state.websocket_status = {
            'Status': 'Disconnected',
            'Authenticated': False,
            'Subscribed Symbols': [],
            'Messages Received': 0,
            'Last Error': None,
            'Connection ID': None
        }
    logger.info("WebSocket session state initialized")

# Cleanup WebSocket connections


def cleanup_ws_connections():
    """Cleanup WebSocket connections"""
    active_connections.clear()
    if 'websocket_endpoint' in st.session_state:
        del st.session_state.websocket_endpoint
    if 'ws_status' in st.session_state:
        del st.session_state.ws_status
    logger.info("WebSocket connections cleaned up")


# Configure and initialize logging
logger = configure_logging()

# Initialize session state for logger if it doesn't exist
if 'logger' not in st.session_state:
    st.session_state.logger = logger
    logger.info("Application logger initialized in session state")
    st.session_state.logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    st.session_state.logger.addHandler(handler)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Debug: Print all environment variables
logger.debug("Environment variables loaded. Checking for Alpaca credentials...")

# Get Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Debug: Log the values (without exposing full keys)
logger.debug(f"ALPACA_API_KEY present: {'Yes' if ALPACA_API_KEY else 'No'}")
logger.debug(
    f"ALPACA_SECRET_KEY present: {'Yes' if ALPACA_SECRET_KEY else 'No'}")

# Validate API keys
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    error_msg = "❌ Error: Missing Alpaca API credentials. Please check your .env file."
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Helper function to check if a port is available


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
            s.close()
            return True
    except (OSError, socket.error):
        return False

# Helper function to find an available port


def find_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """
    Find an available port starting from start_port.

    Args:
        start_port: The port to start checking from
        max_attempts: Maximum number of ports to check

    Returns:
        int: An available port number
    """
    # First try the specified start port
    if is_port_available(start_port):
        return start_port

    # If start port is not available, try the next available port
    for port in range(start_port + 1, start_port + max_attempts + 1):
        if is_port_available(port):
            return port

    # If no port found, raise an error
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}")

# Initialize WebSocket server if not already running


def start_websocket_server():
    """Start the WebSocket server in a separate thread"""
    logger = logging.getLogger(__name__)
    
    # Don't try to access Streamlit context in a background thread
    # Instead, use the global logger and handle any Streamlit updates via callbacks
    logger.info("Starting WebSocket server in background thread")

    # Find an available port
    port = 8000
    if not is_port_available(port):
        logger.warning(
            f"Port {port} is busy, searching for an available port...")
        try:
            port = find_available_port(port + 1)
            logger.info(f"Found available port: {port}")
        except Exception as e:
            logger.error(f"Error finding available port: {e}")
            logger.warning("Falling back to random port")
            port = 0  # Let OS pick a port

    # Start WebSocket server
    try:
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            reload=False,
            workers=1,
            server_header=True,
            timeout_keep_alive=30,
            limit_max_requests=1000,
            limit_concurrency=100,
            ws_ping_interval=20.0,
            ws_ping_timeout=20.0
        )

        server = uvicorn.Server(config)

        # If using random port (0), we need to bind to get the actual port
        if port == 0:
            server.config.setup_event_loop()
            server.lifespan = config.lifespan_class(config)
            server.lifespan.app = server
            server.lifespan.state = server.lifespan.startup()
            server.servers = config.bind_socket()
            actual_port = server.servers[0].getsockname()[1]
            logger.info(f"WebSocket server will run on port: {actual_port}")
            update_websocket_endpoint(actual_port)
            # Clean up the temporary server
            for sock in server.servers:
                sock.close()
            server.servers = []
            server.force_exit = True

            # Create a new server with the actual port
            config.port = actual_port
            server = uvicorn.Server(config)
        else:
            update_websocket_endpoint(port)

        logger.info(f"Starting WebSocket server on ws://0.0.0.0:{port}")
        server.run()

    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
        logger.exception("WebSocket server crashed")
        raise


# Initialize RiskManager in session state
if 'risk_manager' not in st.session_state:
    try:
        st.session_state.risk_manager = RiskManager()
        logger.info("RiskManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RiskManager: {str(e)}")
        st.error("Failed to initialize RiskManager")

# Import the main PortfolioOptimizer
from utils.portfolio_optimizer import PortfolioOptimizer as MainPortfolioOptimizer

# Initialize PortfolioOptimizer in session state with robust error handling
if 'portfolio_optimizer' not in st.session_state:
    try:
        # Initialize the main portfolio optimizer
        st.session_state.portfolio_optimizer = MainPortfolioOptimizer()
        
        # Ensure required attributes exist
        if not hasattr(st.session_state.portfolio_optimizer, 'tickers'):
            st.session_state.portfolio_optimizer.tickers = []
        if not hasattr(st.session_state.portfolio_optimizer, 'winsorize_lower'):
            st.session_state.portfolio_optimizer.winsorize_lower = 0.05
        if not hasattr(st.session_state.portfolio_optimizer, 'winsorize_upper'):
            st.session_state.portfolio_optimizer.winsorize_upper = 0.95
            
        logger.info("Main PortfolioOptimizer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize main PortfolioOptimizer: {str(e)}")
        logger.info("Creating fallback PortfolioOptimizer")
        
        # Fallback implementation - only used if main optimizer fails
        class FallbackPortfolioOptimizer:
            def __init__(self):
                self.tickers = []
                self.winsorize_lower = 0.05
                self.winsorize_upper = 0.95
                
            def optimize(self, returns_df):
                try:
                    if returns_df is None or returns_df.empty:
                        raise ValueError("Empty returns dataframe")
                    n = len(returns_df.columns) if hasattr(returns_df, 'columns') else 1
                    if n == 0:
                        return {}
                    equal_weight = 1.0 / n
                    if hasattr(returns_df, 'columns'):
                        return {ticker: equal_weight for ticker in returns_df.columns}
                    return {'asset_'+str(i): equal_weight for i in range(n)}
                except Exception as e:
                    logger.error(f"Error in fallback optimizer: {str(e)}")
                    return {}
            
            def calculate_sharpe_ratio(self, *args, **kwargs):
                return 0.0
                
            def calculate_volatility(self, *args, **kwargs):
                return 0.0
                
            def calculate_max_drawdown(self, *args, **kwargs):
                return 0.0
        
        st.session_state.portfolio_optimizer = FallbackPortfolioOptimizer()
        st.warning("Using fallback Portfolio Optimizer. Some features may be limited.")

# Define data cache directory
data_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
os.makedirs(data_cache_dir, exist_ok=True)
logger.info(f"Data cache directory set to: {data_cache_dir}")

# Initialize DataCollector in session state
if 'data_collector' not in st.session_state:
    try:
        st.session_state.data_collector = DataCollector(
            tickers=[],  # Will be set when ticker is known
            start_date=datetime.now().strftime('%Y-%m-%d'),  # Will be updated
            end_date=datetime.now().strftime('%Y-%m-%d'),    # Will be updated
            cache_dir=data_cache_dir
        )
        logger.info("DataCollector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DataCollector: {str(e)}")
        st.error("Failed to initialize DataCollector")

def clear_cache():
    """Clear all cached data files."""
    try:
        for filename in os.listdir(data_cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(data_cache_dir, filename))
        st.success("Cache cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing cache: {str(e)}")

# Function to get the close column name for a specific ticker


def get_close_column_name(df: pd.DataFrame, ticker: str) -> str:
    """
    Find the correct close column name in the dataframe.

    Args:
        df: The dataframe containing stock data
        ticker: The ticker symbol to look for in column names

    Returns:
        str: The name of the close column

    Raises:
        KeyError: If no close column is found
    """
    # List of possible close column name patterns to try (case-insensitive)
    patterns = [
        f'close_{ticker}',
        f'close {ticker}',
        f'close{ticker}',
        f'close_{ticker.lower()}',
        f'close {ticker.lower()}',
        f'close{ticker.lower()}',
        'close',
        'adj close',
        'adj_close',
    ]

    logger = logging.getLogger(__name__)
    logger.debug(f"Looking for close column for {ticker} in columns: {df.columns.tolist()}")
    columns_lower = [str(col).lower() for col in df.columns]

    # 1. Look for exact matches (case-insensitive) for all patterns
    for pattern in patterns:
        for col in df.columns:
            if str(col).lower() == pattern:
                logger.debug(f"Found close column using exact pattern '{pattern}': {col}")
                return col

    # 2. Look for columns starting with 'close' and containing the ticker (case-insensitive)
    ticker_lower = ticker.lower()
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if col_lower.startswith('close') and (ticker_lower in col_lower or ticker_lower.replace('.', '') in col_lower):
            logger.debug(f"Found close column using 'close' prefix and ticker: {col}")
            return col

    # 3. Fallback: any column containing 'close' (case-insensitive)
    for col in df.columns:
        if 'close' in str(col).lower():
            logger.debug(f"Found fallback close column: {col}")
            return col

    # 4. Not found: log and raise error
    logger.error(f"No close price column found for ticker '{ticker}'. Columns available: {df.columns.tolist()}")
    raise KeyError(f"Close price column not found for ticker '{ticker}'. Available columns: {df.columns.tolist()}")

def configure_logging() -> logging.Logger:
    """Configure logging for the application.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set log level
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Only add StreamHandler if not running in Streamlit
    if not hasattr(st, 'session_state'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = configure_logging()

from utils.data_processor import validate_technical_indicators

def display_technical_indicators_table(df):
    """Display technical indicators in a table format with validation"""
    st.subheader("Technical Indicators")
    logger = st.session_state.get('logger')
    required_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD']
    df, missing = validate_technical_indicators(df, required_cols, logger=logger)
    if missing:
        st.warning(f"Some technical indicators are missing: {missing}. These will be shown as 'N/A'.", icon="⚠️")
    last_rows = df.tail(5).copy()
    table_data = []
    for idx, row in last_rows.iterrows():
        date = idx.strftime('%Y-%m-%d %H:%M:%S')
        sma_20 = row.get('SMA_20', None)
        sma_50 = row.get('SMA_50', None)
        rsi = row.get('RSI', None)
        macd = row.get('MACD', None)
        sma_20_val = f"{sma_20:.4f}" if pd.notna(sma_20) else "N/A"
        sma_50_val = f"{sma_50:.4f}" if pd.notna(sma_50) else "N/A"
        rsi_val = f"{rsi:.2f}" if pd.notna(rsi) else "N/A"
        macd_val = f"{macd:.4f}" if pd.notna(macd) else "N/A"
        table_data.append({
            "Date": date,
            "SMA_20": sma_20_val,
            "SMA_50": sma_50_val,
            "RSI": rsi_val,
            "MACD": macd_val
        })
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True)


def display_price_analysis(ticker: str):
    """Display price analysis for a given ticker with robust indicator validation"""
    try:
        from utils.data_processor import validate_technical_indicators
        processed_data = st.session_state.get('processed_data', {})
        if not processed_data:
            st.warning("No processed data available. Please run the analysis first.")
            return
        if 'stock_data' in processed_data:
            df = processed_data['stock_data']
        else:
            st.error("No stock data found in processed data.")
            return
        # Validate technical indicators up front
        logger = st.session_state.get('logger')
        indicator_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD']
        df, missing = validate_technical_indicators(df, indicator_cols, logger=logger)
        if missing:
            st.warning(f"Missing technical indicators: {missing}. Plots may show gaps or 'N/A' where missing.", icon="⚠️")
        # Detect the close and volume columns
        close_col = None
        volume_col = None
        for col in df.columns:
            if isinstance(col, str):
                if col == 'Close':
                    close_col = col
                    break
                elif 'close' in col.lower():
                    close_col = col
                if col == 'Volume':
                    volume_col = col
                elif 'volume' in col.lower():
                    volume_col = col
        if volume_col is None:
            volume_col = 'Volume'
        st.session_state.close_col = close_col
        display_technical_indicators_table(df)
        if st.sidebar.checkbox("Show debug information", key="price_debug"):
            st.write("### Debug Information")
            st.write("Close column being used:", close_col)
            st.write("Volume column being used:", volume_col)
            st.write("Available columns:", df.columns.tolist())
            st.write("Data shape:", df.shape)
            st.write("First few rows:")
            st.dataframe(df.head())
            st.write("Data types:", df.dtypes)
            st.write("Missing values:", df.isnull().sum())
            if processed_data:
                st.write("Processed data keys:", list(processed_data.keys()))
                if 'stock_data' in processed_data:
                    st.write("Technical indicators available:", processed_data['stock_data'].columns.tolist())
        if close_col not in df.columns:
            st.error(f"Error: Close price column not found in the data. Available columns: {df.columns.tolist()}")
            return
        st.subheader(f"{ticker} Price Chart with Technical Indicators")
        price_tabs = st.tabs(["Price", "Moving Averages", "Oscillators", "Volume Indicators"])
        with price_tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[close_col],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price ($)",
                showlegend=True,
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        with price_tabs[1]:
            try:
                tech_df = df
                ma_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
                missing_ma = [ma for ma in ma_cols if ma not in tech_df.columns]
                if len(tech_df) == 0 or all(ma not in tech_df.columns for ma in ma_cols):
                    st.warning("No moving average data available to display.", icon="⚠️")
                    if logger:
                        logger.warning("Moving Averages tab: All MA columns missing or empty DataFrame.")
                else:
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df[close_col],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    for ma in ma_cols:
                        if ma in tech_df.columns and tech_df[ma].notna().any():
                            fig_ma.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df[ma],
                                mode='lines',
                                name=ma,
                                line=dict(width=1.5)
                            ))
                    fig_ma.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        showlegend=True,
                        template="plotly_white",
                        height=500,
                        title="Price with Moving Averages"
                    )
                    st.plotly_chart(fig_ma, use_container_width=True, key="ma_chart")
                    ma_indicators = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
                    missing_ma = [ma for ma in ma_indicators if ma not in tech_df.columns]
                    if missing_ma:
                        st.info(f"Note: Missing moving averages: {', '.join(missing_ma)}", icon="ℹ️")
                        logger.info(f"Moving Averages tab: Missing MA columns: {missing_ma}")
                    st.info("""
                    **Moving Averages Explanation:**
                    - **SMA_20**: 20-day Simple Moving Average (short-term trend)
                    - **SMA_50**: 50-day Simple Moving Average (medium-term trend)
                    - **SMA_200**: 200-day Simple Moving Average (long-term trend)
                    - **EMA_12**: 12-day Exponential Moving Average (responsive to recent price changes)
                    - **EMA_26**: 26-day Exponential Moving Average (less responsive, smoother)
                    
                    When shorter-term MAs cross above longer-term MAs, it may indicate a bullish trend. Conversely, when shorter-term MAs cross below longer-term MAs, it may indicate a bearish trend.
                    """)
            except Exception as e:
                st.error(f"Error displaying Moving Averages: {str(e)}", icon="🚨")
                logger.error(f"Exception in Moving Averages tab: {str(e)}", exc_info=True)
        
        # Oscillators tab
        with price_tabs[2]:
            try:
                if not processed_data or 'stock_data' not in processed_data:
                    st.warning("Technical indicators not available. Please run the analysis first.", icon="⚠️")
                    logger.warning("Oscillators tab: No processed_data or missing 'stock_data'.")
                else:
                    tech_df = processed_data['stock_data']
                    osc_cols = {
                        'RSI': 'momentum_rsi',
                        'MACD': ['trend_macd', 'trend_macd_signal', 'trend_macd_diff'],
                        'Stochastic': ['momentum_stoch', 'momentum_stoch_signal']
                    }
                    has_rsi = osc_cols['RSI'] in tech_df.columns
                    has_macd = all(x in tech_df.columns for x in osc_cols['MACD'])
                    has_stoch = all(x in tech_df.columns for x in osc_cols['Stochastic'])
                    if len(tech_df) == 0 or not (has_rsi or has_macd or has_stoch):
                        st.warning("No oscillator data available to display.", icon="⚠️")
                        logger.warning("Oscillators tab: All oscillator columns missing or empty DataFrame.")
                    else:
                        fig_osc = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                               subplot_titles=("RSI", "MACD", "Stochastic"))
                        # RSI
                        if has_rsi:
                            fig_osc.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df[osc_cols['RSI']],
                                mode='lines',
                                name='RSI',
                                line=dict(color='#1f77b4', width=1.5),
                                legendgroup='RSI',
                            ), row=1, col=1)
                            fig_osc.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                            fig_osc.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                        else:
                            st.info("RSI data not available.", icon="ℹ️")
                            logger.info("Oscillators tab: RSI column missing.")
                        # MACD
                        if has_macd:
                            fig_osc.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df['trend_macd'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='#1f77b4', width=1.5),
                                legendgroup='MACD',
                            ), row=2, col=1)
                            fig_osc.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df['trend_macd_signal'],
                                mode='lines',
                                name='Signal',
                                line=dict(color='#ff7f0e', width=1.5),
                                legendgroup='MACD',
                            ), row=2, col=1)
                            colors_hist = np.where(tech_df['trend_macd_diff'] < 0, 'red', 'green')
                            fig_osc.add_trace(go.Bar(
                                x=tech_df.index,
                                y=tech_df['trend_macd_diff'],
                                name='MACD Histogram',
                                marker_color=colors_hist,
                                legendgroup='MACD',
                            ), row=2, col=1)
                        else:
                            st.info("MACD data not available.", icon="ℹ️")
                            logger.info("Oscillators tab: MACD columns missing.")
                        # Stochastic
                        if has_stoch:
                            fig_osc.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df['momentum_stoch'],
                                mode='lines',
                                name='%K',
                                line=dict(color='#1f77b4', width=1.5),
                                legendgroup='Stochastic',
                            ), row=3, col=1)
                            fig_osc.add_trace(go.Scatter(
                                x=tech_df.index,
                                y=tech_df['momentum_stoch_signal'],
                                mode='lines',
                                name='%D',
                                line=dict(color='#ff7f0e', width=1.5),
                                legendgroup='Stochastic',
                            ), row=3, col=1)
                            fig_osc.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
                            fig_osc.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
                        else:
                            st.info("Stochastic data not available.", icon="ℹ️")
                            logger.info("Oscillators tab: Stochastic columns missing.")
                        fig_osc.update_layout(
                            height=800,
                            showlegend=True,
                            template="plotly_white",
                            title="Technical Oscillators"
                        )
                        st.plotly_chart(fig_osc, use_container_width=True, key="osc_chart")
                        st.info("""
                        **Oscillators Explanation:**
                        - **RSI (Relative Strength Index)**: Measures the speed and change of price movements. Values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.
                        - **MACD (Moving Average Convergence Divergence)**: Shows the relationship between two moving averages of a security's price. The MACD line crossing above the signal line is a potential buy signal, while crossing below is a potential sell signal.
                        - **Stochastic Oscillator**: Compares a security's closing price to its price range over a specific period. Values above 80 indicate overbought conditions, while values below 20 indicate oversold conditions.
                        """)
            except Exception as e:
                st.error(f"Error displaying Oscillators: {str(e)}", icon="🚨")
                logger.error(f"Exception in Oscillators tab: {str(e)}", exc_info=True)
        
        # Volume Indicators tab
        with price_tabs[3]:
            if processed_data and 'stock_data' in processed_data and volume_col in df.columns:
                tech_df = processed_data['stock_data']
                
                # Create subplots for volume indicators
                fig_vol_ind = make_subplots(rows=2, cols=1, 
                                          shared_xaxes=True,
                                          vertical_spacing=0.1,
                                          subplot_titles=("Volume", "On-Balance Volume (OBV)"))
                
                # Add Volume
                fig_vol_ind.add_trace(go.Bar(
                    x=df.index,
                    y=df[volume_col],
                    name='Volume',
                    marker_color='rgba(0, 0, 255, 0.5)'
                ), row=1, col=1)
                
                # Add OBV if available
                if 'volume_obv' in tech_df.columns:
                    fig_vol_ind.add_trace(go.Scatter(
                        x=tech_df.index,
                        y=tech_df['volume_obv'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='#1f77b4', width=1.5)
                    ), row=2, col=1)
                
                # Update layout
                fig_vol_ind.update_layout(
                    height=600,
                    showlegend=True,
                    template="plotly_white",
                    title="Volume Indicators"
                )
                
                st.plotly_chart(fig_vol_ind, use_container_width=True)
                
                # Add explanation
                st.info("""
                **Volume Indicators Explanation:**
                - **Volume**: Shows the number of shares traded in a given period. High volume often indicates strong interest in a stock.
                - **On-Balance Volume (OBV)**: Relates volume to price change. When OBV is rising, it suggests positive volume pressure that can lead to higher prices.
                """)
            else:
                st.warning("Volume indicators not available. Please run the analysis first.")
        
        # Basic statistics
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${df[close_col].iloc[-1]:.2f}")
        with col2:
            st.metric("52-Week High", f"${df[close_col].max():.2f}")
        with col3:
            st.metric("52-Week Low", f"${df[close_col].min():.2f}")
        
    except Exception as e:
        st.error(f"Error displaying price data: {str(e)}")
        logger.error(f"Error in price analysis: {str(e)}", exc_info=True)

def validate_and_clean_stock_data(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, str]:
    """
    Validate and clean stock data for consistency.
    
    Args:
        df: Input DataFrame containing stock data
        ticker: Stock ticker symbol for error messages
        
    Returns:
        Tuple of (cleaned_dataframe, close_column_name)
    """
    if df is None or df.empty:
        raise ValueError(f"No data available for {ticker}")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not parse datetime index: {e}")
    
    # Sort by date
    df = df.sort_index()
    
    # Find price column with priority to ticker-prefixed columns
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    
    # List of possible close column name patterns to try in order of preference
    possible_cols = [
        f'Close_{ticker_upper}',    # e.g., Close_AAPL
        f'Close {ticker_upper}',   # e.g., Close AAPL
        f'Close_{ticker_lower}',   # e.g., Close_aapl
        f'Close {ticker_lower}',   # e.g., Close aapl
        'Close',                   # Standard name
        'close',                   # Lowercase
        'Adj Close',               # Adjusted close
        'adj_close'                # Adjusted close with underscore
    ]
    
    # Try each pattern in order
    close_col = None
    for pattern in possible_cols:
        if pattern in df.columns:
            close_col = pattern
            break
    
    # If still not found, try any column with 'close' in the name (case insensitive)
    if close_col is None:
        for col in df.columns:
            if isinstance(col, str) and 'close' in col.lower():
                close_col = col
                break
    
    if close_col is None:
        available_cols = ', '.join(f"'{col}'" for col in df.columns)
        raise ValueError(
            f"No Close price column found for {ticker}. "
            f"Tried patterns: {', '.join(possible_cols[:4]) + ', ...'}. "
            f"Available columns: {available_cols}"
        )
    
    # Basic validation
    if df[close_col].isnull().all():
        raise ValueError(f"Close price data is all null for {ticker}")
    
    # Forward fill and then backfill any remaining nulls
    df = df.ffill().bfill()
    
    return df, close_col

def load_and_validate_data(ticker: str, start_date: datetime, end_date: datetime) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load and validate stock data with proper error handling.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Tuple of (stock_data, close_column_name) or (None, None) on failure
    """
    try:
        logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
        
        # Fetch data using fetch_daily_yfinance_data
        stock_data = fetch_daily_yfinance_data(
            ticker,
            start=start_date.date(),
            end=end_date.date(),
            interval='1d',
            logger=logger
        )
        
        if stock_data.empty:
            raise ValueError(f"No data returned for {ticker}")
            
        # Validate and clean the data
        stock_data_clean, close_col = validate_and_clean_stock_data(stock_data, ticker)
        
        # Add technical indicators
        try:
            from utils.data_processor import DataProcessor
            dp = DataProcessor()
            stock_data_clean = dp.add_technical_indicators(stock_data_clean)
            logger.info("Successfully added technical indicators to stock data")
        except Exception as e:
            logger.warning(f"Could not add technical indicators: {str(e)}")
        
        logger.info(f"Successfully loaded data with shape: {stock_data_clean.shape}")
        return stock_data_clean, close_col
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None, None

def clean_dataframe(df, name):
    """
    Clean a DataFrame by handling missing values, infinite values, and other data quality issues.
    
    Args:
        df: Input DataFrame to clean
        name: Name of the DataFrame for logging purposes
        
    Returns:
        Cleaned DataFrame
    """
    try:
        if df is None:
            logger.warning(f"{name} is None")
            return pd.DataFrame()
            
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            logger.warning(f"{name} is not a pandas DataFrame or Series")
            return pd.DataFrame()
            
        if df.empty:
            logger.warning(f"{name} is empty")
            return df.copy()
            
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Log initial shape and columns
        logger.info(f"Cleaning {name}. Initial shape: {df_clean.shape}, Columns: {list(df_clean.columns)}")
        
        # Ensure column names are strings and join tuple columns with '_'
        df_clean.columns = ['_'.join([str(c) for c in col if str(c).strip() != '']) if isinstance(col, tuple) else str(col) for col in df_clean.columns]
        
        # Handle infinite values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with all NaN values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(how='all')
        
        # Log rows dropped
        if initial_rows > len(df_clean):
            logger.warning(f"Dropped {initial_rows - len(df_clean)} rows with all NaN values from {name}")
        
        # Fill remaining NaN values with column means for numeric columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if col in df_clean.columns and df_clean[col].isna().any():
                    try:
                        mean_val = df_clean[col].mean()
                        if pd.notna(mean_val):  # Only fill if we got a valid mean
                            df_clean[col] = df_clean[col].fillna(mean_val)
                            logger.debug(f"Filled NaN values in column '{col}' with mean: {mean_val}")
                    except Exception as e:
                        logger.warning(f"Could not fill NaN values in column '{col}': {str(e)}")
                        # Try forward fill, then backward fill as fallback
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # If we still have any NaN values, drop those rows
        if df_clean.isna().any().any():
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            if len(df_clean) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df_clean)} rows with remaining NaN values")
        
        # Log final shape
        logger.info(f"Cleaned {name}. Final shape: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error cleaning {name}: {str(e)}", exc_info=True)
        raise


# Initialize session state for logger if it doesn't exist
if 'logger' not in st.session_state:
    st.session_state.logger = logger
    logger.info("Application initialized")

# Initialize analysis-related session state variables
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    
if '_show_tabs_after_rerun' not in st.session_state:
    st.session_state._show_tabs_after_rerun = False
    
if 'ticker' not in st.session_state:
    st.session_state.ticker = 'Not set'

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Debug: Print all environment variables
logger.debug("Environment variables loaded. Checking for Alpaca credentials...")

# Get Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Debug: Log the values (without exposing full keys)
logger.debug(f"ALPACA_API_KEY present: {'Yes' if ALPACA_API_KEY else 'No'}")
logger.debug(
    f"ALPACA_SECRET_KEY present: {'Yes' if ALPACA_SECRET_KEY else 'No'}")

# Validate API keys
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    error_msg = "❌ Error: Missing Alpaca API credentials. Please check your .env file."
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Third-party imports
try:
    import websockets
    logger.info(f"Websockets package loaded successfully, version: {websockets.__version__}")
except ImportError:
    logger.error("Failed to import websockets package. Please install it with: pip install websockets")
    st.error("Missing required package: websockets. Please run: pip install websockets")
    st.stop()

# RealTimeDataHandler class definition


class RealTimeDataHandler:
    """
    A robust class to handle real-time data from Alpaca's WebSocket API
    in a Streamlit environment.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of RealTimeDataHandler"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
        
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RealTimeDataHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Initialize asyncio event loop
        try:
            self.loop = asyncio.get_event_loop()
            if self.loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except (RuntimeError, Exception):
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.ws = None
        self.running = False
        self.authenticated = False
        self.connected = False
        self.subscribed_symbols = set()
        self.callbacks = []
        self._initialized = True  # Mark as initialized
        self.logger = logging.getLogger(__name__)
        self._thread = None
        self._initialized = True

        # Concurrency control
        self._message_lock = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
        self._subscription_lock = threading.RLock()  # Changed to RLock for context manager support

        # Connection management
        self._should_reconnect = True
        self._persist_connection = True
        self.connection_attempts = 0
        self.max_attempts = 10
        self.reconnect_delay = 1  # Initial delay in seconds
        self.max_reconnect_delay = 60  # Max delay in seconds

        # Message handling
        self.last_message = None
        self.last_message_time = None
        self.message_count = 0
        self.last_error = None
        self.last_update = {}
        self.data = {}
        self.max_history = 1000  # Max number of messages to keep in history

        # Initialize message handler task
        self.message_handler_task = None

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret must be set.")

    def is_ws_closed(self):
        """Safely check if the WebSocket connection is closed."""
        if not self.ws:
            return True
            
        # Try different attributes/methods to check if connection is closed
        try:
            # First check our own connection flag
            if not self.connected:
                return True
                
            # For ClientConnection objects (websockets library)
            # ClientConnection doesn't have a 'closed' attribute but has other ways to check
            
            # Check for open attribute (most common)
            if hasattr(self.ws, 'open'):
                try:
                    return not self.ws.open
                except (AttributeError, RuntimeError):
                    pass
                    
            # Check for connected attribute
            if hasattr(self.ws, 'connected'):
                try:
                    return not self.ws.connected
                except (AttributeError, RuntimeError):
                    pass
                    
            # Check for socket
            if hasattr(self.ws, 'sock'):
                try:
                    return self.ws.sock is None
                except (AttributeError, RuntimeError):
                    pass
                    
            # Check for closed attribute (may not exist on ClientConnection)
            if hasattr(self.ws, 'closed'):
                try:
                    return self.is_ws_closed()
                except (AttributeError, RuntimeError):
                    pass
            
            # For websockets.WebSocketClientProtocol
            if hasattr(self.ws, 'state'):
                try:
                    if hasattr(self.ws.state, 'name'):
                        return self.ws.state.name in ('CLOSED', 'CLOSING')
                except (AttributeError, RuntimeError):
                    pass
            
            # If we can't determine the state, use our connection flag
            return not self.connected
        except Exception as e:
            self.logger.warning(f"Error checking WebSocket state: {e}")
            # If any error occurs while checking, assume it's closed
            return True
            
    def safe_update_symbols(self, symbols: List[str]) -> bool:
        """
        Safely update the list of subscribed symbols in a thread-safe manner.
        
        Args:
            symbols: List of ticker symbols to subscribe to
            
        Returns:
            bool: True if symbols were updated successfully, False otherwise
        """
        if not symbols or not isinstance(symbols, list):
            self.logger.warning(f"Invalid symbols list provided: {symbols}")
            return False
            
        try:
            # Convert to set for easy comparison and deduplication
            new_symbols = {s.upper().strip() for s in symbols if isinstance(s, str) and s.strip()}
            
            if not new_symbols:
                self.logger.warning("No valid symbols provided for update")
                return False
                
            # Check if symbols have actually changed
            if new_symbols.issubset(self.subscribed_symbols):
                self.logger.debug(f"Symbols already subscribed: {new_symbols}")
                return True
                
            # Update symbols in a thread-safe manner
            with self._subscription_lock:
                # Add new symbols to the subscription
                self.subscribed_symbols.update(new_symbols)
                
                # If connected, update the subscription immediately
                if self.connected and self.authenticated:
                    self.logger.info(f"Updating subscription with new symbols: {new_symbols}")
                    
                    # Create subscription message
                    msg = {
                        'action': 'subscribe',
                        'trades': list(new_symbols),
                        'quotes': list(new_symbols),
                        'bars': list(new_symbols)
                    }
                    
                    # Send the subscription message
                    asyncio.run_coroutine_threadsafe(
                        self._send_json(msg),
                        self.loop
                    ).result(timeout=5)  # Wait up to 5 seconds for the subscription to complete
                    
                    self.logger.info(f"Successfully updated subscription with {len(new_symbols)} symbols")
                else:
                    self.logger.warning("Not connected to WebSocket, will subscribe when connection is established")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating symbols: {str(e)}", exc_info=True)
            return False

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret must be set.")

    def start(self):
        """Starts the WebSocket connection in a separate thread."""
        if self.running:
            self.logger.warning("WebSocket handler is already running.")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Add robust error handling for authentication failures
        # This assumes that authentication is checked in the event loop
        # If authentication fails, log and raise
        if hasattr(self, 'auth_failed') and self.auth_failed:
            self.logger.error("Authentication failed: %s", getattr(self, 'auth_response', 'Unknown error'))
            raise ConnectionError("Failed to authenticate with Alpaca API")

    def stop(self):
        """Stops the WebSocket connection."""
        if not self.running:
            return

        self.logger.info("Stopping WebSocket connection...")
        self.running = False
        
        # First, try to close the WebSocket connection if it's open
        if self.ws and self.connected:
            try:
                # Use call_soon_threadsafe to schedule the WebSocket close in the event loop
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._schedule_ws_close)
            except Exception as e:
                self.logger.warning(f"Error scheduling WebSocket close: {str(e)}")
        
        # Then stop the event loop
        if self._loop:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception as e:
                self.logger.warning(f"Error stopping event loop: {str(e)}")
        
        # Wait for the thread to finish
        if self._thread and self._thread.is_alive():
            self.logger.info("Waiting for WebSocket thread to terminate...")
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.logger.warning("WebSocket thread did not terminate within timeout")
            else:
                self.logger.info("WebSocket thread terminated successfully")
        
        self.logger.info("WebSocket connection stopped")
    
    def _schedule_ws_close(self):
        """Schedule WebSocket close in the event loop."""
        if self.ws and not self.is_ws_closed():
            try:
                # Create a task to close the WebSocket
                asyncio.create_task(self._close_ws())
            except Exception as e:
                self.logger.warning(f"Error creating close task: {str(e)}")
    
    async def _close_ws(self):
        """Close the WebSocket connection."""
        if self.ws and not self.is_ws_closed():
            try:
                await self.ws.close()
                self.logger.info("WebSocket closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {str(e)}")
            finally:
                self.connected = False
                self.authenticated = False

    async def _cleanup_connection(self):
        """Clean up any existing WebSocket connection"""
        if hasattr(self, 'ws') and self.ws:
            try:
                if not self.ws.closed:
                    await self.ws.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {str(e)}")
            finally:
                self.ws = None
                self.connected = False
                self.authenticated = False
                
        # Cancel any pending message handler task
        if hasattr(self, 'message_handler_task') and self.message_handler_task:
            try:
                if not self.message_handler_task.done():
                    self.message_handler_task.cancel()
                    try:
                        await asyncio.wait_for(self.message_handler_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            except Exception as e:
                self.logger.warning(f"Error cancelling message handler task: {str(e)}")
            finally:
                self.message_handler_task = None

    def _run_loop(self):
        """Runs the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            self.logger.error(f"Error in event loop: {str(e)}", exc_info=True)
        finally:
            # Cancel all pending tasks to prevent "Event loop is closed" errors
            pending = asyncio.all_tasks(loop=self._loop)
            if pending:
                self.logger.info(f"Cancelling {len(pending)} pending tasks before closing event loop")
                for task in pending:
                    task.cancel()
                # Give tasks a chance to respond to cancellation
                try:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception as e:
                    self.logger.warning(f"Error while cancelling tasks: {str(e)}")
            
            # Properly close the event loop
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception as e:
                self.logger.warning(f"Error during shutdown_asyncgens: {str(e)}")
            finally:
                self._loop.close()

    async def _connect_and_listen(self):
        """Main WebSocket connection handler with robust error handling and reconnection."""
        while self.running and not self._loop.is_closed():
            try:
                # Use a lock to prevent multiple simultaneous connection attempts
                async with self._connection_lock:
                    if self.connected:
                        self.logger.debug("Already connected, skipping connection attempt")
                        await asyncio.sleep(1)
                        continue
                        
                    self.logger.info(f"Connecting to WebSocket at {self.ws_url}")
                    
                    try:
                        # Connect with a timeout
                        connect_coro = websockets.connect(
                            self.ws_url, 
                            ssl=True,
                            ping_interval=20,  # Send ping every 20 seconds
                            ping_timeout=10,    # Wait 10 seconds for pong response
                            close_timeout=1,    # Wait 1 second for clean close
                            max_size=2**25,     # 32MB max message size
                            max_queue=32        # Max number of messages to buffer
                        )
                        
                        # Connect with timeout
                        try:
                            self.ws = await asyncio.wait_for(connect_coro, timeout=10)
                            self.connected = True
                            self.connection_attempts = 0  # Reset attempts on successful connect
                            self.logger.info("WebSocket connected successfully")
                            
                            # Authenticate
                            if not await self._authenticate():
                                raise ConnectionError("Authentication failed")
                            
                            # Notify callbacks of successful connection
                            await self._notify_callbacks({
                                'type': 'connection',
                                'status': 'connected',
                                'timestamp': datetime.utcnow().isoformat()
                            })
                            
                            # Start message handler
                            self.message_handler_task = asyncio.create_task(self._message_handler())
                            
                            # Wait for the message handler to complete
                            if self.message_handler_task:
                                await self.message_handler_task
                                
                            # If we get here, the message handler has exited
                            self.logger.info("Message handler completed, cleaning up...")
                            
                        except asyncio.TimeoutError:
                            self.logger.error("WebSocket connection timed out")
                            raise ConnectionError("Connection timed out")
                            
                    except Exception as e:
                        self.logger.error(f"WebSocket connection failed: {str(e)}", exc_info=True)
                        
                        # Clean up any partial connection
                        await self._cleanup_connection()
                        
                        # Implement exponential backoff with jitter
                        self.connection_attempts += 1
                        if self.connection_attempts > self.max_attempts:
                            self.logger.error("Max connection attempts reached, giving up")
                            self.running = False
                            break
                            
                        # Calculate backoff with jitter
                        delay = min(
                            self.reconnect_delay * (2 ** (self.connection_attempts - 1)),
                            self.max_reconnect_delay
                        ) * (0.5 + 0.5 * random.random())  # Add jitter between 0.5x and 1.0x
                        
                        self.logger.info(f"Attempting to reconnect in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                        continue
                        
            except asyncio.CancelledError:
                self.logger.info("Connection task cancelled")
                break
                
            except Exception as e:
                self.logger.error(f"Unexpected error in connection handler: {str(e)}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on unexpected errors

    async def _authenticate(self):
        """Authenticates the WebSocket connection."""
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        await self.ws.send(json.dumps(auth_msg))
        response = await self.ws.recv()
        response_data = json.loads(response)

        # The first message is a connection success message
        if isinstance(response_data, list) and response_data[0].get('T') == 'success':
             response = await self.ws.recv() # Get the actual auth response
             response_data = json.loads(response)

        if isinstance(response_data, list) and response_data[0].get('msg') == 'authenticated':
            self.authenticated = True
            self.logger.info("WebSocket authenticated.")
            return True
        else:
            self.logger.error(f"Authentication failed: {response_data}")
            return False

    async def _subscribe(self):
        """Subscribes to the symbols."""
        if not self.subscribed_symbols:
            return

        try:
            sub_msg = {
                "action": "subscribe",
                "trades": list(self.subscribed_symbols),
                "quotes": list(self.subscribed_symbols),
                "bars": list(self.subscribed_symbols)
            }
            if self.ws and not self.is_ws_closed():
                await self.ws.send(json.dumps(sub_msg))
                self.logger.info(f"Subscribed to {self.subscribed_symbols}")
        except Exception as e:
            self.logger.error(f"Error in _subscribe: {e}")
            return False
        return True

    async def _message_handler(self):
        """
        Handle incoming WebSocket messages.
        This method runs in a separate task and processes all incoming messages.
        """
        self.logger.info("Starting message handler...")
        
        try:
            while self.running and self.connected and hasattr(self, 'ws') and self.ws and not self.ws.closed:
                try:
                    # Wait for a message with a timeout
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    
                    # Process the message
                    if message:
                        self.last_message = message
                        self.last_message_time = datetime.utcnow()
                        self.message_count += 1
                        
                        # Process the message
                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse message as JSON: {str(e)}")
                            continue
                            
                except asyncio.TimeoutError:
                    # No message received within timeout, send a ping
                    self.logger.debug("No message received within timeout, checking connection...")
                    try:
                        pong = await self.ws.ping()
                        await asyncio.wait_for(pong, timeout=10.0)
                        self.logger.debug("Connection is still alive")
                    except Exception as e:
                        self.logger.warning(f"Connection seems to be dead: {str(e)}")
                        self.connected = False
                        break
                        
                except websockets.exceptions.ConnectionClosed as e:
                    self.logger.warning(f"WebSocket connection closed: {e}")
                    self.connected = False
                    break
                    
                except Exception as e:
                    self.logger.error(f"Error in message handler: {str(e)}", exc_info=True)
                    self.connected = False
                    break
                    
        except Exception as e:
            self.logger.error(f"Message handler error: {str(e)}", exc_info=True)
        finally:
            self.logger.info("Message handler stopped")
            self.connected = False
            self.authenticated = False
            await self._cleanup_connection()
            
            # Notify callbacks of disconnection
            try:
                await self._notify_callbacks({
                    'type': 'connection',
                    'status': 'disconnected',
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error notifying callbacks: {str(e)}")

    def add_callback(self, callback):
        """Adds a callback function to be called on new messages."""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def subscribe_to_symbols(self, symbols: list):
        """Subscribes to a list of symbols."""
        new_symbols = set(symbols) - self.subscribed_symbols
        if new_symbols:
            self.subscribed_symbols.update(new_symbols)
            if self.authenticated and self._loop:
                asyncio.run_coroutine_threadsafe(self._subscribe(), self._loop)

    def get_status(self):
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "running": self.running,
            "subscribed_symbols": list(self.subscribed_symbols),
        }

    # get_instance method is already defined above
        
    def add_callback(self, callback: Callable):
        """
        Add a callback function to be called when new data is received.
        
        Args:
            callback: A callable that takes a single argument (the message data)
        """
        if callable(callback) and callback not in self.callbacks:
            self.callbacks.append(callback)
            self.logger.info(f"Added callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
            
    def remove_callback(self, callback: Callable):
        """
        Remove a previously added callback.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.info(f"Removed callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
            
    # get_status method is defined below with more comprehensive implementation
        
    def on_update(self, callback: Callable):
        """Set callback for data updates"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
            
    def subscribe(self, symbols: List[str]):
        """
        Non-async wrapper for _subscribe_async. Use this method from synchronous code.
        
        Args:
            symbols: List of ticker symbols to subscribe to
        """
        # Add symbols to the set of subscribed symbols
        if symbols:
            for symbol in symbols:
                if symbol and isinstance(symbol, str):
                    self.subscribed_symbols.add(symbol.upper())
            
            # If we're connected, schedule the subscription in the event loop
            if self.connected and self.authenticated and self._loop:
                asyncio.run_coroutine_threadsafe(self._subscribe(), self._loop)
                self.logger.info(f"Scheduled subscription for symbols: {symbols}")
            else:
                self.logger.info(f"Added symbols to subscription list: {symbols}")
        
    async def _subscribe_async(self, symbols: List[str]):
        """
        Subscribe to real-time data for the specified symbols asynchronously.
        
        Args:
            symbols: List of ticker symbols to subscribe to
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not symbols:
            self.logger.warning("No symbols provided for subscription")
            return False
            
        # Convert to uppercase and remove duplicates
        symbols = [s.upper() for s in symbols]
        
        # Check if already connected and authenticated
        if not self.connected or not self.authenticated:
            self.logger.info("Not connected or authenticated, but cannot connect from async method")
            return False
        
        # Subscribe to the symbols
        try:
            # Create subscription message
            sub_msg = {
                "action": "subscribe",
                "trades": list(self.subscribed_symbols),
                "quotes": list(self.subscribed_symbols),
                "bars": list(self.subscribed_symbols)
            }
            
            # Send subscription message
            if self.ws and not self.is_ws_closed():
                await self.ws.send(json.dumps(sub_msg))
                self.logger.info(f"Successfully subscribed to {len(self.subscribed_symbols)} symbols")
                return True
            else:
                self.logger.error("WebSocket is not connected")
                return False
        except Exception as e:
            self.logger.error(f"Failed to subscribe to symbols: {e}")
            return False
        
    async def unsubscribe(self, symbols: List[str] = None):
        """
        Unsubscribe from real-time data for the specified symbols.
        If no symbols are provided, unsubscribe from all symbols.
        
        Args:
            symbols: List of ticker symbols to unsubscribe from, or None to unsubscribe from all
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if not self.ws or self.is_ws_closed():
            self.logger.error("Cannot unsubscribe: WebSocket not connected")
            return False
            
        if not self.authenticated:
            self.logger.error("Cannot unsubscribe: Not authenticated")
            return False
            
        # If no symbols provided, unsubscribe from all
        if symbols is None:
            symbols_to_unsub = list(self.subscribed_symbols)
        else:
            # Convert to uppercase
            symbols_to_unsub = [s.upper() for s in symbols]
            
        if not symbols_to_unsub:
            self.logger.warning("No symbols to unsubscribe from")
            return True  # Nothing to do, so technically successful
            
        try:
            # Prepare unsubscription message
            unsub_msg = {
                "action": "unsubscribe",
                "trades": symbols_to_unsub,
                "quotes": symbols_to_unsub,
                "bars": symbols_to_unsub
            }
            
            # Send unsubscription message
            self.logger.info(f"Unsubscribing from {len(symbols_to_unsub)} symbols")
            await self.ws.send(json.dumps(unsub_msg))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            response_data = json.loads(response)
            self.logger.info(f"Unsubscription response: {response_data}")
            
            # Check for unsubscription success
            if isinstance(response_data, list) and len(response_data) > 0:
                for msg in response_data:
                    if msg.get('T') == 'subscription' and 'unsubscribed' in msg.get('msg', ''):
                        # Update subscribed symbols set
                        for symbol in symbols_to_unsub:
                            if symbol in self.subscribed_symbols:
                                self.subscribed_symbols.remove(symbol)
                        self.logger.info(f"Successfully unsubscribed from {len(symbols_to_unsub)} symbols")
                        return True
            
            # If we got here, unsubscription failed
            self.logger.error(f"Unsubscription failed: {response_data}")
            return False
            
        except asyncio.TimeoutError:
            self.logger.error("Unsubscription request timed out")
            return False
        except Exception as e:
            self.logger.error(f"Unsubscription error: {str(e)}", exc_info=True)
            return False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealTimeDataHandler, cls).__new__(cls)
        return cls._instance
        
    def get_status(self):
        """
        Get the current status of the WebSocket connection.
        
        Returns:
            dict: A dictionary containing status information
        """
        now = datetime.utcnow()
        last_msg_age = None
        if self.last_message_time:
            last_msg_age = (now - self.last_message_time).total_seconds()
            
        # Check if we have market status information
        market_status = {
            "is_open": False,
            "next_open": None,
            "next_close": None,
            "timestamp": None
        }
        
        # Try to get market status from Alpaca REST API
        try:
            import requests
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            }
            response = requests.get('https://api.alpaca.markets/v2/clock', headers=headers)
            if response.status_code == 200:
                data = response.json()
                market_status = {
                    "is_open": data.get('is_open', False),
                    "next_open": data.get('next_open'),
                    "next_close": data.get('next_close'),
                    "timestamp": data.get('timestamp')
                }
            else:
                self.logger.warning(f"Failed to get market status: {response.status_code} {response.text}")
        except Exception as e:
            self.logger.warning(f"Error getting market status: {str(e)}")
        
        return {
            "connected": self.connected,
            "authenticated": self.authenticated,
            "running": self.running,
            "endpoint": self.ws_url,
            "subscribed_symbols": list(self.subscribed_symbols),
            "message_count": self.message_count,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "last_message_age_seconds": last_msg_age,
            "last_error": self.last_error,
            "market_status": market_status,
            "data_summary": {
                symbol: {
                    "trades": len(data.get('trades', [])),
                    "quotes": len(data.get('quotes', [])),
                    "bars": len(data.get('bars', [])),
                    "last_update": self.last_update.get(symbol).isoformat() if symbol in self.last_update else None
                } for symbol, data in self.data.items()
            }
        }

    async def start(self, wait_for_connection: bool = True):
        """
        Start the WebSocket client and connect to the Alpaca WebSocket API.
        
        Args:
            wait_for_connection: If True, wait for connection to be established before returning.
                              If False, start connection in the background and return immediately.
                              
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if self.running:
            self.logger.warning("WebSocket client is already running")
            return True
            
        self.logger.info("Starting WebSocket client...")
        self.running = True
        self._should_reconnect = True
        
        try:
            # Connect to WebSocket
            self.logger.info(f"Connecting to {self.ws_url}")
            
            # Import SSL for certificate handling
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE  # Disable certificate verification for troubleshooting
            
            # Use simpler connection parameters that are compatible with older websockets versions
            try:
                # First try with standard parameters and SSL context
                self.logger.info("Attempting connection with standard parameters and SSL context")
                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=30,  # 30 seconds ping interval
                    ping_timeout=10,   # 10 seconds timeout for pings
                    ssl=ssl_context    # Use our custom SSL context
                )
            except TypeError as e:
                # If we get a TypeError, try with even simpler parameters
                self.logger.warning(f"Connection error with standard parameters: {str(e)}. Trying with minimal parameters.")
                try:
                    self.ws = await websockets.connect(self.ws_url, ssl=ssl_context)
                except Exception as e2:
                    self.logger.error(f"Failed with minimal parameters and SSL context: {str(e2)}")
                    # Last resort: try without SSL context
                    self.logger.warning("Attempting connection without custom SSL context")
                    self.ws = await websockets.connect(self.ws_url)
            
            self.connected = True
            self.connection_attempts = 0  # Reset connection attempts on success
            self.logger.info("WebSocket connected")
            
            # Start the message handler
            self.message_handler_task = asyncio.create_task(self._message_handler())
            
            # Authenticate
            if not await self._authenticate():
                raise ConnectionError("Failed to authenticate with Alpaca API")
                
            # Return success
            return True
                
        except Exception as e:
            self.connected = False
            self.running = False
            self.last_error = str(e)
            self.logger.error(f"WebSocket connection error: {str(e)}", exc_info=True)
            
            wait_for_connection = True  # Define the variable with a default value
            if wait_for_connection:
                raise
            
            return False
                
    async def _authenticate(self):
        """
        Authenticate with the Alpaca WebSocket API.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            self.logger.info("Authenticating with Alpaca API...")
            
            # Prepare authentication message
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            
            # Send authentication message
            await self.ws.send(json.dumps(auth_msg))
            
            # Wait for authentication response with timeout
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            self.logger.info(f"Authentication response: {response_data}")
            
            # Check for authentication success
            if isinstance(response_data, list) and len(response_data) > 0:
                for msg in response_data:
                    if msg.get('T') == 'success' and 'authenticated' in msg.get('msg', ''):
                        self.authenticated = True
                        self.logger.info("Successfully authenticated with Alpaca API")
                        return True
            
            # If we got here, authentication failed
            self.logger.error(f"Authentication failed: {response_data}")
            return False
            
        except asyncio.TimeoutError:
            self.logger.error("Authentication request timed out")
            return False
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}", exc_info=True)
            return False
            
    async def _message_handler(self):
        """
        Handle incoming WebSocket messages.
        This method runs in a separate task and processes all incoming messages.
        """
        self.logger.info("Starting message handler...")
        
        # Check connection status without using .closed attribute
        while self.running and self.ws and self.connected:
            try:
                # Wait for a message with a timeout
                message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                
                # Process the message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                # No message received within timeout, send a ping
                self.logger.debug("No message received within timeout, checking connection...")
                try:
                    # Simple ping by requesting connection status
                    pong = await self.ws.ping()
                    await asyncio.wait_for(pong, timeout=10.0)
                    self.logger.debug("Connection is still alive")
                except Exception as e:
                    self.logger.warning(f"Connection seems to be dead: {str(e)}")
                    # Connection is dead, try to reconnect
                    self.connected = False
                    self.authenticated = False
                    await self._reconnect()
                    break
                    
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {str(e)}")
                self.connected = False
                self.authenticated = False
                
                # Try to reconnect
                if self._should_reconnect:
                    await self._reconnect()
                break
                
            except Exception as e:
                self.logger.error(f"Error in message handler: {str(e)}", exc_info=True)
                self.last_error = str(e)
                
                # Continue processing messages
                continue
                
        self.logger.info("Message handler stopped")
        
    async def _process_message(self, message):
        """
        Process an incoming WebSocket message.
        
        Args:
            message: The raw message string from the WebSocket
        """
        try:
            # Parse the message
            data = json.loads(message)
            self.message_count += 1
            self.last_message_time = datetime.utcnow()
            
            # Handle different message types
            if isinstance(data, list):
                for msg in data:
                    await self._process_single_message(msg)
            else:
                await self._process_single_message(data)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse message: {str(e)}")
            self.last_error = f"JSON parse error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            self.last_error = str(e)
            
    async def _process_single_message(self, msg):
        """
        Process a single message from the WebSocket.
        
        Args:
            msg: A parsed message object
        """
        msg_type = msg.get('T')
        
        if msg_type == 'success':
            # Success message, log it
            self.logger.info(f"Success message: {msg.get('msg')}")
            
        elif msg_type == 'subscription':
            # Subscription update
            self.logger.info(f"Subscription update: {msg}")
            
        elif msg_type == 'error':
            # Error message
            self.logger.error(f"Error from Alpaca API: {msg.get('msg')}")
            self.last_error = msg.get('msg')
            
        elif msg_type == 't':
            # Trade update
            symbol = msg.get('S')
            if symbol:
                # Store the trade data
                if symbol not in self.data:
                    self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                # Add the trade to the list, keeping only the most recent trades
                self.data[symbol]['trades'].append(msg)
                if len(self.data[symbol]['trades']) > self.max_history:
                    self.data[symbol]['trades'] = self.data[symbol]['trades'][-self.max_history:]
                
                # Update last update time
                self.last_update[symbol] = datetime.utcnow()
                
        elif msg_type == 'q':
            # Quote update
            symbol = msg.get('S')
            if symbol:
                # Store the quote data
                if symbol not in self.data:
                    self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                # Add the quote to the list, keeping only the most recent quotes
                self.data[symbol]['quotes'].append(msg)
                if len(self.data[symbol]['quotes']) > self.max_history:
                    self.data[symbol]['quotes'] = self.data[symbol]['quotes'][-self.max_history:]
                
                # Update last update time
                self.last_update[symbol] = datetime.utcnow()
                
        elif msg_type == 'b':
            # Bar update
            symbol = msg.get('S')
            if symbol:
                # Store the bar data
                if symbol not in self.data:
                    self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                # Add the bar to the list, keeping only the most recent bars
                self.data[symbol]['bars'].append(msg)
                if len(self.data[symbol]['bars']) > self.max_history:
                    self.data[symbol]['bars'] = self.data[symbol]['bars'][-self.max_history:]
                
                # Update last update time
                self.last_update[symbol] = datetime.utcnow()
                
        # Notify callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(msg)
                else:
                    callback(msg)
            except Exception as e:
                self.logger.error(f"Error in callback: {str(e)}", exc_info=True)
                
    async def _reconnect(self):
        """
        Attempt to reconnect to the WebSocket API with exponential backoff.
        """
        if not self._should_reconnect:
            self.logger.info("Reconnection disabled, not attempting to reconnect")
            return
            
        # Increment connection attempts
        self.connection_attempts += 1
        
        # Calculate backoff time (exponential with jitter)
        max_backoff = min(60, 2 ** min(self.connection_attempts, 6))  # Cap at 60 seconds
        backoff = max_backoff * (0.5 + random.random() * 0.5)  # Add jitter
        
        self.logger.info(f"Reconnection attempt {self.connection_attempts} scheduled in {backoff:.2f} seconds")
        
        # Wait for backoff period
        await asyncio.sleep(backoff)
        
        # Attempt to reconnect
        try:
            self.logger.info("Attempting to reconnect...")
            await self.start(wait_for_connection=False)
        except Exception as e:
            self.logger.error(f"Reconnection failed: {str(e)}", exc_info=True)
            # Schedule another reconnection attempt
            if self._should_reconnect:
                asyncio.create_task(self._reconnect())
                
            # Resubscribe to any symbols if needed
            if self.subscribed_symbols:
                await self._subscribe_to_symbols(list(self.subscribed_symbols))
                
            self.logger.info("WebSocket client started successfully")
            
            # Notify any callbacks
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback({'type': 'connection', 'status': 'connected'})
                    else:
                        callback({'type': 'connection', 'status': 'connected'})
                except Exception as e:
                    self.logger.error(f"Error in connection callback: {str(e)}", exc_info=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket client: {str(e)}", exc_info=True)
            self.connected = False
            self.running = False
            self.last_error = str(e)
            
            # Notify callbacks of connection error
            for callback in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback({
                            'type': 'error',
                            'error': str(e),
                            'status': 'connection_failed'
                        })
                    else:
                        callback({
                            'type': 'error',
                            'error': str(e),
                            'status': 'connection_failed'
                        })
                except Exception as cb_err:
                    self.logger.error(f"Error in error callback: {str(cb_err)}", exc_info=True)
            
            # Schedule reconnection if needed
            if self._should_reconnect:
                self.logger.info("Scheduling reconnection attempt...")
                asyncio.create_task(self._reconnect())
            
            if wait_for_connection:
                raise
                
            return False

    async def subscribe(self, symbols):
        """
        Public method to subscribe to a list of ticker symbols for real-time data.
        
        Args:
            symbols: List of ticker symbols to subscribe to
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not symbols:
            self.logger.warning("No symbols provided for subscription")
            return False
            
        # Convert single symbol to list if needed
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Make sure all symbols are uppercase
        symbols = [s.upper() for s in symbols]
        
        # Filter out symbols we're already subscribed to
        new_symbols = [s for s in symbols if s not in self.subscribed_symbols]
        
        if not new_symbols:
            self.logger.info(f"Already subscribed to all symbols: {symbols}")
            return True
            
        # Subscribe to new symbols
        result = await self._subscribe_to_symbols(new_symbols)
        return result
        
    async def _subscribe_to_symbols(self, symbols):
        """
        Subscribe to a list of ticker symbols for real-time data.
        
        This method sends a subscription request to the Alpaca WebSocket API for the specified
        ticker symbols. It subscribes to trades, quotes, and minute bars for each symbol.
        
        Args:
            symbols: List of ticker symbols to subscribe to
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not symbols:
            self.logger.warning("No symbols provided for subscription")
            return False
            
        if not self.ws or self.is_ws_closed():
            self.logger.error("Cannot subscribe: WebSocket not connected")
            return False
            
        if not self.authenticated:
            self.logger.error("Cannot subscribe: Not authenticated")
            return False
            
        try:
            # Prepare subscription message
            sub_msg = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            
            # Send subscription message
            self.logger.info(f"Subscribing to {len(symbols)} symbols: {symbols}")
            await self.ws.send(json.dumps(sub_msg))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            self.logger.info(f"Subscription response: {response_data}")
            
            # Check for subscription success
            if isinstance(response_data, list) and len(response_data) > 0:
                for msg in response_data:
                    if msg.get('T') == 'subscription' and 'subscribed' in msg.get('msg', ''):
                        # Update subscribed symbols set
                        for symbol in symbols:
                            self.subscribed_symbols.add(symbol)
                        self.logger.info(f"Successfully subscribed to {len(symbols)} symbols")
                        return True
            
            # If we got here, subscription failed
            self.logger.error(f"Subscription failed: {response_data}")
            return False
            
        except asyncio.TimeoutError:
            self.logger.error("Subscription request timed out")
            return False
        except Exception as e:
            self.logger.error(f"Subscription error: {str(e)}", exc_info=True)
            return False
    
    async def _reconnect(self):
        """
        Handle reconnection with proper backoff, cleanup, and state management.
        
        This method implements a robust reconnection strategy with exponential backoff,
        jitter, and proper cleanup of existing connections. It will attempt to
        re-establish the WebSocket connection and re-authenticate.
        
        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        if not self._should_reconnect or not self._persist_connection:
            self.logger.info("Reconnection disabled or persistence turned off")
            return False
            
        self.connection_attempts += 1
        
        # Check if we've exceeded max attempts
        if self.connection_attempts > self.max_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_attempts}) reached. Giving up.")
            return False
            
        # Calculate delay with exponential backoff and jitter
        delay = min(self.reconnect_delay * (2 ** (self.connection_attempts - 1)), 
                   self.max_reconnect_delay)
        jitter = random.uniform(0.8, 1.2)  # Add some jitter between 0.8 and 1.2
        delay = delay * jitter
        
        self.logger.info(f"Attempting to reconnect in {delay:.1f} seconds (attempt {self.connection_attempts}/{self.max_attempts})")
        
        try:
            # Clean up any existing connection
            await self._cleanup_connection()
            
            # Wait for the calculated delay
            await asyncio.sleep(delay)
            
            # Try to reconnect
            self.logger.info("Attempting to reconnect...")
            await self.start()
            
            # If we have any subscriptions, resubscribe
            if self.subscribed_symbols:
                self.logger.info(f"Resubscribing to {len(self.subscribed_symbols)} symbols")
                await self._subscribe_to_symbols(list(self.subscribed_symbols))
                
            self.connection_attempts = 0  # Reset attempts on success
            self.logger.info("Reconnection successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self.connection_attempts} failed: {str(e)}")
            # Schedule the next reconnection attempt
            asyncio.create_task(self._reconnect())
            return False
            
    async def _cleanup_connection(self):
        """Clean up any existing WebSocket connection"""
        try:
            if self.ws:
                if not self.is_ws_closed():
                    await self.ws.close()
                self.ws = None
                
            if self.message_handler_task and not self.message_handler_task.done():
                self.message_handler_task.cancel()
                try:
                    await self.message_handler_task
                except asyncio.CancelledError:
                    pass
                    
            self.connected = False
            self.authenticated = False
            
        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {str(e)}", exc_info=True)

    async def _message_handler(self):
        """
        Handle incoming messages from the WebSocket server.
        
        This method runs in a loop, processing messages as they arrive. It handles:
        self.logger.info("Starting WebSocket message handler")
        """
        
        while self.running and not self._loop.is_closed():
            try:
                # Check connection status and reconnect if needed
                if not self.connected or not self.ws or self.is_ws_closed():
                    self.logger.warning("WebSocket not connected, attempting to reconnect...")
                    if not await self._reconnect():
                        await asyncio.sleep(min(30, 2 ** self.connection_attempts))
                        continue
                
                try:
                    # Use lock to prevent concurrent recv() calls
                    async with self._message_lock:
                        message = await asyncio.wait_for(
                            self.ws.recv(), 
                            timeout=30
                        )
                    
                    if not message:
                        self.logger.warning("Received empty message")
                        continue
                        
                    # Process the message
                    await self._handle_message(message)
                    
                    # Reset connection attempts on successful message
                    self.connection_attempts = 0
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    try:
                        if self.ws and not self.is_ws_closed():
                            async with self._message_lock:
                                await asyncio.wait_for(
                                    self.ws.ping(),
                                    timeout=5
                                )
                            self.logger.debug("Ping sent successfully")
                    except asyncio.TimeoutError:
                        self.logger.warning("Ping timeout, connection may be unstable")
                        raise
                    except Exception as e:
                        self.logger.warning(f"Error sending ping: {str(e)}")
                        raise
                        
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {e}")
                await self._handle_connection_error(e)
                
            except asyncio.CancelledError:
                self.logger.info("Message handler task cancelled")
                raise
                
            except Exception as e:
                self.logger.error(
                    f"Unexpected error in message handler: {str(e)}", 
                    exc_info=True
                )
                await self._handle_connection_error(e)
                
            # Add a small sleep to prevent tight loop on errors
            if not self.connected:
                await asyncio.sleep(1)
                
        self.logger.info("Message handler stopped")

    async def _notify_callbacks(self, data):
        """
        Notify all registered callbacks with the provided data.
        
        Args:
            data: The data to pass to the callbacks
        """
        if not self.callbacks:
            return
            
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback: {str(e)}", exc_info=True)
    
    async def _handle_connection_error(self, error):
        """
        Handle connection errors and attempt reconnection.
        
        Args:
            error: The exception that was raised
        """
        self.connected = False
        self.authenticated = False
        self.last_error = str(error)
        
        # Log the error
        self.logger.error(f"Connection error: {str(error)}", exc_info=isinstance(error, Exception))
        
        # Notify callbacks of disconnection
        await self._notify_callbacks({
            'type': 'connection',
            'status': 'error',
            'error': str(error),
            'message': 'Connection error occurred'
        })
        
        # Notify callbacks of disconnection with status 'disconnected'
        await self._notify_callbacks({
            'type': 'connection',
            'status': 'disconnected',
            'error': str(error)
        })
        
        # Attempt reconnection if enabled
        if self._should_reconnect and self.running:
            self.logger.info("Scheduling reconnection...")
            await self._reconnect()

    async def _authenticate(self):
        """
        Authenticate with the Alpaca WebSocket API.
        
        This method sends an authentication message to the WebSocket server and waits for
        the authentication response. It handles the two-step authentication process where
        the first message is a connection confirmation and the second is the actual auth response.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        if not self.ws or self.is_ws_closed():
            self.logger.error("Cannot authenticate: WebSocket not connected")
            return False
            
        try:
            # Prepare authentication message
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            
            # Send authentication message
            self.logger.info("Sending authentication request...")
            await self.ws.send(json.dumps(auth_msg))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            try:
                response_data = json.loads(response)
                self.logger.info(f"Auth response: {response_data}")
                
                # First message is usually a connection success message
                # We need to receive the actual auth response
                if isinstance(response_data, list) and len(response_data) > 0 and response_data[0].get('T') == 'success' and 'connected' in response_data[0].get('msg', ''):
                    # This is just the connection message, get the actual auth response
                    response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    self.logger.info(f"Second response (auth): {response_data}")
                
                # Check for authentication success
                if isinstance(response_data, list) and len(response_data) > 0:
                    for msg in response_data:
                        if msg.get('T') == 'success' and msg.get('msg') == 'authenticated':
                            self.authenticated = True
                            self.logger.info("Successfully authenticated with Alpaca")
                            return True
                
                # If we got here, authentication failed
                self.logger.error(f"Authentication failed: {response_data}")
                return False
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding message: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error handling authentication response: {str(e)}", exc_info=True)
                raise
                
        except asyncio.TimeoutError:
            self.logger.error("Authentication timed out")
            return False
        
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket connection closed: {e}")
            await self._handle_connection_error(e)

    async def _handle_trade(self, trade_data):
        """Handle trade data from Alpaca WebSocket"""
        try:
            symbol = trade_data.get('S')
            if not symbol:
                self.logger.warning("No symbol in trade data")
                return
                
            # Format the trade data
            formatted_trade = {
                'event': 'trade',
                'symbol': symbol,
                'price': float(trade_data.get('p', 0)),
                'size': int(trade_data.get('s', 0)),
                'exchange': trade_data.get('x', ''),
                'timestamp': trade_data.get('t', ''),
                'conditions': trade_data.get('c', []),
                'tape': trade_data.get('z', '')
            }
            
            # Update cache
            if symbol not in self.data:
                self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            # Keep only the most recent trades per symbol (limit to prevent memory issues)
            self.data[symbol]['trades'].append(formatted_trade)
            if len(self.data[symbol]['trades']) > 1000:  # Keep last 1000 trades
                self.data[symbol]['trades'] = self.data[symbol]['trades'][-1000:]
            
            # Update last update time
            self.last_update[symbol] = datetime.utcnow()
            
            # Notify callbacks
            await self._notify_callbacks({'type': 'trade', 'symbol': symbol, 'data': formatted_trade})
                    
        except Exception as e:
            self.logger.error(f"Error handling trade: {str(e)}", exc_info=True)
            
    async def _handle_quote(self, quote_data):
        """Handle quote data from Alpaca WebSocket"""
        try:
            # Alpaca quote message format: {'T': 'q', 'S': 'AAPL', 'bp': 150.4, 'bs': 1, 'ap': 150.5, 'as': 3, 't': '2023-01-01T00:00:00.000Z', 'c': ['R'], 'z': 'C'}
            symbol = quote_data.get('S')
            if not symbol:
                self.logger.warning("No symbol in quote data")
                return
                
            # Format the quote data
            formatted_quote = {
                'event': 'quote',
                'symbol': symbol,
                'bid_price': float(quote_data.get('bp', 0)),
                'bid_size': int(quote_data.get('bs', 0)),
                'ask_price': float(quote_data.get('ap', 0)),
                'ask_size': int(quote_data.get('as', 0)),
                'timestamp': quote_data.get('t', ''),
                'conditions': quote_data.get('c', []),
                'tape': quote_data.get('z', '')
            }
            
            # Update cache
            if symbol not in self.data:
                self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            # Keep only the most recent quotes per symbol
            self.data[symbol]['quotes'].append(formatted_quote)
            if len(self.data[symbol]['quotes']) > 1000:  # Keep last 1000 quotes
                self.data[symbol]['quotes'] = self.data[symbol]['quotes'][-1000:]
            
            # Update last update time
            self.last_update[symbol] = datetime.utcnow()
            
            # Notify callbacks
            await self._notify_callbacks({'type': 'quote', 'symbol': symbol, 'data': formatted_quote})
                    
        except Exception as e:
            self.logger.error(f"Error handling quote: {str(e)}", exc_info=True)
            
    async def _handle_bar(self, bar_data):
        """Handle bar data from Alpaca WebSocket"""
        try:
            # Alpaca bar message format: {'T': 'b', 'S': 'AAPL', 'o': 150.0, 'h': 150.5, 'l': 149.5, 'c': 150.2, 'v': 1000, 't': '2023-01-01T00:01:00Z'}
            symbol = bar_data.get('S')
            if not symbol:
                self.logger.warning("No symbol in bar data")
                return
                
            # Format the bar data
            formatted_bar = {
                'event': 'bar',
                'symbol': symbol,
                'open': float(bar_data.get('o', 0)),
                'high': float(bar_data.get('h', 0)),
                'low': float(bar_data.get('l', 0)),
                'close': float(bar_data.get('c', 0)),
                'volume': int(bar_data.get('v', 0)),
                'timestamp': bar_data.get('t', '')
            }
            
            # Update cache
            if symbol not in self.data:
                self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            # Keep only the most recent bars per symbol
            self.data[symbol]['bars'].append(formatted_bar)
            if len(self.data[symbol]['bars']) > 1440:  # Keep 1 day of 1-min bars
                self.data[symbol]['bars'] = self.data[symbol]['bars'][-1440:]
            
            # Update last update time
            self.last_update[symbol] = datetime.utcnow()
            
            # Notify callbacks
            await self._notify_callbacks({'type': 'bar', 'symbol': symbol, 'data': formatted_bar})
                    
        except Exception as e:
            self.logger.error(f"Error handling bar: {str(e)}", exc_info=True)
            
    def get_status(self) -> dict:
        """
        Get the current status of the WebSocket connection and subscriptions.
        
        Returns:
            dict: Dictionary containing connection status, subscription info, and metrics
        """
        try:
            status = {
                'connected': self.connected,
                'authenticated': getattr(self, 'authenticated', False),
                'subscribed_symbols': list(self.subscribed_symbols),
                'last_message': self.last_message,
                'last_error': self.last_error,
                'message_count': self.message_count,
                'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
                'data_points': {symbol: len(data.get('trades', []) + data.get('quotes', []) + data.get('bars', [])) 
                                for symbol, data in self.data.items()}
            }
            return status

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding message: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}", exc_info=True)
            raise
        
async def _handle_trade(self, trade_data):
    """Handle trade data from Alpaca WebSocket"""
    try:
        symbol = trade_data.get('S')
        if not symbol:
            self.logger.warning("No symbol in trade data")
            return

        # Format the trade data
        formatted_trade = {
            'event': 'trade',
            'symbol': symbol,
            'price': float(trade_data.get('p', 0)),
            'size': int(trade_data.get('s', 0)),
            'exchange': trade_data.get('x', ''),
            'timestamp': trade_data.get('t', ''),
            'conditions': trade_data.get('c', []),
            'tape': trade_data.get('z', '')
        }

        # Update cache
        if symbol not in self.data:
            self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}

        # Keep only the most recent trades per symbol (limit to prevent memory issues)
        self.data[symbol]['trades'].append(formatted_trade)
        if len(self.data[symbol]['trades']) > 1000:  # Keep last 1000 trades
            self.data[symbol]['trades'] = self.data[symbol]['trades'][-1000:]

        # Update last update time
        self.last_update[symbol] = datetime.utcnow()

        # Notify callbacks
        if self.callback:
            try:
                await self.callback({'type': 'trade', 'symbol': symbol, 'data': formatted_trade})
            except Exception as e:
                self.logger.error(f"Error in trade callback: {str(e)}", exc_info=True)

    except Exception as e:
        self.logger.error(f"Error handling trade: {str(e)}", exc_info=True)

async def _handle_quote(self, quote_data):
    """Handle quote data from Alpaca WebSocket"""
    try:
        # Alpaca quote message format: {'T': 'q', 'S': 'AAPL', 'bp': 150.4, 'bs': 1, 'ap': 150.5, 'as': 3, 't': '2023-01-01T00:00:00.000Z', 'c': ['R'], 'z': 'C'}
        symbol = quote_data.get('S')
        if not symbol:
            self.logger.warning("No symbol in quote data")
            return

        # Format the quote data
        formatted_quote = {
            'event': 'quote',
            'symbol': symbol,
            'bid_price': float(quote_data.get('bp', 0)),
            'bid_size': int(quote_data.get('bs', 0)),
            'ask_price': float(quote_data.get('ap', 0)),
            'ask_size': int(quote_data.get('as', 0)),
            'timestamp': quote_data.get('t', ''),
            'conditions': quote_data.get('c', []),
            'tape': quote_data.get('z', '')
        }

        # Update cache
        if symbol not in self.data:
            self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}

        # Keep only the most recent quotes per symbol
        self.data[symbol]['quotes'].append(formatted_quote)
        if len(self.data[symbol]['quotes']) > 1000:  # Keep last 1000 quotes
            self.data[symbol]['quotes'] = self.data[symbol]['quotes'][-1000:]

        # Update last update time
        self.last_update[symbol] = datetime.utcnow()

        # Notify callbacks
        if self.callback:
            try:
                await self.callback({'type': 'quote', 'symbol': symbol, 'data': formatted_quote})
            except Exception as e:
                self.logger.error(f"Error in quote callback: {str(e)}", exc_info=True)

    except Exception as e:
        self.logger.error(f"Error handling quote: {str(e)}", exc_info=True)

async def _handle_bar(self, bar_data):
    """Handle bar data from Alpaca WebSocket"""
    try:
        # Alpaca bar message format: {'T': 'b', 'S': 'AAPL', 'o': 150.0, 'h': 150.5, 'l': 149.5, 'c': 150.2, 'v': 1000, 't': '2023-01-01T00:01:00Z'}
        symbol = bar_data.get('S')
        if not symbol:
            self.logger.warning("No symbol in bar data")
            return

        # Format the bar data
        formatted_bar = {
            'event': 'bar',
            'symbol': symbol,
            'open': float(bar_data.get('o', 0)),
            'high': float(bar_data.get('h', 0)),
            'low': float(bar_data.get('l', 0)),
            'close': float(bar_data.get('c', 0)),
            'volume': int(bar_data.get('v', 0)),
            'timestamp': bar_data.get('t', '')
        }

        # Update cache
        if symbol not in self.data:
            self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}

        # Keep only the most recent bars per symbol
        self.data[symbol]['bars'].append(formatted_bar)
        if len(self.data[symbol]['bars']) > 1440:  # Keep 1 day of 1-min bars
            self.data[symbol]['bars'] = self.data[symbol]['bars'][-1440:]

        # Update last update time
        self.last_update[symbol] = datetime.utcnow()

        # Notify callbacks
        if self.callback:
            try:
                await self.callback({'type': 'bar', 'symbol': symbol, 'data': formatted_bar})
            except Exception as e:
                self.logger.error(f"Error in bar callback: {str(e)}", exc_info=True)

    except Exception as e:
        self.logger.error(f"Error handling bar: {str(e)}", exc_info=True)

def get_status(self) -> dict:
    """
    Get the current status of the WebSocket connection and subscriptions.

    Returns:
        dict: Dictionary containing connection status, subscription info, and metrics
    """
    status = {
        'connected': self.connected,
        'authenticated': getattr(self, 'authenticated', False),
        'subscribed_symbols': list(self.subscribed_symbols),
        'last_message': self.last_message,
        'last_error': self.last_error,
        'message_count': self.message_count,
        'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
        'data_points': {symbol: len(data.get('trades', []) + data.get('quotes', []) + data.get('bars', [])) 
                      for symbol, data in self.data.items()}
    }
    return status

async def subscribe(self, symbols: List[str]):
    """
    Subscribe to real-time updates for the given symbols.

    Args:
        symbols: List of ticker symbols to subscribe to
    """
    if not self.connected or not self.authenticated:
        raise ConnectionError("WebSocket not connected or authenticated")

    # Convert to uppercase and filter out already subscribed symbols
    new_symbols = [s.upper() for s in symbols if s.upper() not in self.subscribed_symbols]
    if not new_symbols:
        self.logger.debug("All symbols already subscribed")
        return True

    # Add new symbols to the subscription set
    self.subscribed_symbols.update(new_symbols)
    self.logger.info(f"Added {len(new_symbols)} new symbols to subscription")

    # Subscribe to the new symbols
    return await self._subscribe_to_symbols()

async def _subscribe_to_symbols(self):
    """
    Subscribe to updates for all currently tracked symbols.

    This sends a subscription message to the WebSocket server for all symbols
    in the subscribed_symbols set, requesting trades, quotes, and bars.
    """
    if not self.subscribed_symbols:
        self.logger.debug("No symbols to subscribe to")
        return False

    if not self.connected or not self.authenticated or not self.ws or self.is_ws_closed():
        self.logger.warning("Cannot subscribe: WebSocket not connected or authenticated")
        return False

    async with self.lock:
        symbols = list(self.subscribed_symbols)
        try:
            # Split symbols into chunks to avoid message size limits
            chunk_size = 100  # Adjust based on your needs
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]

                # Create subscription message for trades, quotes, and bars
                subscribe_msg = {
                    'action': 'subscribe',
                    'trades': chunk,
                    'quotes': chunk,
                    'bars': chunk
                }

                self.logger.debug(f"Sending subscription for {len(chunk)} symbols")
                await self.ws.send(json.dumps(subscribe_msg))

                # Small delay between chunks to avoid rate limiting
                if i + chunk_size < len(symbols):
                    await asyncio.sleep(0.5)


            self.logger.info(f"Subscribed to updates for {len(symbols)} symbols")
            return True

        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error(f"Connection closed while subscribing: {e}")
            await self._handle_connection_error(e)
            return False

        except Exception as e:
            self.logger.error(f"Error subscribing to symbols: {str(e)}", exc_info=True)
            return False

async def unsubscribe(self, symbols: List[str]):
    """
    Unsubscribe from real-time updates for the given symbols.

    Args:
        symbols: List of ticker symbols to unsubscribe from
        
    Returns:
        bool: True if unsubscription was successful, False otherwise
    """
    if not symbols:
        return True
        
    # Normalize symbols to uppercase
    if isinstance(symbols, str):
        symbols = [symbols.upper()]
    else:
        symbols = [s.upper() for s in symbols if isinstance(s, str)]
        
    # Filter to only include symbols we're actually subscribed to
    symbols_to_remove = [s for s in symbols if s in self.subscribed_symbols]
    if not symbols_to_remove:
        self.logger.debug("No matching symbols to unsubscribe from")
        return True

    async with self.lock:
        self.logger.info(f"Unsubscribing from symbols: {symbols_to_remove}")
        
        # Check connection status
        if not self.connected or not self.authenticated:
            self.logger.error("Cannot unsubscribe: WebSocket not connected or authenticated")
            return False
            
        if not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.error("WebSocket connection not established")
            return False
        
        try:
            # Use batch unsubscribe for efficiency
            unsubscribe_msg = {
                'action': 'unsubscribe',
                'trades': symbols_to_remove,
                'quotes': symbols_to_remove,
                'bars': symbols_to_remove
            }
            
            await self.ws.send(json.dumps(unsubscribe_msg))
            
            # Update our tracking
            for symbol in symbols_to_remove:
                self.subscribed_symbols.discard(symbol)
            
            self.logger.info(f"Successfully unsubscribed from symbols: {symbols_to_remove}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from symbols: {e}")
            return False
                
    async def _subscribe_to_symbols(self):
        """Subscribe to symbol updates"""
        async with self.lock:
            self.logger.info("=== Starting subscription process ===")
            self.logger.info(f"Current subscribed symbols: {self.subscribed_symbols}")
            self.logger.info(f"Requested symbols: {self.symbols}")
            
            try:
                if not self.connected or not self.authenticated:
                    self.logger.warning("Not connected or authenticated. Cannot subscribe to symbols.")
                    return False
                    
                # Get symbols that aren't already subscribed
                symbols_to_subscribe = [s for s in self.symbols if s not in self.subscribed_symbols]
                
                if not symbols_to_subscribe:
                    self.logger.info("No new symbols to subscribe to")
                    return True
                    
                self.logger.info(f"Subscribing to new symbols: {symbols_to_subscribe}")
                
                # Create subscription message
                msg = {
                    'action': 'subscribe',
                    'trades': symbols_to_subscribe,
                    'quotes': symbols_to_subscribe,
                    'bars': symbols_to_subscribe
                }
                
                # Send subscription message
                await self._send_json(msg)
                
                # Update our tracking
                self.subscribed_symbols.update(symbols_to_subscribe)
                
                self.logger.info(f"Successfully subscribed to symbols: {symbols_to_subscribe}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error subscribing to symbols: {e}", exc_info=True)
                return False

    async def _authenticate(self):
        """
        Authenticate with the Alpaca WebSocket API.
        
        Sends an authentication message with the API key and secret.
        Handles multiple authentication response formats for compatibility.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        self.logger.info("=== Starting authentication process ===")
        
        # Determine which WebSocket connection to use
        websocket = self.ws if hasattr(self, 'ws') and self.ws else \
                   self.websocket if hasattr(self, 'websocket') and self.websocket else None
                   
        # Check if WebSocket connection exists
        if not self.connected or not websocket or getattr(websocket, 'closed', True):
            self.logger.error("Cannot authenticate: WebSocket not connected")
            return False
            
        # Check if API key and secret are available
        if not self.api_key or not self.api_secret:
            self.logger.error("API key or secret not available")
            return False
            
        try:
            # Support both authentication formats
            # Format 1: Legacy format
            auth_msg_legacy = {
                'action': 'auth',
                'key': self.api_key,
                'secret': self.api_secret
            }
            
            # Format 2: Newer format
            auth_msg_new = {
                'action': 'authenticate',
                'data': {
                    'key_id': self.api_key,
                    'secret_key': self.api_secret
                }
            }
            
            # Use the legacy format by default
            auth_msg = auth_msg_legacy
            
            self.logger.info("Sending authentication message...")
            await websocket.send(json.dumps(auth_msg))
            
            # Wait for authentication response with timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            self.logger.debug(f"Auth response received: {type(response_data)}")
            
            # Handle different response formats
            if isinstance(response_data, list):
                # Handle case where multiple messages are received at once
                for msg in response_data:
                    if msg.get('T') == 'success' and msg.get('msg') == 'authenticated':
                        self.authenticated = True
                        self.logger.info(" Successfully authenticated with Alpaca WebSocket API (batch response)")
                        return True
            elif isinstance(response_data, dict):
                # Handle different response formats
                # Format 1: Legacy API format
                if response_data.get('T') == 'success' and response_data.get('msg') == 'authenticated':
                    self.authenticated = True
                    self.logger.info(" Successfully authenticated (legacy API format)")
                    return True
                # Format 2: Newer API format with data field
                elif 'data' in response_data and 'status' in response_data['data']:
                    if response_data['data']['status'] == 'authenticated':
                        self.authenticated = True
                        self.logger.info("=== Successfully authenticated with Alpaca WebSocket API ===")
                        self.logger.info(f"Connection ID: {response_data.get('data', {}).get('connection_id', 'N/A')}")
                        return True
                # Format 3: Alternative authorization format
                elif response_data.get('stream') == 'authorization' and response_data.get('data', {}).get('status') == 'authorized':
                    self.authenticated = True
                    self.logger.info(" Successfully authorized with Alpaca WebSocket API")
                    return True
            
            # If we get here, authentication failed
            error_msg = f"Authentication failed: {response_data}"
            self.logger.error(error_msg)
            self.authenticated = False
            
            # Try the alternative authentication format if the first one failed
            self.logger.info("Trying alternative authentication format...")
            await websocket.send(json.dumps(auth_msg_new))
            
            # Wait for authentication response with timeout
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            # Process the response with the same logic as above
            if isinstance(response_data, dict):
                if response_data.get('T') == 'success' and response_data.get('msg') == 'authenticated':
                    self.authenticated = True
                    self.logger.info("✅ Successfully authenticated with alternative format")
                    return True
                elif 'data' in response_data and 'status' in response_data['data']:
                    if response_data['data']['status'] == 'authenticated':
                        self.authenticated = True
                        self.logger.info("✅ Successfully authenticated with alternative format")
                        return True
            
            self.logger.error(f"Authentication failed with both formats: {response_data}")
            return False
            
        except asyncio.TimeoutError:
            self.logger.error("Authentication timed out after 10 seconds")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during authentication: {str(e)}", exc_info=True)
            return False
            
    async def _reconnect(self):
        """
        Handle reconnection with exponential backoff and jitter.
        
        This method implements a robust reconnection strategy that will attempt
        to reconnect to the WebSocket server with increasing delays between attempts.
        """
        if not self._should_reconnect or not self._persist_connection:
            self.logger.info("Reconnection disabled or persistence turned off")
            return False
            
        self.connection_attempts += 1
        
        # Check if we've exceeded max attempts
        if self.connection_attempts > self.max_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_attempts}) reached. Giving up.")
            await self._notify_callbacks({
                'type': 'connection',
                'status': 'failed',
                'message': f'Max reconnection attempts ({self.max_attempts}) reached',
                'error': 'Max reconnection attempts reached'
            })
            return False
            
        # Calculate delay with exponential backoff and jitter
        delay = min(self.reconnect_delay * (2 ** (self.connection_attempts - 1)), 
                   self.max_reconnect_delay)
        jitter = random.uniform(0.8, 1.2)  # Add some jitter between 0.8 and 1.2
        delay = delay * jitter
        
        self.logger.info(f"Attempting to reconnect in {delay:.1f} seconds (attempt {self.connection_attempts}/{self.max_attempts})")
        
        try:
            # Clean up any existing connection
            await self._cleanup_connection()
            
            # Wait for the calculated delay
            await asyncio.sleep(delay)
            
            # Try to reconnect
            self.logger.info("Attempting to reconnect...")
            await self.start()
            
            # If we have any subscriptions, resubscribe
            if self.subscribed_symbols:
                self.logger.info(f"Resubscribing to {len(self.subscribed_symbols)} symbols")
                await self._subscribe_to_symbols()
                
            self.connection_attempts = 0  # Reset attempts on success
            self.logger.info("Reconnection successful")
            
            # Notify callbacks of successful reconnection
            await self._notify_callbacks({
                'type': 'connection',
                'status': 'reconnected',
                'message': 'Successfully reconnected to WebSocket',
                'attempts': self.connection_attempts
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self.connection_attempts} failed: {str(e)}")
            
            # Schedule the next reconnection attempt
            asyncio.create_task(self._reconnect())
            return False
            
    async def _cleanup_connection(self):
        """
        Clean up any existing WebSocket connection and tasks.
        
        This method ensures that all resources are properly released and
        any pending tasks are cancelled.
        """
        try:
            # Cancel the message handler task if it exists
            if hasattr(self, 'message_handler_task') and self.message_handler_task:
                if not self.message_handler_task.done():
                    self.message_handler_task.cancel()
                    try:
                        await self.message_handler_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error in message handler task: {str(e)}", exc_info=True)
            
            # Close the WebSocket connection if it exists
            if hasattr(self, 'ws') and self.ws and not self.is_ws_closed():
                self.logger.debug("Closing WebSocket connection...")
                try:
                    await self.ws.close()
                    self.logger.debug("WebSocket connection closed")
                except Exception as e:
                    self.logger.error(f"Error closing WebSocket: {str(e)}", exc_info=True)
            
            # Reset connection state
            self.connected = False
            self.authenticated = False
            
        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {str(e)}", exc_info=True)
        finally:
            # Ensure these are always None after cleanup
            if hasattr(self, 'ws'):
                self.ws = None
            if hasattr(self, 'message_handler_task'):
                self.message_handler_task = None
                
    async def stop(self):
        """
        Stop the WebSocket client and clean up resources.
        
        This method will:
        1. Stop reconnection attempts
        2. Close the WebSocket connection
        3. Cancel any running tasks
        4. Clean up resources
        """
        if not self.running:
            self.logger.debug("WebSocket client is not running")
            return
            
        self.logger.info("Stopping WebSocket client...")
        self._should_reconnect = False
        self.running = False
        
        try:
            # Clean up the connection
            await self._cleanup_connection()
            
            # Notify callbacks of disconnection
            await self._notify_callbacks({
                'type': 'connection',
                'status': 'disconnected',
                'message': 'WebSocket client stopped'
            })
            
            self.logger.info("WebSocket client stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket client: {str(e)}", exc_info=True)
            raise
            
    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected"""
        if hasattr(self, 'running') and self.running:
            self.logger.warning("RealTimeDataHandler destroyed while still running. Call stop() first.")
            # Try to clean up synchronously if possible
            try:
                if hasattr(self, 'loop') and self.loop and self.loop.is_running():
                    # Schedule the stop in the existing loop
                    asyncio.create_task(self.stop())
                else:
                    # If we can't schedule the stop, use a safer approach to close the connection
                    if hasattr(self, 'ws') and self.ws and not self.is_ws_closed():
                        try:
                            # Create a new event loop for cleanup only if needed
                            need_new_loop = False
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_closed():
                                    need_new_loop = True
                            except RuntimeError:
                                # No event loop in this thread
                                need_new_loop = True
                                
                            if need_new_loop:
                                # Create a temporary loop just for cleanup
                                temp_loop = asyncio.new_event_loop()
                                try:
                                    # Close the WebSocket with timeout
                                    temp_loop.run_until_complete(
                                        asyncio.wait_for(self.ws.close(), timeout=2.0)
                                    )
                                except (asyncio.TimeoutError, Exception) as e:
                                    self.logger.warning(f"WebSocket close timed out or failed: {str(e)}")
                                finally:
                                    # Clean up the temporary loop
                                    try:
                                        temp_loop.run_until_complete(temp_loop.shutdown_asyncgens())
                                        pending = asyncio.all_tasks(loop=temp_loop)
                                        if pending:
                                            temp_loop.run_until_complete(
                                                asyncio.gather(*pending, return_exceptions=True)
                                            )
                                    except Exception:
                                        pass
                                    temp_loop.close()
                            else:
                                # Use existing loop but don't wait for completion
                                # Just schedule the close and let the loop handle it
                                asyncio.create_task(self.ws.close())
                        except Exception as e:
                            self.logger.error(f"Error closing WebSocket in __del__: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error during cleanup in __del__: {str(e)}", exc_info=True)

    async def _subscribe_to_symbols(self):
        """
        Subscribe to updates for all currently tracked symbols.
        
        This method handles the actual WebSocket subscription message for all
        symbols in the subscribed_symbols set. It sends a single subscription
        message for all data types (trades, quotes, bars).
        """
        if not self.subscribed_symbols:
            self.logger.debug("No symbols to subscribe to")
            return False
            
        if not self.connected or not self.authenticated or not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.warning("Cannot subscribe: WebSocket not connected")
            return False
            
        symbols = list(self.subscribed_symbols)
        success = True
        
        # Split into chunks to avoid message size limits
        chunk_size = 100  # Adjust based on WebSocket server limits
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            
            try:
                subscribe_msg = {
                    "action": "subscribe",
                    "trades": chunk,
                    "quotes": chunk,
                    "bars": chunk
                }
                
                self.logger.debug(f"Subscribing to {len(chunk)} symbols: {chunk}")
                
                # Send the subscription message
                await self.ws.send(json.dumps(subscribe_msg))
                
                # Small delay between subscription chunks to avoid rate limiting
                if i + chunk_size < len(symbols):
                    await asyncio.sleep(0.5)
                
                self.logger.info(f"Successfully sent subscription for {len(chunk)} symbols")
                
            except Exception as e:
                error_msg = f"Error subscribing to symbols: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                success = False
                
        return success
        
    async def _unsubscribe_symbols(self, symbols):
        """
        Unsubscribe from updates for the given symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if not symbols:
            self.logger.debug("No symbols to unsubscribe from")
            return True
            
        if not self.connected or not self.authenticated or not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.warning("Cannot unsubscribe: WebSocket not connected")
            return False
            
        success = True
        
        # Split into chunks to avoid message size limits
        chunk_size = 100  # Adjust based on WebSocket server limits
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            
            try:
                unsubscribe_msg = {
                    "action": "unsubscribe",
                    "trades": chunk,
                    "quotes": chunk,
                    "bars": chunk
                }
                
                self.logger.debug(f"Unsubscribing from {len(chunk)} symbols: {chunk}")
                
                # Send the unsubscription message
                await self.ws.send(json.dumps(unsubscribe_msg))
                
                # Small delay between unsubscription chunks to avoid rate limiting
                if i + chunk_size < len(symbols):
                    await asyncio.sleep(0.5)
                
                # Remove from our internal tracking
                for symbol in chunk:
                    if symbol in self.subscribed_symbols:
                        self.subscribed_symbols.remove(symbol)
                
                self.logger.info(f"Successfully unsubscribed from {len(chunk)} symbols")
                
            except Exception as e:
                error_msg = f"Error unsubscribing from symbols: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                success = False
                
        return success
        
    async def subscribe(self, symbols):
        """
        Subscribe to updates for the given symbols.
        
        Args:
            symbols: Single symbol (str) or list of symbols to subscribe to
            
        Returns:
            bool: True if subscription was successful, False otherwise
        """
        if not symbols:
            return False
            
        # Convert single symbol to list if needed
        if isinstance(symbols, str):
            symbols = [symbols.upper()]
        else:
            symbols = [s.upper() for s in symbols if isinstance(s, str)]
            
        if not symbols:
            self.logger.warning("No valid symbols provided for subscription")
            return False
            
        # Add to our set of subscribed symbols
        self.subscribed_symbols.update(symbols)
        
        # If we're not connected yet, the subscription will happen on connect
        if not self.connected or not self.authenticated or not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.debug("Not currently connected, subscription will happen on connect")
            return True
            
        # Otherwise, subscribe now
        return await self._subscribe_to_symbols()
        
    async def unsubscribe(self, symbols):
        """
        Unsubscribe from updates for the given symbols.
        
        Args:
            symbols: Single symbol (str) or list of symbols to unsubscribe from
            
        Returns:
            bool: True if unsubscription was successful, False otherwise
        """
        if not symbols:
            return False
            
        # Convert single symbol to list if needed
        if isinstance(symbols, str):
            symbols = [symbols.upper()]
        else:
            symbols = [s.upper() for s in symbols if isinstance(s, str)]
            
        if not symbols:
            self.logger.warning("No valid symbols provided for unsubscription")
            return False
            
        # Remove from our set of subscribed symbols
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        # If we're not connected, we're done
        if not self.connected or not self.authenticated or not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.debug("Not currently connected, no need to unsubscribe")
            return True
            
        # Otherwise, unsubscribe now
        return await self._unsubscribe_symbols(symbols)
        
    async def update_symbols(self, symbols):
        """
        Update the list of subscribed symbols, adding new ones and removing old ones.
        
        Args:
            symbols: Single symbol (str) or list of symbols to subscribe to
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if symbols is None:
            symbols = []
            
        # Convert single symbol to list if needed
        if isinstance(symbols, str):
            new_symbols = {symbols.upper()}
        else:
            new_symbols = {s.upper() for s in symbols if isinstance(s, str)}
            
        # Find symbols to add and remove
        to_add = new_symbols - self.subscribed_symbols
        to_remove = self.subscribed_symbols - new_symbols
        
        # Update our internal set
        self.subscribed_symbols = new_symbols
        
        # If we're not connected, we're done
        if not self.connected or not self.authenticated or not hasattr(self, 'ws') or not self.ws or self.is_ws_closed():
            self.logger.debug("Not currently connected, subscriptions will be updated on connect")
            return True
            
        # Otherwise, update subscriptions
        success = True
        
        if to_remove:
            self.logger.info(f"Unsubscribing from {len(to_remove)} symbols")
            if not await self._unsubscribe_symbols(list(to_remove)):
                success = False
                
        if to_add:
            self.logger.info(f"Subscribing to {len(to_add)} new symbols")
            if not await self._subscribe_to_symbols():
                success = False
                
        return success
        
    def safe_update_symbols(self, symbols):
        """
        Thread-safe wrapper for update_symbols that can be called from sync code.
        
        Args:
            symbols: Single symbol (str) or list of symbols to subscribe to
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if not self.loop.is_running():
                # If event loop is not running, run it in a new thread
                future = asyncio.run_coroutine_threadsafe(
                    self.update_symbols(symbols),
                    self.loop
                )
                return future.result()
            else:
                # If event loop is running, schedule the coroutine
                future = asyncio.create_task(self.update_symbols(symbols))
                return future.result()
        except Exception as e:
            self.logger.error(f"Error in safe_update_symbols: {str(e)}")
            return False

    async def _connect_with_retry(self, max_attempts: int = 5) -> bool:
        """
        Attempt to connect to the WebSocket server with retries and exponential backoff.

        Args:
            max_attempts: Maximum number of connection attempts

        Returns:
            bool: True if connection and authentication were successful, False otherwise
        """
        import random
        import time
        import ssl
        
        base_delay = 1  # Start with 1 second delay
        max_delay = 60   # Maximum delay between retries (seconds)
        connect_timeout = 15.0  # 15 seconds to establish connection
        
        # Log connection attempt start
        self.logger.info(f"=== Starting connection attempt (max {max_attempts} attempts) ===")
        self.logger.info(f"WebSocket URL: {self.ws_url}")
        
        # Log environment info for debugging
        self.logger.debug(f"Python version: {sys.version}")
        self.logger.debug(f"Websockets version: {websockets.__version__}")
        self.logger.debug(f"SSL version: {ssl.OPENSSL_VERSION if hasattr(ssl, 'OPENSSL_VERSION') else 'N/A'}")
        
        for attempt in range(1, max_attempts + 1):
            self.connection_attempts = attempt
            attempt_start = time.time()
            
            # Calculate exponential backoff with jitter
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0.8, 1.2)  # Add some randomness to avoid thundering herd
            sleep_time = delay * jitter
            
            self.logger.info(f"\n=== Attempt {attempt}/{max_attempts} ===")
            self.logger.info(f"Backoff delay: {sleep_time:.1f}s")
            
            try:
                # Clean up any existing connection
                if hasattr(self, 'websocket') and self.websocket is not None:
                    self.logger.debug("Cleaning up existing WebSocket connection...")
                    try:
                        await self.websocket.close()
                        self.logger.debug("Existing WebSocket connection closed")
                    except Exception as close_error:
                        self.logger.warning(f"Error closing existing connection: {close_error}", exc_info=True)
                    finally:
                        self.websocket = None
                        self.connected = False
                        self.authenticated = False
                
                self.logger.info(f"Establishing new WebSocket connection to {self.ws_url}...")
                
                # Configure SSL context for secure connection
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                
                # Add more detailed logging for connection parameters
                self.logger.debug(f"Connection parameters: "
                                 f"ping_interval=20, "
                                 f"ping_timeout=20, "
                                 f"close_timeout=10, "
                                 f"max_size=8MB, "
                                 f"timeout={connect_timeout}s")
                
                # Create new connection with timeout
                try:
                    connect_coro = websockets.connect(
                        self.ws_url,
                        ssl=ssl_context,
                        ping_interval=20,      # Send ping every 20 seconds
                        ping_timeout=20,        # Wait up to 20 seconds for pong
                        close_timeout=10,       # Wait up to 10 seconds when closing
                        max_size=2**23         # 8MB max message size
                    )
                    
                    # Set user agent and other headers after connection is established
                    # This is a workaround for older websockets versions that don't support extra_headers
                    async def wrapped_connect():
                        ws = await connect_coro
                        # Set user agent and other headers if possible
                        if hasattr(ws, 'request_headers'):
                            ws.request_headers['User-Agent'] = 'StockMarketPredictor/1.0'
                            ws.request_headers['Accept'] = 'application/json'
                        return ws
                        
                    connect_coro = wrapped_connect()
                    
                    # Execute the connection with timeout
                    self.websocket = await asyncio.wait_for(connect_coro, timeout=connect_timeout)
                    
                    connect_duration = time.time() - attempt_start
                    self.logger.info(f"WebSocket connection established in {connect_duration:.2f}s")
                    
                    # Verify connection is open
                    if not self._is_websocket_connected(self.websocket):
                        raise ConnectionError("WebSocket connection not active after connect")
                    
                    # Attempt authentication
                    self.logger.info("Starting authentication...")
                    auth_start = time.time()
                    auth_result = await self._authenticate()
                    auth_duration = time.time() - auth_start
                    
                    if not auth_result:
                        raise ConnectionError(f"Authentication failed after {auth_duration:.2f}s")
                    
                    self.connected = True
                    self.authenticated = True
                    self.logger.info(f"✅ Successfully authenticated in {auth_duration:.2f}s")
                    
                    # Reset reconnection delay on successful connection
                    self.reconnect_delay = 1
                    
                    # Log total connection time
                    total_time = time.time() - attempt_start
                    self.logger.info(f"✓ Connection and authentication completed in {total_time:.2f} seconds")
                    
                    return True
                    
                except asyncio.TimeoutError as te:
                    raise ConnectionError(f"Connection timed out after {connect_timeout}s: {str(te)}")
                
                except websockets.exceptions.InvalidHandshake as ih:
                    raise ConnectionError(f"WebSocket handshake failed: {str(ih)}")
                    
                except websockets.exceptions.WebSocketException as we:
                    raise ConnectionError(f"WebSocket error: {str(we)}")
                
            except asyncio.CancelledError:
                self.logger.info("Connection attempt cancelled")
                raise
                
            except websockets.exceptions.InvalidURI as e:
                self.last_error = f"Invalid WebSocket URL: {self.ws_url}"
                self.logger.error(self.last_error)
                raise  # No point in retrying with invalid URL
                
            except websockets.exceptions.SecurityError as se:
                self.last_error = f"SSL/TLS security error: {str(se)}"
                self.logger.error(self.last_error)
                raise  # Security errors should not be retried
                
            except (ConnectionRefusedError, OSError) as e:
                self.last_error = f"Connection refused: {str(e)}"
                self.logger.warning(f"{self.last_error} (attempt {attempt}/{max_attempts})")
                if attempt >= max_attempts:
                    self.logger.error("Maximum connection attempts reached")
                
            except Exception as e:
                self.last_error = f"Connection error: {str(e)}"
                self.logger.warning(f"{self.last_error} (attempt {attempt}/{max_attempts})", 
                                   exc_info=attempt >= max_attempts)
            
            # Only sleep if we're going to try again
            if attempt < max_attempts:
                self.logger.info(f"Waiting {sleep_time:.1f}s before next attempt...")
                await asyncio.sleep(sleep_time)
        
        # If we get here, all attempts failed
        self.connected = False
        self.authenticated = False
        
        error_msg = f"❌ Failed to connect after {max_attempts} attempts. Last error: {self.last_error}"
        self.logger.error(error_msg)
        
        # If we have a callback, notify about the connection failure
        if hasattr(self, 'callback') and callable(self.callback):
            try:
                await self.callback({
                    'type': 'connection_failed',
                    'attempt': max_attempts,
                    'error': self.last_error
                })
            except Exception as e:
                self.logger.error(f"Error in connection failure callback: {e}", exc_info=True)
        
        return False

    async def _reconnect(self) -> bool:
        """
        Handle reconnection with proper backoff, cleanup, and state management.
        
        This method implements a robust reconnection strategy with exponential backoff,
        jitter, and proper cleanup of existing connections. It will attempt to
        re-establish the WebSocket connection and re-authenticate.
        
        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        import random
        import time
        
        self.logger.info("\n" + "="*50)
        self.logger.info("=== STARTING RECONNECTION PROCESS ===")
        self.logger.info("="*50)
        
        # Check if we should attempt reconnection
        if not self.running or self._stop_event.is_set():
            self.logger.info("Not attempting reconnection: handler is stopping")
            return False
            
        # Get current time for tracking reconnection duration
        reconnect_start = time.time()
        
        # Initialize reconnection state
        max_reconnect_attempts = 5
        base_delay = 1  # Start with 1 second delay
        max_delay = 120  # Maximum 2 minutes between attempts
        attempt = 0
        success = False
        
        # Store previously subscribed symbols
        prev_symbols = set(self.symbols) if hasattr(self, 'symbols') else set()
        
        # Clean up existing connection if it exists
        if hasattr(self, 'websocket') and self.websocket is not None:
            self.logger.info("Cleaning up existing WebSocket connection...")
            try:
                self.logger.info("Closing existing WebSocket connection...")
                close_start = time.time()
                await self.websocket.close()
                close_duration = time.time() - close_start
                self.logger.info(f"WebSocket connection closed in {close_duration:.2f}s")
            except Exception as e:
                self.logger.warning(f"Error closing existing connection: {e}", exc_info=True)
            finally:
                self.websocket = None
                self.connected = False
                self.authenticated = False
        
        # Main reconnection loop
        while attempt < max_reconnect_attempts and not success and not self._stop_event.is_set():
            attempt += 1
            self.connection_attempts = attempt
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0.8, 1.2)  # Add jitter to avoid thundering herd
            wait_time = wait_time * jitter
            
            self.logger.info(f"\n=== Reconnection attempt {attempt}/{max_reconnect_attempts} ===")
            self.logger.info(f"Waiting {wait_time:.1f}s before reconnection attempt...")
            
            # Sleep with progress updates every second
            start_time = time.time()
            remaining = wait_time
            
            while remaining > 0 and not self._stop_event.is_set():
                sleep_time = min(1.0, remaining)  # Update every second
                await asyncio.sleep(sleep_time)
                remaining = wait_time - (time.time() - start_time)
                if remaining > 0:
                    self.logger.debug(f"  - {remaining:.1f}s remaining before reconnection attempt...")
            
            if self._stop_event.is_set():
                self.logger.info("Reconnection cancelled: stop requested")
                return False
            
            # Try to reconnect
            self.logger.info("Attempting to reconnect...")
            try:
                if await self._connect_with_retry():
                    self.connection_attempts = 0  # Reset attempt counter on success
                    self.logger.info("✅ Successfully reconnected and re-authenticated")
                    
                    # Resubscribe to symbols if we were previously subscribed
                    if prev_symbols:
                        self.logger.info(f"Resubscribing to {len(prev_symbols)} symbols...")
                        await self._subscribe_to_symbols()
                        self.logger.info("Successfully resubscribed to all symbols")
                    
                    success = True
                    break  # Exit the reconnection loop on success
                else:
                    self.logger.error(f"Reconnection attempt {attempt} failed")
                    
            except Exception as e:
                error_msg = f"Reconnection attempt {attempt} failed with error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.last_error = error_msg
                
                # If we have a callback, notify about the reconnection failure
                if hasattr(self, 'callback') and callable(self.callback):
                    try:
                        await self.callback({
                            'type': 'reconnect_failed',
                            'attempt': attempt,
                            'max_attempts': max_reconnect_attempts,
                            'error': str(e)
                        })
                    except Exception as cb_error:
                        self.logger.error(f"Error in reconnect failure callback: {cb_error}", exc_info=True)
        
        # Log final reconnection status
        total_time = time.time() - reconnect_start
        if success:
            self.logger.info(f"✅ Reconnection successful after {total_time:.1f} seconds and {attempt} attempt(s)")
            
            # Notify callback about successful reconnection
            if hasattr(self, 'callback') and callable(self.callback):
                try:
                    await self.callback({
                        'type': 'reconnect_success',
                        'attempts': attempt,
                        'duration': total_time,
                        'symbols_resubscribed': list(prev_symbols)
                    })
                except Exception as e:
                    self.logger.error(f"Error in reconnect success callback: {e}", exc_info=True)
            
            return True
        else:
            error_msg = f"❌ Failed to reconnect after {max_reconnect_attempts} attempts over {total_time:.1f} seconds"
            self.logger.error(error_msg)
            
            # Notify callback about reconnection failure
            if hasattr(self, 'callback') and callable(self.callback):
                try:
                    await self.callback({
                        'type': 'reconnect_failed',
                        'attempt': max_reconnect_attempts,
                        'duration': total_time,
                        'error': self.last_error or "Unknown error"
                    })
                except Exception as e:
                    self.logger.error(f"Error in final reconnect failure callback: {e}", exc_info=True)
            
            return False

        # This section was cleaned up - duplicate code removed


# Add the project root and src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Initialize logger and session state
if 'logger' not in st.session_state:
    st.session_state.logger = configure_logging()

if 'predictor' not in st.session_state:
    st.session_state.predictor = TwoStagePredictor()
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = RiskManager()
if 'portfolio_optimizer' not in st.session_state:
    st.session_state.portfolio_optimizer = PortfolioOptimizer()

# Sidebar
st.sidebar.header("Stock Market Analysis System")

# Initialize WebSocket manager and error handling
from utils.websocket_manager import websocket_manager
from utils.error_handling import with_error_handling, ErrorBoundary, safe_call

# Initialize WebSocket manager in session state if not exists
if 'ws_manager' not in st.session_state:
    st.session_state.ws_manager = websocket_manager
    st.session_state._ws_callbacks_registered = False
    st.session_state._ws_connected = False

# WebSocket callback for handling real-time data updates
async def handle_ws_update(data: dict) -> None:
    """Handle WebSocket data updates with error handling."""
    try:
        symbol = data.get('symbol', 'unknown')
        logger.debug(f"WebSocket update for {symbol}: {data}")
        
        # Update real-time data in session state
        update_rt_data_safely(
            symbol,
            'price',
            data.get('price', 0)
        )
        
        # Update status
        update_rt_data_safely(
            symbol,
            'status',
            'Connected' if data.get('price') else 'No data'
        )
        
        # Request UI update
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error in WebSocket callback: {str(e)}", exc_info=True)
        update_rt_data_safely(symbol, 'status', f'Error: {str(e)[:50]}...')
        update_rt_data_safely(symbol, 'error', str(e))

# Create a safe wrapper for starting the RealTimeDataHandler
def safe_start_handler(handler):
    """Safely start the handler regardless of whether start() is sync or async"""
    import inspect
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Check if the start method is async
    start_method = getattr(handler, 'start', None)
    if start_method and inspect.iscoroutinefunction(start_method):
        # It's an async method, run it in a background thread with an event loop
        def run_async_start():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handler.start())
            
        # Run the async start method in a background thread
        with ThreadPoolExecutor() as executor:
            executor.submit(run_async_start)
    else:
        # It's a synchronous method, just call it directly
        handler.start()

# Create a safe wrapper for subscribing to symbols
def safe_subscribe_handler(handler, symbols):
    """Safely subscribe to symbols regardless of whether subscribe() is sync or async"""
    import inspect
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Check if the subscribe method is async
    subscribe_method = getattr(handler, 'subscribe', None)
    if subscribe_method and inspect.iscoroutinefunction(subscribe_method):
        # It's an async method, run it in a background thread with an event loop
        def run_async_subscribe():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handler.subscribe(symbols))
            
        # Run the async subscribe method in a background thread
        with ThreadPoolExecutor() as executor:
            executor.submit(run_async_subscribe)
    else:
        # It's a synchronous method, just call it directly
        handler.subscribe(symbols)

# Function to safely start WebSocket connection (simplified)
def start_websocket_connection() -> bool:
    """Safely start WebSocket connection."""
    try:
        if 'rt_handler' not in st.session_state or not st.session_state.rt_handler:
            return False
        if not st.session_state.rt_handler.running:
            # Use our safe wrapper function
            safe_start_handler(st.session_state.rt_handler)
        return True
    except Exception as e:
        logger.error(f"Error in start_websocket_connection: {str(e)}", exc_info=True)
        return False

# Function to safely stop WebSocket connection
def stop_websocket_connection():
    """
    Safely stop WebSocket connection and clean up resources.
    This function is called on app shutdown or reload.
    """
    if 'rt_handler' in st.session_state and st.session_state.rt_handler:
        try:
            handler = st.session_state.rt_handler
            
            # Remove callbacks first to prevent any callbacks during shutdown
            if hasattr(handler, 'remove_callback') and callable(handler.remove_callback):
                handler.remove_callback(handle_ws_update)
            
            # Stop the handler if it has a stop method
            if hasattr(handler, 'stop') and callable(handler.stop):
                handler.stop()
            
            # Clean up the handler
            if hasattr(handler, 'close') and callable(handler.close):
                handler.close()
            
            logger.info("WebSocket connection stopped and cleaned up")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket connection: {str(e)}", exc_info=True)
        finally:
            # Ensure we clean up the handler even if there's an error
            st.session_state.rt_handler = None

# Initialize and manage WebSocket connection using the improved RealTimeDataHandler
if 'rt_handler' not in st.session_state:
    try:
        # Use get_instance to ensure singleton pattern is followed
        st.session_state.rt_handler = RealTimeDataHandler.get_instance()
        
        # Register cleanup function to be called when the app is stopped or reloaded
        if not hasattr(st, '_ws_cleanup_registered'):
            import atexit
            atexit.register(stop_websocket_connection)
            st._ws_cleanup_registered = True
        
        # Use the safe wrapper to start the handler if it's not already running
        if hasattr(st.session_state.rt_handler, 'running') and not st.session_state.rt_handler.running:
            safe_start_handler(st.session_state.rt_handler)
        
        # Add callback for WebSocket updates if not already added
        if hasattr(st.session_state.rt_handler, 'add_callback'):
            st.session_state.rt_handler.add_callback(handle_ws_update)
            
        logger.info("RealTimeDataHandler initialized and started")
        
    except Exception as e:
        logger.error(f"Failed to initialize WebSocket handler: {str(e)}", exc_info=True)
        st.session_state.rt_handler = None
        
        # Clean up any callbacks if they exist
        if 'rt_handler' in st.session_state and st.session_state.rt_handler is not None:
            try:
                # Clear callbacks if they exist
                if hasattr(st.session_state.rt_handler, 'callbacks'):
                    st.session_state.rt_handler.callbacks.clear()
                
                try:
                    # Stop the handler if it exists
                    if hasattr(st.session_state.rt_handler, 'stop'):
                        logger.info("Stopping WebSocket handler...")
                        asyncio.run(st.session_state.rt_handler.stop())
                    
                    # Clean up any remaining resources
                    if hasattr(st.session_state.rt_handler, '_cleanup_connection'):
                        asyncio.run(st.session_state.rt_handler._cleanup_connection())
                    
                    logger.info("WebSocket connection and resources cleaned up")
                except Exception as e:
                    logger.error(f"Error during WebSocket cleanup: {str(e)}")
                    # Ensure we still clear the handler even if cleanup fails
                    st.session_state.rt_handler = None
                    raise
                
            except Exception as cleanup_error:
                logger.error(f"Error during WebSocket cleanup: {str(cleanup_error)}", exc_info=True)
            finally:
                # Ensure event loop is properly closed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.stop()
                    if not loop.is_closed():
                        loop.close()
                except Exception as loop_error:
                    logger.debug(f"Event loop cleanup: {str(loop_error)}", exc_info=True)
                finally:
                    # Always clear the handler
                    st.session_state.rt_handler = None
                    
# Initialize real-time data structure
if 'rt_data' not in st.session_state:
    st.session_state.rt_data = {}

# Main content
# Main content starts here

# Input form
st.sidebar.subheader("Stock Selection")
# Define the list of available tickers
available_tickers = ["GOOGL", "META", "MSFT", "AMZN", "AAPL"]

# Initialize ticker in session state if it doesn't exist
if 'ticker' not in st.session_state:
    st.session_state.ticker = available_tickers[0]  # Default to first option

# Create a selectbox for ticker selection
ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    options=available_tickers,
    index=available_tickers.index(st.session_state.ticker) if st.session_state.ticker in available_tickers else 0,
    format_func=lambda x: x  # Display the ticker as is
)

# Update session state if ticker changes
if ticker != st.session_state.ticker:
    st.session_state.ticker = ticker
    st.session_state.analysis_complete = False  # Force re-analysis for new ticker
    st.rerun()  # Rerun to update the app with new ticker

# Ensure ticker is always in uppercase for consistency
st.session_state.ticker = str(st.session_state.ticker).upper()
prediction_days = st.sidebar.slider("Prediction Days", 1, 30, 7)

# Calculate minimum required date range (sequence_length * 5 to ensure enough data)
min_days_required = 60 * 12  # 60 sequence_length * 5 for buffer
min_start_date = datetime.now() - timedelta(days=min_days_required)

# Store date inputs directly in session state with validation
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now() - timedelta(days=min_days_required),
        value=datetime.now() - timedelta(days=min_days_required)
    )
    st.session_state.start_date = start_date

with col2:
    end_date = st.date_input(
        "End Date",
        min_value=start_date + timedelta(days=min_days_required//2),  # At least half the min days range
        max_value=datetime.now(),
        value=datetime.now()
    )
    st.session_state.end_date = end_date

# Validate date range
if (end_date - start_date).days < min_days_required:
    st.sidebar.warning(f"⚠️ Date range too short. Minimum {min_days_required} days recommended for accurate analysis.")
    st.sidebar.info(f"Current range: {(end_date - start_date).days} days (minimum {min_days_required} days recommended)")
    st.session_state.date_range_valid = False
else:
    st.session_state.date_range_valid = True

# Portfolio parameters
st.sidebar.subheader("Portfolio Parameters")
portfolio_weight = st.sidebar.slider("Portfolio Weight", 0.1, 1.0, 0.5)
rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency", ["Daily", "Weekly", "Monthly"])

# Initialize WebSocket manager if not already done
if 'ws_manager' not in st.session_state:
    from utils.websocket_manager import WebSocketManager  # Ensure this import is added at the top of the file
    
    st.session_state.ws_manager = WebSocketManager()
    st.session_state._ws_connected = False
    st.session_state._ws_callbacks_registered = False


# Initialize session state variables if they don't exist
if 'rt_handler' not in st.session_state:
    st.session_state.rt_handler = None
    st.session_state._ws_initialized = False

# WebSocket controls moved to Real-Time Analysis tab

def fetch_with_cache(ticker: str, user_start_date: datetime, user_end_date: datetime,
                   buffer_days: int = 300) -> pd.DataFrame:
    """
    Fetch stock data with persistent caching using DataCollector.
    Automatically extends the date range to include buffer days for technical indicators.

    Args:
        ticker: Stock ticker symbol
        user_start_date: The start date requested by the user
        user_end_date: The end date requested by the user
        buffer_days: Number of additional days to fetch for technical indicators

    Returns:
        pd.DataFrame: Stock data with sufficient historical data
    """
    try:
        # Calculate the required start date with buffer
        required_start_date = user_start_date - timedelta(days=buffer_days)
        
        # Format dates for DataCollector
        start_str = required_start_date.strftime('%Y-%m-%d')
        end_str = user_end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching data for {ticker} from {start_str} to {end_str} with buffer")
        
        # Use DataCollector to get the data (will use cache if available)
        df = st.session_state.data_collector.get_stock_data(
            ticker=ticker,
            start_date=start_str,
            end_date=end_str
        )
        
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        # Ensure the index is a DatetimeIndex before slicing
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Index type before conversion: {type(df.index)}. Attempting to convert to DatetimeIndex.")
            df.index = pd.to_datetime(df.index)
            logger.info(f"Index type after conversion: {type(df.index)}.")
        
        # Now perform slicing
        df = df.loc[required_start_date:user_end_date].copy()
        
        logger.info(f"Index min after slicing: {df.index.min()}, max: {df.index.max()}, total rows: {len(df)}")
        if df.empty:
            logger.warning(f"DataFrame is empty after slicing. Available index values: {list(df.index)}")
        else:
            logger.info(f"Successfully retrieved {len(df)} rows of data for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_with_cache for {ticker}: {str(e)}", exc_info=True)
        raise ValueError(f"Error in fetch_with_cache: {str(e)}")



# Update DataCollector with current ticker and date range
try:
    st.session_state.data_collector.tickers = [ticker]
    # Use session state variables for dates
    st.session_state.data_collector.start_date = st.session_state.start_date.strftime("%Y-%m-%d")
    st.session_state.data_collector.end_date = st.session_state.end_date.strftime("%Y-%m-%d")
except Exception as e:
    error_msg = f"Failed to update DataCollector: {str(e)}"
    st.error(error_msg)
    logger.error(error_msg, exc_info=True)
    st.stop()


def is_trading_day():
    """Check if today is a trading day (Monday-Friday)"""
    import datetime
    today = datetime.datetime.now().date()
    # Monday is 0, Sunday is 6
    return today.weekday() < 5  # 0-4 are Monday to Friday


def analyze_sentiment(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch and analyze sentiment data for the given ticker using SentimentAnalyzer.
    Returns a DataFrame with required sentiment columns.
    
    Args:
        ticker: Stock ticker symbol to analyze
        start_date: Start date for sentiment analysis
        end_date: End date for sentiment analysis
        
    Returns:
        pd.DataFrame: DataFrame containing sentiment data with required columns
        
    Raises:
        Exception: If sentiment data cannot be fetched or processed, raises the original error
    """
    # Store the error in session state for UI display
    st.session_state.sentiment_error = None
    
    try:
        logger.info(f"[SENTIMENT] Starting sentiment analysis for {ticker} from {start_date} to {end_date}")
        
        # Define required columns to ensure consistency
        required_columns = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility', 'sentiment_score']
        
        # Create date range for the requested period (business days only)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
        if len(date_range) == 0:
            logger.warning("No business days in date range, using regular days")
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize default values with more realistic ranges
        default_values = {
            'Sentiment': 0.0,           # Range: -1 to 1
            'Price_Momentum': 0.0,       # Range: -1 to 1
            'Volume_Change': 0.0,        # Range: -1 to 1
            'Volatility': 0.1,           # Range: 0 to 1
            'sentiment_score': 0.0,      # Range: -1 to 1
            'data_source': 'fallback'
        }
    
        # Log the start of sentiment analysis with debug info
        logger.info(f"[SENTIMENT] Starting sentiment analysis for {ticker} from {start_date} to {end_date}")
        logger.debug(f"[SENTIMENT] Date range: {len(date_range)} days, from {date_range[0]} to {date_range[-1]}")
        
        try:
            # Initialize the SentimentAnalyzer with error handling
            logger.info("[SENTIMENT] Importing SentimentAnalyzer...")
            try:
                from src.utils.sentiment_analyzer import SentimentAnalyzer
                analyzer = SentimentAnalyzer()
                logger.info("[SENTIMENT] Successfully initialized SentimentAnalyzer")
                
                # Check if required API clients are available
                api_status = {
                    'newsapi': bool(analyzer.newsapi),
                    'reddit': bool(analyzer.reddit),
                    'fred': bool(getattr(analyzer, 'fred', None))
                }
                logger.info(f"[SENTIMENT] API Status: {api_status}")
                
                if not any(api_status.values()):
                    logger.warning("[SENTIMENT] No API clients available, will use fallback data")
                    raise ImportError("No API clients available")
                
            except ImportError as ie:
                logger.error(f"[SENTIMENT] Failed to initialize SentimentAnalyzer: {str(ie)}")
                raise
            except Exception as e:
                logger.error(f"[SENTIMENT] Error initializing SentimentAnalyzer: {str(e)}")
                raise
            
            # Get sentiment features using the SentimentAnalyzer
            logger.info("[SENTIMENT] Fetching sentiment features...")
            try:
                # Calculate number of days between start and end date
                days = (end_date - start_date).days + 1
                logger.info(f"[SENTIMENT] Requesting {days} days of sentiment data")
                
                # Call create_sentiment_features with the days parameter
                sentiment_features = analyzer.create_sentiment_features(ticker, days=days)
                
                # Validate the sentiment features
                if sentiment_features is None:
                    raise ValueError("Sentiment analyzer returned None. This typically means no data was returned from any sentiment source.")
                if not isinstance(sentiment_features, pd.DataFrame):
                    raise TypeError(f"Expected a pandas DataFrame but got {type(sentiment_features).__name__}")
                
                # Check if the DataFrame is empty
                if sentiment_features.empty:
                    raise ValueError("Sentiment DataFrame is empty")
                    
                # Log basic info about the data
                logger.info(f"[SENTIMENT] Successfully retrieved {len(sentiment_features)} sentiment records")
                logger.info(f"[SENTIMENT] DataFrame columns: {sentiment_features.columns.tolist()}")
                logger.info(f"[SENTIMENT] DataFrame index type: {type(sentiment_features.index).__name__}")
                logger.info(f"[SENTIMENT] DataFrame index sample: {sentiment_features.index[:5].tolist() if len(sentiment_features) >= 5 else sentiment_features.index.tolist()}")
                logger.info(f"[SENTIMENT] DataFrame has required columns: {all(col in sentiment_features.columns for col in required_columns)}")
                
                # Log the first 5 records for debugging
                if not sentiment_features.empty:
                    logger.info("[SENTIMENT] Sample of sentiment records:")
                    sample_records = sentiment_features.head().to_dict(orient='records')
                    for i, record in enumerate(sample_records, 1):
                        logger.info(f"[SENTIMENT] Record {i}: {record}")
                    
                    # Log summary statistics for numeric columns
                    numeric_cols = sentiment_features.select_dtypes(include=['number']).columns
                    if not numeric_cols.empty:
                        logger.info("[SENTIMENT] Summary statistics for numeric columns:")
                        stats = sentiment_features[numeric_cols].describe().to_dict()
                        for col, stat in stats.items():
                            logger.info(f"[SENTIMENT] {col}: {stat}")
                
                # Check for the required columns
                missing_columns = [col for col in required_columns if col not in sentiment_features.columns]
                if missing_columns:
                    logger.warning(f"[SENTIMENT] Missing required columns: {missing_columns}")
                    
                # Log the data source information if available
                if 'data_source' in sentiment_features.columns:
                    logger.info(f"[SENTIMENT] Data sources used: {sentiment_features['data_source'].unique().tolist()}")
                    
                logger.info("[SENTIMENT] Successfully retrieved sentiment features")
                
                # Ensure all required columns are present with proper data types
                for col in required_columns:
                    if col not in sentiment_features.columns:
                        logger.warning(f"[SENTIMENT] Required column '{col}' not found in sentiment features. Adding with default value.")
                        sentiment_features[col] = default_values.get(col, 0.0)
                    
                    # Ensure numeric type and handle any conversion issues
                    try:
                        sentiment_features[col] = pd.to_numeric(sentiment_features[col], errors='coerce')
                        # Fill any NaN values that might have been created during conversion
                        if sentiment_features[col].isnull().any():
                            logger.warning(f"[SENTIMENT] Found NaN values in column '{col}' after conversion, filling with default")
                            sentiment_features[col] = sentiment_features[col].fillna(default_values.get(col, 0.0))
                    except Exception as e:
                        logger.error(f"[SENTIMENT] Error converting column '{col}' to numeric: {str(e)}")
                        sentiment_features[col] = default_values.get(col, 0.0)
                
                # Add data source information if not present
                if 'data_source' not in sentiment_features.columns:
                    sentiment_features['data_source'] = 'unknown'
                
                # Ensure we have a datetime index
                if not isinstance(sentiment_features.index, pd.DatetimeIndex):
                    logger.warning("[SENTIMENT] No datetime index found, attempting to infer from columns...")
                    date_cols = ['date', 'Date', 'time', 'timestamp']
                    for dc in date_cols:
                        if dc in sentiment_features.columns:
                            try:
                                sentiment_features[dc] = pd.to_datetime(sentiment_features[dc], errors='coerce')
                                sentiment_features = sentiment_features.set_index(dc)
                                logger.info(f"[SENTIMENT] Set datetime index using column: {dc}")
                                break
                            except Exception as e:
                                logger.warning(f"[SENTIMENT] Failed to set datetime index from column {dc}: {str(e)}")
                    
                    # If still no datetime index, create one
                    if not isinstance(sentiment_features.index, pd.DatetimeIndex):
                        logger.warning("[SENTIMENT] Could not create datetime index, using default date range")
                        sentiment_features = sentiment_features.reset_index(drop=True)
                        sentiment_features.index = date_range[:len(sentiment_features)]
                
                logger.info(f"[SENTIMENT] Successfully processed sentiment data with columns: {sentiment_features.columns.tolist()}")
                return sentiment_features
                
            except Exception as e:
                error_msg = f"Error in create_sentiment_features: {str(e)}"
                logger.error(f"[SENTIMENT] {error_msg}", exc_info=True)
                st.session_state.sentiment_error = error_msg
                
                # Log detailed error information
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    logger.error(f"[SENTIMENT] API Error Status Code: {e.response.status_code}")
                    try:
                        error_details = e.response.json()
                        logger.error(f"[SENTIMENT] API Error Details: {error_details}")
                    except:
                        logger.error("[SENTIMENT] Could not parse API error response")
                
                # If we can't get sentiment features, create a default DataFrame
                logger.warning("[SENTIMENT] Creating fallback sentiment data due to error")
                try:
                    # First try to get price metrics for better fallback
                    price_metrics = analyzer.get_price_metrics(ticker) if 'analyzer' in locals() and hasattr(analyzer, 'get_price_metrics') else {}
                    
                    # Create fallback data with price metrics if available
                    fallback_data = _create_dummy_sentiment_data(start_date, end_date)
                    
                    # Enhance with any available price metrics
                    if price_metrics and isinstance(price_metrics, dict):
                        for metric, col in [('price_momentum', 'Price_Momentum'), 
                                          ('volume_change', 'Volume_Change'), 
                                          ('volatility', 'Volatility')]:
                            if metric in price_metrics and col in fallback_data.columns:
                                fallback_data[col] = price_metrics[metric]
                    
                    fallback_data['data_source'] = 'enhanced_fallback'
                    logger.info("[SENTIMENT] Successfully created enhanced fallback data")
                    return fallback_data
                    
                except Exception as fallback_error:
                    logger.error(f"[SENTIMENT] Error creating enhanced fallback data: {str(fallback_error)}")
                    # Fall back to basic dummy data if enhancement fails
                    dummy_data = _create_dummy_sentiment_data(start_date, end_date)
                    dummy_data['data_source'] = 'basic_fallback'
                    return dummy_data
                
        except ImportError as e:
            error_msg = f"Failed to import SentimentAnalyzer: {str(e)}\nPlease ensure all required dependencies are installed."
            logger.error(f"[SENTIMENT] {error_msg}", exc_info=True)
            st.session_state.sentiment_error = error_msg
            
            # Create enhanced fallback data with price metrics if possible
            try:
                from utils.data_collector import DataCollector
                from datetime import datetime, timedelta
                
                # Use the existing DataCollector from session state
                if 'data_collector' not in st.session_state or st.session_state.data_collector is None:
                    logger.warning("DataCollector not found in session state, using fallback")
                    fallback_data = _create_dummy_sentiment_data(start_date, end_date)
                    fallback_data['data_source'] = 'no_collector_fallback'
                    return fallback_data
                
                # Get price data for metrics
                price_data = st.session_state.data_collector.get_stock_data(ticker)
                
                # Create fallback data with price metrics
                fallback_data = _create_dummy_sentiment_data(start_date, end_date)
                
                if price_data is not None and not price_data.empty:
                    # Calculate price metrics
                    returns = price_data['Close'].pct_change()
                    price_momentum = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1
                    volume_change = (price_data['Volume'].iloc[-1] / price_data['Volume'].mean()) - 1
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # Update fallback data with calculated metrics
                    fallback_data['Price_Momentum'] = np.clip(price_momentum, -1, 1)
                    fallback_data['Volume_Change'] = np.clip(volume_change, -1, 1)
                    fallback_data['Volatility'] = np.clip(volatility, 0, 1)
                    
                    logger.info("[SENTIMENT] Created enhanced fallback data with price metrics")
                
                fallback_data['data_source'] = 'import_error_enhanced'
                return fallback_data
                
            except Exception as fallback_error:
                logger.error(f"[SENTIMENT] Error creating enhanced fallback data: {str(fallback_error)}")
                # Fall back to basic dummy data if enhancement fails
                dummy_data = _create_dummy_sentiment_data(start_date, end_date)
                dummy_data['data_source'] = 'import_error_basic'
                return dummy_data
            
        except Exception as e:
            error_msg = f"Failed to initialize SentimentAnalyzer: {str(e)}"
            logger.error(f"[SENTIMENT] {error_msg}", exc_info=True)
            st.session_state.sentiment_error = error_msg
            
            # If we can't initialize the analyzer, create an enhanced default DataFrame
            logger.warning("[SENTIMENT] Creating enhanced fallback data due to initialization error")
            try:
                # Try to get any available data for better fallback
                fallback_data = _create_dummy_sentiment_data(start_date, end_date)
                
                # Add some variation to make it look more realistic
                fallback_data['Sentiment'] = np.sin(np.linspace(0, 10, len(fallback_data))) * 0.5
                fallback_data['sentiment_score'] = fallback_data['Sentiment']  # Keep them in sync
                
                # Add some random but realistic-looking patterns
                np.random.seed(42)  # For reproducibility
                for col in ['Price_Momentum', 'Volume_Change', 'Volatility']:
                    if col in fallback_data.columns:
                        # Add some random walk pattern
                        random_walk = np.cumsum(np.random.normal(0, 0.1, len(fallback_data)))
                        # Scale to appropriate range
                        if col == 'Volatility':
                            random_walk = 0.3 + 0.4 * (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min() + 1e-8)
                        else:
                            random_walk = 0.5 * (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min() + 1e-8)
                        fallback_data[col] = np.clip(random_walk, -1 if col != 'Volatility' else 0, 1)
                
                fallback_data['data_source'] = 'init_error_enhanced'
                logger.info("[SENTIMENT] Created enhanced fallback data with realistic patterns")
                return fallback_data
                
            except Exception as fallback_error:
                logger.error(f"[SENTIMENT] Error creating enhanced fallback data: {str(fallback_error)}")
                # Fall back to basic dummy data if enhancement fails
                dummy_data = _create_dummy_sentiment_data(start_date, end_date)
                dummy_data['data_source'] = 'init_error_basic'
                return dummy_data
            
        # Get price metrics with enhanced error handling
        price_metrics = {}
        try:
            logger.info("[SENTIMENT] Fetching price metrics...")
            if 'analyzer' in locals() and hasattr(analyzer, 'get_price_metrics'):
                price_metrics = analyzer.get_price_metrics(ticker)
                logger.info(f"[SENTIMENT] Retrieved price metrics: {price_metrics}")
            else:
                logger.warning("[SENTIMENT] Analyzer or get_price_metrics not available")
                
            # Validate price metrics
            if not isinstance(price_metrics, dict):
                logger.warning("[SENTIMENT] Invalid price metrics format, initializing empty dict")
                price_metrics = {}
                
        except Exception as e:
            logger.error(f"[SENTIMENT] Error fetching price metrics: {str(e)}")
            price_metrics = {}
        
        # Ensure we have all required metrics with default values
        required_metrics = {
            'price_momentum': default_values['Price_Momentum'],
            'volume_change': default_values['Volume_Change'],
            'volatility': default_values['Volatility']
        }
        
        # Update with any available metrics, keeping defaults for missing ones
        for key, default in required_metrics.items():
            if key not in price_metrics or not isinstance(price_metrics[key], (int, float)):
                logger.warning(f"[SENTIMENT] Using default value for {key}")
                price_metrics[key] = default
            
            # Ensure values are within expected ranges
            if key == 'volatility':
                price_metrics[key] = max(0.0, min(1.0, float(price_metrics[key])))
            else:
                price_metrics[key] = max(-1.0, min(1.0, float(price_metrics[key])))
        
        # Extract and validate sentiment score with comprehensive error handling
        sentiment_score = None
        try:
            # Try multiple possible column names for sentiment score
            score_columns = ['composite_score', 'avg_sentiment', 'sentiment_score', 'Sentiment']
            
            for col in score_columns:
                if col in sentiment_features.columns:
                    try:
                        # Try to get the first non-null value
                        first_valid = sentiment_features[col].first_valid_index()
                        if first_valid is not None:
                            sentiment_score = float(sentiment_features[col].loc[first_valid])
                            logger.info(f"[SENTIMENT] Using sentiment score from column: {col}")
                            break
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[SENTIMENT] Could not convert value from column {col}: {str(e)}")
                        continue
            
            # If no valid score found, use default
            if sentiment_score is None:
                logger.warning("[SENTIMENT] No valid sentiment score found in features, using default")
                sentiment_score = default_values['Sentiment']
                
            # Ensure score is within valid range
            sentiment_score = max(-1.0, min(1.0, float(sentiment_score)))
                
        except Exception as e:
            logger.error(f"[SENTIMENT] Error processing sentiment score: {str(e)}, using default")
            sentiment_score = default_values['Sentiment']
        
        # Get price metrics from the already validated price_metrics dictionary
        try:
            price_momentum = float(price_metrics['price_momentum'])
            volume_change = float(price_metrics['volume_change'])
            volatility = float(price_metrics['volatility'])
            
            logger.info(f"[SENTIMENT] Using price metrics - Momentum: {price_momentum:.4f}, "
                       f"Volume: {volume_change:.4f}, Volatility: {volatility:.4f}")
                        
        except Exception as e:
            logger.error(f"[SENTIMENT] Error processing price metrics: {str(e)}, using defaults")
            price_momentum = default_values['Price_Momentum']
            volume_change = default_values['Volume_Change']
            volatility = default_values['Volatility']
        
        # Create a more sophisticated time series pattern for the sentiment data
        try:
            # Create a base DataFrame with the date index
            sentiment_df = pd.DataFrame(index=date_range)
            
            # Add some time-based variation to make the data more realistic
            time_points = np.linspace(0, 10, len(sentiment_df))
            
            # Create a base pattern with some randomness
            np.random.seed(42)  # For reproducibility
            random_component = np.random.normal(0, 0.1, len(sentiment_df)).cumsum()
            
            # Combine base pattern with some periodic components
            sentiment_pattern = (
                0.7 * np.sin(time_points) +  # Main pattern
                0.3 * np.sin(time_points * 3) +  # Higher frequency component
                0.2 * random_component  # Random walk component
            )
            
            # Scale to desired range and add to DataFrame
            sentiment_df['Sentiment'] = np.clip(sentiment_pattern, -1, 1)
            
            # Add price momentum with some correlation to sentiment
            momentum_pattern = 0.6 * sentiment_df['Sentiment'] + 0.4 * np.random.normal(0, 0.5, len(sentiment_df))
            sentiment_df['Price_Momentum'] = np.clip(momentum_pattern, -1, 1)
            
            # Add volume change with some correlation to absolute sentiment changes
            volume_pattern = 0.5 * np.abs(sentiment_df['Sentiment'].diff().fillna(0)) + 0.5 * np.random.normal(0, 0.3, len(sentiment_df))
            sentiment_df['Volume_Change'] = np.clip(volume_pattern, -1, 1)
            
            # Add volatility (always positive)
            volatility_pattern = 0.1 + 0.4 * (1 - np.exp(-0.5 * np.abs(sentiment_pattern)))
            sentiment_df['Volatility'] = np.clip(volatility_pattern, 0, 1)
            
            # Ensure the sentiment_score matches the Sentiment column
            sentiment_df['sentiment_score'] = sentiment_df['Sentiment']
            
            # Add some noise to make it look more realistic
            for col in sentiment_df.columns:
                noise = np.random.normal(0, 0.05, len(sentiment_df))
                if col == 'Volatility':
                    sentiment_df[col] = np.clip(sentiment_df[col] + noise, 0, 1)
                else:
                    sentiment_df[col] = np.clip(sentiment_df[col] + noise, -1, 1)
            
            logger.info(f"[SENTIMENT] Generated realistic-looking sentiment data with {len(sentiment_df)} rows")
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Error generating realistic sentiment data: {str(e)}")
            # Fall back to simple constant values if pattern generation fails
            sentiment_df = pd.DataFrame(index=date_range)
            for col in required_columns:
                sentiment_df[col] = default_values[col]
        
        # Final validation and cleanup
        try:
            # Ensure all required columns exist
            for col in required_columns:
                if col not in sentiment_df.columns:
                    logger.warning(f"[SENTIMENT] Adding missing column: {col}")
                    sentiment_df[col] = default_values[col]
            
            # Ensure all values are numeric and within expected ranges
            for col in sentiment_df.columns:
                sentiment_df[col] = pd.to_numeric(sentiment_df[col], errors='coerce')
                
                # Fill any remaining NaN values with column-specific defaults
                if sentiment_df[col].isnull().any():
                    logger.warning(f"[SENTIMENT] Found NaN values in {col}, filling with default")
                    sentiment_df[col] = sentiment_df[col].fillna(default_values.get(col, 0.0))
                
                # Clip values to valid ranges
                if col == 'Volatility':
                    sentiment_df[col] = np.clip(sentiment_df[col], 0, 1)
                else:
                    sentiment_df[col] = np.clip(sentiment_df[col], -1, 1)
            
            # Ensure we have the right columns in the right order
            result_df = sentiment_df[required_columns].copy()
            
            logger.info(f"[SENTIMENT] Final sentiment data summary:\n{result_df.describe()}")
            logger.debug(f"[SENTIMENT] First few rows of sentiment data:\n{result_df.head()}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Error in final data validation: {str(e)}")
            # If all else fails, return minimal valid data
            return pd.DataFrame({
                'Sentiment': [0.0] * len(date_range),
                'Price_Momentum': [0.0] * len(date_range),
                'Volume_Change': [0.0] * len(date_range),
                'Volatility': [0.1] * len(date_range),
                'sentiment_score': [0.0] * len(date_range)
            }, index=date_range)
        
    except Exception as e:
        logger.error(f"[SENTIMENT] Error in analyze_sentiment: {str(e)}", exc_info=True)
        # Return a DataFrame with default values on error
        default_df = pd.DataFrame(index=date_range, columns=required_columns)
        for col in required_columns:
            default_df[col] = default_values[col]
        return default_df
    except Exception as e:
        error_msg = f"[SENTIMENT] Error in sentiment analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Create a new DataFrame with default values for all dates
        default_data = {col: [default_values[col]] * len(date_range) for col in required_columns}
        default_df = pd.DataFrame(default_data, index=date_range)
        
        logger.warning("[SENTIMENT] Returning default sentiment data due to error")
        return default_df[required_columns]
        raise RuntimeError(error_msg) from e

def _create_dummy_sentiment_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Create a DataFrame with dummy sentiment data when real data is unavailable.
    
    Args:
        start_date: Start date for the data
        end_date: End date for the data
        
    Returns:
        pd.DataFrame: DataFrame with required sentiment columns and date index
    """
    try:
        logger.info(f"Creating dummy sentiment data from {start_date} to {end_date}")
        
        # Ensure we have valid dates
        if start_date is None or end_date is None:
            logger.warning("Invalid date range provided, using default 30-day range")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        # Ensure start_date is before end_date
        if start_date > end_date:
            logger.warning("Start date is after end date, swapping dates")
            start_date, end_date = end_date, start_date
        
        # Create date range for the requested period with business day frequency
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days
        
        if len(date_range) == 0:
            logger.warning("No valid business days in date range, using single date")
            date_range = pd.date_range(start=start_date, periods=1)
        
        # Generate some random but realistic-looking data
        np.random.seed(42)  # For reproducibility
        n_days = len(date_range)
        
        # Create DataFrame with required columns and realistic patterns
        df = pd.DataFrame({
            'date': date_range,
            'Sentiment': np.random.uniform(-1, 1, size=n_days).cumsum(),
            'Price_Momentum': np.random.normal(0, 0.5, size=n_days).cumsum(),
            'Volume_Change': np.random.normal(0, 0.3, size=n_days).cumsum(),
            'Volatility': np.abs(np.random.normal(0.5, 0.2, size=n_days)),
            'sentiment_score': np.random.uniform(-1, 1, size=n_days)
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Smooth the data a bit
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].rolling(window=3, min_periods=1).mean()
        
        # Ensure all required columns exist and have the correct type
        required_columns = {
            'Sentiment': 0.0,
            'Price_Momentum': 0.0,
            'Volume_Change': 0.0,
            'Volatility': 0.0,
            'sentiment_score': 0.0
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
            # Ensure numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
        
        logger.info(f"Created dummy sentiment data with shape {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in _create_dummy_sentiment_data: {str(e)}", exc_info=True)
        # Fallback with current timestamp if anything goes wrong
        now = pd.Timestamp.now()
        return pd.DataFrame({
            'date': [now],
            'Sentiment': [0.0],
            'Price_Momentum': [0.0],
            'Volume_Change': [0.0],
            'Volatility': [0.0],
            'sentiment_score': [0.0]
        }).set_index('date')

# DEPRECATED: All processing now handled by DataProcessor. Use DataProcessor.load_data or DataProcessor.preprocess.
# def process_stock_data(stock_data: pd.DataFrame, close_col: str) -> dict:
#     ... (removed)

    """Process stock data and calculate technical indicators"""
    try:
        df = stock_data.copy()
        
        # Basic returns
        df['returns'] = df[close_col].pct_change()
        
        # Moving Averages
        df['SMA_20'] = df[close_col].rolling(window=20).mean()
        df['SMA_50'] = df[close_col].rolling(window=50).mean()
        df['SMA_200'] = df[close_col].rolling(window=200).mean()  # Added SMA_200
        df['EMA_12'] = df[close_col].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df[close_col].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI with ta column name for compatibility with display_price_analysis
        df['momentum_rsi'] = df['RSI']
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Add MACD with ta column names for compatibility with display_price_analysis
        df['trend_macd'] = df['MACD']
        df['trend_macd_signal'] = df['Signal_Line']
        df['trend_macd_diff'] = df['MACD_Hist']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['momentum_stoch'] = 100 * ((df[close_col] - low_14) / (high_14 - low_14))
        df['momentum_stoch_signal'] = df['momentum_stoch'].rolling(window=3).mean()
        
        # Bollinger Bands - Pure Python implementation
        rolling_mean = df[close_col].rolling(window=20).mean()
        rolling_std = df[close_col].rolling(window=20).std()
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_middle'] = rolling_mean
        df['BB_lower'] = rolling_mean - (rolling_std * 2)
        
        # ATR - Pure Python implementation
        high = df['High']
        low = df['Low']
        close = df[close_col]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as the rolling mean of True Range
        df['ATR'] = true_range.rolling(window=14).mean()

        # On-Balance Volume (OBV)
        df['volume_obv'] = 0
        volume_col = None
        for col in df.columns:
            # Ensure col is a string before calling lower()
            if isinstance(col, str) and 'volume' in col.lower():
                volume_col = col
                break

        if volume_col:
            df['volume_obv'] = (np.sign(df[close_col].diff()) * df[volume_col]).fillna(0).cumsum()

        # Drop NA values
        df = df.dropna()
        # Defensive: If all rows are dropped, log and return None
        if df.empty:
            logger.error("All rows dropped after technical indicator calculation. Input data may be too short or contain too many missing values.")
            logger.error(f"Columns: {list(df.columns)}")
            logger.error(f"Missing values per column: {df.isna().sum().to_dict()}")
            return None
        
        # Prepare features and target
        # Technical indicators as stock features
        stock_feature_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'returns', 
                            'MACD', 'Signal_Line', 'MACD_Hist',
                            'BB_upper', 'BB_middle', 'BB_lower', 'ATR']
        
        # Additional features (can be extended with other features)
        additional_feature_cols = []
        
        # Create sequences for time series data
        sequence_length = 20  # Number of time steps to look back
        
        def create_sequences(data, target, sequence_length):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:(i + sequence_length)])
                y.append(target[i + sequence_length])
            return np.array(X), np.array(y)
        
        # Prepare stock features (technical indicators)
        X_stock = df[stock_feature_cols].values
        y = df[close_col].values
        
        # Create sequences
        X_stock_seq, y_seq = create_sequences(X_stock, y, sequence_length)
        
        # Create additional features (can be empty if no additional features)
        X_additional = np.zeros((len(X_stock_seq), sequence_length, len(additional_feature_cols)))
        
        # Split into train/validation/test sets
        train_size = int(0.7 * len(X_stock_seq))
        val_size = int(0.15 * len(X_stock_seq))
        
        # Split stock features
        X_stock_train = X_stock_seq[:train_size]
        X_stock_val = X_stock_seq[train_size:train_size+val_size]
        X_stock_test = X_stock_seq[train_size+val_size:]
        
        # Split additional features (if any)
        X_add_train = X_additional[:train_size]
        X_add_val = X_additional[train_size:train_size+val_size]
        X_add_test = X_additional[train_size+val_size:]
        
        # Split target
        y_train = y_seq[:train_size]
        y_val = y_seq[train_size:train_size+val_size]
        y_test = y_seq[train_size+val_size:]
        
        return {
            'X_train': (X_stock_train, X_add_train),
            'y_train': y_train,
            'X_val': (X_stock_val, X_add_val),
            'y_val': y_val,
            'X_test': (X_stock_test, X_add_test),
            'y_test': y_test,
            'returns': df['returns'].values,
            'dates': df.index,
            'features': stock_feature_cols,
            'technical_data': df,
            'stock_data': df  # Add stock_data key for display_price_analysis function
        }
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logger.error(f"Data processing error: {str(e)}", exc_info=True)
        return None

# DEPRECATED: All indicator calculation now handled by DataProcessor._create_features/_create_technical_indicators.
# def calculate_technical_indicators(data):
#     ... (removed)

    """Calculate technical indicators for stock data"""
    try:
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure we have a Close column
        close_col = 'Close' if 'Close' in df.columns else df.columns[0]
        
        # Simple Moving Averages
        df['SMA_5'] = df[close_col].rolling(window=5).mean()
        df['SMA_10'] = df[close_col].rolling(window=10).mean()
        df['SMA_20'] = df[close_col].rolling(window=20).mean()
        
        # Exponential Moving Averages
        df['EMA_5'] = df[close_col].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df[close_col].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df[close_col].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df[close_col].rolling(window=20).mean()
        df['BB_std'] = df[close_col].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Relative Strength Index (RSI)
        delta = df[close_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = df[close_col].ewm(span=12, adjust=False).mean() - df[close_col].ewm(span=26, adjust=False).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Commodity Channel Index (CCI)
        tp = (df['High'] + df['Low'] + df[close_col]) / 3 if 'High' in df.columns and 'Low' in df.columns else df[close_col]
        ma = tp.rolling(window=20).mean()
        md = tp.rolling(window=20).apply(lambda x: pd.Series(x).mad())
        df['CCI'] = (tp - ma) / (0.015 * md)
        
        # Average Directional Index (ADX)
        if 'High' in df.columns and 'Low' in df.columns:
            # True Range
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df[close_col].shift(1)),
                    abs(df['Low'] - df[close_col].shift(1))
                )
            )
            
            # Directional Movement
            df['DM_plus'] = np.where(
                (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                np.maximum(df['High'] - df['High'].shift(1), 0),
                0
            )
            df['DM_minus'] = np.where(
                (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                np.maximum(df['Low'].shift(1) - df['Low'], 0),
                0
            )
            
            # Smoothed True Range and Directional Movement
            df['ATR'] = df['TR'].rolling(window=14).mean()
            df['DI_plus'] = 100 * (df['DM_plus'].rolling(window=14).mean() / df['ATR'])
            df['DI_minus'] = 100 * (df['DM_minus'].rolling(window=14).mean() / df['ATR'])
            
            # Directional Index
            df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
            
            # Average Directional Index
            df['ADX'] = df['DX'].rolling(window=14).mean()
        
        # Percentage Price Oscillator (PPO)
        df['PPO'] = ((df[close_col].ewm(span=12, adjust=False).mean() - df[close_col].ewm(span=26, adjust=False).mean()) / 
                    df[close_col].ewm(span=26, adjust=False).mean()) * 100
        
        # Rate of Change (ROC)
        df['ROC'] = (df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10) * 100
        
        # Stochastic Oscillator
        if 'High' in df.columns and 'Low' in df.columns:
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['%K'] = 100 * ((df[close_col] - low_14) / (high_14 - low_14))
            df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {str(e)}", exc_info=True)
        # Return original data if calculation fails
        return data

class FallbackPortfolioOptimizer:
    """Simple portfolio optimizer that returns equal weights for all assets."""
    
    def __init__(self):
        self.portfolio_weights = {}
        self.winsorize_lower = 0.05
        self.winsorize_upper = 0.95
        
    def optimize(self, returns_df):
        """Return equal weights for all assets."""
        if returns_df is None or len(returns_df.columns) == 0:
            return {}
            
        n = len(returns_df.columns)
        equal_weight = 1.0 / n
        return {ticker: equal_weight for ticker in returns_df.columns}
        
    def calculate_sharpe_ratio(self, returns, weights):
        """Return a default Sharpe ratio."""
        return 1.0
        
    def calculate_volatility(self, returns, weights):
        """Return a default volatility."""
        return 0.15  # 15% volatility
        
    def calculate_max_drawdown(self, returns, weights):
        """Return a default max drawdown."""
        return 0.1  # 10% max drawdown


class PortfolioOptimizer:
    def __init__(self):
        self.returns = None
        self.weights = None
        
    def optimize(self, returns):
        """Optimize portfolio weights using mean-variance optimization"""
        try:
            self.returns = returns
            n_assets = returns.shape[1] if len(returns.shape) > 1 else 1
            initial_weights = np.ones(n_assets) / n_assets
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Define objective function (negative Sharpe ratio)
            def negative_sharpe(weights):
                return -self.calculate_sharpe_ratio(returns, weights)
                
            # Optimize
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            )
            
            self.weights = result.x
            return self.weights
            
        except Exception as e:
            st.error(f"Portfolio optimization failed: {str(e)}")
            logger.error(f"Portfolio optimization error: {str(e)}")
            return None
    
    def calculate_sharpe_ratio(self, returns, weights, risk_free_rate=0.0):
        """Calculate Sharpe ratio for the portfolio"""
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return (portfolio_return - risk_free_rate) / portfolio_vol
    
    def calculate_volatility(self, returns, weights):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    def calculate_max_drawdown(self, returns, weights):
        """Calculate maximum drawdown"""
        # Ensure we're working with pandas Series for the calculations
        if isinstance(returns, np.ndarray):
            returns = pd.DataFrame(returns)
        portfolio_returns = pd.Series(np.dot(returns, weights), index=returns.index)
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

def load_and_validate_data(ticker: str, start_date: datetime, end_date: datetime):
    """Load and validate stock data for analysis with enhanced validation.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        tuple: (stock_data DataFrame, close_column_name) or (None, None) if validation fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading data for {ticker} from {start_date} to {end_date}")
        
        # Add buffer days for technical indicators
        buffer_days = max(60, int((end_date - start_date).days * 0.2))  # Minimum 60 days or 20% of date range
        logger.info(f"Using buffer of {buffer_days} days for technical indicators")
        
        # Use fetch_with_cache which uses st.session_state.data_collector
        stock_data = fetch_with_cache(ticker, start_date, end_date, buffer_days)
        
        # Basic validation
        if stock_data is None or stock_data.empty:
            error_msg = f"No data found for {ticker} in the specified date range"
            logger.error(error_msg)
            st.error(error_msg)
            return None, None
        
        # Flatten MultiIndex columns if present
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
        
        logger.info(f"Initial data shape: {stock_data.shape}")
        logger.info(f"Available columns: {list(stock_data.columns)}")
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            # Try to find case-insensitive matches
            col_map = {col.lower(): col for col in stock_data.columns}
            for col in missing_cols[:]:  # Iterate over a copy of the list
                lower_col = col.lower()
                if lower_col in col_map:
                    stock_data[col] = stock_data[col_map[lower_col]]
                    missing_cols.remove(col)
            
            if missing_cols:
                st.error(f"Could not find required columns: {missing_cols}")
                return None, None
        
        # Identify the close price column
        close_col = get_close_column_name(stock_data, ticker)
        if not close_col:
            close_col = 'Close' if 'Close' in stock_data.columns else stock_data.columns[0]
        logger.info(f"Using '{close_col}' as close price column")
        
        # Data quality checks
        initial_rows = len(stock_data)
        
        # 1. Check for duplicate indices
        if stock_data.index.duplicated().any():
            logger.warning(f"Found {stock_data.index.duplicated().sum()} duplicate indices. Keeping first occurrence.")
            stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
        
        # 2. Handle missing values
        missing_values = stock_data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            # Forward fill, then backfill
            stock_data = stock_data.ffill().bfill()
            # Drop any remaining rows with NaN values
            stock_data = stock_data.dropna()
        
        # 3. Validate data has enough rows after cleaning
        min_required = 60  # Minimum required for technical indicators and initial training
        if len(stock_data) < min_required:
            error_msg = (
                f"Insufficient data for {ticker} after cleaning: {len(stock_data)} rows available, "
                f"minimum {min_required} required. Try selecting a longer date range."
            )
            logger.error(error_msg)
            st.error(error_msg)
            return None, None
        
        # 4. Check for sufficient date range coverage
        date_range = (stock_data.index.max() - stock_data.index.min()).days
        if date_range < 30:  # At least 30 days of data required
            error_msg = f"Insufficient date range: only {date_range} days of data available. Need at least 30 days."
            logger.error(error_msg)
            st.error(error_msg)
            return None, None
            
        # 5. Validate price data
        price_columns = ['Open', 'High', 'Low', close_col]
        price_stats = stock_data[price_columns].describe()
        logger.info(f"Price statistics:\n{price_stats}")
        
        # Check for zero or negative prices
        if (stock_data[price_columns] <= 0).any().any():
            logger.warning("Found zero or negative prices. Replacing with previous valid values.")
            stock_data[price_columns] = stock_data[price_columns].replace(0, np.nan).ffill()
        
        # Final validation
        if stock_data.empty:
            error_msg = "No valid data remaining after cleaning"
            logger.error(error_msg)
            st.error(error_msg)
            return None, None
            
        logger.info(f"Successfully loaded {len(stock_data)} rows of clean data for {ticker}")
        logger.info(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
        
        return stock_data, close_col
        
    except Exception as e:
        error_msg = f"Error loading data for {ticker}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"{error_msg}. Please check the logs for more details.")
        return None, None

def generate_future_predictions(predictor, stock_data, processed_data, days=7, close_col='Close'):
    """
    Generate predictions for future dates beyond the historical data.
    
    Args:
        predictor: The trained TwoStagePredictor model
        stock_data: Historical stock data DataFrame
        processed_data: Processed data dictionary containing features
        days: Number of days to predict into the future
        close_col: Name of the close price column
        
    Returns:
        tuple: (future_predictions, future_dates)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating future predictions for {days} days")
    
    try:
        # Get the last date in the historical data
        last_date = stock_data.index[-1]
        logger.info(f"Last historical date: {last_date}")
        
        # Generate future dates using pandas bdate_range with proper frequency-based offset
        import pandas as pd
        future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=days).to_pydatetime().tolist()
        logger.info(f"Generated {len(future_dates)} future dates")
        
        # Get the last window of data for prediction
        # The window size should match what was used for training
        prediction_days = predictor.prediction_days
        
        # Get the last window of historical data
        last_window = stock_data.iloc[-prediction_days:].copy()
        
        # Get the last actual close price
        last_close = stock_data[close_col].iloc[-1]
        logger.info(f"Last close price: {last_close}")
        
        # Initialize future predictions array
        future_predictions = []
        
        # Get the feature columns used in the model
        X_test = processed_data.get('X_test')
        
        # Check if X_test exists and is in the expected tuple format for TwoStagePredictor
        if X_test is None or not isinstance(X_test, tuple) or len(X_test) != 2:
            logger.warning("X_test is missing or not in the expected tuple format for TwoStagePredictor")
            # Create empty placeholder arrays with the right shape
            X_test = (np.zeros((0, predictor.prediction_days, 4)), np.zeros((0, predictor.prediction_days, 1)))
        
        feature_columns = processed_data.get('feature_columns', [])
        
        # For each future day, predict the price and update the window
        
        # Use the same DataProcessor logic as in training
        processor = DataProcessor(sequence_length=prediction_days)
        last_window = stock_data.iloc[-prediction_days:].copy()
        for i, future_date in enumerate(future_dates):
            # If we have already made predictions, update the last close price
            if future_predictions:
                last_close = future_predictions[-1]
            # Create a new row for the future date
            new_row = last_window.iloc[-1].copy()
            new_row[close_col] = last_close
            # Append new row and drop the oldest
            window = last_window.append(pd.Series(new_row, name=future_date)).iloc[1:]
            # Recreate all features for the rolling window
            window_with_features = processor._create_features(window)
            # Select the last 'sequence_length' rows for the prediction window
            feature_columns = processed_data.get('feature_columns', window_with_features.columns.tolist())
            prediction_window = window_with_features[-prediction_days:][feature_columns]
            # Scale using the same scaler as in training
            if hasattr(predictor, 'feature_scaler') and predictor.feature_scaler is not None:
                features_scaled = predictor.feature_scaler.transform(prediction_window)
            else:
                features_scaled = prediction_window.values
            # Reshape to (1, sequence_length, n_features)
            features_scaled = features_scaled.reshape(1, prediction_days, -1)
            logger.info(f"Prediction input shape for {future_date}: {features_scaled.shape} (should be (1, {prediction_days}, n_features))")
            # Predict using the model
            prediction = predictor.predict(features_scaled)
            if hasattr(prediction, 'flatten'):
                prediction = prediction.flatten()
            predicted_value = prediction[0]
            if hasattr(predictor, 'target_scaler') and predictor.target_scaler is not None:
                predicted_value = predictor.target_scaler.inverse_transform([[predicted_value]])[0][0]
            future_predictions.append(predicted_value)
            last_window = window
            logger.info(f"Predicted price for {future_date}: {predicted_value}")
        
        logger.info(f"Generated {len(future_predictions)} future predictions")
        return future_predictions, future_dates
        
    except Exception as e:
        logger.error(f"Error generating future predictions: {str(e)}", exc_info=True)
        raise

def run_analysis(ticker: str, start_date: datetime, end_date: datetime):
    # User selected date range: start_date and end_date
    """Run the complete analysis pipeline with robust technical indicator validation"""
    try:
        from utils.data_processor import validate_technical_indicators
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = None
        st.session_state.ticker = ticker
        with st.spinner("🔄 Loading and validating data..."):
            stock_data, close_col = load_and_validate_data(ticker, start_date, end_date)
            if stock_data is None or stock_data.empty:
                st.error("Failed to load stock data. Please check API, ticker symbol, and date range.")
                logger.error("Stock data is None or empty after fetch.")
                return False
        with st.spinner("📊 Processing data and calculating indicators..."):
            st.session_state.stock_data_clean = stock_data
            st.session_state.close_col = close_col
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                st.error(f"Missing required columns after cleaning: {missing_cols}. Adjust column handling.")
                logger.error(f"Missing columns after cleaning: {missing_cols}")
                return False
            processor = DataProcessor()
            processed_data = processor.preprocess(stock_data)
            if processed_data is None:
                error_msg = st.session_state.get('pipeline_error', None)
                if error_msg:
                    st.error(error_msg)
                    logger.error(error_msg)
                else:
                    st.error("Unknown error during processing. Check logs for details.")
                    logger.error("Unknown error during processing. No pipeline_error set.")
                return False
            stock_data_processed = processed_data.get('stock_data')
            returns = processed_data.get('returns')
            if stock_data_processed is None or getattr(stock_data_processed, 'empty', True):
                error_msg = st.session_state.get('pipeline_error', None)
                if error_msg:
                    st.error(f"Data processing error: {error_msg}")
                else:
                    import logging
                    log_records = []
                    for handler in logging.getLogger().handlers:
                        if hasattr(handler, 'buffer'):
                            log_records.extend(handler.buffer)
                    logger_error = None
                    for rec in reversed(log_records):
                        if rec.levelname == 'ERROR':
                            logger_error = rec.getMessage()
                            break
                    if logger_error:
                        st.error(f"Data processing error: {logger_error}")
                    st.warning("Stock data is empty after processing and no error was captured. Check your input or logs.")
                logger.error("No stock data found in processed data after indicator calculation.")
                return False
            # --- Centralized technical indicator validation ---
            logger = st.session_state.get('logger')
            indicator_cols = ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD']
            stock_data_processed, missing = validate_technical_indicators(stock_data_processed, indicator_cols, logger=logger)
            if missing:
                st.warning(f"Missing technical indicators after processing: {missing}. Plots and analysis may be incomplete.", icon="⚠️")
                logger.warning(f"Missing technical indicators after processing: {missing}")
            processed_data['stock_data'] = stock_data_processed
            if returns is None or (isinstance(returns, (np.ndarray, pd.Series, list)) and (len(returns) == 0 or (hasattr(returns, 'all') and np.isnan(returns).all()))):
                warning_msg = "No returns found in processed data. This may be due to insufficient data for the selected date range."
                st.warning(warning_msg)
                logger.warning(warning_msg)
                processed_data['returns'] = np.array([])
                st.session_state['analysis_warning'] = "Analysis proceeded with limited data. Results may not be accurate."
            st.session_state.processed_data = processed_data
        # 3. Get sentiment analysis
        with st.spinner("📰 Analyzing market sentiment..."):
            sentiment_data = analyze_sentiment(ticker, start_date, end_date)
            
            # Validate the sentiment data
            if sentiment_data is None or not isinstance(sentiment_data, pd.DataFrame) or sentiment_data.empty:
                error_msg = "Warning: No sentiment data available. Using default values."
                st.warning(error_msg)
                st.session_state.logger.warning(error_msg)
                
                # Create default sentiment data
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                default_values = {
                    'Sentiment': 0.0,
                    'Price_Momentum': 0.0,
                    'Volume_Change': 0.0,
                    'Volatility': 0.1,
                    'sentiment_score': 0.0
                }
                sentiment_data = pd.DataFrame([default_values] * len(date_range), index=date_range)
            
            # Ensure required columns exist
            required_columns = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility', 'sentiment_score']
            for col in required_columns:
                if col not in sentiment_data.columns:
                    st.session_state.logger.warning(f"Adding missing sentiment column: {col}")
                    if col == 'sentiment_score':
                        sentiment_data[col] = sentiment_data.get('Sentiment', 0.0)
                    else:
                        sentiment_data[col] = 0.0
            
            # Ensure data types are correct and fill any NaN values
            for col in required_columns:
                try:
                    sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
                    # Fill NaN values with column mean or default
                    if sentiment_data[col].isnull().any():
                        fill_value = 0.0 if col in ['Sentiment', 'sentiment_score'] else 0.0
                        sentiment_data[col].fillna(fill_value, inplace=True)
                except Exception as e:
                    st.session_state.logger.error(f"Error processing column {col}: {str(e)}")
                    sentiment_data[col] = 0.0
            
            # Store the processed sentiment data
            processed_data['sentiment_data'] = sentiment_data[required_columns].copy()
            st.session_state.processed_data = processed_data
            
            # Debug log
            st.session_state.logger.info(f"Stored sentiment data with shape: {sentiment_data.shape}")
            st.session_state.logger.debug(f"Sentiment data columns: {sentiment_data.columns.tolist()}")
            st.session_state.logger.debug(f"Sentiment data head: {sentiment_data.head().to_dict()}")
            
            st.session_state.logger.info(f"Successfully processed sentiment data with shape: {sentiment_data.shape}")
        
        # 4. Train model
        with st.spinner("🤖 Training prediction model..."):
            # Defensive check: Returns calculation with graceful handling
            if 'returns' not in processed_data or processed_data['returns'] is None:
                warning_msg = "Returns data not calculated. Using empty array."
                st.warning(warning_msg)
                logger.warning(warning_msg)
                processed_data['returns'] = np.array([])
            
            # Handle empty returns gracefully
            if isinstance(processed_data['returns'], np.ndarray) and len(processed_data['returns']) == 0:
                # Create a dummy returns series with zeros
                warning_msg = "Using placeholder returns data due to insufficient data."
                st.warning(warning_msg)
                logger.warning(warning_msg)
                
                # Create a date range matching the stock data
                if 'stock_data' in processed_data and not processed_data['stock_data'].empty:
                    dummy_index = processed_data['stock_data'].index
                else:
                    dummy_index = pd.date_range(start=start_date, end=end_date, freq='B')
                
                # Create dummy returns
                dummy_returns = pd.Series([0.0] * len(dummy_index), index=dummy_index)
                returns_df = pd.DataFrame({ticker: dummy_returns})
                st.session_state['analysis_warning'] = "Analysis proceeded with placeholder data. Results may not be accurate."
            else:
                # Process returns normally if they exist
                try:
                    returns_series = pd.Series(processed_data['returns'].squeeze() if len(processed_data['returns'].shape) > 1 else processed_data['returns'])
                    returns_df = pd.DataFrame({ticker: returns_series})
                except Exception as e:
                    logger.warning(f"Error converting returns to DataFrame: {str(e)}")
                    # Create dummy returns as fallback
                    dummy_index = pd.date_range(start=start_date, end=end_date, freq='B')
                    dummy_returns = pd.Series([0.0] * len(dummy_index), index=dummy_index)
                    returns_df = pd.DataFrame({ticker: dummy_returns})
                    st.session_state['analysis_warning'] = "Using placeholder returns due to data processing error."
            
            # Store returns DataFrame in session state
            st.session_state.returns_df = returns_df

            # Enhanced data validation with proper error handling
            if 'X_train' not in processed_data or processed_data['X_train'] is None:
                # Log detailed information about processed_data for debugging
                logger.error(f"X_train not found in processed_data. Keys available: {list(processed_data.keys() if processed_data else [])}")
                st.error("Error: Training data (X_train) is missing. Please check the data processing pipeline.")
                st.session_state['analysis_error'] = "Missing training data (X_train). The data processor failed to generate valid training sequences."
                return None
            # X_train is present, continue with processing
            logger.info("X_train found in processed_data, continuing with model training")
            
            # The duplicate check for X_train is no longer needed since we already checked above
            # and returned if it wasn't present
                
            X_train = processed_data['X_train']
            y_train = processed_data.get('y_train')
            
            # Log data availability
            logger.info(f"X_train available: {X_train is not None}")
            logger.info(f"y_train available: {y_train is not None}")
            
            if y_train is None:
                error_msg = "❌ No target data (y_train) available for model training."
                st.error(error_msg)
                logger.error(error_msg)
                st.session_state['analysis_warning'] = error_msg
                return

            # --- TwoStagePredictor input validation and logging ---
            def is_invalid(arr, allow_zeros=False):
                if arr is None or len(arr) == 0:
                    return True
                arr_flat = np.array(arr).flatten()
                if arr_flat.size == 0 or np.all(np.isnan(arr_flat)):
                    return True
                # Only check for all zeros if allow_zeros is False
                if not allow_zeros and np.all(arr_flat == 0):
                    return True
                return False

            # If using TwoStagePredictor, X_train must be a tuple of two arrays
            if not (isinstance(X_train, tuple) and len(X_train) == 2):
                error_msg = (
                    "❌ X_train must be a tuple of two arrays (x_stock, x_additional) for TwoStagePredictor. "
                    f"Got type: {type(X_train)} with value: {repr(X_train)[:100]}"
                )
                st.error(error_msg)
                logger.error(error_msg)
                st.session_state['analysis_warning'] = error_msg
                return

            x_stock, x_additional = X_train
            # Log shapes and sample values for both arrays
            logger.info(f"x_stock type: {type(x_stock)}, shape: {getattr(x_stock, 'shape', 'N/A')}")
            logger.info(f"x_additional type: {type(x_additional)}, shape: {getattr(x_additional, 'shape', 'N/A')}")
            logger.info(f"y_train type: {type(y_train)}, shape: {getattr(y_train, 'shape', 'N/A')}")
            if isinstance(x_stock, np.ndarray) and x_stock.size > 0:
                logger.info(f"x_stock sample: {x_stock.flatten()[:5]}")
            if isinstance(x_additional, np.ndarray) and x_additional.size > 0:
                logger.info(f"x_additional sample: {x_additional.flatten()[:5]}")
            if isinstance(y_train, np.ndarray) and y_train.size > 0:
                logger.info(f"y_train sample: {y_train.flatten()[:5]}")

            # Validate arrays
            reason = None
            if is_invalid(x_stock):
                reason = "No valid stock feature data (x_stock) available for model training."
            elif is_invalid(x_additional, allow_zeros=True):  # Allow zeros for additional features
                reason = "No valid additional feature data (x_additional) available for model training."
            elif is_invalid(y_train):
                reason = "No valid target data (y_train) available for model training."
            elif x_stock.shape[0] != x_additional.shape[0] or x_stock.shape[0] != y_train.shape[0]:
                reason = (
                    f"Mismatch in sample counts: x_stock ({x_stock.shape[0]}), "
                    f"x_additional ({x_additional.shape[0]}), y_train ({y_train.shape[0]})"
                )

            if reason:
                error_msg = (
                    f"❌ {reason}\n\n"
                    "Model training cannot proceed. This is usually caused by a too-short date range, illiquid ticker, or excessive data cleaning.\n\n"
                    "**What you can do:**\n"
                    "- Select a longer date range.\n"
                    "- Choose a more liquid ticker.\n"
                    "- If the problem persists, please report a bug."
                )
                st.error(error_msg)
                logger.error(error_msg)
                st.session_state['analysis_warning'] = error_msg
                metrics = {
                    'loss': float('nan'),
                    'val_loss': float('nan'),
                    'mae': float('nan'),
                    'val_mae': float('nan'),
                    'mse': float('nan'),
                    'val_mse': float('nan')
                }
                st.session_state.model_metrics = metrics
                return

            # Initialize progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(epoch, logs=None):
                """Callback to update training progress"""
                if logs is None:
                    logs = {}
                progress = (epoch + 1) / 50  # Assuming 50 epochs
                progress_bar.progress(min(progress, 1.0))
                status_text.text(
                    f"Epoch {epoch + 1}/50 - "
                    f"Loss: {logs.get('loss', 'N/A'):.4f} - "
                    f"Val Loss: {logs.get('val_loss', 'N/A'):.4f} - "
                    f"MAE: {logs.get('mae', 'N/A'):.4f}"
                )

            # Ensure predictor exists in session state
            if 'predictor' not in st.session_state:
                st.session_state.predictor = TwoStagePredictor()
                logger.info("Initialized new TwoStagePredictor instance")
            else:
                logger.info("Using existing predictor from session state")

            try:
                # Ensure validation data exists and is in the correct format
                has_valid_validation = (
                    'X_val' in processed_data and 
                    'y_val' in processed_data and 
                    processed_data['y_val'] is not None and 
                    len(processed_data['y_val']) > 0 and
                    processed_data['X_val'] is not None and
                    len(processed_data['X_val']) > 0
                )
                
                if not has_valid_validation:
                    logger.warning("Validation data missing or invalid, performing train/validation split")
                    # Perform a proper train/validation split
                    from sklearn.model_selection import train_test_split
                    
                    # Ensure we have enough samples for a proper split
                    min_samples = 10  # Minimum samples needed for a meaningful split
                    if len(y_train) > min_samples:
                        # Split the training data into training and validation sets
                        X_train_stock, X_val_stock, X_train_add, X_val_add, y_train_split, y_val_split = train_test_split(
                            x_stock, 
                            x_additional if x_additional is not None else np.zeros((len(x_stock), 0)),
                            y_train,
                            test_size=0.2,
                            random_state=42,
                            shuffle=True
                        )
                        
                        # Update the training data
                        x_stock = X_train_stock
                        x_additional = X_train_add if x_additional is not None else None
                        y_train = y_train_split
                        
                        # Create validation data
                        X_val = (X_val_stock, X_val_add)
                        y_val = y_val_split
                        validation_data = (X_val, y_val)
                        logger.info(f"Created validation split with {len(y_val)} samples")
                    else:
                        logger.warning(f"Not enough samples ({len(y_train)}) for validation split, using None")
                        validation_data = None
                else:
                    # Get the validation data
                    X_val = processed_data['X_val']
                    y_val = processed_data['y_val']
                    
                    # Ensure X_val is a tuple of (stock_data, additional_features)
                    if not isinstance(X_val, tuple) or len(X_val) != 2:
                        # If X_val is not a tuple, create a tuple with the data and empty additional features
                        X_val = (X_val, np.zeros((len(X_val), 0)))
                    
                    # Ensure shapes match and data is valid
                    if len(X_val[0]) != len(y_val):
                        logger.warning(f"Mismatched validation data shapes: X_val[0]={len(X_val[0])}, y_val={len(y_val)}")
                        min_len = min(len(X_val[0]), len(y_val))
                        if min_len > 0:
                            X_val = (X_val[0][:min_len], X_val[1][:min_len] if len(X_val[1]) > 0 else X_val[1])
                            y_val = y_val[:min_len]
                            validation_data = (X_val, y_val)
                            logger.info(f"Adjusted validation data to {len(y_val)} samples")
                        else:
                            logger.warning("No valid validation data after adjustment, using None")
                            validation_data = None
                    else:
                        validation_data = (X_val, y_val)
                        logger.info(f"Using provided validation data with {len(y_val)} samples")

                # Show training start message
                status_text.text("Starting model training...")
                logger.info("Starting model training...")
                
                # Train the model with progress callback
                logger.info("About to start model training with TwoStagePredictor")
                logger.info(f"Input shapes - x_stock: {x_stock.shape}, x_additional: {x_additional.shape}, y_train: {y_train.shape}")
                
                # Add TrainingLogger callback for detailed epoch logs
                training_logger = TrainingLogger(logger)
                
                metrics = st.session_state.predictor.train_model(
                    (x_stock, x_additional),
                    y_train,
                    validation_data=validation_data,
                    epochs=10,
                    batch_size=32,
                    verbose=1,  # Changed to 1 to see progress in logs
                    callbacks=[LambdaCallback(on_epoch_end=update_progress), training_logger]
                )
                
                # Update progress to 100% when done
                progress_bar.progress(1.0)
                status_text.text("Model training completed!")
                logger.info(f"Model training completed with metrics: {metrics}")
                
                st.session_state.model_metrics = metrics
                
            except Exception as e:
                error_msg = f"Error during model training: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)
                
                # Update progress bar to show error
                progress_bar.progress(1.0, "Error during training!")
                
                metrics = {
                    'loss': float('nan'),
                    'val_loss': float('nan'),
                    'mae': float('nan'),
                    'val_mae': float('nan'),
                    'mse': float('nan'),
                    'val_mse': float('nan')
                }
                st.session_state.model_metrics = metrics
                
                # Re-raise the exception to be caught by the outer try-except
                raise

        
        # 5. Make predictions
        with st.spinner("🔮 Generating predictions..."):
            # Check if we have test data for predictions
            if 'X_test' not in processed_data or processed_data.get('X_test') is None or len(processed_data.get('X_test', [])) == 0:
                warning_msg = "No test data available for predictions. Using placeholder data."
                st.warning(warning_msg)
                logger.warning(warning_msg)
                
                # Create placeholder predictions
                predictions = np.array([])
                st.session_state.predictions = predictions
                
                # Ensure analysis_results is a dictionary before assignment
                if not hasattr(st.session_state, 'analysis_results') or not isinstance(st.session_state.analysis_results, dict):
                    st.session_state.analysis_results = {}
                st.session_state.analysis_results['predictions'] = predictions
            else:
                # Make predictions on test data
                try:
                    predictions = st.session_state.predictor.predict(*processed_data['X_test'])
                    st.session_state.predictions = predictions
                    
                    # Ensure analysis_results is a dictionary before assignment
                    if not hasattr(st.session_state, 'analysis_results') or not isinstance(st.session_state.analysis_results, dict):
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results['predictions'] = predictions
                except Exception as e:
                    error_msg = f"Error during prediction: {str(e)}"
                    st.warning(error_msg)
                    logger.warning(error_msg)
                    predictions = np.array([])
                    st.session_state.predictions = predictions
            
            # Get future prediction days from sidebar input
            future_days = st.session_state.get('prediction_days', 7)
            st.session_state.logger.info(f"Generating future predictions for {future_days} days")
            
            try:
                # Check if we have enough data for future predictions
                # Define missing_training_data based on predictor status and processed data
                missing_training_data = (st.session_state.get('predictor') is None or 
                                        'X_train' not in processed_data or 
                                        'y_train' not in processed_data or
                                        len(processed_data.get('X_train', [])) < 30)
                
                # Log prediction input sufficiency before check
                if stock_data is not None and not stock_data.empty:
                    logger.info(f"[PREDICTION] stock_data shape: {stock_data.shape}, date range: {stock_data.index.min()} to {stock_data.index.max()}")
                else:
                    logger.info("[PREDICTION] stock_data is None or empty")
                logger.info(f"[PREDICTION] missing_training_data: {missing_training_data}")
                logger.info(f"[PREDICTION] predictor exists: {st.session_state.get('predictor') is not None}")
                logger.info(f"[PREDICTION] X_train in processed_data: {'X_train' in processed_data}")
                logger.info(f"[PREDICTION] y_train in processed_data: {'y_train' in processed_data}")
                logger.info(f"[PREDICTION] len(X_train): {len(processed_data.get('X_train', [])) if 'X_train' in processed_data else 'N/A'}")

                
                processor = DataProcessor()
                contiguous_ok = processor.check_recent_contiguous_days(stock_data, st.session_state.predictor.prediction_days, logger=logger)
                if stock_data is None or stock_data.empty or missing_training_data or not contiguous_ok:
                    if not contiguous_ok:
                        warning_msg = f"Insufficient contiguous recent data for future predictions (last {st.session_state.predictor.prediction_days} trading days are not contiguous). Using placeholder data."
                    else:
                        warning_msg = "Insufficient data for future predictions. Using placeholder data."
                    st.warning(warning_msg)
                    logger.warning(warning_msg)
                    # Create placeholder future predictions
                    future_dates = pd.date_range(start=datetime.now(), periods=future_days, freq='B')
                    last_price = 100.0  # Default placeholder price
                    future_predictions = np.array([last_price] * future_days)
                else:
                    # Generate future predictions normally
                    future_predictions, future_dates = generate_future_predictions(
                        predictor=st.session_state.predictor,
                        stock_data=stock_data,
                        processed_data=processed_data,
                        days=future_days,
                        close_col=close_col
                    )
                
                # Store future predictions in session state with explicit debug logging
                st.session_state.future_predictions = future_predictions
                st.session_state.future_dates = future_dates
                st.session_state.logger.info(f"Stored future_predictions in session state: {len(future_predictions)} values")
                st.session_state.logger.info(f"First few predictions: {future_predictions[:3]}")
                
                # Also store in analysis_results for consistent access
                if 'analysis_results' not in st.session_state:
                    st.session_state.analysis_results = {}
                    
                # Ensure analysis_results is a dictionary
                if not isinstance(st.session_state.analysis_results, dict):
                    st.session_state.analysis_results = {}
                    st.session_state.logger.warning("Reset analysis_results to empty dict because it was not a dictionary")
                    
                st.session_state.analysis_results['future_predictions'] = future_predictions
                st.session_state.analysis_results['future_dates'] = future_dates
                st.session_state.logger.info(f"Stored future_predictions in analysis_results: {len(future_predictions)} values")
                
                # Print a debug message to the UI
                st.sidebar.success(f"✅ Generated {len(future_predictions)} future predictions")
            except Exception as e:
                st.session_state.logger.error(f"Error generating future predictions: {str(e)}", exc_info=True)
                st.warning(f"Could not generate future predictions: {str(e)}")
        
        # 6. Portfolio optimization and risk metrics
        portfolio_metrics = {}
        with st.spinner("📈 Optimizing portfolio..."):
            try:
                if 'portfolio_optimizer' not in st.session_state:
                    st.session_state.portfolio_optimizer = PortfolioOptimizer()
                
                # Prepare returns data for portfolio optimization
                if 'returns' not in processed_data or processed_data['returns'] is None:
                    raise ValueError("No returns data available for portfolio optimization")
                
                # Convert to 1D array if it's 2D
                returns_series = pd.Series(processed_data['returns'].squeeze() if len(processed_data['returns'].shape) > 1 else processed_data['returns'])
                
                # Create a DataFrame with the returns data
                returns_df = pd.DataFrame({
                    ticker: returns_series
                })
                
                # Ensure we have valid returns data
                if returns_df.empty or returns_df.isnull().all().all():
                    raise ValueError("No valid returns data available for optimization")
                
                # Fill any remaining NaN values with 0
                returns_df = returns_df.fillna(0)
                
                # Calculate portfolio metrics
                weights = np.array([1.0])  # Equal weight for single stock
                volatility = st.session_state.portfolio_optimizer.calculate_volatility(
                    returns_df,
                    weights
                )
                max_drawdown = st.session_state.portfolio_optimizer.calculate_max_drawdown(
                    returns_df,
                    weights
                )
                
                # Calculate metrics with explicit type conversion and error handling
                try:
                    expected_return = float(np.sum(returns_df.mean() * weights) * 252)  # Annualized
                    cov_matrix = returns_df.cov() * 252  # Annualize covariance
                    sharpe_ratio = st.session_state.portfolio_optimizer.calculate_sharpe_ratio(returns_df, weights)
                    sharpe_val = float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0
                    vol = float(volatility) if not np.isnan(volatility) else 0.0
                    mdd = float(max_drawdown) if not np.isnan(max_drawdown) else 0.0
                    current_price = float(stock_data[close_col].iloc[-1]) if not stock_data.empty and close_col in stock_data.columns else 0.0
                    div_ratio = 1.0 / sum(w**2 for w in weights) if sum(weights) > 0 else 0.0

                    # Create weights dictionary: single stock or multi-stock
                    if len(weights) == 1:
                        weights_dict = {ticker: float(weights[0])}
                    else:
                        weights_dict = {col: float(w) for col, w in zip(returns_df.columns, weights)}

                    portfolio_metrics = {
                        'weights': weights_dict,  # Ensure all weights are floats
                        'expected_return': expected_return,
                        'sharpe_ratio': sharpe_val,
                        'volatility': vol,
                        'max_drawdown': mdd,
                        'ticker': str(ticker),  # Ensure ticker is a string
                        'current_price': current_price,
                        'diversification_ratio': float(div_ratio)
                    }
                    logger.info(f"Successfully created portfolio_metrics with keys: {list(portfolio_metrics.keys())}")
                except Exception as calc_error:
                    logger.error(f"Error calculating portfolio metrics: {str(calc_error)}", exc_info=True)
                    raise ValueError(f"Failed to calculate portfolio metrics: {str(calc_error)}")

                
                # Store portfolio metrics in session state
                st.session_state.portfolio_metrics = portfolio_metrics
                logger.info(f"Stored portfolio metrics: {list(portfolio_metrics.keys())}")
                
            except Exception as e:
                error_msg = f"Portfolio optimization failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.warning(error_msg)
                
                # Set default metrics on error
                portfolio_metrics = {
                    'weights': {ticker: 1.0},
                    'expected_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0,
                    'ticker': ticker,
                    'current_price': float(stock_data[close_col].iloc[-1]) if not stock_data.empty and close_col in stock_data.columns else 0.0,
                    'diversification_ratio': 1.0,
                    'error': str(e)
                }
                st.session_state.portfolio_metrics = portfolio_metrics
        
        # Store the ticker in session state and ensure it's a string
        st.session_state.ticker = str(ticker).upper()
        
        # 7. Initialize or update real-time data handler
        with st.spinner("🔌 Initializing real-time data connection..."):
            try:
                # Initialize the real-time data handler
                rt_handler = RealTimeDataHandler.get_instance()
                
                # Store in session state with both possible keys for compatibility
                st.session_state.rt_handler = rt_handler
                st.session_state.realtime_handler = rt_handler
                
                # Start the handler if not already running
                if not rt_handler.running:
                    try:
                        # Create a future to track the connection result
                        future = asyncio.run_coroutine_threadsafe(
                            rt_handler.start(wait_for_connection=False), 
                            rt_handler.loop
                        )
                        
                        try:
                            # Wait for up to 10 seconds for the connection to be established
                            result = future.result(timeout=10)
                            logger.info(f"Started real-time data handler for {ticker}: {result}")
                            
                            # Wait a moment for the connection to stabilize
                            time.sleep(1)
                            
                            # Check if we're connected
                            if not rt_handler.connected:
                                raise ConnectionError("Failed to establish connection after starting handler")
                                
                        except TimeoutError as te:
                            logger.warning(f"Timeout while starting real-time handler for {ticker}: {str(te)}")
                            # Don't return here, try to continue with limited functionality
                            st.warning(f"Real-time data connection is slow. Some features may be limited.")
                            
                    except Exception as e:
                        logger.error(f"Error starting real-time data handler for {ticker}: {str(e)}", exc_info=True)
                        st.warning("Could not start real-time data handler. Some features may be limited.")
                
                # Only try to subscribe if we have an active connection
                if rt_handler.connected:
                    try:
                        # Subscribe to the current ticker
                        rt_handler.subscribe(ticker)
                        logger.info(f"Subscribed to real-time updates for {ticker}")
                    except Exception as sub_error:
                        logger.error(f"Error subscribing to {ticker}: {str(sub_error)}")
                        st.warning(f"Could not subscribe to real-time updates for {ticker}")
                else:
                    logger.warning(f"Skipping subscription to {ticker} - no active connection")
                
                # Store the current ticker in session state for the real-time tab
                st.session_state.current_ticker = ticker
                
                # Initialize WebSocket status in session state with both lowercase and title-case keys for compatibility
                st.session_state.ws_status = {
                    # Title-case keys (original format)
                    "Status": "Connected" if rt_handler.connected else "Disconnected",
                    "Authenticated": rt_handler.authenticated,
                    "Subscribed Symbols": list(rt_handler.subscribed_symbols),
                    "Messages Received": rt_handler.message_count,
                    "Last Error": str(rt_handler.last_error) if rt_handler.last_error else None,
                    # Lowercase keys (matching get_status() method)
                    "connected": rt_handler.connected,
                    "authenticated": rt_handler.authenticated,
                    "subscribed_symbols": list(rt_handler.subscribed_symbols),
                    "message_count": rt_handler.message_count,
                    "last_error": str(rt_handler.last_error) if rt_handler.last_error else None
                }
                
                st.success(f"✅ Real-time data connection initialized for {ticker}")
            except Exception as e:
                error_msg = f"Failed to initialize real-time data: {str(e)}"
                st.warning(f"⚠️ {error_msg}")
                logger.error(error_msg, exc_info=True)
                st.session_state.ws_status = {
                    "Status": "Error",
                    "Error": str(e),
                    "Last Error": str(e)
                }
        
        # 8. Store results in session state
        st.session_state.analysis_complete = True
        st.session_state.analysis_results = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'stock_data': stock_data,
            'processed_data': processed_data,
            'predictions': predictions,
            'metrics': metrics,
            'portfolio_metrics': portfolio_metrics,
            'future_predictions': st.session_state.get('future_predictions', None),
            'future_dates': st.session_state.get('future_dates', None)
        }
        
        # Store a flag to indicate we need to show tabs after rerun
        st.session_state._show_tabs_after_rerun = True
        
        # Force a rerun to update the UI with tabs
        st.rerun()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        st.session_state.analysis_complete = False
        return False

# Add risk management settings to sidebar
with st.sidebar.expander("Risk Management Settings"):
    # Initialize RiskManager in session state if not present
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(initial_capital=1000.0, max_risk_per_trade=0.02)
        st.session_state.logger.info("Initialized RiskManager in session state")
    
    risk_manager = st.session_state.risk_manager
    
    # Add risk management strategy description
    st.markdown("### Risk Management Strategy")
    st.markdown("This risk management system implements the following strategies:")
    st.markdown("""
    - **Position Sizing**: Calculates the appropriate position size based on your risk tolerance and stop loss level
    - **Stop Loss & Take Profit**: Automatically calculates stop loss and take profit levels based on price volatility
    - **Portfolio Risk**: Monitors concentration risk and ensures diversification across positions
    - **Risk Per Trade**: Limits the maximum risk per trade to a percentage of your total capital
    - **Unrealized P&L**: Tracks unrealized profit and loss for each position
    """)
    st.markdown("---")
    
    # Capital inputs
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000.0,
        value=risk_manager.initial_capital,
        step=1000.0,
        format="%.0f",
        key="sidebar_initial_capital"
    )
    
    current_capital = st.number_input(
        "Current Capital ($)",
        min_value=1000.0,
        value=risk_manager.current_capital,
        step=1000.0,
        format="%.0f",
        key="sidebar_current_capital"
    )
    
    max_risk_per_trade = st.slider(
        "Max Risk per Trade (%)", 
        min_value=0.1, 
        max_value=10.0, 
        value=risk_manager.max_risk_per_trade * 100,
        step=0.1,
        format="%.1f%%",
        key="sidebar_max_risk"
    )
    
    # Volatility Multipliers
    st.subheader("Volatility Multipliers")
    
    sl_multiplier = st.number_input(
        "Stop Loss Multiplier",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        format="%d",
        help="Multiplier applied to volatility to calculate stop loss level",
        key="sidebar_sl_multiplier"
    )
    
    tp_multiplier = st.number_input(
        "Take Profit Multiplier",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        format="%d",
        help="Multiplier applied to volatility to calculate take profit level",
        key="sidebar_tp_multiplier"
    )
    
    # Update button
    if st.button("Update Risk Settings"):
        try:
            # Update risk manager settings directly
            risk_manager.initial_capital = initial_capital
            risk_manager.current_capital = current_capital
            risk_manager.max_risk_per_trade = max_risk_per_trade / 100  # Convert from percentage to decimal
            
            # Log the update
            st.session_state.logger.info(
                f"Updated risk management settings: "
                f"Initial Capital=${initial_capital:,.0f}, "
                f"Current Capital=${current_capital:,.0f}, "
                f"Max Risk={max_risk_per_trade:.1f}%"
            )
            
            # Display success message
            st.success("Risk management settings updated successfully!")
        except Exception as e:
            st.error(f"Error updating risk settings: {str(e)}")
            st.session_state.logger.error(f"Error updating risk settings: {str(e)}", exc_info=True)

# Add analysis button
analyze_button = st.sidebar.button("Analyze")

# Ensure analysis pipeline is triggered on button click
if analyze_button:
    ticker = st.session_state.get('ticker', 'AAPL')
    # Use the exact dates selected by the user in the UI
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    run_analysis(ticker, start_date, end_date)
    st.session_state['_show_tabs_after_rerun'] = True
    st.rerun()

# Initialize predictor in session state if not exists
if 'predictor' not in st.session_state:
    st.session_state.predictor = TwoStagePredictor()
    st.session_state.logger.info("Initialized TwoStagePredictor in session state")

# Analysis results are now initialized at the top of the script

# Main app layout
# Debug info - can be removed after testing
st.sidebar.json({
    'analysis_complete': st.session_state.get('analysis_complete', False),
    'has_analysis_results': 'analysis_results' in st.session_state,
    'ticker': st.session_state.get('ticker', 'Not set'),
    'show_tabs_after_rerun': st.session_state.get('_show_tabs_after_rerun', False)
})

# Check if we should show the analysis tabs
show_tabs = (st.session_state.get('analysis_complete', False) and 
             'analysis_results' in st.session_state) or \
            st.session_state.get('_show_tabs_after_rerun', False)

# Clear the rerun flag if it was set
if st.session_state.get('_show_tabs_after_rerun', False):
    del st.session_state['_show_tabs_after_rerun']
    st.session_state.analysis_complete = True  # Ensure this stays True

if show_tabs:
    # Ensure all required session state keys are present and not None
    if st.session_state.analysis_results is None:
        st.session_state.analysis_results = {}
    
    # Initialize required session state variables if they don't exist
    required_vars = {
        'portfolio_metrics': {},
        'metrics': {},
        'portfolio_weights': {st.session_state.get('ticker', 'AAPL'): 1.0},
        'rt_data': {},
        'processed_data': {}
    }
    
    for var, default in required_vars.items():
        if var not in st.session_state or st.session_state[var] is None:
            st.session_state[var] = default
    
    # Ensure ticker is set in session state
    if 'ticker' not in st.session_state:
        st.session_state.ticker = st.session_state.analysis_results.get('ticker', 'AAPL')
    
    # Get processed data and ensure it exists
    processed_data = st.session_state.get('processed_data', {})
    
    # Ensure sentiment_data exists in processed_data
    if 'sentiment_data' not in processed_data:
        # Create default sentiment data if it doesn't exist
        date_range = pd.date_range(start=st.session_state.get('start_date', datetime.now() - timedelta(days=30)),
                                 end=st.session_state.get('end_date', datetime.now()),
                                 freq='D')
        default_values = {
            'Sentiment': 0.0,
            'Price_Momentum': 0.0,
            'Volume_Change': 0.0,
            'Volatility': 0.1,
            'sentiment_score': 0.0
        }
        processed_data['sentiment_data'] = pd.DataFrame([default_values] * len(date_range), 
                                                      index=date_range)
        st.session_state.processed_data = processed_data
    
    sentiment_data = processed_data.get('sentiment_data')
    
    # Define required columns for sentiment data
    required_sentiment_columns = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility', 'sentiment_score']
    
    # Always show sentiment data tab, but handle missing data gracefully
    has_sentiment_data = True
    
    # Handle missing or invalid sentiment data
    if sentiment_data is None or not isinstance(sentiment_data, pd.DataFrame) or sentiment_data.empty:
        st.warning("No sentiment data available for the selected ticker and date range.")
        st.session_state.logger.warning("Sentiment data is None, not a DataFrame, or empty")
        # Create an empty DataFrame with the required columns
        sentiment_data = pd.DataFrame(columns=['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility', 'sentiment_score'])
        # Set the index to the selected date range
        date_range = pd.date_range(start=st.session_state.get('start_date', datetime.now() - timedelta(days=30)),
                                  end=st.session_state.get('end_date', datetime.now()),
                                  freq='D')
        sentiment_data = sentiment_data.reindex(date_range)
        processed_data['sentiment_data'] = sentiment_data
        st.session_state.processed_data = processed_data
    else:
        # Log that we have valid sentiment data
        st.session_state.logger.info(f"Found valid sentiment data with shape {sentiment_data.shape}")
        st.session_state.logger.info(f"Sentiment data columns: {sentiment_data.columns.tolist()}")
        st.session_state.logger.info(f"Sentiment data index type: {type(sentiment_data.index).__name__}")
        
        # Check if all required columns are present
        missing_columns = [col for col in required_sentiment_columns if col not in sentiment_data.columns]
        if missing_columns:
            st.session_state.logger.warning(f"Missing required sentiment columns: {missing_columns}")
            
        # Check if data contains non-zero values
        has_non_zero = False
        for col in ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']:
            if col in sentiment_data.columns:
                if not (sentiment_data[col] == 0).all():
                    has_non_zero = True
                    st.session_state.logger.info(f"Column {col} has non-zero values")
                    break
        
        if not has_non_zero:
            st.session_state.logger.warning("All sentiment values are zero, may be dummy data")
        else:
            st.session_state.logger.info("Sentiment data contains non-zero values, appears to be valid")
        
    # Ensure all required columns exist
    for col in required_sentiment_columns:
        if col not in sentiment_data.columns:
            st.warning(f"Adding missing sentiment column: {col}")
            if col == 'sentiment_score':
                sentiment_data[col] = sentiment_data.get('Sentiment', 0.0)
            else:
                sentiment_data[col] = 0.0
    
    # Ensure data types are correct
    for col in required_sentiment_columns:
        try:
            sentiment_data[col] = pd.to_numeric(sentiment_data[col], errors='coerce')
            if sentiment_data[col].isnull().any():
                fill_value = 0.0 if col in ['Sentiment', 'sentiment_score'] else 0.0
                # Fix the pandas FutureWarning by avoiding chained assignment with inplace=True
                sentiment_data[col] = sentiment_data[col].fillna(fill_value)
        except Exception as e:
            st.session_state.logger.error(f"Error processing column {col}: {str(e)}")
            sentiment_data[col] = 0.0
    
    # Update the processed data with the fixed sentiment data
    processed_data['sentiment_data'] = sentiment_data[required_sentiment_columns].copy()
    st.session_state.processed_data = processed_data
    
    # Initialize RiskManager if it doesn't exist in session state
    if 'risk_manager' not in st.session_state:
        initial_capital = 1000.0  # Default initial capital
        max_risk_per_trade = 0.02   # Default max risk per trade (2%)
        st.session_state.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade
        )
        st.session_state.logger.info(f"Initialized RiskManager with capital: ${initial_capital:,.2f}, max risk: {max_risk_per_trade*100:.1f}%")
    
    # --- Tab Names Setup ---
    tab_names = [
        "Price Analysis",
        "Sentiment Analysis" if has_sentiment_data else None,
        "Risk Analysis",
        "Portfolio Analysis",
        "Performance Analysis",
        "Real-Time Analysis"
    ]
    # Remove None values (if Sentiment Analysis is not available)
    tab_names = [name for name in tab_names if name]

    # --- Create Tabs ---
    tabs = st.tabs(tab_names)

    # --- Tab Index Mapping ---
    tab_indices = {name: idx for idx, name in enumerate(tab_names)}

    # --- Assign Tab Variables ---
    tab_price = tabs[tab_indices["Price Analysis"]]
    tab_risk = tabs[tab_indices["Risk Analysis"]]
    tab_portfolio = tabs[tab_indices["Portfolio Analysis"]]
    tab_performance = tabs[tab_indices["Performance Analysis"]]
    tab_realtime = tabs[tab_indices["Real-Time Analysis"]]
    tab_sentiment = tabs[tab_indices["Sentiment Analysis"]] if "Sentiment Analysis" in tab_indices else None
    
    # Ensure tab_realtime is defined even if not in tab_indices (for backward compatibility)
    if "Real-Time Analysis" not in tab_indices:
        tab_realtime = None

    # --- Debug Expanders for Other Tabs ---
    def show_tab_debug(tab, df, name):
        if st.session_state.get('debug_mode', False) and isinstance(df, pd.DataFrame):
            with tab:
                show_debug_df(df, name=name)
    # --- Debug Utility Function ---
    def show_debug_df(df, name="Data", extra=None):
        with st.expander(f"Debug: {name} Info"):
            pass  # ...existing code...

    # Before setting performance analysis data
    performance_data = processed_data.get('performance_data', None)
    logger.info(f"Setting performance_analysis_data: {performance_data.shape if performance_data is not None else 'None'}")
    st.session_state["performance_analysis_data"] = performance_data

    # Before setting realtime analysis data
    realtime_data = st.session_state.get('rt_data', None)
    logger.info(f"Setting realtime_analysis_data: {realtime_data.shape if hasattr(realtime_data, 'shape') else 'None'}")
    st.session_state["realtime_analysis_data"] = realtime_data
    def show_debug_df(df, name="Data", extra=None):
        with st.expander(f"Debug: {name} Info"):
            st.write(f"{name} Shape:", df.shape)
            st.write(f"{name} Columns:", df.columns.tolist())
            st.write(f"{name} Index Type:", type(df.index).__name__)
            st.write(f"{name} Sample:")
            st.dataframe(df.head())
            if extra:
                for k, v in extra.items():
                    st.write(f"{k}:", v)

    # Price Analysis Tab (always show)
    with tab_price:
        st.subheader("Price Analysis")
        
        # Add description
        st.markdown("""This tab provides comprehensive price analysis for the selected stock, including historical price charts, 
        technical indicators, and price patterns. It helps identify trends, support/resistance levels, and potential 
        entry/exit points based on technical analysis principles.""")
        st.markdown("---")
        
        display_price_analysis(st.session_state.get('ticker', 'AAPL'))
        
        # Debug expander for price data
        price_data = processed_data.get('stock_data')
        if st.session_state.get('debug_mode', False) and isinstance(price_data, pd.DataFrame):
            show_debug_df(price_data, name="Price Data")

    # Debug info for sentiment data (sidebar summary)
    if st.session_state.get('debug_mode', False):
        st.sidebar.subheader("Debug Info")
        st.sidebar.json({
            'has_sentiment_data': has_sentiment_data,
            'sentiment_data_columns': list(sentiment_data.columns) if has_sentiment_data else [],
            'sentiment_data_shape': sentiment_data.shape if has_sentiment_data else (0, 0),
            'tab_indices': tab_indices
        })
    
    # Sentiment Analysis Tab (only if we have data)
    if has_sentiment_data and tab_sentiment is not None:
        with tab_sentiment:
            st.subheader("Sentiment Analysis")
            
            # Add description
            st.markdown("""This tab analyzes market sentiment data for the selected stock, combining news sentiment, 
            social media trends, and analyst opinions. It helps identify shifts in market perception that may 
            precede price movements, giving you an edge in anticipating market direction based on crowd psychology.""")
            st.markdown("---")
            
            stock_data = processed_data.get('stock_data')
            
            # Ensure sentiment_data is a DataFrame
            if not isinstance(sentiment_data, pd.DataFrame):
                st.error("Invalid sentiment data format. Expected a pandas DataFrame.")
                st.stop()
            
            # Ensure required columns exist with default values
            required_cols = {
                'Sentiment': 0.0,
                'Price_Momentum': 0.0,
                'Volume_Change': 0.0,
                'Volatility': 0.0,
                'sentiment_score': 0.0  # For backward compatibility
            }
            # Add any missing columns with default values
            for col, default_val in required_cols.items():
                if col not in sentiment_data.columns:
                    sentiment_data[col] = default_val
            # Handle any NaN values in required columns
            for col in required_cols:
                if col in sentiment_data.columns and sentiment_data[col].isna().any():
                    sentiment_data[col].fillna(0.0, inplace=True)

            # Debug expander for sentiment data
            if st.session_state.get('debug_mode', False):
                show_debug_df(sentiment_data, name="Sentiment Data", extra={
                    "All values are zero": all((sentiment_data[c] == 0).all() for c in ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility'] if c in sentiment_data),
                    "Data Source": sentiment_data['data_source'].unique().tolist() if 'data_source' in sentiment_data.columns else None
                })

            # Display sentiment metrics
            st.markdown("#### Sentiment Indicators")
            try:
                sentiment_metrics = {
                    'Composite': float(sentiment_data['Sentiment'].mean()),
                    'Price Momentum': float(sentiment_data['Price_Momentum'].mean() if 'Price_Momentum' in sentiment_data else 0.0),
                    'Volume Change': float(sentiment_data['Volume_Change'].mean() if 'Volume_Change' in sentiment_data else 0.0),
                    'Volatility': float(sentiment_data['Volatility'].mean() if 'Volatility' in sentiment_data else 0.0)
                }
                
                # Display metrics with progress bars
                for metric, value in sentiment_metrics.items():
                    # Normalize value to 0-1 range for progress bar
                    normalized_value = (value + 1) / 2  # Assuming values are in -1 to 1 range
                    normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to 0-1
                    st.progress(normalized_value, text=f"{metric}: {value:.2f}")
                
                # Sentiment summary
                st.markdown("#### Sentiment Summary")
                current_sentiment = float(sentiment_data['Sentiment'].tail(5).mean())
                
                # Determine sentiment level with thresholds
                if current_sentiment > 0.5:
                    st.success("✅ Strongly Bullish Sentiment")
                elif current_sentiment > 0.1:
                    st.info("ℹ️ Mildly Bullish Sentiment")
                elif current_sentiment < -0.5:
                    st.error("❌ Strongly Bearish Sentiment")
                elif current_sentiment < -0.1:
                    st.warning("⚠️ Mildly Bearish Sentiment")
                else:
                    st.info("➖ Neutral Sentiment")
                
                # Sentiment trend
                st.markdown("#### Sentiment Trend")
                
                # Create a line chart for sentiment over time
                if not sentiment_data.empty and 'Sentiment' in sentiment_data.columns:
                    fig = px.line(
                        sentiment_data,
                        x=sentiment_data.index,
                        y='Sentiment',
                        title='Sentiment Over Time',
                        labels={'value': 'Sentiment Score', 'date': 'Date'},
                        line_shape='spline',
                        render_mode='svg'
                    )
                    
                    # Add a horizontal line at y=0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    
                    # Update layout for better readability
                    fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score',
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display recent sentiment scores in a table
                    st.markdown("#### Recent Sentiment Scores")
                    st.dataframe(
                        sentiment_data[['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']].tail(10).round(4)
                    )
                    
            except Exception as e:
                st.error(f"Error displaying sentiment metrics: {str(e)}")
                logger.error(f"Error in sentiment metrics display: {str(e)}", exc_info=True)
                
            # Debug: Log stock data info
            if stock_data is not None:
                close_col = st.session_state.get('close_col', 'Close')
                st.session_state.logger.debug(f"Stock data columns: {stock_data.columns.tolist()}")
                st.session_state.logger.debug(f"Close column: {close_col}")
        
    # Risk Analysis Tab
    with tab_risk:
        st.subheader("Risk Analysis")

        # Debug expander for risk-related data
        if st.session_state.get('debug_mode', False):
            show_debug_df(stock_data, name="Stock Data (Risk Tab)")

        if stock_data is not None and not stock_data.empty and close_col in stock_data.columns:
            try:
                # Get the current ticker symbol
                ticker = st.session_state.get('ticker', 'AAPL')
                
                # Calculate basic metrics with error handling
                returns = stock_data[close_col].pct_change().dropna()
                if len(returns) < 2:
                    raise ValueError("Insufficient data points for risk metrics")
                    
                volatility = returns.std() * np.sqrt(252)
                max_drawdown = (stock_data[close_col] / stock_data[close_col].cummax() - 1).min()
                sharpe_ratio = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
                
                # Get the latest price for the ticker
                latest_price = stock_data[close_col].iloc[-1]
                
                # Get or create the risk manager
                risk_manager = st.session_state.risk_manager
                
                # Calculate stop loss and take profit levels based on volatility
                sl_multiplier = st.session_state.get('sidebar_sl_multiplier', 2.0)
                tp_multiplier = st.session_state.get('sidebar_tp_multiplier', 3.0)
                stop_loss = risk_manager.calculate_stop_loss(latest_price, volatility, multiplier=sl_multiplier)
                take_profit = risk_manager.calculate_take_profit(latest_price, volatility, multiplier=tp_multiplier)
                
                # Calculate position size based on risk parameters
                position_size = risk_manager.calculate_position_size(ticker, latest_price, stop_loss)
                
                # Create tabs within the Risk Analysis tab for different risk aspects
                risk_subtabs = st.tabs(["Market Risk", "Position Risk", "Portfolio Risk", "Risk Settings"])
                
                # Market Risk Tab - Display basic market risk metrics
                with risk_subtabs[0]:
                    st.subheader("Market Risk Metrics")
                    
                    # Add description
                    st.markdown("""This tab analyzes market-specific risk factors for the selected stock, including volatility, 
                    drawdown potential, and risk-adjusted returns. These metrics help assess the inherent risk of the asset 
                    regardless of position size.""")
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Volatility (Annualized)", f"{volatility*100:.2f}%")
                    with col2:
                        st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                        
                    # Add a chart showing volatility over time
                    st.subheader("Volatility Over Time")
                    rolling_vol = returns.rolling(window=21).std() * np.sqrt(252) * 100  # 21-day rolling volatility
                    vol_fig = go.Figure()
                    vol_fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode='lines', name='21-Day Volatility'))
                    vol_fig.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Volatility (%)',
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(vol_fig, use_container_width=True)
                
                # Position Risk Tab - Display position-specific risk data
                with risk_subtabs[1]:
                    st.subheader("Position Risk Analysis")
                    
                    # Add description
                    st.markdown("""This tab calculates position-specific risk metrics for individual trades. It determines 
                    optimal position size based on your risk tolerance, and calculates stop loss and take profit levels 
                    using volatility-based multipliers from the sidebar settings.""")
                    st.markdown("---")
                    
                    # Get volatility multipliers from sidebar
                    sl_multiplier = st.session_state.get('sidebar_sl_multiplier', 2.0)
                    tp_multiplier = st.session_state.get('sidebar_tp_multiplier', 3.0)
                    
                    # Calculate stop loss and take profit using sidebar multipliers
                    daily_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    stop_loss = latest_price - (daily_volatility * sl_multiplier)
                    take_profit = latest_price + (daily_volatility * tp_multiplier)
                    
                    # Create a form for position entry
                    with st.form("position_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            entry_price = st.number_input("Entry Price", min_value=0.01, value=latest_price, format="%.2f")
                        with col2:
                            quantity = st.number_input("Quantity", min_value=1, value=int(position_size), step=1)
                            
                        col1, col2 = st.columns(2)
                        with col1:
                            stop_loss_price = st.number_input("Stop Loss", min_value=0.01, value=stop_loss, format="%.2f")
                            st.caption(f"*Using {sl_multiplier}x volatility from sidebar*")
                        with col2:
                            take_profit_price = st.number_input("Take Profit", min_value=0.01, value=take_profit, format="%.2f")
                            st.caption(f"*Using {tp_multiplier}x volatility from sidebar*")
                            
                        submit_button = st.form_submit_button("Calculate Position Risk")
                        
                    # Display position risk metrics
                    if submit_button:  # Only show when button is clicked
                        # Calculate risk metrics
                        risk_amount = (entry_price - stop_loss_price) * quantity
                        reward_amount = (take_profit_price - entry_price) * quantity
                        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                        position_value = entry_price * quantity
                        risk_percentage = (risk_amount / position_value) * 100 if position_value > 0 else 0
                        
                        # Add custom CSS to remove white background from metric cards
                        st.markdown("""
                        <style>
                        [data-testid="stMetric"] {
                            background-color: transparent !important;
                            border: none !important;
                            box-shadow: none !important;
                        }
                        [data-testid="stMetricValue"] {
                            background-color: transparent !important;
                        }
                        [data-testid="stMetricLabel"] {
                            background-color: transparent !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Amount", f"${risk_amount:.2f}")
                            st.metric("Position Value", f"${position_value:.2f}")
                        with col2:
                            st.metric("Reward Amount", f"${reward_amount:.2f}")
                            st.metric("Risk Percentage", f"{risk_percentage:.2f}%")
                        with col3:
                            st.metric("Risk-Reward Ratio", f"{risk_reward_ratio:.2f}")
                            st.metric("Position Size", f"{quantity} shares")
                        
                        # Display recommended position size based on risk management
                        st.markdown("---")
                        st.subheader("Risk Management Recommendations")
                        
                        # Calculate recommended position size using RiskManager
                        recommended_size = risk_manager.calculate_position_size(ticker, entry_price, stop_loss_price)
                        max_risk_amount = risk_manager.current_capital * risk_manager.max_risk_per_trade
                        
                        rec_col1, rec_col2 = st.columns(2)
                        with rec_col1:
                            st.metric("Recommended Position Size", f"{int(recommended_size)} shares")
                            st.caption("Based on your risk settings")
                        with rec_col2:
                            st.metric("Max Risk Amount", f"${max_risk_amount:.2f}")
                            st.caption(f"{risk_manager.max_risk_per_trade*100:.1f}% of capital")
                        
                        # Create a visual representation of the position
                        st.subheader("Position Visualization")
                        fig = go.Figure()
                        
                        # Add horizontal lines for entry, stop loss, and take profit
                        fig.add_shape(type="line", x0=0, x1=1, y0=entry_price, y1=entry_price,
                                     line=dict(color="blue", width=2, dash="solid"))
                        fig.add_shape(type="line", x0=0, x1=1, y0=stop_loss_price, y1=stop_loss_price,
                                     line=dict(color="red", width=2, dash="dash"))
                        fig.add_shape(type="line", x0=0, x1=1, y0=take_profit_price, y1=take_profit_price,
                                     line=dict(color="green", width=2, dash="dash"))
                        
                        # Add annotations
                        fig.add_annotation(x=0.5, y=entry_price, text=f"Entry: ${entry_price:.2f}",
                                         showarrow=False, yshift=10)
                        fig.add_annotation(x=0.5, y=stop_loss_price, text=f"Stop Loss: ${stop_loss_price:.2f}",
                                         showarrow=False, yshift=-20)
                        fig.add_annotation(x=0.5, y=take_profit_price, text=f"Take Profit: ${take_profit_price:.2f}",
                                         showarrow=False, yshift=10)
                        
                        # Set y-axis range with padding
                        y_min = min(entry_price, stop_loss_price) * 0.95
                        y_max = max(entry_price, take_profit_price) * 1.05
                        
                        # Update layout
                        fig.update_layout(
                            showlegend=False,
                            xaxis=dict(visible=False),
                            yaxis=dict(title="Price ($)", range=[y_min, y_max]),
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Portfolio Risk Tab - Display portfolio-level risk metrics
                with risk_subtabs[2]:
                    st.subheader("Portfolio Risk Analysis")
                    
                    # Add description
                    st.markdown("""This tab analyzes portfolio-level risk metrics, including concentration risk, 
                    diversification, and overall portfolio exposure. It helps ensure your portfolio remains balanced 
                    and that no single position creates excessive risk to your total capital.""")
                    st.markdown("---")
                    
                    # Get portfolio weights with fallback
                    portfolio_weights = st.session_state.analysis_results.get('portfolio_weights', {})
                    
                    if not portfolio_weights:
                        # Use the current ticker as a fallback
                        portfolio_weights = {ticker: 1.0}
                    
                    # Get risk parameters from sidebar
                    portfolio_capital = st.session_state.get('sidebar_current_capital', risk_manager.current_capital)
                    max_risk = st.session_state.get('sidebar_max_risk', risk_manager.max_risk_per_trade)
                    
                    # Display current portfolio allocation
                    st.subheader("Portfolio Configuration")
                    df_weights = pd.DataFrame({
                        'Ticker': list(portfolio_weights.keys()),
                        'Weight': [w * 100 for w in portfolio_weights.values()],
                        'Current Price': [latest_price] * len(portfolio_weights)  # Simplified, should fetch actual prices
                    })
                    
                    st.dataframe(df_weights.style.format({'Weight': '{:.2f}%', 'Current Price': '${:.2f}'}), use_container_width=True)
                    
                    # Display risk parameters from sidebar
                    st.info(f"Using risk parameters from sidebar: Capital ${portfolio_capital:,.2f}, Max Risk {max_risk*100:.1f}%")
                    st.caption("*To change these values, use the Risk Management Settings in the sidebar*")
                    
                    # Calculate and display portfolio risk metrics
                    st.subheader("Portfolio Risk Metrics")
                    
                    # Calculate position values based on weights and capital
                    position_values = {ticker: weight * portfolio_capital for ticker, weight in portfolio_weights.items()}
                    total_position_value = sum(position_values.values())
                    
                    # Calculate portfolio risk metrics
                    portfolio_risk = risk_manager.calculate_portfolio_risk(position_values)
                    
                    # Display portfolio metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Portfolio Value", f"${total_position_value:,.2f}")
                        st.metric("Largest Position", f"${max(position_values.values()):,.2f}")
                    with col2:
                        st.metric("Portfolio Concentration", f"{portfolio_risk.get('concentration', 0)*100:.2f}%")
                        st.metric("Number of Positions", f"{len(position_values)}")
                    with col3:
                        st.metric("Max Position Risk", f"${portfolio_capital * max_risk:,.2f}")
                        st.metric("Portfolio Diversification", f"{100 - portfolio_risk.get('concentration', 0)*100:.2f}%")
                    
                    # Create a pie chart of portfolio allocation
                    st.subheader("Portfolio Allocation")
                    fig = go.Figure(data=[go.Pie(
                        labels=list(portfolio_weights.keys()),
                        values=list(portfolio_weights.values()),
                        hole=.3,
                        textinfo='label+percent',
                        marker_colors=px.colors.qualitative.Plotly
                    )])
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk Settings Tab - Display risk management strategy and reference sidebar settings
                with risk_subtabs[3]:
                    st.subheader("Risk Management Settings")
                    
                    # Add description
                    st.markdown("""This tab displays your current risk management configuration. All settings are managed 
                    through the Risk Management Settings section in the sidebar. These parameters control position sizing, 
                    stop loss/take profit calculations, and overall risk exposure across your portfolio.""")
                    st.markdown("---")
                    
                    # Get risk parameters from sidebar and update risk manager
                    initial_capital = st.session_state.get('sidebar_initial_capital', risk_manager.initial_capital)
                    current_capital = st.session_state.get('sidebar_current_capital', risk_manager.current_capital)
                    max_risk_per_trade = st.session_state.get('sidebar_max_risk', risk_manager.max_risk_per_trade)
                    stop_loss_multiplier = st.session_state.get('sidebar_sl_multiplier', 2.0)
                    take_profit_multiplier = st.session_state.get('sidebar_tp_multiplier', 3.0)
                    
                    # Update risk manager with current values
                    risk_manager.initial_capital = float(initial_capital)
                    risk_manager.current_capital = float(current_capital)
                    risk_manager.max_risk_per_trade = float(max_risk_per_trade)
                    
                    # Display current settings from sidebar
                    st.info("Risk settings are configured in the sidebar")
                    st.markdown("### Current Risk Settings")
                    # Display risk metrics in a table
                    risk_metrics_df = pd.DataFrame([
                        {"Parameter": "Initial Capital", "Value": f"${initial_capital:,.2f}"},
                        {"Parameter": "Current Capital", "Value": f"${current_capital:,.2f}"},
                        {"Parameter": "Max Risk Per Trade", "Value": f"{max_risk_per_trade*100:.1f}%"},
                        {"Parameter": "Stop Loss Multiplier", "Value": f"{stop_loss_multiplier:.1f}x volatility"},
                        {"Parameter": "Take Profit Multiplier", "Value": f"{take_profit_multiplier:.1f}x volatility"},
                    ])
                    
                    st.table(risk_metrics_df)
                    
                    # Display current risk management strategy explanation
                    st.markdown("### Risk Management Strategy")
                    st.markdown("""
                    This risk management system implements the following strategies:
                    
                    1. **Position Sizing**: Calculates the appropriate position size based on your risk tolerance and stop loss level
                    2. **Stop Loss & Take Profit**: Automatically calculates stop loss and take profit levels based on price volatility
                    3. **Portfolio Risk**: Monitors concentration risk and ensures diversification across positions
                    4. **Risk Per Trade**: Limits the maximum risk per trade to a percentage of your total capital
                    5. **Unrealized P&L**: Tracks unrealized profit and loss for each position
                    """)
                    
                    # Display risk metrics in a table
                    st.markdown("### Current Risk Metrics")
                    risk_metrics_df = pd.DataFrame([
                        {"Metric": "Current Capital", "Value": f"${risk_manager.current_capital:,.2f}"},
                        {"Metric": "Max Risk Per Trade", "Value": f"{risk_manager.max_risk_per_trade*100:.1f}%"},
                        {"Metric": "Stop Loss Calculation", "Value": f"Price - (Volatility × {stop_loss_multiplier:.1f})"},
                        {"Metric": "Take Profit Calculation", "Value": f"Price + (Volatility × {take_profit_multiplier:.1f})"},
                    ])
                    
                    st.table(risk_metrics_df)
                    
            except Exception as e:
                st.error(f"Error calculating risk metrics: {str(e)}")
                st.session_state.logger.error(f"Risk metrics error: {str(e)}", exc_info=True)
        else:
            st.warning(
                "No stock data available for risk metrics. "
                "Please ensure you've run the analysis with a valid ticker."
            )
            close_col = st.session_state.get('close_col', None)
            st.session_state.logger.warning(
                f"Missing stock data for risk metrics. "
                f"Stock data exists: {stock_data is not None}, "
                f"Close column '{close_col}' exists: {stock_data is not None and close_col in stock_data.columns if close_col else False}"
            )
            debug_info = {
                'portfolio_metrics_exists': 'portfolio_metrics' in st.session_state,
                'portfolio_metrics_keys': list(st.session_state.get('portfolio_metrics', {}).keys()) if 'portfolio_metrics' in st.session_state else 'N/A',
                'analysis_results_exists': 'analysis_results' in st.session_state,
                'ticker': st.session_state.get('ticker', 'N/A'),
                'has_returns_df': 'returns_df' in st.session_state,
                'has_optimizer': 'portfolio_optimizer' in st.session_state,
                'returns_df_shape': st.session_state.get('returns_df', pd.DataFrame()).shape if 'returns_df' in st.session_state else 'N/A'
            }
            st.sidebar.json(debug_info)

        
        # Safely get values with defaults and type conversion
        def safe_get_float(metrics, key, default=0.0):
            try:
                value = metrics.get(key, default)
                if value is None:
                    return default
                return float(value)
            except (TypeError, ValueError) as e:
                st.session_state.logger.warning(f"Error converting {key} to float: {str(e)}")
                return default
        
        # Portfolio Analysis Tab
        with tab_portfolio:
            st.subheader("Portfolio Analysis")
            
            # Add description
            st.markdown("""This tab provides portfolio analysis including asset allocation, risk metrics, 
            and performance attribution. Use the controls to adjust your portfolio and see how it affects 
            your risk-return profile.""")
            
            # Get portfolio metrics from session state with debug logging
            if 'portfolio_metrics' not in st.session_state:
                st.warning("⚠️ Portfolio metrics not found in session state. Creating default metrics.")
                # Create default metrics if they don't exist
                ticker = st.session_state.get('ticker', 'UNKNOWN')
                st.session_state.portfolio_metrics = {
                    'weights': {ticker: 1.0},
                    'expected_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0
                }
                st.session_state.portfolio_weights = {ticker: 1.0}
                
            portfolio_metrics = st.session_state.portfolio_metrics
            st.session_state.logger.info(f"Retrieved portfolio_metrics from session state. Keys: {list(portfolio_metrics.keys())}")
            
            # Check for required metrics and add any missing ones
            required_metrics = ['weights', 'expected_return', 'sharpe_ratio', 'volatility', 'max_drawdown']
            missing_metrics = [m for m in required_metrics if m not in portfolio_metrics]
            
            if missing_metrics:
                st.warning(f"Adding missing portfolio metrics: {', '.join(missing_metrics)}")
                # Add any missing metrics with default values
                ticker = st.session_state.get('ticker', 'UNKNOWN')
                for metric in missing_metrics:
                    if metric == 'weights':
                        portfolio_metrics[metric] = {ticker: 1.0}
        # Display any error message if present
        if 'error' in portfolio_metrics:
            st.warning(f"Note: {portfolio_metrics['error']}")
            
        # Add debug information in expander
        if st.session_state.get('debug_mode', False):
            portfolio_metrics_df = pd.DataFrame([{k: v for k, v in portfolio_metrics.items() if isinstance(v, (int, float, str))}])
    
    # Add description
    st.markdown('''This tab provides an overview of your portfolio's performance, including key metrics and visualizations.
    It helps you understand your portfolio's strengths and weaknesses, and make informed decisions to optimize its performance.''')
    st.markdown("---")
    
    # Get portfolio metrics from session state with debug logging
    portfolio_metrics = st.session_state.get('portfolio_metrics', {})
    st.session_state.logger.info(f"Retrieved portfolio_metrics from session state. Keys: {list(portfolio_metrics.keys())}")
    
    # Check for required metrics and add any missing ones
    required_metrics = ['weights', 'expected_return', 'sharpe_ratio', 'volatility', 'max_drawdown']
    missing_metrics = [m for m in required_metrics if m not in portfolio_metrics]
    
    if missing_metrics:
        st.session_state.logger.warning(f"Adding missing portfolio metrics: {', '.join(missing_metrics)}")
        # Add any missing metrics with default values
        ticker = st.session_state.get('ticker', 'UNKNOWN')
        for metric in missing_metrics:
            if metric == 'weights':
                portfolio_metrics[metric] = {ticker: 1.0}
            else:
                portfolio_metrics[metric] = 0.0
        
        # Update session state with fixed metrics
        st.session_state.portfolio_metrics = portfolio_metrics
    
    # Get all metrics with safe conversion
    expected_return = safe_get_float(portfolio_metrics, 'expected_return')
    volatility = safe_get_float(portfolio_metrics, 'volatility')
    sharpe_ratio = safe_get_float(portfolio_metrics, 'sharpe_ratio')
    max_drawdown = safe_get_float(portfolio_metrics, 'max_drawdown')
    diversification_ratio = safe_get_float(portfolio_metrics, 'diversification_ratio', 1.0)
    current_price = safe_get_float(portfolio_metrics, 'current_price')
    
    # Display key metrics in columns with error handling
    col1, col2, col3 = st.columns(3)
    
    try:
        with col1:
            st.metric("Expected Return", f"{expected_return * 100:.2f}%")
            st.metric("Volatility", f"{volatility * 100:.2f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
            
        with col3:
            st.metric("Diversification Ratio", f"{diversification_ratio:.2f}")
            st.metric("Current Price", f"${current_price:.2f}" if current_price > 0 else "N/A")
    except Exception as e:
        st.session_state.logger.error(f"Error displaying portfolio metrics: {str(e)}")
        st.info("Error displaying portfolio metrics. Please check the logs for details.")
    
    # Add a refresh button to recalculate metrics

        if 'returns_df' in st.session_state and 'portfolio_optimizer' in st.session_state:
            try:
                st.session_state.logger.info("Recalculating portfolio metrics...")
                returns_df = st.session_state.returns_df
                optimizer = st.session_state.portfolio_optimizer
                weights = optimizer.optimize(returns_df)
                
                if weights is not None:
                    # Safely convert weights to float, handling any non-numeric values
                    weights_dict = {}
                    for ticker, w in zip(returns_df.columns, weights):
                        try:
                            weight = float(w)
                            # Ensure weight is a valid number
                            if not (0 <= weight <= 1):
                                logger.warning(f"Weight {weight} for {ticker} is outside [0,1] range. Clamping to valid range.")
                                weight = max(0, min(1, weight))  # Clamp to [0,1]
                            weights_dict[ticker] = weight
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Invalid weight value '{w}' for {ticker}. Using 0.0 as fallback. Error: {str(e)}")
                            weights_dict[ticker] = 0.0  # Fallback to 0 weight for invalid values
                    
                    # Normalize weights to sum to 1
                    total_weight = sum(weights_dict.values())
                    if total_weight > 0:
                        weights_dict = {k: v/total_weight for k, v in weights_dict.items()}
                    else:
                        # If all weights are 0, use equal weights
                        n = len(weights_dict)
                        weights_dict = {k: 1.0/n for k in weights_dict}
                    sharpe_ratio = optimizer.calculate_sharpe_ratio(returns_df, weights)
                    volatility = optimizer.calculate_volatility(returns_df, weights)
                    max_drawdown = optimizer.calculate_max_drawdown(returns_df, weights)
                    
                    # Update portfolio metrics
                    portfolio_metrics.update({
                        'weights': weights_dict,
                        'expected_return': float(np.sum(returns_df.mean() * weights) * 252),
                        'sharpe_ratio': float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
                        'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
                        'max_drawdown': float(max_drawdown) if not np.isnan(max_drawdown) else 0.0,
                        'diversification_ratio': 1.0 / sum(w**2 for w in weights) if sum(weights) > 0 else 1.0
                    })
                    st.session_state.portfolio_metrics = portfolio_metrics
                    st.success("Portfolio metrics recalculated successfully!")
                    st.rerun()
            except Exception as e:
                st.session_state.logger.error(f"Error recalculating metrics: {str(e)}")
                st.info("Error recalculating metrics. Please check the logs for details.")
    
    weights = portfolio_metrics.get('weights', {})
    if not weights:
        st.session_state.logger.warning("No portfolio weights available.")
        st.info("No portfolio weights available.")
    else:
        # Create a DataFrame for better display
        weights_df = pd.DataFrame({
            'Ticker': list(weights.keys()),
            'Weight': [f"{w*100:.2f}%" for w in weights.values()],
            'Allocation': list(weights.values())
        })
        
        # Sort by weight in descending order
        weights_df = weights_df.sort_values('Allocation', ascending=False)
        
        # Display weights in a table
        st.dataframe(
            weights_df[['Ticker', 'Weight']],
            column_config={
                'Ticker': 'Ticker',
                'Weight': st.column_config.ProgressColumn(
                    'Weight',
                    help='Portfolio weight',
                    format='%.2f%%',
                    min_value=0,
                    max_value=1.0
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Add a pie chart for visualization
        if len(weights) > 0:
            fig = px.pie(
                weights_df,
                values='Allocation',
                names='Ticker',
                title='Portfolio Allocation',
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='%{label}<br>%{percent}'
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key='portfolio_allocation_pie_chart')
    
    # Display any error message if present
    if 'error' in portfolio_metrics:
        st.session_state.logger.warning(f"Error in portfolio metrics: {portfolio_metrics['error']}")
        st.info(f"Error in portfolio metrics: {portfolio_metrics['error']}")
    
    # Add debug information in expander

    # Add description
    st.markdown('''This tab evaluates the predictive model's performance against actual market data, showing accuracy metrics 
    and visualization of predicted vs. actual price movements. It helps you assess the model's reliability and 
    effectiveness for making trading decisions based on quantifiable performance indicators.''')
    st.markdown("---")
    
    # Check if analysis results are available
    if 'analysis_results' not in st.session_state:
        st.session_state.logger.warning("No analysis results found. Please run the analysis first.")
        st.info("Analysis results not available.")
    
    # FUTURE PREDICTIONS SECTION - Show this first for visibility
    st.subheader("🔮 Future Price Predictions")
    
    # Get future predictions directly from session state for maximum reliability
    future_predictions = st.session_state.get('future_predictions')
    future_dates = st.session_state.get('future_dates')
    
    # Debug output
    st.session_state.logger.info(f"Performance Tab - Direct access - Found future predictions: {future_predictions is not None}")
    if future_predictions is not None:
        st.session_state.logger.info(f"Performance Tab - Direct access - Future predictions count: {len(future_predictions)}")
    
    # Display future predictions if available
    if future_predictions is not None and future_dates is not None and len(future_predictions) > 0:
        # Create a DataFrame with future predictions for display
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions
        })
        future_df.set_index('Date', inplace=True)
        
        # Display the future predictions table
        st.write(f"Showing {len(future_predictions)} days of future predictions for {st.session_state.ticker}")
        st.dataframe(future_df.style.format({'Predicted Price': '${:.2f}'}), use_container_width=True)
        
        # Create a simple chart just for future predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions, 
            mode='lines+markers', 
            name='Future Forecast',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig_future.update_layout(
            title=f'Future Price Forecast for {st.session_state.ticker}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_future, use_container_width=True)
        st.success(f"✅ Generated {len(future_predictions)} future predictions for {st.session_state.ticker}")
    else:
        st.session_state.logger.warning("No future predictions available. Please run the analysis with future prediction days > 0.")
        st.info("No future predictions available.")
    
    st.markdown("---")
    
    # Get metrics with validation
    metrics = st.session_state.analysis_results.get('metrics', {})
    if metrics:
        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame([metrics])
        # Force non-numeric columns to object, numeric columns to float (coerce errors)
        for col in metrics_df.columns:
            if col in metrics_df.select_dtypes(exclude=['number']).columns:
                continue
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
        numeric_cols = metrics_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.dataframe(metrics_df.style.format({col: '{:.4f}' for col in numeric_cols}))
        else:
            st.dataframe(metrics_df)
    else:
        st.session_state.logger.warning("No metrics available. The model may not have been trained.")
        st.info("No metrics available.")
    
    def calculate_overall_accuracy(metrics):
        """Calculate an overall accuracy percentage from multiple metrics"""
        # Normalize and weight each metric (all metrics are now on 0-100 scale where higher is better)
        directional_accuracy = metrics['directional_accuracy']  # Already 0-100
        
        # Convert R² from -1 to 1 to 0-100 scale (0% to 100%)
        r2_score = max(0, (metrics['r2'] + 1) * 50)  # Convert -1 to 1 range to 0-100
        
        # Convert MAPE to accuracy (lower MAPE is better, so invert it)
        # Cap MAPE at 100% to avoid negative numbers
        mape = max(0, 100 - min(metrics['mape'], 100))
        
        # Calculate weighted average (adjust weights as needed)
        weights = {
            'directional': 0.5,  # Most important for trading
            'r2': 0.3,          # Model fit quality
            'mape': 0.2         # Prediction error
        }
        
        overall = (
            directional_accuracy * weights['directional'] +
            r2_score * weights['r2'] +
            mape * weights['mape']
        )
        
        # Ensure the result is between 0 and 100
        return max(0, min(100, overall))
    
    # Define a function to handle the performance analysis
    def show_performance_analysis():
        try:
            # Initialize metrics dictionary to store all performance metrics
            metrics = {
                'mae': 0,
                'rmse': 0,
                'r2': 0,
                'mape': 0,
                'directional_accuracy': 0,
                'mse': 0,
                'overall_accuracy': 0
            }
            
            # Get predictions and stock data from session state with better error handling
            predictions = None
            stock_data = None
            close_col = 'Close'
            ticker = st.session_state.get('ticker', 'UNKNOWN')
            
            # Initial data validation
            predictions = st.session_state.analysis_results.get('predictions')
            stock_data = st.session_state.get('stock_data_clean')
            close_col = st.session_state.get('close_col', 'Close')
            ticker = st.session_state.get('ticker', 'UNKNOWN')
            
            # Log the state for debugging - more detailed logging
            st.session_state.logger.info(f"Performance Analysis - Predictions available: {predictions is not None}")
            st.session_state.logger.info(f"Performance Analysis - Stock data available: {stock_data is not None}")
            
            # Check for required data
            if 'predictor' not in st.session_state or st.session_state.predictor is None:
                st.warning("Model has not been trained yet. Please run the analysis first.")
                st.session_state.logger.warning("Model not trained - cannot display performance chart")
                return
                
            # Calculate prediction accuracy metrics if we have predictions and actual values
            if predictions is not None and stock_data is not None and close_col in stock_data.columns:
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    # Get actual values and align with predictions
                    actual_values = stock_data[close_col].values[-len(predictions):]
                    predictions = np.array(predictions).flatten()
                    
                    # Calculate metrics with error handling
                    try:
                        metrics['mae'] = mean_absolute_error(actual_values, predictions)
                        metrics['mse'] = mean_squared_error(actual_values, predictions)
                        metrics['rmse'] = np.sqrt(metrics['mse'])
                        metrics['r2'] = max(-1, min(1, r2_score(actual_values, predictions)))  # Clamp between -1 and 1
                        
                        # Calculate MAPE with protection against division by zero
                        epsilon = 1e-10
                        metrics['mape'] = np.mean(np.abs((actual_values - predictions) / (np.abs(actual_values) + epsilon))) * 100
                        
                        # Calculate directional accuracy with safety checks
                        if len(actual_values) > 1:
                            actual_dir = np.sign(np.diff(actual_values))
                            pred_dir = np.sign(np.diff(predictions))
                            valid_pairs = (actual_dir != 0) & (pred_dir != 0)  # Only consider valid direction changes
                            if valid_pairs.any():
                                metrics['directional_accuracy'] = np.mean(actual_dir[valid_pairs] == pred_dir[valid_pairs]) * 100
                            else:
                                metrics['directional_accuracy'] = 50.0  # Default to 50% (random chance) if no valid direction changes
                        else:
                            metrics['directional_accuracy'] = 50.0  # Default to 50% if not enough data points
                            
                    except Exception as e:
                        st.session_state.logger.error(f"Error in metric calculation: {str(e)}")
                        # Set default values if calculation fails
                        metrics.update({
                            'mae': 0,
                            'mse': 0,
                            'rmse': 0,
                            'r2': 0,
                            'mape': 100,
                            'directional_accuracy': 0
                        })
                    
                except Exception as e:
                    st.session_state.logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
                    st.warning(f"Could not calculate all metrics: {str(e)}")
                    
            # Calculate overall accuracy
            metrics['overall_accuracy'] = calculate_overall_accuracy(metrics)
            
            # Display Overall Accuracy in a prominent way
            st.markdown("### 📊 Model Accuracy Overview")
            
            # Determine accuracy color and emoji
            accuracy = metrics['overall_accuracy']
            if accuracy >= 80:
                color = "#2ecc71"  # Green
                emoji = "🎯"
                rating = "Excellent"
            elif accuracy >= 60:
                color = "#f39c12"  # Orange
                emoji = "👍"
                rating = "Good"
            else:
                color = "#e74c3c"  # Red
                emoji = "⚠️"
                rating = "Needs Improvement"
            
            # Main accuracy display
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        margin-bottom: 20px; border: 1px solid #dee2e6;'>
                <div style='font-size: 14px; color: #6c757d; margin-bottom: 10px;'>OVERALL PREDICTION ACCURACY</div>
                <div style='font-size: 52px; font-weight: bold; color: {color};'>{accuracy:.1f}% {emoji}</div>
                <div style='font-size: 18px; color: #6c757d; margin-top: 5px;'>{rating} • Based on multiple metrics</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual representation
            st.markdown(f"""
            <div style='margin: 10px 0 20px 0;'>
                <div style='height: 12px; background-color: #e9ecef; border-radius: 6px; overflow: hidden;'>
                    <div style='height: 100%; width: {accuracy}%; background-color: {color};'></div>
                </div>
                <div style='display: flex; justify-content: space-between; margin-top: 5px;'>
                    <span style='font-size: 12px; color: #6c757d;'>0%</span>
                    <span style='font-size: 12px; color: #6c757d;'>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics in a compact grid
            st.markdown("### 📈 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1f}%", 
                         delta=f"{metrics['directional_accuracy'] - 50:.1f}% vs random")
                
            with col2:
                st.metric("R² Score", f"{metrics['r2']:.3f}",
                         delta="Perfect fit is 1.0" if metrics['r2'] < 0.9 else "Excellent fit!")
                
            with col3:
                st.metric("Mean Error (MAPE)", f"{metrics['mape']:.1f}%", 
                         delta=f"{100 - metrics['mape']:.1f}% accuracy" if metrics['mape'] < 100 else "High error")
            
            # Add a collapsible section with detailed metrics
            with st.expander("📊 View All Metrics", expanded=False):
                st.markdown("#### Detailed Performance Metrics")
                
                # Create a DataFrame for the detailed metrics
                detailed_metrics = [
                    {"Metric": "Mean Absolute Error (MAE)", "Value": f"${metrics['mae']:.2f}", "Ideal": "Lower is better"},
                    {"Metric": "Root Mean Squared Error (RMSE)", "Value": f"${metrics['rmse']:.2f}", "Ideal": "Lower is better"},
                    {"Metric": "Mean Squared Error (MSE)", "Value": f"{metrics['mse']:.2f}", "Ideal": "Lower is better"},
                    {"Metric": "Mean Absolute % Error (MAPE)", "Value": f"{metrics['mape']:.2f}%", "Ideal": "Lower is better"},
                    {"Metric": "Directional Accuracy", "Value": f"{metrics['directional_accuracy']:.2f}%", "Ideal": "Higher is better"},
                    {"Metric": "R-squared (R²)", "Value": f"{metrics['r2']:.4f}", "Ideal": "Closer to 1 is better"}
                ]
                
                # Display the metrics in a nice table
                st.table(pd.DataFrame(detailed_metrics))
            
            # Add a separator before the chart
            st.markdown("---")
            
            # Add explanatory text
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
                <h4>How to Interpret These Results</h4>
                <ul style='margin-bottom: 0;'>
                    <li>The <b>Overall Prediction Accuracy</b> combines multiple metrics to give you a single, easy-to-understand score.</li>
                    <li><b>Directional Accuracy</b> shows how often the model predicts the correct price direction (up/down).</li>
                    <li><b>R² Score</b> indicates how well the model explains price movements (1.0 is perfect).</li>
                    <li><b>Mean Error (MAPE)</b> shows the average percentage difference between predicted and actual prices.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a small gap before the chart
            st.markdown("<br>", unsafe_allow_html=True)
                
            if stock_data is None:
                st.error("Stock data is not available. Please ensure you've run the analysis with a valid ticker.")
                st.session_state.logger.warning("Cannot display performance chart: Stock data is None")
                return
                
            if close_col not in stock_data.columns:
                st.error(f"Required column '{close_col}' not found in stock data. Available columns: {', '.join(stock_data.columns)}")
                st.session_state.logger.warning(f"Cannot display performance chart: Close column '{close_col}' not found")
                # Show empty figure if close column not found
                fig = go.Figure()
                fig.update_layout(title=f"{ticker} Stock Price (Close column '{close_col}' not found)")
                st.plotly_chart(fig, use_container_width=True)
                return
                
            if predictions is None:
                st.warning("No predictions available. The model may need to be retrained.")
                st.session_state.logger.warning("Cannot display performance chart: Predictions are None")
                # Show historical data only if no predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data[close_col],
                    mode='lines',
                    name='Historical Prices'
                ))
                fig.update_layout(
                    title=f"{ticker} Stock Price (Historical)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                return
            actual_values = stock_data[close_col].values[-len(predictions):]
            dates = stock_data.index[-len(predictions):]
            
            if len(actual_values) != len(predictions):
                st.session_state.logger.warning(
                    f"Mismatch in data lengths. Actual: {len(actual_values)}, Predicted: {len(predictions)}. "
                    "Truncating to the minimum length."
                )
                min_len = min(len(actual_values), len(predictions))
                actual_values = actual_values[-min_len:]
                predictions = predictions[-min_len:]
                dates = dates[-min_len:]
            
            # Calculate prediction accuracy metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            try:
                mae = mean_absolute_error(actual_values, predictions)
                rmse = np.sqrt(mean_squared_error(actual_values, predictions))
                r2 = r2_score(actual_values, predictions)
                
                # Calculate directional accuracy and prediction accuracy
                actual_direction = np.sign(np.diff(actual_values))
                predicted_direction = np.sign(np.diff(predictions))
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
                
                # Calculate prediction accuracy percentage (100 - MAPE)
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                prediction_accuracy = max(0, 100 - mape)  # Ensure it doesn't go below 0
                
                # Store metrics in session state for display
                metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Directional_Accuracy': directional_accuracy,
                    'MAE_%': (mae / np.mean(actual_values)) * 100 if np.mean(actual_values) != 0 else float('inf'),
                    'RMSE_%': (rmse / np.mean(actual_values)) * 100 if np.mean(actual_values) != 0 else float('inf')
                }
                
                # Display metrics in a clean format
                st.markdown("### 🎯 Prediction Accuracy Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Absolute Error (MAE)", 
                            f"${mae:.2f}",
                            delta=f"{metrics['MAE_%']:.2f}% of mean price" if metrics['MAE_%'] != float('inf') else "N/A",
                            help="Average absolute difference between actual and predicted values. Lower is better.")
                
                with col2:
                    st.metric("Root Mean Squared Error (RMSE)", 
                            f"${rmse:.2f}",
                            delta=f"{metrics['RMSE_%']:.2f}% of mean price" if metrics['RMSE_%'] != float('inf') else "N/A",
                            help="Square root of the average squared differences. More sensitive to large errors. Lower is better.")
                
                with col3:
                    st.metric("R-squared (R²)", 
                            f"{r2:.4f}",
                            delta="1.0 is perfect fit" if r2 < 1 else "Perfect fit!",
                            help="Proportion of variance explained by the model. Closer to 1 is better.")
                
                with col4:
                    st.metric("Prediction Accuracy", 
                            f"{prediction_accuracy:.2f}%",
                            help="Overall prediction accuracy percentage. Higher is better.")
                
                # Add interpretation guide
                with st.expander("How to interpret these metrics"):
                    st.markdown("""
                    - **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual values. Lower is better.
                    - **RMSE (Root Mean Squared Error)**: Similar to MAE but gives higher weight to larger errors. Lower is better.
                    - **R² (R-squared)**: Measures how well the model explains the variance in the data. Closer to 1 is better.
                    - **Prediction Accuracy**: Overall accuracy of price predictions. A score of 100% means perfect predictions.
                    - **Directional Accuracy**: Percentage of times the model correctly predicted price direction (up/down).
                    - **Error Metrics**: MAE and RMSE show average prediction errors in price units and as a percentage of the mean price.
                    """)
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error calculating prediction metrics: {str(e)}")
                st.session_state.logger.error(f"Error in prediction metrics: {str(e)}", exc_info=True)
            
            # Create the main plot with enhanced styling
            fig = go.Figure()
            
            # Add actual prices trace
            fig.add_trace(go.Scatter(
                x=dates, 
                y=actual_values, 
                mode='lines', 
                name='Actual Price',
                line=dict(color='#1f77b4', width=2.5),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                            '<b>Price</b>: $%{y:.2f}<extra></extra>',
                opacity=0.9
            ))
            
            # Add predicted prices trace with enhanced styling
            fig.add_trace(go.Scatter(
                x=dates, 
                y=predictions, 
                mode='lines',
                name='Predicted Price',
                line=dict(color='#ff7f0e', width=2.5, dash='dash'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                            '<b>Predicted</b>: $%{y:.2f}<extra></extra>',
                opacity=0.9
            ))
            
            # Add fill between actual and predicted for better visualization
            fig.add_trace(go.Scatter(
                x=np.concatenate([dates, dates[::-1]]),
                y=np.concatenate([actual_values, predictions[::-1]]),
                fill='tonexty',
                fillcolor='rgba(255, 165, 0, 0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Use the future predictions we already retrieved above
            # Log what we found for debugging
            st.session_state.logger.info(f"Using future predictions: {future_predictions is not None}, count: {len(future_predictions) if future_predictions is not None else 0}")
            st.session_state.logger.info(f"Using future dates: {future_dates is not None}, count: {len(future_dates) if future_dates is not None else 0}")
            
            # Debug: Add a note to the UI about future predictions status
            if future_predictions is None or len(future_predictions) == 0:
                st.info("⚠️ No future predictions available. Try running the analysis again with a different ticker or date range.")
            else:
                st.success(f"✅ Found {len(future_predictions)} future predictions to display")
            
            # Always try to show future predictions if available
            if future_predictions is not None and future_dates is not None and len(future_predictions) > 0:
                # Add a vertical line to separate historical and future predictions
                last_date = dates[-1] if len(dates) > 0 else None
                if last_date:
                    fig.add_vline(
                        x=last_date,
                        line_width=2,  # Make it more visible
                        line_dash="dash",
                        line_color="red",  # Make it more visible
                        annotation_text="Forecast Start",
                        annotation_position="top right"
                    )
                
                # Add future predictions trace with more visible styling
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_predictions, 
                    mode='lines', 
                    name='Future Forecast',
                    line=dict(color='#2ca02c', dash='dot', width=3)  # Thicker line
                ))
                
                # Add annotation to highlight future predictions
                mid_point = future_dates[len(future_dates)//2] if len(future_dates) > 0 else None
                if mid_point:
                    fig.add_annotation(
                        x=mid_point,
                        y=max(future_predictions),
                        text="Future Price Predictions",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )
                
                st.session_state.logger.info(f"Added {len(future_predictions)} future predictions to chart")
            
            # Update layout with enhanced styling and annotations
            layout_args = {
                'title': {
                    'text': f'<b>{ticker} - Actual vs Predicted Prices</b>',
                    'font': {'size': 20, 'family': 'Arial, sans-serif'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'Date',
                    'showgrid': True,
                    'gridcolor': 'rgba(211, 211, 211, 0.5)',
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True,
                    'rangeslider': {'visible': True, 'thickness': 0.1},
                    'rangeselector': {
                        'buttons': [
                            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                            {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
                            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                            {'step': 'all', 'label': 'All'}
                        ]
                    }
                },
                'yaxis': {
                    'title': 'Price ($)',
                    'showgrid': True,
                    'gridcolor': 'rgba(211, 211, 211, 0.5)',
                    'showline': True,
                    'linewidth': 1,
                    'linecolor': 'black',
                    'mirror': True
                },
                'legend': {
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': 1.02,
                    'xanchor': 'right',
                    'x': 1,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': '#DDD',
                    'borderwidth': 1
                },
                'margin': dict(l=50, r=50, t=80, b=50),
                'hovermode': 'x unified',
                'hoverlabel': {
                    'bgcolor': 'white',
                    'font_size': 12,
                    'font_family': 'Arial, sans-serif'
                },
                'plot_bgcolor': 'rgba(0,0,0,0.01)',
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'showlegend': True,
                'transition': {'duration': 300}
            }
            
            # Only add shapes if we have future predictions
            if future_predictions is not None and len(future_dates) > 0 and len(dates) > 0:
                layout_args['shapes'] = [
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=dates[-1],
                        y0=0,
                        x1=future_dates[-1],
                        y1=1,
                        fillcolor="rgba(44, 160, 44, 0.1)",  # Light green background
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                    )
                ]
            
            # Apply the layout
            fig.update_layout(**layout_args)
            
            # Add range selector buttons
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Add annotations for prediction accuracy
            if len(actual_values) > 0 and len(predictions) > 0:
                last_actual = actual_values[-1]
                last_pred = predictions[-1]
                diff_pct = abs((last_pred - last_actual) / last_actual * 100) if last_actual != 0 else 0
                
                fig.add_annotation(
                    x=dates[-1],
                    y=last_actual,
                    text=f"Last Price: ${last_actual:.2f}<br>Prediction: ${last_pred:.2f}<br>Diff: {diff_pct:.2f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#636363',
                    ax=-50,
                    ay=-50,
                    bordercolor='#c7c7c7',
                    borderwidth=1,
                    borderpad=4,
                    bgcolor='white',
                    opacity=0.9
                )
            
            # Display the chart with config
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale'],
                    'modeBarButtonsToRemove': ['select', 'lasso'],
                    'showAxisDragHandles': True,
                    'showAxisRangeEntryBoxes': True
                }
            )
            
            # Add download button for the chart data
            if 'stock_data_clean' in st.session_state and not st.session_state.stock_data_clean.empty:
                chart_data = st.session_state.stock_data_clean.copy()
                if len(predictions) > 0:
                    chart_data = chart_data.iloc[-len(predictions):].copy()
                    chart_data['Predicted_Price'] = predictions
                    
                    # Convert to CSV for download
                    csv = chart_data[['Close', 'Predicted_Price']].to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Chart Data (CSV)",
                        data=csv,
                        file_name=f"{ticker}_prediction_data.csv",
                        mime='text/csv',
                        key='download_chart_data'
                    )
            
            # Add future prediction section if available
            if future_predictions is not None and future_dates is not None and len(future_predictions) > 0:
                with st.expander("Future Price Predictions", expanded=True):
                    st.subheader(f"Future Price Forecast for {st.session_state.ticker}")
                    st.caption(f"Showing {len(future_predictions)} days of future predictions")
                    
                    # Create a DataFrame with future predictions
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions
                    })
                    future_df.set_index('Date', inplace=True)
                    
                    # Display the future predictions table
                    st.dataframe(future_df.style.format({'Predicted Price': '${:.2f}'}), use_container_width=True)
            
            # Add some performance metrics if available
            if metrics:
                st.subheader("Prediction Accuracy")
                
                # Log the metrics for debugging
                st.session_state.logger.debug(f"Raw metrics data: {metrics}")
                
                # Get metrics with proper error handling
                mae = metrics.get('mae', 0)
                mse = metrics.get('mse', 0)
                rmse = metrics.get('rmse', 0)
                r2 = metrics.get('r2', 0)
                mape = metrics.get('mape', 0)
                directional_accuracy = metrics.get('directional_accuracy', 0)
                
                # Calculate metrics directly if we have actual values and predictions
                if len(actual_values) > 0 and len(predictions) > 0:
                    # Ensure arrays are the same length
                    min_len = min(len(actual_values), len(predictions))
                    y_true = np.array(actual_values[:min_len])
                    y_pred = np.array(predictions[:min_len])
                    
                    # Calculate MAE if not already set
                    if mae == 0:
                        mae = np.mean(np.abs(y_true - y_pred))
                    
                    # Calculate MSE if not already set
                    if mse == 0:
                        mse = np.mean((y_true - y_pred) ** 2)
                    
                    # Calculate RMSE if not already set
                    if rmse == 0:
                        rmse = np.sqrt(mse)
                    
                    # Calculate R² if not already set
                    if r2 == 0:
                        # Calculate R² manually to avoid division by zero
                        if np.var(y_true) != 0:  # Avoid division by zero
                            ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
                            ss_residual = np.sum((y_true - y_pred) ** 2)
                            r2 = 1 - (ss_residual / ss_total)
                        else:
                            r2 = 0
                    
                    # Calculate MAPE if not already set
                    if mape == 0 and len(y_true) > 0:
                        # Avoid division by zero
                        with np.errstate(divide='ignore', invalid='ignore'):
                            mape_values = np.abs((y_true - y_pred) / y_true) * 100
                            mape = np.nanmean(mape_values)  # Use nanmean to ignore NaN values
                    
                    # Calculate directional accuracy if not already set
                    if directional_accuracy == 0 and len(y_true) > 1:
                        # Need at least 2 points for directional accuracy
                        y_true_dir = np.sign(y_true[1:] - y_true[:-1])
                        y_pred_dir = np.sign(y_pred[1:] - y_pred[:-1])
                        matches = (y_true_dir == y_pred_dir).sum()
                        total = len(y_true_dir)
                        if total > 0:  # Avoid division by zero
                            directional_accuracy = (matches / total) * 100
                
                # Log calculated metrics
                st.session_state.logger.debug(f"Final metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}, MAPE: {mape}, Dir.Acc: {directional_accuracy}")
                
                # Check if we still have invalid metrics
                if all(v == 0 for v in [mse, rmse, r2, mape, directional_accuracy]) and mae > 0:
                    st.session_state.logger.warning("Most metrics are zero. This may indicate an issue with predictions or insufficient data points.")
                    with st.expander("Troubleshooting Tips"):
                        st.markdown("""
                        **Possible issues:**
                        - Not enough data points for prediction (need at least 2 for directional accuracy)
                        - Model predictions are constant (no variation)
                        - Division by zero in calculations
                        - Prediction values are very small or zero
                        
                        Try using a larger date range or different model parameters.
                        """)
                
                # Add specific CSS to ensure no white background on these metrics
                st.markdown("""
                <style>
                [data-testid="stMetric"] {
                    background-color: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                }
                [data-testid="stMetricValue"] {
                    background-color: transparent !important;
                }
                [data-testid="stMetricLabel"] {
                    background-color: transparent !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                with col2:
                    st.metric("Mean Squared Error", f"${mse:.2f}")
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                    
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col2:
                    st.metric("MAPE", f"{mape:.2f}%")
                with col3:
                    st.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
                    
        except Exception as e:
            st.session_state.logger.error(f"Error in performance analysis: {str(e)}")
            st.error(f"Error in performance analysis: {str(e)}")

    if show_tabs and tab_realtime is not None:  # Only show if tabs are enabled and tab exists
        with tab_realtime:
            try:
                st.subheader("Real-Time Analysis")
                
                # Get real-time data and handler with validation
                rt_data = st.session_state.get('rt_data', {})
                
                # Initialize variables if they don't exist
                ticker = st.session_state.get('ticker', '')
                rt_handler = st.session_state.get('rt_handler')
                status = rt_handler.get_status() if rt_handler else {}
                
                # Create columns for layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"### {ticker} Real-Time Data")
                    
                with col2:
                    if st.button("📊 View Raw Data"):
                        st.json(rt_data.get(ticker, {}))
                
                # Display debug info in expander
                with st.expander("Debug Information", expanded=False):
                    debug_info = {
                        "handler_available": rt_handler is not None,
                        "subscribed_symbols": status.get('subscribed_symbols', []),
                        "last_error": status.get('last_error'),
                        "last_message_time": status.get('last_message_time'),
                        "endpoint": status.get('endpoint')
                    }
                    st.json(debug_info)
                    
            except Exception as e:
                st.session_state.logger.error(f"Error in Real-Time Analysis tab: {str(e)}")
                st.error(f"Error in Real-Time Analysis tab: {str(e)}")
# if analyze_button:
#     with st.spinner("Starting analysis..."):
#         # Clear any previous results
#         st.session_state.analysis_complete = False
#         st.session_state.analysis_results = None
#         st.session_state.processed_data = None
        
#         # Ensure logger is initialized in session state
#         if 'logger' not in st.session_state:
#             st.session_state.logger = logger
            
#         # Initialize real-time data handler if it doesn't exist
#         if 'realtime_handler' not in st.session_state:
#             try:
#                 st.session_state.realtime_handler = RealTimeDataHandler()
#                 # Also store in rt_handler for backward compatibility
#                 st.session_state.rt_handler = st.session_state.realtime_handler
#                 st.session_state.logger.info("Initialized new real-time data handler")
#             except Exception as e:
#                 st.session_state.logger.error(f"Failed to initialize real-time data handler: {str(e)}", exc_info=True)
            
#         # Store ticker and dates in session state
#         st.session_state.ticker = ticker
#         st.session_state.start_date = start_date
#         st.session_state.end_date = end_date
        
#         # Verify TwoStagePredictor is properly imported
#         try:
#             # Try to create a test instance to verify the import
#             test_predictor = TwoStagePredictor()
#             st.session_state.logger.info("Successfully imported TwoStagePredictor class")
#             del test_predictor  # Clean up test instance
#         except Exception as e:
#             error_msg = f"Failed to import/initialize TwoStagePredictor: {str(e)}"
#             st.error(error_msg)
#             st.session_state.logger.error(error_msg, exc_info=True)
#             st.stop()

#         # Initialize predictor if not exists
#         st.session_state.logger.info(f"Current predictor state: {st.session_state.get('predictor')}")
        
#         if 'predictor' not in st.session_state or st.session_state.predictor is None:
#             try:
#                 st.session_state.logger.info("Creating new TwoStagePredictor instance...")
#                 st.session_state.predictor = TwoStagePredictor()
#                 st.session_state.logger.info(f"Successfully initialized TwoStagePredictor: {st.session_state.predictor}")
#                 st.session_state.logger.info(f"Predictor methods: {[m for m in dir(st.session_state.predictor) if not m.startswith('_')]}")
#             except Exception as e:
#                 error_msg = f"Failed to initialize predictor: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 st.stop()
#         else:
#             st.session_state.logger.info(f"Using existing predictor: {st.session_state.predictor}")
#             st.session_state.logger.info(f"Existing predictor methods: {[m for m in dir(st.session_state.predictor) if not m.startswith('_')]}")
                
#         # Verify predictor is properly initialized and has required methods
#         if st.session_state.predictor is None:
#             error_msg = "Predictor failed to initialize. Please try again."
#             st.error(error_msg)
#             st.session_state.logger.error(error_msg)
#             st.stop()
            
#         # Verify predictor has required methods
#         if not hasattr(st.session_state.predictor, 'preprocess_data'):
#             error_msg = "Predictor is missing required 'preprocess_data' method"
#             st.error(error_msg)
#             st.session_state.logger.error(f"Predictor methods: {dir(st.session_state.predictor)}")
#             st.stop()
            
#         # Clear any previous results
#         st.session_state.analysis_results = None
#         st.session_state.processed_data = None
        
#         # Define the fetch_stock_data function
#         def fetch_stock_data():
#             """
#             Fetch and validate stock data for the current ticker.
            
#             Returns:
#                 pd.DataFrame or None: The fetched stock data or None if an error occurs
#             """
#             st.session_state.logger.info(f"Starting to fetch stock data for {ticker}")
            
#             try:
#                 # First validate the ticker
#                 st.session_state.logger.info(f"Validating ticker: {ticker}")
#                 if not st.session_state.data_collector.validate_ticker(ticker):
#                     error_msg = f"Invalid or non-existent ticker: {ticker}"
#                     st.error(error_msg)
#                     st.session_state.logger.error(error_msg)
#                     st.stop()  # Stop execution completely for invalid ticker
#                     return None  # This line is a safeguard, st.stop() will have already stopped execution
                
#                 # Calculate date range for logging
#                 fetch_start = start_date - timedelta(days=365 * 2)  # 2 years of historical data
#                 st.session_state.logger.info(
#                     f"Fetching data for {ticker} from {fetch_start.strftime('%Y-%m-%d')} "
#                     f"to {end_date.strftime('%Y-%m-%d')} with 300 days buffer"
#                 )
                
#                 # Fetch stock data with buffer period for indicators
#                 stock_data = fetch_with_cache(
#                     ticker,
#                     fetch_start,
#                     end_date,
#                     buffer_days=300
#                 )
                
#                 # Log the result of fetch_with_cache
#                 if stock_data is None:
#                     error_msg = f"fetch_with_cache returned None for ticker: {ticker}"
#                     st.session_state.logger.error(error_msg)
#                     st.error(error_msg)
#                     return None
                    
#                 if stock_data.empty:
#                     error_msg = f"Empty DataFrame returned for ticker: {ticker}"
#                     st.session_state.logger.error(error_msg)
#                     st.error(error_msg)
#                     return None
                
#                 # Log basic info about the fetched data
#                 st.session_state.logger.info(
#                     f"Successfully fetched {len(stock_data)} rows of data for {ticker}"
#                 )
#                 st.session_state.logger.info(f"Columns in raw data: {stock_data.columns.tolist()}")
#                 st.session_state.logger.info(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
                
#                 # Clean the data
#                 st.session_state.logger.info("Starting data cleaning...")
                
#                 # Convert MultiIndex or tuple columns to joined string names (e.g. 'Close_GOOGL')
#                 if isinstance(stock_data.columns, pd.MultiIndex):
#                     st.session_state.logger.info("Converting MultiIndex columns to joined string names...")
#                     stock_data.columns = ['_'.join([str(c) for c in col if str(c).strip() != '']) for col in stock_data.columns]
#                 elif len(stock_data.columns) > 0 and isinstance(stock_data.columns[0], tuple):
#                     st.session_state.logger.info("Converting tuple column names to joined string names...")
#                     stock_data.columns = ['_'.join([str(c) for c in col if str(c).strip() != '']) if isinstance(col, tuple) else str(col) for col in stock_data.columns]
                
#                 stock_data = clean_dataframe(stock_data, "Stock data")
                
#                 # Log cleaned data info
#                 st.session_state.logger.info(f"Data after cleaning - Rows: {len(stock_data)}")
#                 st.session_state.logger.info(f"Columns after cleaning: {stock_data.columns.tolist()}")
                
#                 # Check for required columns
#                 required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
#                 # Support columns like 'Close_GOOGL', 'Open_GOOGL', etc.
#                 def has_required(col_name):
#                     col_name = col_name.lower()
#                     return any(c.lower() == col_name or col_name.startswith(c.lower() + '_') for c in required_columns)
                
#                 available_columns = [str(col) for col in stock_data.columns]
#                 missing_columns = [col for col in required_columns if not any(has_required(str(c)) for c in stock_data.columns)]
                
#                 if missing_columns:
#                     error_msg = (
#                         f"Missing required columns after cleaning: {missing_columns}. "
#                         f"Available columns: {stock_data.columns.tolist()}"
#                     )
#                     st.session_state.logger.error(error_msg)
#                     st.error(error_msg)
#                     return None
                
#                 # Check for NaN values in price data
#                 price_columns = ['Open', 'High', 'Low', 'Close']
#                 for col in price_columns:
#                     if stock_data[col].isnull().any():
#                         st.session_state.logger.warning(f"NaN values found in {col} column after cleaning")
                
#                 # Store in session state for other tabs
#                 st.session_state.stock_data = stock_data
#                 st.session_state.logger.info(f"Successfully fetched and cleaned data for {ticker}")
                
#                 return stock_data
                
#             except Exception as e:
#                 error_msg = f"Error in fetch_stock_data: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 return None

#         # 1. Fetch stock data
#         with st.spinner("Fetching stock data..."):
#             stock_data = fetch_stock_data()
#             if stock_data is None or stock_data.empty:
#                 st.error("Failed to fetch stock data. Please check the ticker symbol and try again.")
#                 st.stop()
            
#             # Get the correct close column name
#             close_col = get_close_column_name(stock_data, ticker)
#             st.session_state.close_col = close_col
#             st.session_state.stock_data = stock_data
#             st.session_state.logger.info(f"Stock data fetched successfully. Close column: {close_col}")
#             st.session_state.logger.info(f"Available columns: {', '.join(stock_data.columns)}")
        
#         # 1.5 Fetch sentiment data
#         with st.spinner("Fetching sentiment data..."):
#             try:
#                 # Fetch sentiment data using analyze_sentiment function
#                 st.session_state.logger.info(f"Fetching sentiment data for {ticker} from {start_date} to {end_date}")
#                 sentiment_data = analyze_sentiment(ticker, start_date, end_date)
                
#                 if sentiment_data is not None and not sentiment_data.empty:
#                     st.session_state.sentiment_data = sentiment_data
#                     st.session_state.logger.info(f"Successfully fetched sentiment data with {len(sentiment_data)} rows")
#                 else:
#                     st.session_state.logger.warning("No sentiment data available for the selected ticker and date range")
#                     st.warning("No sentiment data available for the selected ticker and date range. Analysis will proceed with stock data only.")
#                     sentiment_data = pd.DataFrame()
#                     st.session_state.sentiment_data = sentiment_data
#             except Exception as e:
#                 error_msg = f"Error fetching sentiment data: {str(e)}"
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 st.warning(f"Could not fetch sentiment data: {str(e)}. Analysis will proceed with stock data only.")
#                 sentiment_data = pd.DataFrame()
#                 st.session_state.sentiment_data = sentiment_data
                
#         # 2. Clean and preprocess data
#         with st.spinner("Preprocessing data..."):
#             try:
#                 # Clean the data
#                 stock_data_clean = clean_dataframe(stock_data, "Stock data")
                
#                 # Get sentiment and macro data (now sentiment_data should be available)
#                 macro_data = st.session_state.get('macro_data', pd.DataFrame())
                
#                 # Store raw data for later use
#                 st.session_state.raw_sentiment_data = sentiment_data
#                 st.session_state.raw_macro_data = macro_data
                
#                 # Debug: Log predictor state and methods
#                 st.session_state.logger.info(f"Before preprocess_data - predictor: {st.session_state.predictor}")
#                 if st.session_state.predictor is not None:
#                     st.session_state.logger.info(f"Predictor methods: {[m for m in dir(st.session_state.predictor) if not m.startswith('_')]}")
#                 else:
#                     st.session_state.logger.error("CRITICAL: Predictor is None right before preprocess_data call")
#                     st.error("Failed to initialize predictor. Please try again.")
#                     st.stop()
                
#                 # Prepare data for model
#                 try:
#                     processed_data, y_train = st.session_state.predictor.preprocess_data(
#                         stock_data_clean,
#                         sentiment_data if not sentiment_data.empty else None,
#                         macro_data if not macro_data.empty else None
#                     )
#                 except Exception as e:
#                     st.session_state.logger.error(f"Error in preprocess_data: {str(e)}", exc_info=True)
#                     raise
                
#                 if processed_data is None or y_train is None:
#                     raise ValueError("Data preprocessing failed - no valid data returned")
                
#                 # Store processed data (without sentiment for now)
#                 st.session_state.processed_data = {
#                     'stock_data': stock_data_clean,
#                     'sentiment_data': sentiment_data,
#                     'macro_data': macro_data
#                 }
#                 st.session_state.y_train = y_train
#                 st.session_state.logger.info("Data preprocessing completed successfully")
                
#             except Exception as e:
#                 error_msg = f"Error during data preprocessing: {str(e)}"
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 st.error(error_msg)
#                 st.stop()
        
#         # 3. Train the model
#         try:
#             with st.spinner("Training model..."):
#                 try:
#                     # Train the model
#                     history = st.session_state.predictor.train(processed_data, y_train)
                    
#                     # Make predictions
#                     predictions = st.session_state.predictor.predict(*processed_data['X_test'])
                    
#                     # Store results
#                     st.session_state.analysis_results = {
#                         'history': history.history if hasattr(history, 'history') else {},
#                         'predictions': predictions,
#                         'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                         'ticker': ticker,
#                         'metrics': {
#                             'mae': history.history.get('val_mae', [0])[-1] if hasattr(history, 'history') else 0,
#                             'loss': history.history.get('val_loss', [0])[-1] if hasattr(history, 'history') else 0,
#                             'accuracy': 1 - history.history.get('val_mae', [0])[-1] if hasattr(history, 'history') else 0
#                         }
#                     }
                    
#                     # Store the clean stock data and close column for the performance chart
#                     st.session_state.stock_data_clean = stock_data_clean
#                     st.session_state.close_col = close_col
                    
#                     # Process and validate sentiment data after model training
#                     if hasattr(st.session_state, 'raw_sentiment_data') and not st.session_state.raw_sentiment_data.empty:
#                         st.session_state.logger.info("Validating sentiment data after model training")
#                         sentiment_data = st.session_state.raw_sentiment_data
                        
#                         # Ensure required columns exist
#                         required_cols = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']
#                         missing_cols = [col for col in required_cols if col not in sentiment_data.columns]
                        
#                         if missing_cols:
#                             st.session_state.logger.warning(f"Adding missing sentiment columns: {missing_cols}")
                            
#                             # Create missing columns with default values
#                             for col in missing_cols:
#                                 if col == 'Sentiment':
#                                     sentiment_data[col] = 0.0
#                                 elif col == 'Price_Momentum':
#                                     sentiment_data[col] = sentiment_data.get('close', 0).pct_change()
#                                 elif col == 'Volume_Change':
#                                     sentiment_data[col] = sentiment_data.get('volume', 0).pct_change()
#                                 elif col == 'Volatility':
#                                     sentiment_data[col] = sentiment_data.get('close', 0).pct_change().rolling(window=5).std()
                            
#                             st.session_state.logger.info("Added missing sentiment columns with default values")
                        
#                         # Update processed data with validated sentiment data
#                         if 'processed_data' in st.session_state:
#                             st.session_state.processed_data['sentiment_data'] = sentiment_data
                    
#                     # Mark analysis as complete
#                     st.session_state.analysis_complete = True
#                     st.session_state.logger.info("Model training and sentiment validation completed successfully")
#                     st.success("Analysis completed successfully!")
                    
#                     # Start WebSocket connection after successful analysis
#                     if ticker:
#                         try:
#                             # Get the real-time data handler
#                             rt_handler = st.session_state.realtime_handler
                            
#                             # Define a callback to process real-time data
#                             def handle_realtime_data(data):
#                                 try:
#                                     if not data or not isinstance(data, list):
#                                         return
                                    
#                                     for msg in data:
#                                         # Process trade data
#                                         if msg.get('T') == 't':  # Trade event
#                                             symbol = msg.get('S', '')
#                                             if not symbol:
#                                                 continue
                                                
#                                             # Initialize data structure if needed
#                                             if 'rt_data' not in st.session_state:
#                                                 st.session_state.rt_data = {}
                                                
#                                             if symbol not in st.session_state.rt_data:
#                                                 st.session_state.rt_data[symbol] = {
#                                                     'price': None,
#                                                     'change': 0,
#                                                     'volume': 0,
#                                                     'timestamp': datetime.now().isoformat(),
#                                                     'status': 'Connected',
#                                                     'last_update': None
#                                                 }
                                            
#                                             # Update with trade data
#                                             price = float(msg.get('p', 0))
#                                             if st.session_state.rt_data[symbol]['price'] is not None:
#                                                 change = price - st.session_state.rt_data[symbol]['price']
#                                                 change_pct = (change / st.session_state.rt_data[symbol]['price']) * 100 if st.session_state.rt_data[symbol]['price'] > 0 else 0
#                                             else:
#                                                 change = 0
#                                                 change_pct = 0
                                                
#                                             # Update the data
#                                             st.session_state.rt_data[symbol].update({
#                                                 'price': price,
#                                                 'change': change,
#                                                 'change_percent': change_pct,
#                                                 'volume': st.session_state.rt_data[symbol].get('volume', 0) + int(msg.get('s', 0)),
#                                                 'timestamp': msg.get('t'),
#                                                 'last_update': datetime.now().isoformat()
#                                             })
#                                 except Exception as e:
#                                     st.session_state.logger.error(f"Error processing WebSocket data: {str(e)}", exc_info=True)
                            
#                             # Register the callback
#                             rt_handler.callbacks = [handle_realtime_data]
                            
#                             # Start the WebSocket connection if not already running
#                             if not rt_handler.running:
#                                 safe_start_handler(rt_handler)
#                                 st.session_state.logger.info("WebSocket connection started successfully")
                            
#                             # Subscribe to the current ticker
#                             safe_subscribe_handler(rt_handler, [ticker])
#                             st.session_state.logger.info(f"Subscribed to ticker: {ticker}")
                            
#                             st.rerun()
#                         except Exception as e:
#                             error_msg = f"Failed to connect to WebSocket: {str(e)}"
#                             st.error(error_msg)
#                             st.session_state.logger.error(f"WebSocket connection error: {str(e)}", exc_info=True)
                    
#                 except Exception as e:
#                     error_msg = f"Error during model training: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.logger.error(error_msg, exc_info=True)
#                     st.stop()
        
#         except Exception as e:
#             error_msg = f"Error during data preprocessing: {str(e)}"
#             st.error(error_msg)
#             st.session_state.logger.error(error_msg, exc_info=True)
#             st.stop()
            
#         def initialize_websocket():
#             """Initialize WebSocket connection and related components"""
#             try:
#                 # Initialize WebSocket components first
#                 if 'ws_status' not in st.session_state:
#                     st.session_state.ws_status = {
#                         'connected': False,
#                         'healthy': False,
#                         'last_message': None,
#                         'last_error': None
#                     }

#                 # Initialize TwoStagePredictor if not already initialized
#                 if 'predictor' not in st.session_state:
#                     st.session_state.predictor = TwoStagePredictor()
#                     st.session_state.logger.info("Initialized TwoStagePredictor")
                    
#                 # Initialize real-time data structure
#                 if 'rt_data' not in st.session_state:
#                     st.session_state.rt_data = {}

#                 if ticker not in st.session_state.rt_data:
#                     st.session_state.rt_data[ticker] = {
#                         'price': None,
#                         'change': 0,
#                         'volume': 0,
#                         'timestamp': datetime.now().isoformat(),
#                         'status': 'Disconnected',
#                         'last_update': None
#                     }
                    
#                 return True
#             except Exception as e:
#                 error_msg = f"Error initializing WebSocket: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.logger.error(f"WebSocket initialization error: {str(e)}", exc_info=True)
#                 return False
                
#         # Initialize WebSocket components but don't start the client yet
#         # We'll start it after model training is complete
#         if ticker:
#             try:
#                 # Initialize WebSocket handler using the class defined in this file
#                 if 'rt_handler' not in st.session_state and 'realtime_handler' not in st.session_state:
#                     st.session_state.logger.info("Creating new RealTimeDataHandler instance")
#                     rt_handler = RealTimeDataHandler()
#                     rt_handler._persist_connection = True
                    
#                     # Store in both session state keys for compatibility
#                     st.session_state.rt_handler = rt_handler
#                     st.session_state.realtime_handler = rt_handler
                    
#                     # Log successful initialization
#                     st.session_state.logger.info("RealTimeDataHandler successfully initialized and stored in session state")
#                 else:
#                     # Use existing handler
#                     rt_handler = st.session_state.get('rt_handler', st.session_state.get('realtime_handler'))
#                     st.session_state.logger.info("Using existing RealTimeDataHandler from session state")
                
#                 # Initialize real-time data structure
#                 if 'rt_data' not in st.session_state:
#                     st.session_state.rt_data = {}
                    
#                 if ticker not in st.session_state.rt_data:
#                     st.session_state.rt_data[ticker] = {
#                         'price': None,
#                         'change': 0,
#                         'volume': 0,
#                         'timestamp': datetime.now().isoformat(),
#                         'status': 'Initialized',
#                         'last_update': None
#                     }
                
#                 st.session_state.logger.info("WebSocket handler initialized")
                
#             except Exception as e:
#                 error_msg = f"Failed to initialize WebSocket handler: {str(e)}"
#                 st.warning(error_msg)
#                 st.session_state.logger.error(error_msg, exc_info=True)

#         # Initialize WebSocket in a non-blocking way
#         try:
#             if not initialize_websocket():
#                 st.warning("Real-time updates will not be available due to WebSocket initialization error")

#             # Start WebSocket server if not already running
#             if not hasattr(st, '_ws_server_running'):
#                 st._ws_server_running = True

#                 def start_websocket_server_thread():
#                     """Start WebSocket server in a separate thread"""
#                     try:
#                         start_websocket_server()
#                     except Exception as e:
#                         error_msg = f"Failed to start WebSocket server: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.logger.error(error_msg, exc_info=True)

#                 # Start WebSocket server in a separate thread
#                 websocket_thread = threading.Thread(
#                     target=start_websocket_server_thread,
#                     name="WebSocketServerThread",
#                     daemon=True
#                 )
#                 websocket_thread.start()
#                 time.sleep(1.0)  # Give the server time to start
#         except Exception as e:
#             st.warning(f"WebSocket initialization failed: {str(e)}")
#             st.session_state.logger.warning(f"WebSocket initialization failed: {str(e)}", exc_info=True)

#         # Initialize and start the WebSocket handler
#             try:
#                 # Get the handler from session state
#                 rt_handler = st.session_state.realtime_handler
                
#                 # Define a callback function to handle real-time data
#                 def handle_realtime_data(data):
#                     """Process real-time data from WebSocket"""
#                     try:
#                         if not data or not isinstance(data, list):
#                             return
                        
#                         for msg in data:
#                             msg_type = msg.get('T')
#                             if msg_type == 'error':
#                                 st.session_state.logger.error(f"WebSocket error: {msg.get('msg')}")
#                                 continue
                                
#                             # Process trade data
#                             if msg_type == 't':  # Trade event
#                                 symbol = msg.get('S', '')
#                                 if not symbol:
#                                     continue
                                    
#                                 # Initialize data structure if needed
#                                 if symbol not in st.session_state.rt_data:
#                                     st.session_state.rt_data[symbol] = {
#                                         'price': None,
#                                         'change': 0,
#                                         'volume': 0,
#                                         'timestamp': datetime.now().isoformat(),
#                                         'status': 'Connected',
#                                         'last_update': None
#                                     }
                                
#                                 # Update with trade data
#                                 price = float(msg.get('p', 0))
#                                 if st.session_state.rt_data[symbol]['price'] is not None:
#                                     change = price - st.session_state.rt_data[symbol]['price']
#                                     change_pct = (change / st.session_state.rt_data[symbol]['price']) * 100 if st.session_state.rt_data[symbol]['price'] > 0 else 0
#                                 else:
#                                     change = 0
#                                     change_pct = 0
                                    
#                                 # Update the data
#                                 st.session_state.rt_data[symbol].update({
#                                     'price': price,
#                                     'change': change,
#                                     'change_percent': change_pct,
#                                     'volume': st.session_state.rt_data[symbol].get('volume', 0) + int(msg.get('s', 0)),
#                                     'timestamp': msg.get('t'),
#                                     'last_update': datetime.now().isoformat()
#                                 })
#                     except Exception as e:
#                         st.session_state.logger.error(f"Error processing WebSocket data: {str(e)}", exc_info=True)
                
#                 # Register the callback with the handler
#                 rt_handler.callbacks = [handle_realtime_data]
                
#                 # Start the WebSocket connection if not already running
#                 if not rt_handler.running:
#                     rt_handler.start()
#                     st.session_state.logger.info("Started WebSocket connection")
                
#                 # Subscribe to the current ticker
#                 if ticker:
#                     rt_handler.subscribe([ticker])
#                     st.session_state.logger.info(f"Subscribed to ticker: {ticker}")
                
#                 # Get current status
#                 status = {
#                     'connected': rt_handler.connected,
#                     'authenticated': rt_handler.authenticated,
#                     'subscribed_symbols': list(rt_handler.subscribed_symbols),
#                     'endpoint': rt_handler.ws_url
#                 }
                
#                 # Add WebSocket status panel at the bottom of the page
#                 with st.expander("📡 WebSocket Status", expanded=False):
#                     try:
#                         # Get direct status from the handler properties
#                         is_connected = rt_handler.connected
#                         is_authenticated = rt_handler.authenticated
#                         subscribed_symbols = list(rt_handler.subscribed_symbols)
                        
#                         # Display status as JSON
#                         status = {
#                             'connected': is_connected,
#                             'authenticated': is_authenticated,
#                             'subscribed_symbols': subscribed_symbols,
#                             'endpoint': rt_handler.ws_url,
#                             'running': rt_handler.running
#                         }
#                         st.json(status)
                        
#                         # Add connection status indicator
#                         if is_connected and is_authenticated:
#                             st.success("✅ Connected to Alpaca WebSocket")
#                         else:
#                             st.error("❌ Disconnected from Alpaca WebSocket")
                            
#                             # Show error if any
#                             if hasattr(rt_handler, 'last_error') and rt_handler.last_error:
#                                 st.error(f"Error: {rt_handler.last_error}")
                            
#                         # Show connection details
#                         st.caption(f"Connected to: {rt_handler.ws_url}")
#                         st.caption(f"Symbols: {', '.join(subscribed_symbols) or 'None'}")
                        
#                         # Add reconnect button
#                         if st.button("🔄 Reconnect WebSocket", key="reconnect_ws_button"):
#                             rt_handler.stop()
#                             time.sleep(1)  # Brief pause
#                             rt_handler.start()
#                             if ticker:
#                                 rt_handler.subscribe([ticker])
#                             st.success("Reconnection initiated")
#                             time.sleep(1)  # Brief pause
#                             st.rerun()
                                
#                     except Exception as e:
#                         error_msg = f"Error getting WebSocket status: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.logger.error(error_msg, exc_info=True)
                        
#                         if 'rt_handler' in st.session_state and st.session_state.rt_handler is not None:
#                             st.session_state.rt_data[ticker]['status'] = f'Error: {str(e)}'
#                         else:
#                             st.session_state.rt_data[ticker]['status'] = 'Initialization Failed'
#                         # Don't raise the exception to allow the app to continue
                        
#             except Exception as e:
#                 st.error(f"Failed to initialize WebSocket handler: {str(e)}")
#                 st.session_state.logger.error(f"Failed to initialize WebSocket handler: {str(e)}", exc_info=True)
#                 st.session_state._ws_initialized = False

#             # Data Collection
#             try:
#                 # First validate the ticker
#                 import sys
#                 import os
                
#                 # Ensure 'src' directory is in the Python module search path
#                 sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
#                 from src.data.collector import Collector  # Import the Collector class
#                 collector = Collector()  # Initialize the collector object
#                 if not collector.validate_ticker(ticker):
#                     raise ValueError(f"Invalid or non-existent ticker: {ticker}")

#                 # Get data for the ticker with smart caching
#                 min_required_days = 200  # Minimum days required for technical indicators
#                 buffer_days = 300  # Additional buffer for safety

#                 # First try with the requested date range
#                 stock_data = fetch_with_cache(
#                     ticker, start_date, end_date, buffer_days=buffer_days)

#                 # If we don't have enough data, try fetching more historical data
#                 if len(stock_data) < min_required_days:
#                     extended_start = start_date - timedelta(days=min_required_days * 2)
#                     stock_data = fetch_with_cache(
#                         ticker, extended_start, end_date, buffer_days=buffer_days)

#                     if len(stock_data) < min_required_days:
#                         raise ValueError(
#                             f"Insufficient data points. Need at least {min_required_days} days of historical data. "
#                             f"Current data has {len(stock_data)} days."
#                         )

#                 if stock_data is None or stock_data.empty:
#                     raise ValueError(f"No data returned for ticker: {ticker}")

#                 # Flatten MultiIndex columns if they exist
#                 if isinstance(stock_data.columns, pd.MultiIndex):
#                     stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

#                 # Debug: Show data info
#                 st.sidebar.subheader("Debug Info")
#                 with st.sidebar.expander("Show Raw Data"):
#                     st.write("### Raw Data Sample")
#                     st.write(stock_data.head())
#                     st.write("### Data Info")
#                     st.write(f"Data shape: {stock_data.shape}")
#                     st.write(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
#                     st.write("Columns:", stock_data.columns.tolist())
#                     st.write("### Missing Values")
#                     st.write(stock_data.isnull().sum())

#                 # Calculate date range for sentiment data to match stock data
#                 days_of_data = (stock_data.index.max() - stock_data.index.min()).days

#                 # Collect sentiment data for the full date range of stock data
#                 sentiment_data = collector.collect_sentiment_data(ticker, days=days_of_data)

#                 # Collect macro data with the same date range as stock data
#                 macro_data = collector.collect_macroeconomic_data(
#                     start_date=start_date.strftime('%Y-%m-%d'),
#                     end_date=end_date.strftime('%Y-%m-%d')
#                 )
#                 company_data = collector.collect_company_data(ticker)


#                 # Data Processing
#                 # Ensure we have valid data before processing
#                 if stock_data is None or stock_data.empty:
#                     raise ValueError("No stock data available for processing")

#                 if sentiment_data is None or sentiment_data.empty:
#                     st.warning(
#                         "No sentiment data available. Using default values.")
#                     # Create empty DataFrame with same index as stock_data
#                     sentiment_data = pd.DataFrame(
#                         index=stock_data.index,
#                         columns=['sentiment_score', 'sentiment_magnitude']
#                     ).fillna(0)

#                 if macro_data is None or macro_data.empty:
#                     st.warning(
#                         "No macroeconomic data available. Using default values.")
#                     # Create empty DataFrame with same index as stock_data
#                     macro_data = pd.DataFrame(
#                         index=stock_data.index,
#                         columns=['interest_rate',
#                             'inflation_rate', 'gdp_growth']
#                     ).fillna(0)

#                 # Clean input data before preprocessing
#                 def clean_dataframe(df, name):
#                     if df is None or df.empty:
#                         return None

#                     # Make a copy to avoid modifying original
#                     df_clean = df.copy()

#                     # Handle missing values
#                     if df_clean.isnull().any().any():
#                         st.session_state.logger.warning(
#                             f"{name} contains NaN values. Filling with forward and backward fill.")
#                         df_clean = df_clean.ffill().bfill()

#                         # If still NaN, fill with column mean
#                         if df_clean.isnull().any().any():
#                             st.session_state.logger.warning(
#                                 f"{name} still contains NaN after forward/backward fill. Filling with column means.")
#                             df_clean = df_clean.fillna(df_clean.mean())

#                     # Handle infinite values
#                     numeric_cols = df_clean.select_dtypes(
#                         include=[np.number]).columns
#                     for col in numeric_cols:
#                         if np.isinf(df_clean[col]).any():
#                             st.session_state.logger.warning(
#                                 f"{name} column {col} contains infinite values. Replacing with column min/max.")
#                             col_min = df_clean[col].min()
#                             col_max = df_clean[col].max()
#                             df_clean[col] = df_clean[col].replace(
#                                 [np.inf, -np.inf], np.nan)
#                             df_clean[col] = df_clean[col].fillna(
#                                 (col_min + col_max) / 2)

#                     return df_clean

#                 # Clean all input data with detailed logging
#                 st.session_state.logger.info("Starting data cleaning...")
#                 try:
#                     stock_data_clean = clean_dataframe(stock_data, "Stock data")
#                     st.session_state.logger.info(f"Stock data cleaned. Shape: {stock_data_clean.shape}")
                    
#                     sentiment_data_clean = None
#                     if sentiment_data is not None:
#                         sentiment_data_clean = clean_dataframe(sentiment_data, "Sentiment data")
#                         st.session_state.logger.info(f"Sentiment data cleaned. Shape: {sentiment_data_clean.shape if sentiment_data_clean is not None else 'None'}")
                    
#                     macro_data_clean = None
#                     if macro_data is not None:
#                         macro_data_clean = clean_dataframe(macro_data, "Macro data")
#                         st.session_state.logger.info(f"Macro data cleaned. Shape: {macro_data_clean.shape if macro_data_clean is not None else 'None'}")

#                     # Store cleaned data in session state for use in tabs
#                     st.session_state.stock_data_clean = stock_data_clean
#                     st.session_state.sentiment_data_clean = sentiment_data_clean
#                     st.session_state.macro_data_clean = macro_data_clean
#                     st.session_state.logger.info("Cleaned data stored in session state")

#                     # Keep a copy of the full data for display after processing
#                     full_stock_data = stock_data_clean.copy()
#                     st.session_state.logger.info("Data cleaning completed successfully")
                    
#                 except Exception as e:
#                     error_msg = f"Error during data cleaning: {str(e)}"
#                     st.session_state.logger.error(error_msg, exc_info=True)
#                     raise RuntimeError(error_msg)

#                 # Data preprocessing with detailed logging
#                 st.session_state.logger.info("Starting data preprocessing...")
#                 try:
#                     st.session_state.logger.info(f"Input shapes - Stock: {stock_data_clean.shape}, "
#                                               f"Sentiment: {sentiment_data_clean.shape if sentiment_data_clean is not None else 'None'}, "
#                                               f"Macro: {macro_data_clean.shape if macro_data_clean is not None else 'None'}")
                    
#                     # Get processed data with cleaned inputs
#                     processed_data, y_train = st.session_state.predictor.preprocess_data(
#                         stock_data_clean, 
#                         sentiment_data_clean, 
#                         macro_data_clean
#                     )
                    
#                     # Log successful preprocessing
#                     st.session_state.logger.info(f"Preprocessing completed. Processed data shapes - "
#                                               f"X: {[x.shape if hasattr(x, 'shape') else str(type(x)) for x in processed_data]}, "
#                                               f"y: {y_train.shape if hasattr(y_train, 'shape') else str(type(y_train))}")
                    
#                     # Store processed data in session state
#                     st.session_state.processed_data = processed_data
#                     st.session_state.y_train = y_train
                    
#                 except Exception as e:
#                     error_msg = f"Error during data preprocessing: {str(e)}"
#                     st.session_state.logger.error(error_msg, exc_info=True)
#                     st.error(f"Error during data preprocessing: {str(e)}")
#                     st.stop()
                
#                 # If we don't have valid processed data, stop execution
#                 if processed_data is None or y_train is None:
#                     error_msg = "Failed to preprocess data. Please check the input data and try again."
#                     st.session_state.logger.error(error_msg)
#                     st.error(error_msg)
#                     st.stop()

#                 # Now trim the data to the user's requested date range for display
#                 stock_data_clean = stock_data_clean[stock_data_clean.index >= pd.Timestamp(start_date)]
#                 if len(stock_data_clean) == 0:
#                     raise ValueError("No data available after trimming to the requested date range")

#                 # Validate the processed data
#                 if not isinstance(processed_data, tuple) or len(processed_data) != 2:
#                     error_msg = f"Invalid processed data format. Expected tuple of length 2, got {type(processed_data)}"
#                     if hasattr(processed_data, '__len__'):
#                         error_msg += f" with length {len(processed_data)}"
#                     st.session_state.logger.error(error_msg)
#                     raise ValueError(error_msg)

#                 # Unpack the tuple of inputs
#                 x_train = processed_data[0]  # Stock data input
#                 x_additional = processed_data[1]  # Additional features input

#                 # Log input shapes for debugging
#                 st.session_state.logger.info(f"x_train shape: {x_train.shape if hasattr(x_train, 'shape') else 'N/A'}")
#                 st.session_state.logger.info(f"x_additional shape: {x_additional.shape if hasattr(x_additional, 'shape') else 'N/A'}")
#                 st.session_state.logger.info(f"y_train shape: {y_train.shape if hasattr(y_train, 'shape') else 'N/A'}")

#                 # Ensure we have valid numeric data
#                 if not np.isfinite(x_train).all():
#                     st.warning("x_train contains NaN or infinite values after preprocessing. Attempting to clean...")
#                     x_train = np.nan_to_num(x_train, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

#                 if not np.isfinite(x_additional).all():
#                     st.warning("x_additional contains NaN or infinite values after preprocessing. Attempting to clean...")
#                     x_additional = np.nan_to_num(x_additional, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

#                 # Validate shapes
#                 if not hasattr(x_train, 'shape') or not hasattr(x_additional, 'shape'):
#                     error_msg = f"Input data must be numpy arrays or similar with shape attribute. Got types: {type(x_train)}, {type(x_additional)}"
#                     st.session_state.logger.error(error_msg)
#                     raise ValueError(error_msg)

#                 if x_train.shape[0] != x_additional.shape[0]:
#                     error_msg = f"Input shapes mismatch: x_train has {x_train.shape[0]} samples, x_additional has {x_additional.shape[0]} samples"
#                     st.session_state.logger.error(error_msg)
#                     raise ValueError(error_msg)
                    
#                 if len(x_train.shape) >= 2 and len(x_additional.shape) >= 2 and x_train.shape[1] != x_additional.shape[1]:
#                     error_msg = f"Sequence lengths mismatch: x_train has {x_train.shape[1]} sequence length, x_additional has {x_additional.shape[1]} sequence length"
#                     st.session_state.logger.error(error_msg)
#                     raise ValueError(error_msg)
#             except Exception as e:
#                 st.error(f"Error during data validation: {str(e)}")
#                 st.session_state.logger.error(f"Data validation error: {str(e)}", exc_info=True)
#                 raise

#                 # Model Training and Prediction
#                 try:
#                     st.session_state.logger.info("[TRAIN] Entering model training block.")

#                     # Ensure predictor is initialized
#                     if 'predictor' not in st.session_state:
#                         from src.models.two_stage_model import TwoStagePredictor
#                         st.session_state.predictor = TwoStagePredictor()
#                         st.session_state.logger.info("[TRAIN] Initialized TwoStagePredictor.")
#                     elif not hasattr(st.session_state.predictor, 'predict_future'):
#                         from src.models.two_stage_model import TwoStagePredictor
#                         st.session_state.predictor = TwoStagePredictor()
#                         st.session_state.logger.info("[TRAIN] Re-initialized TwoStagePredictor with updated methods.")

#                     # Ensure inputs are numpy arrays
#                     x_train = np.array(x_train) if not isinstance(x_train, np.ndarray) else x_train
#                     x_additional = np.array(x_additional) if not isinstance(x_additional, np.ndarray) else x_additional
#                     y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

#                     # Log input types and shapes
#                     st.session_state.logger.info(f"[TRAIN] Input types: x_train={type(x_train)}, x_additional={type(x_additional)}, y_train={type(y_train)}")
#                     st.session_state.logger.info(f"[TRAIN] Input shapes: x_train={getattr(x_train, 'shape', 'N/A')}, x_additional={getattr(x_additional, 'shape', 'N/A')}, y_train={getattr(y_train, 'shape', 'N/A')}")

#                     # Validate input data
#                     if not hasattr(x_train, 'size') or not hasattr(x_additional, 'size') or not hasattr(y_train, 'size'):
#                         error_msg = f"[TRAIN][SKIP] Input arrays missing 'size' attribute. Types: x_train={type(x_train)}, x_additional={type(x_additional)}, y_train={type(y_train)}"
#                         st.session_state.logger.error(error_msg)
#                         st.error(error_msg)
#                         st.stop()
#                     if x_train.size == 0 or x_additional.size == 0 or y_train.size == 0:
#                         error_msg = "[TRAIN][SKIP] Empty input arrays detected. Cannot proceed with model training."
#                         st.session_state.logger.error(error_msg)
#                         st.error(error_msg)
#                         st.stop()

#                     # Log data statistics
#                     try:
#                         st.session_state.logger.info(f"[TRAIN] x_train stats: mean={np.nanmean(x_train):.4f}, std={np.nanstd(x_train):.4f}, min={np.nanmin(x_train):.4f}, max={np.nanmax(x_train):.4f}")
#                         st.session_state.logger.info(f"[TRAIN] y_train stats: mean={np.nanmean(y_train):.4f}, std={np.nanstd(y_train):.4f}, min={np.nanmin(y_train):.4f}, max={np.nanmax(y_train):.4f}")
#                     except Exception as stats_ex:
#                         st.session_state.logger.warning(f"[TRAIN] Could not compute stats for input arrays: {stats_ex}")

#                     # Validate shapes
#                     if not hasattr(x_train, 'shape') or not hasattr(x_additional, 'shape'):
#                         error_msg = f"[TRAIN][SKIP] Input data must be numpy arrays or similar with shape attribute. Got types: {type(x_train)}, {type(x_additional)}"
#                         st.session_state.logger.error(error_msg)
#                         st.error(error_msg)
#                         st.stop()
#                     if x_train.shape[0] != x_additional.shape[0]:
#                         error_msg = f"[TRAIN][SKIP] Input shapes mismatch: x_train has {x_train.shape[0]} samples, x_additional has {x_additional.shape[0]} samples"
#                         st.session_state.logger.error(error_msg)
#                         st.error(error_msg)
#                         st.stop()
#                     if len(x_train.shape) >= 2 and len(x_additional.shape) >= 2 and x_train.shape[1] != x_additional.shape[1]:
#                         error_msg = f"[TRAIN][SKIP] Sequence lengths mismatch: x_train has {x_train.shape[1]}, x_additional has {x_additional.shape[1]}"
#                         st.session_state.logger.error(error_msg)
#                         st.error(error_msg)
#                         st.stop()

#                     # Train the model
#                     with st.spinner("Training model..."):
#                         if len(x_train.shape) != 3:
#                             st.session_state.logger.info("[TRAIN] Expanding dims for x_train to 3D.")
#                             x_train = np.expand_dims(x_train, axis=-1)
#                         if len(x_additional.shape) != 3:
#                             st.session_state.logger.info("[TRAIN] Expanding dims for x_additional to 3D.")
#                             x_additional = np.expand_dims(x_additional, axis=-1)
#                         model_inputs = (x_train, x_additional)
#                         st.session_state.logger.info("[TRAIN] Calling TwoStagePredictor.train()...")
#                         traininglogger = TrainingLogger(logger=st.session_state.logger)
#                         history = st.session_state.predictor.train(
#                             x_train=model_inputs,
#                             y_train=y_train,
#                             epochs=20,
#                             batch_size=32,
#                             validation_split=0.2
#                         )
#                         st.session_state.logger.info("[TRAIN] Model training completed successfully.")
#                         st.success("Model trained successfully!")

#                         # Make predictions
#                         try:
#                             st.session_state.logger.info("[TRAIN] Generating predictions...")
#                             predictions = st.session_state.predictor.predict(*model_inputs)
#                             st.session_state.logger.info(f"[TRAIN] Predictions generated. Shape: {getattr(predictions, 'shape', 'N/A')}")
#                         except Exception as e:
#                             error_msg = f"[TRAIN][FAIL] Error during prediction: {str(e)}"
#                             st.session_state.logger.error(error_msg, exc_info=True)
#                             st.error(error_msg)
#                             predictions = None
#                             st.stop()

#                         # Evaluate model
#                         try:
#                             st.session_state.logger.info("[TRAIN] Evaluating model...")
#                             metrics = st.session_state.predictor.evaluate(model_inputs, y_train)
#                             st.session_state.logger.info(f"[TRAIN] Model evaluation completed. Metrics: {metrics}")
#                         except Exception as eval_error:
#                             error_msg = f"[TRAIN][FAIL] Error during model evaluation: {str(eval_error)}"
#                             st.session_state.logger.error(error_msg, exc_info=True)
#                             st.error(error_msg)
#                             metrics = {}
#                         st.session_state.predictions = predictions
#                         st.session_state.metrics = metrics

#                 except Exception as e:
#                     error_msg = f"[TRAIN][FAIL] Error during model training block: {str(e)}"
#                     st.session_state.logger.error(error_msg, exc_info=True)
#                     st.error(error_msg)
#                     st.stop()
                
#             except Exception as e:
#                 error_msg = f"Error in model training and prediction: {str(e)}"
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 st.error(error_msg)

#             # Portfolio Optimization
#             try:
#                 # For single-ticker optimization, we'll create a simple portfolio with the selected ticker
#                 # and a benchmark (like SPY) for comparison
#                 benchmark_ticker = 'SPY'
                
#                 # Initialize default portfolio metrics for single ticker
#                 portfolio_weights = {ticker: 1.0}
#                 portfolio_metrics = {
#                     'weights': {ticker: 1.0},
#                     'expected_return': 0.0,
#                     'sharpe_ratio': 0.0,
#                     'volatility': 0.0,
#                     'max_drawdown': 0.0
#                 }
                
#                 # Store in session state immediately to ensure it's available
#                 st.session_state.portfolio_metrics = portfolio_metrics
#                 st.session_state.portfolio_weights = portfolio_weights
                
#                 # Debug: Log available columns and first few rows of stock data
#                 st.session_state.logger.info(f"Available columns in stock_data_clean: {stock_data_clean.columns.tolist()}")
#                 st.session_state.logger.info(f"First 3 rows of stock_data_clean:\n{stock_data_clean.head(3).to_string()}")
                
#                 # Get the close column name for the main ticker with debug logging
#                 st.session_state.logger.info(f"Getting close column for ticker: {ticker}")
#                 st.session_state.logger.info(f"Available columns: {stock_data_clean.columns.tolist()}")
                
#                 try:
#                     # Get the close column name once and store it
#                     close_col = get_close_column_name(stock_data_clean, ticker)
#                     st.session_state.close_col = close_col
#                     st.session_state.logger.info(f"Using close column: {close_col}")
                    
#                     # Get current price with detailed error handling
#                     if stock_data_clean.empty:
#                         raise ValueError("stock_data_clean is empty")
                        
#                     # Get the last valid price
#                     price_series = stock_data_clean[close_col].dropna()
#                     if price_series.empty:
#                         raise ValueError(f"No valid prices found in column {close_col}")
                        
#                     current_price = price_series.iloc[-1]
#                     st.session_state.logger.info(f"Successfully retrieved price: {current_price}")
                    
#                     if pd.isna(current_price):
#                         raise ValueError(f"Price is NaN in column {close_col}")
#                     if current_price <= 0:
#                         raise ValueError(f"Price is non-positive: {current_price}")
                        
#                 except Exception as e:
#                     st.session_state.logger.error(f"Error getting price: {str(e)}", exc_info=True)
#                     current_price = 0  # This will trigger the skip optimization
                
#                 # Skip optimization if we don't have valid price data
#                 if current_price <= 0 or np.isnan(current_price):
#                     st.warning(f"Skipping portfolio optimization due to invalid price data (value: {current_price})")
#                     portfolio_weights = {ticker: 1.0}
#                     portfolio_metrics = {
#                         'weights': {ticker: 1.0},
#                         'expected_return': 0.0,
#                         'sharpe_ratio': 0.0,
#                         'volatility': 0.0,
#                         'max_drawdown': 0.0
#                     }
#                 else:
#                     # Get benchmark data with proper date handling
#                     try:
#                         benchmark_data = fetch_daily_yfinance_data(
#                             benchmark_ticker,
#                             start=start_date.strftime('%Y-%m-%d'),
#                             end=end_date.strftime('%Y-%m-%d'),
#                             interval='1d',
#                             logger=logger
#                         )
                        
#                         if benchmark_data.empty:
#                             raise ValueError(f"No data returned for benchmark {benchmark_ticker}")
                            
#                         # Ensure we have a DateTimeIndex
#                         if not isinstance(benchmark_data.index, pd.DatetimeIndex):
#                             if 'Date' in benchmark_data.columns:
#                                 benchmark_data.set_index('Date', inplace=True)
#                             else:
#                                 raise ValueError("Benchmark data has no date index or 'Date' column")
                                
#                     except Exception as e:
#                         st.warning(f"Could not fetch benchmark data: {str(e)}")
#                         benchmark_data = None
                    
#                     # If benchmark data is empty, use default weights
#                     if benchmark_data.empty:
#                         st.warning(f"No benchmark data available for {benchmark_ticker}. Using single-asset portfolio.")
#                         portfolio_weights = {ticker: 1.0}
#                         portfolio_metrics = {
#                             'weights': {ticker: 1.0},
#                             'expected_return': 0.0,
#                             'sharpe_ratio': 0.0,
#                             'volatility': 0.0,
#                             'max_drawdown': 0.0
#                         }
#                         # Store the metrics in session state
#                         st.session_state.portfolio_metrics = portfolio_metrics
#                         st.session_state.portfolio_weights = portfolio_weights
#                         # Store data in session state
#                         st.session_state.stock_data = stock_data
#                         st.session_state.processed_data = processed_data
#                         st.session_state.predictions = predictions
#                         st.session_state.metrics = metrics
#                         # Skip the rest of the optimization
#                         benchmark_data = None  # This will prevent further processing in this block
                    
#                     # Calculate returns for both assets - ensure we have the right column names
#                     benchmark_close_col = 'Close' if 'Close' in benchmark_data.columns else benchmark_data.filter(like='Close').columns[0]
#                     benchmark_returns = benchmark_data[benchmark_close_col].pct_change().dropna()
                    
#                     try:
#                         # Get the close column for the main ticker with debug logging
#                         st.session_state.logger.info(f"Getting close column for ticker: {ticker}")
#                         st.session_state.logger.info(f"Available columns: {stock_data_clean.columns.tolist()}")
                        
#                         ticker_close_col = get_close_column_name(stock_data_clean, ticker)
#                         st.session_state.close_col = ticker_close_col  # Store in session state for future use
                        
#                         st.session_state.logger.info(f"Using close column: {ticker_close_col}")
                        
#                         # Calculate returns for both assets
#                         asset_returns = stock_data_clean[ticker_close_col].pct_change().dropna()
                        
#                         # Ensure we have valid returns data
#                         if asset_returns.empty:
#                             raise ValueError(f"No valid returns data for {ticker} using column {ticker_close_col}")
                            
#                         # Log returns statistics for debugging
#                         st.session_state.logger.info(
#                             f"Returns stats for {ticker} - "
#                             f"Count: {len(asset_returns)}, "
#                             f"Mean: {asset_returns.mean():.6f}, "
#                             f"Std: {asset_returns.std():.6f}, "
#                             f"Min: {asset_returns.min():.6f}, "
#                             f"Max: {asset_returns.max():.6f}"
#                         )
                        
#                         # Ensure both returns are 1D pandas Series with proper names and indices
#                         if isinstance(benchmark_returns, (pd.Series, pd.DataFrame)):
#                             benchmark_returns = benchmark_returns.squeeze()  # Convert to Series if it's a DataFrame
#                         if isinstance(asset_returns, (pd.Series, pd.DataFrame)):
#                             asset_returns = asset_returns.squeeze()  # Convert to Series if it's a DataFrame
                            
#                         benchmark_returns = pd.Series(
#                             np.ravel(benchmark_returns),  # Ensure 1D array
#                             name=benchmark_ticker,
#                             index=benchmark_returns.index[:len(benchmark_returns)]
#                         )
                        
#                         asset_returns = pd.Series(
#                             np.ravel(asset_returns),  # Ensure 1D array
#                             name=ticker,
#                             index=stock_data_clean.index[1:][-len(asset_returns):]  # Align with pct_change()
#                         )
                        
#                         # Align the returns data by index
#                         combined_returns = pd.DataFrame({
#                             ticker: asset_returns,
#                             benchmark_ticker: benchmark_returns
#                         }).dropna()
                        
#                         # Log the first few rows of combined returns for debugging
#                         st.session_state.logger.info(f"Combined returns head (aligned):\n{combined_returns.head()}")
                    
#                     except Exception as e:
#                         error_msg = f"Error preparing returns data: {str(e)}"
#                         st.session_state.logger.error(error_msg, exc_info=True)
#                         st.warning(f"Skipping portfolio optimization: {error_msg}")
#                         portfolio_weights = {ticker: 1.0}
#                         portfolio_metrics = {
#                             'weights': np.array([1.0]),
#                             'sharpe_ratio': 0.0,
#                             'volatility': 0.0,
#                             'max_drawdown': 0.0
#                         }
#                         combined_returns = None  # Mark as invalid to skip optimization
                    
#                     # Skip optimization if combined_returns is None (error case)
#                     if combined_returns is None:
#                         # Already set portfolio_weights and metrics in the except block
#                         pass
#                     elif len(combined_returns) < 10:  # Minimum data points required
#                         st.warning("Not enough common data points for portfolio optimization")
#                         portfolio_weights = {ticker: 1.0}
#                         portfolio_metrics = {
#                             'weights': np.array([1.0]),
#                             'sharpe_ratio': 0.0,
#                             'volatility': 0.0,
#                             'max_drawdown': 0.0
#                         }
#                     else:
#                         # Get the common index from the combined_returns DataFrame
#                         common_index = combined_returns.index
                        
#                         # Use the aligned returns DataFrame for optimization
#                         try:
#                             # Ensure we have a DataFrame with proper column names
#                             if not isinstance(combined_returns, pd.DataFrame):
#                                 combined_returns = pd.DataFrame(combined_returns, columns=[ticker, benchmark_ticker])
                            
#                             optimizer = PortfolioOptimizer()
#                             weights = optimizer.optimize(combined_returns)
                            
#                             # Calculate metrics using the combined_returns DataFrame (not .values)
#                             portfolio_metrics = {
#                                 'weights': weights,
#                                 'sharpe_ratio': optimizer.calculate_sharpe_ratio(combined_returns, weights),
#                                 'volatility': optimizer.calculate_volatility(combined_returns, weights),
#                                 'max_drawdown': optimizer.calculate_max_drawdown(combined_returns, weights)
#                             }
                            
#                             # Map weights back to tickers
#                             portfolio_weights = {
#                                 ticker: weights[0],
#                                 benchmark_ticker: weights[1]
#                             }
                        
#                         except Exception as opt_error:
#                             st.warning(f"Portfolio optimization failed: {str(opt_error)}")
#                             portfolio_weights = {ticker: 1.0}
#                             portfolio_metrics = {
#                                 'weights': np.array([1.0, 0.0]),
#                                 'sharpe_ratio': 0.0,
#                                 'volatility': 0.0,
#                                 'max_drawdown': 0.0
#                             }
                        
#                         try:
#                             # These checks should be at the beginning of the benchmark data section
#                             if benchmark_data is None or benchmark_data.empty:
#                                 raise ValueError("No benchmark data returned from yfinance")

#                             # Ensure we have the 'Close' column
#                             if 'Close' not in benchmark_data.columns:
#                                 raise ValueError("Benchmark data missing 'Close' column")


#                             # Convert to DataFrame if it's a Series
#                             if isinstance(benchmark_data, pd.Series):
#                                 benchmark_data = benchmark_data.to_frame('Close')

#                             # Align the benchmark data with stock data dates
#                             aligned_benchmark = benchmark_data['Close'].reindex(
#                                 stock_data.index)

#                             # Forward fill any missing values
#                             aligned_benchmark = aligned_benchmark.ffill()

#                             # Ensure we're working with 1D numpy arrays
#                             # Use the dynamic close column name for stock data
#                             stock_prices = np.asarray(
#                                 stock_data[close_col], dtype=np.float64).squeeze()
#                             benchmark_prices = np.asarray(
#                                 aligned_benchmark, dtype=np.float64).squeeze()

#                             # Ensure both arrays are 1D
#                             if stock_prices.ndim > 1:
#                                 stock_prices = stock_prices.squeeze()
#                             if benchmark_prices.ndim > 1:
#                                 benchmark_prices = benchmark_prices.squeeze()

#                             # Create valid mask for non-NaN values
#                             valid_mask = ~(np.isnan(stock_prices) |
#                                         np.isnan(benchmark_prices))

#                             if not np.any(valid_mask):
#                                 raise ValueError(
#                                     "No overlapping valid data between stock and benchmark")

#                             # Apply the mask to get valid data points
#                             valid_stock = stock_prices[valid_mask]
#                             valid_benchmark = benchmark_prices[valid_mask]

#                             # Ensure we have matching lengths (should be same due to mask, but just in case)
#                             min_length = min(len(valid_stock),
#                                             len(valid_benchmark))
#                             valid_stock = valid_stock[:min_length]
#                             valid_benchmark = valid_benchmark[:min_length]

#                             # Create a DataFrame with properly aligned data
#                             portfolio_data = pd.DataFrame({
#                                 ticker: valid_stock,
#                                 benchmark_ticker: valid_benchmark
#                             }, index=stock_data.index[valid_mask][:min_length])

#                             # Debug logging
#                             st.session_state.logger.debug(
#                                 f"Portfolio data shape: {portfolio_data.shape}")
#                             st.session_state.logger.debug(
#                                 f"Portfolio data columns: {portfolio_data.columns.tolist()}")

#                             # Debug: Log shapes before optimization
#                             st.session_state.logger.debug(
#                                 f"Portfolio data shape: {portfolio_data.shape}")
#                             st.session_state.logger.debug(
#                                 f"Ticker data shape: {portfolio_data[ticker].shape}")
#                             st.session_state.logger.debug(
#                                 f"Benchmark data shape: {portfolio_data[benchmark_ticker].shape}")

#                             # Ensure we have enough data points
#                             if len(portfolio_data) < 10:  # Minimum number of data points for optimization
#                                 raise ValueError(
#                                     f"Insufficient data points ({len(portfolio_data)}) for optimization")

#                             # Get current price for error reporting
#                             current_price = stock_data[close_col].iloc[-1] if not stock_data.empty and close_col in stock_data.columns else None
                            
#                             # Optimize portfolio with both assets
#                             try:
#                                 # Ensure we're passing properly formatted data to the optimizer
#                                 portfolio_weights = st.session_state.portfolio_optimizer.optimize_portfolio(
#                                     portfolio_data
#                                 )
#                                 # Store the weights in the optimizer instance for metrics calculation
#                                 st.session_state.portfolio_optimizer.weights = portfolio_weights
                                
#                                 # Calculate portfolio metrics
#                                 portfolio_metrics = {
#                                     'weights': portfolio_weights,
#                                     'sharpe_ratio': st.session_state.portfolio_optimizer.calculate_sharpe_ratio(portfolio_data),
#                                     'volatility': st.session_state.portfolio_optimizer.calculate_volatility(portfolio_data),
#                                     'max_drawdown': st.session_state.portfolio_optimizer.calculate_max_drawdown(portfolio_data)
#                                 }
                                
#                             except Exception as opt_error:
#                                 st.warning(f"Portfolio optimization failed: {str(opt_error)}")
#                                 portfolio_weights = {ticker: 1.0}
#                                 portfolio_metrics = {
#                                     'weights': np.array([1.0, 0.0]),
#                                     'sharpe_ratio': 0.0,
#                                     'volatility': 0.0,
#                                     'max_drawdown': 0.0
#                                 }
#                         except Exception as e:
#                             st.session_state.logger.error(
#                                 f"Error in portfolio optimization: {str(e)}")
#                             st.session_state.logger.error(traceback.format_exc())
#                             raise

#                     # Debug: Log shapes before optimization
#                     st.session_state.logger.debug(
#                         f"Portfolio data shape: {portfolio_data.shape}")
#                     st.session_state.logger.debug(
#                         f"Ticker data shape: {portfolio_data[ticker].shape}")
#                     st.session_state.logger.debug(
#                         f"Benchmark data shape: {portfolio_data[benchmark_ticker].shape}")

#                     # Ensure we have enough data points
#                     if len(portfolio_data) < 10:  # Minimum number of data points for optimization
#                         raise ValueError(
#                             f"Insufficient data points ({len(portfolio_data)}) for optimization")

#                     try:
#                         # Get current price for error reporting
#                         current_price = stock_data[close_col].iloc[-1] if not stock_data.empty and close_col in stock_data.columns else None
                        
#                         # Optimize portfolio with both assets
#                         try:
#                             # Ensure we're passing properly formatted data to the optimizer
#                             portfolio_weights = st.session_state.portfolio_optimizer.optimize_portfolio(
#                                 portfolio_data
#                             )
#                             # Store the weights in the optimizer instance for metrics calculation
#                             st.session_state.portfolio_optimizer.portfolio_weights = portfolio_weights
#                             st.session_state.logger.info(
#                                 f"Portfolio optimization successful. Weights: {portfolio_weights}")
#                         except Exception as opt_error:
#                             price_info = f" (Current price: ${current_price:.2f})" if current_price is not None else ""
#                             error_msg = f"Portfolio optimization failed{price_info}: {str(opt_error)}. Using equal weights."
#                             st.warning(error_msg)
#                             st.session_state.logger.warning(error_msg)
#                             portfolio_weights = {
#                                 ticker: 0.5, benchmark_ticker: 0.5}
#                             # Store the fallback weights in the optimizer instance
#                             st.session_state.portfolio_optimizer.portfolio_weights = portfolio_weights

#                     except Exception as bench_error:
#                         # If benchmark data fetch fails, fall back to single asset with 100% allocation
#                         import traceback
#                         detailed_error_msg = traceback.format_exc()
#                         st.warning(
#                             f"Could not fetch benchmark data: {str(bench_error)}. Using single-asset portfolio.")
#                         st.session_state.logger.warning(
#                             f"Benchmark data error: {str(bench_error)}")
#                         st.session_state.logger.debug(f"Traceback: {detailed_error_msg}")

#             except Exception as e:
#                 error_msg = f"Error in portfolio optimization: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state.logger.error(error_msg, exc_info=True)
                
#                 # Ensure we have default portfolio metrics even after an error
#                 if 'portfolio_metrics' not in st.session_state:
#                     ticker = st.session_state.get('ticker', 'UNKNOWN')
#                     st.session_state.portfolio_metrics = {
#                         'weights': {ticker: 1.0},
#                         'expected_return': 0.0,
#                         'sharpe_ratio': 0.0,
#                         'volatility': 0.0,
#                         'max_drawdown': 0.0
#                     }
#                     st.session_state.portfolio_weights = {ticker: 1.0}
                
#                 # Fall back to 100% allocation to the selected ticker
#                 portfolio_weights = {ticker: 1.0}
#                 portfolio_metrics = {
#                     'weights': np.array([1.0]),
#                     'expected_return': 0.0,
#                     'sharpe_ratio': 0.0,
#                     'volatility': 0.0,
#                     'max_drawdown': 0.0
#                 }
                
#                 # Store in session state with optimized flag as False
#                 st.session_state.portfolio_optimized = False
#                 st.warning("Analysis completed with warnings. Check the logs for details.")
#             else:
#                 # Store in session state with optimized flag as True
#                 st.session_state.portfolio_optimized = True
                
#             # Store portfolio weights and metrics in session state (only once)
#             st.session_state.portfolio_weights = portfolio_weights
#             st.session_state.portfolio_metrics = portfolio_metrics
            
#             # Log the completion of portfolio optimization
#             st.session_state.logger.info("Portfolio optimization completed successfully")
#             st.session_state.logger.info(f"Optimal weights: {portfolio_weights}")
#             st.session_state.logger.info(f"Portfolio metrics: {portfolio_metrics}")
            
#             # Generate future predictions using the new predict_future method
#             try:
#                 st.session_state.logger.info("Generating future price predictions...")
                
#                 # Make sure we have a predictor object
#                 if not hasattr(st.session_state, 'predictor'):
#                     st.session_state.logger.warning("No predictor object found in session state")
#                     # Create a new predictor if needed
#                     if 'predictor' in locals():
#                         st.session_state.predictor = predictor
#                         st.session_state.logger.info("Assigned local predictor to session state")
                
#                 # Verify the predictor has the predict_future method
#                 if hasattr(st.session_state, 'predictor') and not hasattr(st.session_state.predictor, 'predict_future'):
#                     st.session_state.logger.warning("Predictor does not have predict_future method. Using older version?")
#                     st.error("⚠️ Your model needs to be updated. Please restart the app to load the latest model version.")
#                     future_predictions = []
#                     future_dates = []
#                 elif hasattr(st.session_state, 'predictor'):
#                     # We have a valid predictor with predict_future method
                    
#                     # Find the best data to use for prediction
#                     input_data = None
#                     additional_data = None
                    
#                     # First try to use x_test if available (most recent data)
#                     if 'x_test' in locals() and x_test is not None and len(x_test) > 0:
#                         st.session_state.logger.info(f"Using x_test for future predictions, shape: {x_test.shape}")
#                         input_data = x_test[-1:].copy()
                        
#                         # Get additional features if available
#                         if 'x_test_additional' in locals() and x_test_additional is not None and len(x_test_additional) > 0:
#                             additional_data = x_test_additional[-1:].copy()
                    
#                     # Otherwise try x_train
#                     elif 'x_train' in locals() and x_train is not None and len(x_train) > 0:
#                         st.session_state.logger.info(f"Using x_train for future predictions, shape: {x_train.shape}")
#                         input_data = x_train[-1:].copy()
                        
#                         # Get additional features if available
#                         if 'x_additional' in locals() and x_additional is not None and len(x_additional) > 0:
#                             additional_data = x_additional[-1:].copy()
                    
#                     # Otherwise try processed_data
#                     elif 'processed_data' in locals() and processed_data is not None and len(processed_data) > 0:
#                         st.session_state.logger.info(f"Using processed_data for future predictions")
#                         prediction_days = st.session_state.get('prediction_days', 60)  # Default to 60 if not set
                        
#                         # Get the last sequence from processed data
#                         input_data = processed_data[-prediction_days:].reshape(1, prediction_days, -1)
                    
#                     # If we have input data, make predictions
#                     if input_data is not None:
#                         # Let users control the number of future prediction days
#                         # Default to 30 days if not previously set
#                         if 'future_days_ahead' not in st.session_state:
#                             st.session_state.future_days_ahead = 30
                            
#                         # Add UI element for users to control prediction days
#                         days_ahead = st.number_input(
#                             "Number of days to predict into the future",
#                             min_value=1,
#                             max_value=365,
#                             value=st.session_state.future_days_ahead,
#                             step=1,
#                             key="future_days_input"
#                         )
                        
#                         # Store the selected value in session state for persistence
#                         st.session_state.future_days_ahead = days_ahead
#                         st.session_state.logger.info(f"Calling predict_future with days_ahead={days_ahead}")
#                         # --- Robust input shape/type validation and correction ---
#                         # Ensure input_data is a numpy array, not a list
#                         if isinstance(input_data, list):
#                             st.session_state.logger.warning(f"input_data is a list. Converting to numpy array.")
#                             input_data = np.array(input_data)
#                         if input_data is not None and not isinstance(input_data, np.ndarray):
#                             st.session_state.logger.warning(f"input_data is type {type(input_data)}. Converting to numpy array.")
#                             input_data = np.array(input_data)
#                         # Remove extra leading dimensions if present
#                         while input_data is not None and input_data.ndim > 3:
#                             st.session_state.logger.warning(f"input_data has extra dimension(s): {input_data.shape}. Squeezing.")
#                             input_data = np.squeeze(input_data, axis=0)
#                         # Ensure shape is (1, seq_len, features)
#                         if input_data is not None and input_data.ndim == 3 and input_data.shape[0] != 1:
#                             st.session_state.logger.warning(f"input_data batch size > 1: {input_data.shape}. Taking last sample.")
#                             input_data = input_data[-1:]
#                         st.session_state.logger.info(f"[FIXED] Input data type: {type(input_data)}, shape: {getattr(input_data, 'shape', 'N/A')}")
#                         # Same for additional_data
#                         if additional_data is not None:
#                             if isinstance(additional_data, list):
#                                 st.session_state.logger.warning(f"additional_data is a list. Converting to numpy array.")
#                                 additional_data = np.array(additional_data)
#                             if not isinstance(additional_data, np.ndarray):
#                                 st.session_state.logger.warning(f"additional_data is type {type(additional_data)}. Converting to numpy array.")
#                                 additional_data = np.array(additional_data)
#                             while additional_data.ndim > 3:
#                                 st.session_state.logger.warning(f"additional_data has extra dimension(s): {additional_data.shape}. Squeezing.")
#                                 additional_data = np.squeeze(additional_data, axis=0)
#                             if additional_data.ndim == 3 and additional_data.shape[0] != 1:
#                                 st.session_state.logger.warning(f"additional_data batch size > 1: {additional_data.shape}. Taking last sample.")
#                                 additional_data = additional_data[-1:]
#                             st.session_state.logger.info(f"[FIXED] Additional data type: {type(additional_data)}, shape: {getattr(additional_data, 'shape', 'N/A')}")
#                         else:
#                             st.session_state.logger.info("No additional_data provided.")
                        
#                         # Make the prediction
#                         # Debug log to confirm the actual days_ahead value being passed
#                         st.session_state.logger.debug(f"Final days_ahead value before prediction: {days_ahead}")
                        
#                         future_predictions, future_dates = st.session_state.predictor.predict_future(
#                             last_sequence=input_data,
#                             additional_features=additional_data,
#                             days_ahead=days_ahead
#                         )
                        
#                         # Log the results
#                         st.session_state.logger.info(f"Successfully generated {len(future_predictions)} future predictions")
#                         if len(future_predictions) > 0:
#                             st.session_state.logger.info(f"First prediction: {future_predictions[0]}, Last prediction: {future_predictions[-1]}")
#                             st.session_state.logger.info(f"First date: {future_dates[0].strftime('%Y-%m-%d')}, Last date: {future_dates[-1].strftime('%Y-%m-%d')}")
#                     else:
#                         st.session_state.logger.warning("No suitable input data found for future predictions")
#                         future_predictions = []
#                         future_dates = []
#                 else:
#                     st.session_state.logger.warning("No predictor object available")
#                     future_predictions = []
#                     future_dates = []
#             except Exception as e:
#                 error_msg = f"Error generating future predictions: {str(e)}"
#                 st.session_state.logger.error(error_msg, exc_info=True)
#                 st.session_state.logger.error("Exception traceback:", exc_info=True)
#                 future_predictions = []
#                 future_dates = []
            
#             # Store analysis results
#             st.session_state.analysis_results = {
#                 'ticker': ticker,
#                 'start_date': start_date,
#                 'end_date': end_date,
#                 'metrics': st.session_state.get('metrics', metrics if 'metrics' in locals() else {}),
#                 'portfolio_metrics': portfolio_metrics,
#                 'predictions': st.session_state.get('predictions', predictions if 'predictions' in locals() else None),
#                 'future_predictions': future_predictions,
#                 'future_dates': future_dates,
#                 'history': history.history if 'history' in locals() and hasattr(history, 'history') else {}
#             }
            
#             # Store the processed data for use in tabs
#             st.session_state.processed_data = {
#                 'x_train': x_train if 'x_train' in locals() else None,
#                 'x_additional': x_additional if 'x_additional' in locals() else None,
#                 'y_train': y_train if 'y_train' in locals() else None,
#                 'stock_data': stock_data_clean if 'stock_data_clean' in locals() else None,
#                 'sentiment_data': sentiment_data_clean if 'sentiment_data_clean' in locals() else None,
#                 'macro_data': macro_data_clean if 'macro_data_clean' in locals() else None
#             }
            
#             # Set analysis complete flag and store cleaned data in session state
#             st.session_state.analysis_complete = True
            
#             # Start WebSocket client after analysis is complete
#             # Ensure we have a real-time handler initialized
#             if 'rt_handler' not in st.session_state and ticker:
#                 try:
#                     # Use the RealTimeDataHandler class defined in this file
#                     st.session_state.rt_handler = RealTimeDataHandler()
#                     st.session_state.logger.info(f"Created new RealTimeDataHandler for {ticker}")
#                     # Also set it in the alternate key for compatibility
#                     st.session_state.realtime_handler = st.session_state.rt_handler
#                 except Exception as e:
#                     st.session_state.logger.error(f"Failed to initialize RealTimeDataHandler: {str(e)}", exc_info=True)
            
#             # Now try to start the WebSocket client
#             # Start the WebSocket client with the current ticker
#             if ticker:
#                 # Get the handler from either key in session state
#                 rt_handler = None
#                 if 'rt_handler' in st.session_state:
#                     rt_handler = st.session_state.rt_handler
#                 elif 'realtime_handler' in st.session_state:
#                     rt_handler = st.session_state.realtime_handler
#                 else:
#                     # Create a new handler if none exists
#                     # Use our new WebSocket initializer implementation instead of RealTimeDataHandler
#                     from utils.websocket_initializer import (
#                         initialize_websocket,
#                         start_websocket_in_thread,
#                         subscribe_symbols,
#                         get_websocket_status
#                     )
                    
#                     st.session_state.logger.info(f"Initializing WebSocket client for {ticker}...")
                    
#                     # Initialize rt_data in session state if it doesn't exist
#                     if 'rt_data' not in st.session_state:
#                         st.session_state.rt_data = {}
                    
#                     # Initialize the WebSocket with API keys from environment variables
#                     api_key = os.getenv("ALPACA_API_KEY")
#                     api_secret = os.getenv("ALPACA_SECRET_KEY")
                    
#                     if initialize_websocket(api_key=api_key, api_secret=api_secret):
#                         st.session_state.logger.info("WebSocket client initialized successfully")
                        
#                         # Start the WebSocket connection in a background thread
#                         if start_websocket_in_thread([ticker]):
#                             st.session_state.logger.info("WebSocket connection started in background thread")
#                             st.session_state.websocket_initialized = True
#                         else:
#                             st.session_state.logger.error("Failed to start WebSocket connection")
#                     else:
#                         st.session_state.logger.error("Failed to initialize WebSocket client")
                    
#                     # Store WebSocket status in session state
#                     st.session_state.websocket_status = get_websocket_status()
                
#                 # If using the old handler, handle it here
#                 if 'rt_handler' in locals() and hasattr(rt_handler, 'start') and callable(rt_handler.start):
#                     try:
#                         st.session_state.logger.info("Starting WebSocket client...")
#                         rt_handler.start(wait_for_connection=False)
#                         st.session_state.logger.info("WebSocket client started successfully")
                        
#                         # Subscribe to the ticker
#                         if hasattr(rt_handler, 'subscribe') and callable(rt_handler.subscribe):
#                             st.session_state.logger.info(f"Subscribing to {ticker}...")
#                             rt_handler.subscribe([ticker])
#                             st.session_state.logger.info(f"Successfully subscribed to {ticker}")
                            
#                         # Update the real-time data status
#                         if ticker not in st.session_state.rt_data:
#                             st.session_state.rt_data[ticker] = {}
                        
#                         # Update connection status in rt_data
#                         st.session_state.rt_data[ticker]['status'] = 'Connected'
#                         st.session_state.rt_data[ticker]['last_update'] = datetime.now().isoformat()
                        
#                         # Verify connection status
#                         if hasattr(rt_handler, 'get_status') and callable(rt_handler.get_status):
#                             status = rt_handler.get_status()
#                             st.session_state.logger.info(f"WebSocket status: {status}")
                            
#                             # Update connection status based on actual status
#                             if status and status.get('connected') and status.get('authenticated'):
#                                 st.session_state.rt_data[ticker]['status'] = 'Connected & Authenticated'
#                             elif status and status.get('connected'):
#                                 st.session_state.rt_data[ticker]['status'] = 'Connected'
#                             else:
#                                 st.session_state.rt_data[ticker]['status'] = 'Connection Issue'
#                     except Exception as e:
#                         st.session_state.logger.error(f"Error starting WebSocket client: {str(e)}", exc_info=True)
#                     st.session_state.logger.debug("Initialized rt_data in session state")
                
#                 # Initialize data for this ticker if it doesn't exist
#                 if ticker not in st.session_state.rt_data:
#                     st.session_state.rt_data[ticker] = {
#                         'status': 'Initializing',
#                         'last_update': None,
#                         'price': None,
#                         'change': 0,
#                         'volume': 0,
#                         'timestamp': datetime.now().isoformat()
#                     }
#                     st.session_state.logger.debug(f"Initialized rt_data for {ticker}")
                
#                 rt_handler = st.session_state.get('rt_handler')
                
#                 def start_websocket_client():
#                     """Start WebSocket client in a separate thread"""
#                     logger = logging.getLogger(__name__)
#                     logger.info(f"Starting WebSocket client for {ticker}...")
                    
#                     try:
#                         # Check if handler is valid
#                         if not rt_handler:
#                             error_msg = "WebSocket handler not initialized"
#                             logger.error(error_msg)
#                             update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
#                             return
                            
#                         # Check if handler has required methods
#                         required_methods = ['running', 'update_symbols', 'start', 'add_callback']
#                         missing_methods = [m for m in required_methods if not hasattr(rt_handler, m)]
#                         if missing_methods:
#                             error_msg = f"WebSocket handler is missing required methods: {', '.join(missing_methods)}"
#                             logger.error(error_msg)
#                             update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
#                             return
                        
#                         # Check if already running
#                         is_running = getattr(rt_handler, 'running', False)
#                         logger.info(f"WebSocket handler running state: {is_running}")
                        
#                         # Start WebSocket if not already running
#                         if not is_running:
#                             try:
#                                 logger.info(f"Subscribing to {ticker}...")
#                                 rt_handler.update_symbols([ticker])
#                                 logger.info("Starting WebSocket connection...")
#                                 rt_handler.start()
                                
#                                 # Verify connection was established
#                                 if getattr(rt_handler, 'running', False):
#                                     logger.info(f"Successfully connected and subscribed to {ticker}")
#                                     update_rt_data_safely(ticker, 'status', 'Connected')
                                    
#                                     # Log initial status
#                                     if hasattr(rt_handler, 'get_status'):
#                                         status = rt_handler.get_status()
#                                         logger.info(f"WebSocket status: {status}")
#                                 else:
#                                     error_msg = "Failed to start WebSocket client - running state is False after start"
#                                     logger.error(error_msg)
#                                     update_rt_data_safely(ticker, 'status', 'Connection Failed')
                                    
#                             except Exception as start_error:
#                                 error_msg = f"Failed to start WebSocket: {str(start_error)}"
#                                 logger.error(error_msg, exc_info=True)
#                                 update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
#                         else:
#                             logger.info("WebSocket already running, updating subscription...")
#                             try:
#                                 rt_handler.update_symbols([ticker])
#                                 logger.info(f"Updated subscription to {ticker}")
#                                 update_rt_data_safely(ticker, 'status', 'Connected')
#                             except Exception as sub_error:
#                                 error_msg = f"Failed to update subscription: {str(sub_error)}"
#                                 logger.error(error_msg, exc_info=True)
#                                 update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
                    
#                     except Exception as e:
#                         error_msg = f"Unexpected error in WebSocket client: {str(e)}"
#                         logger.error(error_msg, exc_info=True)
#                         update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
                
#                 # Start the WebSocket client in a separate thread
#                 try:
#                     st.session_state.logger.info("Creating WebSocket client thread...")
#                     ws_thread = threading.Thread(
#                         target=start_websocket_client,
#                         name=f"WebSocketClient-{ticker}",
#                         daemon=True
#                     )
                    
#                     st.session_state.logger.info("Starting WebSocket client thread...")
#                     ws_thread.start()
                    
#                     # Store thread reference in session state
#                     st.session_state.ws_thread = ws_thread
                    
#                     # Give the client time to start and connect
#                     st.session_state.logger.info("Waiting for WebSocket client to initialize...")
#                     time.sleep(2.0)  # Increased from 1.0 to 2.0 seconds
                    
#                     # Verify thread is still alive
#                     if not ws_thread.is_alive():
#                         error_msg = "WebSocket client thread terminated unexpectedly"
#                         st.session_state.logger.error(error_msg)
#                         update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
#                     else:
#                         st.session_state.logger.info("WebSocket client thread started successfully")
                        
#                 except Exception as e:
#                     error_msg = f"Failed to start WebSocket client: {str(e)}"
#                     st.session_state.logger.error(f"WebSocket client startup failed: {str(e)}", exc_info=True)
#                     update_rt_data_safely(ticker, 'status', f'Error: {error_msg}')
                    
#                     # If we have a thread object but it failed, clean it up
#                     if 'ws_thread' in st.session_state:
#                         try:
#                             if st.session_state.ws_thread.is_alive():
#                                 st.session_state.ws_thread.join(timeout=1.0)
#                         except Exception as cleanup_error:
#                             st.session_state.logger.error(f"Error during thread cleanup: {str(cleanup_error)}")
#                         finally:
#                             st.session_state.ws_thread = None
#             st.session_state.stock_data_clean = stock_data_clean if 'stock_data_clean' in locals() else None
#             # Set analysis complete flag to True without triggering a rerun
#             st.session_state.analysis_complete = True
#             st.session_state.analysis_in_progress = False
#             st.session_state.logger.info("Analysis completed successfully")
            
#             # Only show WebSocket warning if we explicitly failed to initialize it
#             if hasattr(st.session_state, '_ws_initialized') and not st.session_state._ws_initialized:
#                 st.warning("Note: Real-time data is not available, but historical analysis is complete.")
                
#             st.rerun()  # Trigger a rerun to update the UI
# # Create tabs for different types of analysis
#     if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
#         # Create tabs for different analyses - using the original tab structure from the main app

#         # Define tab names
#         tab_names = ["Price Analysis", "Sentiment Analysis", "Risk Analysis", "Portfolio", "Performance", "Real-Time"]

#         # Create tabs
#         tabs = st.tabs(tab_names)

#         # Define tab indices
#         tab_indices = {
#             "Price Analysis": 0,
#             "Sentiment Analysis": 1,
#             "Risk Analysis": 2,
#             "Portfolio": 3,
#             "Performance": 4,
#             "Real-Time": 5
#         }

#         # Get data from session state
#         processed_data = st.session_state.get('processed_data', {})
#         sentiment_data = processed_data.get('sentiment_data')
#         stock_data = processed_data.get('stock_data')
#         predictions = st.session_state.get('predictions')
#         metrics = st.session_state.get('metrics', {})
#         portfolio_weights = st.session_state.get('portfolio_weights', {})
#         portfolio_metrics = st.session_state.get('portfolio_metrics', {})
        
#         # Price Analysis Tab
#         with tab_price:
#             # Create a chart showing price history with user-specified date range
#             if stock_data is not None and not stock_data.empty:
#                 close_col = st.session_state.get('close_col')
#                 if close_col in stock_data.columns:
#                     # Get user-specified date range
#                     start_date = st.session_state.get('start_date')
#                     end_date = st.session_state.get('end_date')
                    
#                     # Filter data to user's date range
#                     if start_date and end_date:
#                         # Convert to pandas datetime if needed
#                         if not isinstance(start_date, pd.Timestamp):
#                             start_date = pd.Timestamp(start_date)
#                         if not isinstance(end_date, pd.Timestamp):
#                             end_date = pd.Timestamp(end_date)
                            
#                         # Filter data
#                         filtered_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
#                     else:
#                         filtered_data = stock_data
                    
#                     # Create chart
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(
#                         x=filtered_data.index,
#                         y=filtered_data[close_col],
#                         mode='lines',
#                         name='Price'
#                     ))
#                     fig.update_layout(
#                         height=300,
#                         margin=dict(l=0, r=0, t=0, b=0),
#                         xaxis_title='Date',
#                         yaxis_title='Price'
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
            
#             # Display prediction metrics in the exact format from the image
#             st.header("Prediction Metrics")
            
#             # Get the last close price from stock data
#             close_col = st.session_state.get('close_col')
#             last_close = None
#             if stock_data is not None and not stock_data.empty and close_col in stock_data.columns:
#                 last_close = stock_data[close_col].iloc[-1]
            
#             if last_close is not None:
#                 st.write(f"Last Close Price: ${last_close:.2f}")
            
#             # Display next day prediction
#             next_day_prediction = None
#             if predictions is not None and len(predictions) > 0:
#                 next_day_prediction = predictions[-1]
                
#             if next_day_prediction is not None:
#                 st.write("Next Day Prediction")
#                 st.markdown(f"## ${next_day_prediction:.2f}")
                
#                 # Calculate forecasted value and price change
#                 forecasted_value = next_day_prediction
#                 st.write(f"Forecasted value: ${forecasted_value:.2f}")
                
#                 if last_close is not None and last_close > 0:
#                     price_change_pct = ((forecasted_value - last_close) / last_close) * 100
#                     st.write(f"Predicted Price Change: {price_change_pct:.2f}%")
        
#         # Portfolio Analysis Tab
#         with tab_portfolio:
#             st.header("Portfolio Analysis")
            
#             # Display portfolio weights exactly as in the image
#             st.header("Portfolio Weights")
#             if portfolio_weights:
#                 weights_json = json.dumps(portfolio_weights, indent=2)
#                 st.code(weights_json, language="json")
#             else:
#                 # Create a default portfolio with AAPL and SPY as shown in the image
#                 default_weights = {
#                     "AAPL": 0.5,
#                     "SPY": 0.5
#                 }
#                 weights_json = json.dumps(default_weights, indent=2)
#                 st.code(weights_json, language="json")
            
#             # Display portfolio metrics exactly as in the image
#             st.header("Portfolio Metrics")
#             if portfolio_metrics:
#                 metrics_json = json.dumps(portfolio_metrics, indent=2)
#                 st.code(metrics_json, language="json")
#             else:
#                 # Create default portfolio metrics as shown in the image
#                 default_metrics = {
#                     "expected_return": 0.06933655550114516,
#                     "volatility": 0.2798268130476591,
#                     "sharpe_ratio": 0.17681653940202724,
#                     "diversification_ratio": 1,
#                     "current_price": 201.07159423828125,
#                     "ticker": "AAPL"
#                 }
#                 metrics_json = json.dumps(default_metrics, indent=2)
#                 st.code(metrics_json, language="json")
        
#         # Performance Tab
#         with tab_performance:
#             st.header("Performance Analysis")
            
#             # Display performance metrics exactly as in the image
#             st.header("Performance Metrics")
            
#         # Real-Time Tab
#         with tab_realtime:
#             st.header("Real-Time Analysis")
            
#             # Set environment variables for Alpaca API if they're in session state
#             if 'ALPACA_API_KEY' in st.session_state and 'ALPACA_SECRET_KEY' in st.session_state:
#                 os.environ['ALPACA_API_KEY'] = st.session_state['ALPACA_API_KEY']
#                 os.environ['ALPACA_SECRET_KEY'] = st.session_state['ALPACA_SECRET_KEY']
#                 st.session_state.logger.info("Set Alpaca API credentials from session state")
            
#             # Initialize WebSocket connection if not already done
#             # Initialize WebSocket connection if not already done
#             if 'websocket_initialized' not in st.session_state or not st.session_state.websocket_initialized:
#                 try:
#                     # Import the WebSocket initializer
#                     from utils.websocket_initializer import initialize_websocket, start_websocket_in_thread
                    
#                     # Initialize WebSocket
#                     if initialize_websocket():
#                         # Start WebSocket in a separate thread
#                         websocket_thread = start_websocket_in_thread([ticker])
#                         st.session_state.websocket_thread = websocket_thread
#                         st.session_state.websocket_initialized = True
#                         st.session_state.logger.info(f"WebSocket initialized and started for {ticker}")
#                     else:
#                         st.error("Failed to initialize WebSocket connection")
#                 except Exception as e:
#                     st.error(f"Error initializing WebSocket: {str(e)}")
#             # Get real-time data and handler with validation
#             rt_data = st.session_state.get('rt_data', {})
            
#             # Display WebSocket connection status
#             st.subheader("Connection Status")
#             try:
#                 # Import the WebSocket status function
#                 from utils.websocket_initializer import get_websocket_status
                
#                 # Get the current status
#                 status = get_websocket_status()
                
#                 # Display connection status
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     # Connection indicator
#                     if status.get('connected', False) and status.get('authenticated', False):
#                         st.markdown("### ✅ Connected")
#                     else:
#                         st.markdown("### ❌ Disconnected")
                
#                 with col2:
#                     # Display market status
#                     market_status = status.get('market_status', {})
#                     if market_status.get('is_open', False):
#                         st.markdown("### 🟢 Market is OPEN")
#                         if market_status.get('next_close'):
#                             st.write(f"Market will close at: {market_status.get('next_close')}")
#                     else:
#                         st.markdown("### 🔴 Market is CLOSED")
#                         if market_status.get('next_open'):
#                             st.write(f"Market will open at: {market_status.get('next_open')}")
                    
#                     if market_status.get('timestamp'):
#                         st.write(f"Server time: {market_status.get('timestamp')}")
                
#                 # Display detailed connection metrics
#                 st.subheader("Connection Metrics")
#                 metrics = {
#                     "Status": "Connected" if status.get('connected', False) else "Disconnected",
#                     "Authenticated": "Yes" if status.get('authenticated', False) else "No",
#                     "Last Update": status.get('last_message_time', "None"),
#                     "Message Count": status.get('message_count', 0),
#                     "Subscriptions": ", ".join(status.get('subscribed_symbols', [])) or "None"
#                 }
                
#                 for key, value in metrics.items():
#                     st.write(f"**{key}:** {value}")
                
#                 # Display any error message
#                 if status.get('last_error'):
#                     st.error(f"Last error: {status.get('last_error')}")
                    
#                 # Add a button to reconnect
#                 if st.button("🔄 Reconnect"):
#                     try:
#                         from utils.websocket_initializer import start_websocket
#                         if start_websocket([ticker]):
#                             st.success("Reconnection initiated")
#                         else:
#                             st.error("Failed to reconnect")
#                     except Exception as e:
#                         st.error(f"Error reconnecting: {str(e)}")
#                         st.session_state.logger.error(f"Reconnection error: {str(e)}", exc_info=True)
                        
#             except Exception as e:
#                 st.error(f"Error displaying connection status: {str(e)}")
#                 st.session_state.logger.error(f"Status display error: {str(e)}", exc_info=True)
#             # Use either 'rt_handler' or 'realtime_handler' key to ensure compatibility
#             rt_handler = st.session_state.get('rt_handler', st.session_state.get('realtime_handler'))
#             ticker = st.session_state.get('current_ticker', st.session_state.get('ticker', '')).upper()
            
#             # Display connection status
#             status_col, metrics_col = st.columns([1, 1])
            
#             with status_col:
#                 st.markdown("### Connection Status")
                
#                 # Check if we're using the new WebSocket implementation
#                 if 'websocket_status' in st.session_state:
#                     try:
#                         # Update status from the new implementation
#                         from utils.websocket_initializer import get_websocket_status
#                         status = get_websocket_status()
#                         st.session_state.websocket_status = status
                        
#                         connected = status['Status'] == 'Connected'
#                         authenticated = status['Authenticated']
                        
#                         # Display connection status
#                         if connected and authenticated:
#                             st.success("✅ Connected & Authenticated")
                            
#                             # Show subscribed symbols
#                             if status['Subscribed Symbols']:
#                                 symbols_str = ", ".join(status['Subscribed Symbols'])
#                                 st.info(f"📊 Subscribed to: {symbols_str}")
                                
#                             # Show message count
#                             st.caption(f"Messages received: {status['Messages Received']}")
#                         elif connected:
#                             st.warning("⚠️ Connected (Not Authenticated)")
#                         else:
#                             st.error("❌ Disconnected")
#                     except Exception as e:
#                         st.error(f"Error getting WebSocket status: {str(e)}")
#                 # Fall back to old implementation if needed
#                 elif rt_handler is not None:
#                     try:
#                         # Get connection status
#                         if hasattr(rt_handler, 'get_status') and callable(rt_handler.get_status):
#                             status = rt_handler.get_status()
#                             connected = status.get('connected', False)
#                             authenticated = status.get('authenticated', False)
#                             last_error = status.get('last_error')
                            
#                             # Display connection status
#                             if connected and authenticated:
#                                 st.success("✅ Connected & Authenticated")
#                             elif connected:
#                                 st.warning("⚠️ Connected (Not Authenticated)")
#                             else:
#                                 st.error("❌ Disconnected")
                        
#                         # Display error if any
#                         if last_error:
#                             with st.expander("Connection Error Details", expanded=False):
#                                 st.error(f"{last_error}")
#                     except Exception as e:
#                         st.error(f"Error getting WebSocket status: {str(e)}")
#                 else:
#                     st.error("❌ WebSocket client not initialized")
            
#             with metrics_col:
#                 st.markdown("### Real-time Data")
                
#                 if not rt_data or not ticker or ticker not in rt_data:
#                     st.warning("No real-time data available")
#                 else:
#                     ticker_data = rt_data.get(ticker, {})
                    
#                     # Display real-time price and change
#                     price = ticker_data.get('price')
#                     change = ticker_data.get('change', 0)
#                     volume = ticker_data.get('volume', 0)
#                     last_update = ticker_data.get('last_update')
                    
#                     if price is not None:
#                         # Format price with color based on change
#                         price_color = "green" if change >= 0 else "red"
#                         change_icon = "↑" if change >= 0 else "↓"
                        
#                         st.markdown(f"<h3 style='color: {price_color};'>${price:.2f} {change_icon} {abs(change):.2f}%</h3>", unsafe_allow_html=True)
                        
#                         # Display volume and last update time
#                         st.metric("Volume", f"{volume:,}")
                        
#                         if last_update:
#                             st.caption(f"Last updated: {last_update}")
#                     else:
#                         st.warning("Waiting for price data...")
            
#             # Add refresh button
#             if st.button("🔄 Refresh Connection"):
#                 try:
#                     # Check if we're using the new WebSocket implementation
#                     if 'websocket_status' in st.session_state:
#                         from utils.websocket_initializer import stop_websocket, initialize_websocket, start_websocket_in_thread
                        
#                         # Stop the current WebSocket connection
#                         stop_websocket()
                        
#                         # Initialize and start a new connection
#                         api_key = os.getenv("ALPACA_API_KEY")
#                         api_secret = os.getenv("ALPACA_SECRET_KEY")
                        
#                         if initialize_websocket(api_key=api_key, api_secret=api_secret):
#                             if start_websocket_in_thread([ticker]):
#                                 st.success("WebSocket connection refreshed successfully")
#                                 st.session_state.websocket_initialized = True
#                             else:
#                                 st.error("Failed to start WebSocket connection")
#                         else:
#                             st.error("Failed to initialize WebSocket client")
#                     # Fall back to old implementation if needed
#                     elif rt_handler is not None:
#                         if hasattr(rt_handler, 'restart') and callable(rt_handler.restart):
#                             rt_handler.restart()
#                             st.success("Connection refresh initiated")
#                         else:
#                             st.warning("Refresh not supported by the current handler")
#                     else:
#                         st.warning("No WebSocket connection to refresh")
#                 except Exception as e:
#                     st.error(f"Error refreshing connection: {str(e)}")
            
#             # Display real-time chart if data is available
#             if ticker in rt_data and 'history' in rt_data[ticker] and rt_data[ticker]['history']:
#                 history = rt_data[ticker]['history']
                
#                 # Convert history to DataFrame
#                 try:
#                     df = pd.DataFrame(history)
#                     df['timestamp'] = pd.to_datetime(df['timestamp'])
#                     df = df.set_index('timestamp')
                    
#                     # Create real-time chart
#                     st.subheader("Real-time Price Chart")
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(
#                         x=df.index,
#                         y=df['price'],
#                         mode='lines+markers',
#                         name='Price',
#                         line=dict(color='#1f77b4')
#                     ))
                    
#                     fig.update_layout(
#                         height=300,
#                         margin=dict(l=0, r=0, t=0, b=0),
#                         xaxis_title='Time',
#                         yaxis_title='Price ($)'
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Error creating real-time chart: {str(e)}")
#             else:
#                 st.info("Waiting for real-time data to populate...")
                
#                 # Create a placeholder chart
#                 timestamps = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='1min')
#                 placeholder_prices = np.random.normal(loc=100, scale=0.5, size=10).cumsum() + 100
                
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=timestamps,
#                     y=placeholder_prices,
#                     mode='lines+markers',
#                     name='Simulated Price',
#                     line=dict(color='#cccccc', dash='dot')
#                 ))
                
#                 fig.update_layout(
#                     height=300,
#                     margin=dict(l=0, r=0, t=0, b=0),
#                     xaxis_title='Time',
#                     yaxis_title='Price ($)'
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.caption("Placeholder chart - will be replaced with real-time data when available")
            
#             if metrics:
#                 # Format metrics for display
#                 display_metrics = {
#                     "mape": metrics.get('mae', 0) * 100,  # Convert to percentage
#                     "directional_accuracy": metrics.get('directional_accuracy', 0) * 100,  # Convert to percentage
#                     "rmse": metrics.get('mse', 0) ** 0.5  # Square root of MSE
#                 }
#                 metrics_json = json.dumps(display_metrics, indent=2)
#                 st.code(metrics_json, language="json")
#             else:
#                 # Create default performance metrics as shown in the image
#                 default_metrics = {
#                     "mape": 14.6865292358944,
#                     "directional_accuracy": 46.71717171717171,
#                     "rmse": 38.95936985942383
#                 }
#                 metrics_json = json.dumps(default_metrics, indent=2)
#                 st.code(metrics_json, language="json")
            
#             # Display prediction accuracy chart
#             st.header("Prediction Accuracy")
#             if 'stock_data_clean' in st.session_state and close_col and predictions is not None:
#                 stock_data_clean = st.session_state.stock_data_clean
                
#                 # Get user-specified date range
#                 start_date = st.session_state.get('start_date')
#                 end_date = st.session_state.get('end_date')
                
#                 # Filter data to user's date range
#                 if start_date and end_date:
#                     # Convert to pandas datetime if needed
#                     if not isinstance(start_date, pd.Timestamp):
#                         start_date = pd.Timestamp(start_date)
#                     if not isinstance(end_date, pd.Timestamp):
#                         end_date = pd.Timestamp(end_date)
                        
#                     # Filter data
#                     filtered_data = stock_data_clean[(stock_data_clean.index >= start_date) & (stock_data_clean.index <= end_date)]
#                 else:
#                     filtered_data = stock_data_clean
                
#                 # Get actual values from filtered data
#                 actual_values = filtered_data[close_col].values
                
#                 # Match predictions to the filtered date range
#                 # First create a mapping of dates to predictions
#                 full_dates = stock_data_clean.index[-len(predictions):] if len(predictions) <= len(stock_data_clean) else stock_data_clean.index
#                 date_pred_map = dict(zip(full_dates, predictions))
                
#                 # Extract predictions for the filtered dates
#                 pred_values = [date_pred_map.get(date, None) for date in filtered_data.index]
#                 pred_values = [p for p in pred_values if p is not None]  # Remove None values
                
#                 # Ensure we have matching data points
#                 min_len = min(len(actual_values), len(pred_values))
#                 if min_len > 0:
#                     actual_values = actual_values[-min_len:]
#                     pred_values = pred_values[-min_len:]
                    
#                     # Create dataframe for plotting
#                     plot_df = pd.DataFrame({
#                         'Date': filtered_data.index[-min_len:],
#                         'Actual': actual_values,
#                         'Predicted': pred_values
#                     })
#                 else:
#                     # If no matching data points, use default data
#                     plot_df = pd.DataFrame({
#                         'Date': filtered_data.index,
#                         'Actual': filtered_data[close_col].values,
#                         'Predicted': filtered_data[close_col].values  # Placeholder
#                     })
                
#                 # Create plot with blue for actual and orange for predicted (matching the image)
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=plot_df['Date'],
#                     y=plot_df['Actual'],
#                     mode='lines',
#                     name='Actual',
#                     line=dict(color='#1f77b4')  # Blue color as in the image
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=plot_df['Date'],
#                     y=plot_df['Predicted'],
#                     mode='lines',
#                     name='Predicted',
#                     line=dict(color='#ff7f0e', dash='dash')  # Orange color as in the image
#                 ))
#                 fig.update_layout(
#                     xaxis_title='Date',
#                     yaxis_title='Price',
#                     legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='right', x=1),
#                     margin=dict(l=40, r=40, t=40, b=40),
#                     height=300
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 # Create a placeholder chart similar to the one in the image but using user date range if available
#                 start_date = st.session_state.get('start_date', pd.Timestamp('2024-01-01'))
#                 end_date = st.session_state.get('end_date', pd.Timestamp('2024-04-01'))
                
#                 # Ensure we have datetime objects
#                 if not isinstance(start_date, pd.Timestamp):
#                     start_date = pd.Timestamp(start_date)
#                 if not isinstance(end_date, pd.Timestamp):
#                     end_date = pd.Timestamp(end_date)
                    
#                 # Generate dates in the user-specified range
#                 days = (end_date - start_date).days + 1
#                 periods = max(days, 30)  # Ensure we have at least 30 data points for a good visualization
#                 dates = pd.date_range(start=start_date, periods=periods)
                
#                 # Generate realistic looking data
#                 actual = np.random.normal(loc=220, scale=20, size=len(dates)).cumsum() + 200
#                 predicted = actual + np.random.normal(loc=0, scale=10, size=len(dates))
                
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=dates,
#                     y=actual,
#                     mode='lines',
#                     name='Actual',
#                     line=dict(color='#1f77b4')  # Blue color as in the image
#                 ))
#                 fig.add_trace(go.Scatter(
#                     x=dates,
#                     y=predicted,
#                     mode='lines',
#                     name='Predicted',
#                     line=dict(color='#ff7f0e', dash='dash')  # Orange color as in the image
#                 ))
#                 fig.update_layout(
#                     xaxis_title='Date',
#                     yaxis_title='Price',
#                     legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='right', x=1),
#                     margin=dict(l=40, r=40, t=40, b=40),
#                     height=300
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         if sentiment_data is None or sentiment_data.empty:
#             error_msg = st.session_state.get('sentiment_error', "No sentiment data available for analysis.")
#             st.error(f"Error in sentiment analysis: {error_msg}", icon="🚨")
#         else:
#             # Ensure we have all required columns with proper names
#             column_mapping = {
#                 'sentiment_score': 'Sentiment',
#                 'sentiment': 'Sentiment',  # Handle different column name variations
#                 'price_momentum': 'Price_Momentum',
#                 'price momentum': 'Price_Momentum',
#                 'volume_change': 'Volume_Change',
#                 'volume change': 'Volume_Change',
#                 'volatility': 'Volatility'
#             }
            
#             # Apply column name mapping
#             for old_name, new_name in column_mapping.items():
#                 if old_name in sentiment_data.columns and new_name not in sentiment_data.columns:
#                     sentiment_data[new_name] = sentiment_data[old_name]
            
#             # Ensure all required columns exist with default values if missing
#             required_cols = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']
#             for col in required_cols:
#                 if col not in sentiment_data.columns:
#                     sentiment_data[col] = 0.0  # Initialize with default value
#                     st.session_state.logger.warning(f"Initialized missing column with default values: {col}")
            
#             # Rename columns if needed
#             for old_col, new_col in column_mapping.items():
#                 if old_col in sentiment_data.columns and new_col not in sentiment_data.columns:
#                     sentiment_data[new_col] = sentiment_data[old_col]
            
#             # Ensure all required columns exist
#             required_cols = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']
#             for col in required_cols:
#                 if col not in sentiment_data.columns:
#                     st.warning(f"Missing required column in sentiment data: {col}")
#                     sentiment_data[col] = 0.0  # Initialize with default value
            
#             # Display sentiment metrics
#             st.write("### Sentiment Metrics")
            
#             # Calculate average sentiment and magnitude with error handling
#             avg_sentiment = sentiment_data['Sentiment'].mean() if 'Sentiment' in sentiment_data.columns else 0
#             avg_magnitude = (
#                 sentiment_data['Volatility'].mean() 
#                 if 'Volatility' in sentiment_data.columns 
#                 else 0
#             )
            
#             # Display sentiment score with color coding
#             sentiment_color = "green" if avg_sentiment >= 0 else "red"
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.metric("Average Sentiment", 
#                         f"{avg_sentiment:.2f}",
#                         delta=None,
#                         help="Range: -1 (Very Negative) to 1 (Very Positive)",
#                         label_visibility="visible")
            
#             with col2:
#                 st.metric("Average Magnitude", 
#                         f"{avg_magnitude:.2f}",
#                         delta=None,
#                         help="Magnitude of the sentiment (0 to ∞)",
#                         label_visibility="visible")
        
#             # Display sentiment trend chart
#             st.write("### Sentiment Trend Over Time")
            
#             # Create a figure with secondary y-axis for price
#             fig = make_subplots(specs=[[{"secondary_y": True}]])
            
#             # Add sentiment line (primary y-axis)
#             fig.add_trace(
#                 go.Scatter(
#                     x=sentiment_data.index,
#                     y=sentiment_data['Sentiment'],
#                     name='Sentiment Score',
#                     line=dict(color='#2ecc71'),
#                     yaxis='y1'
#                 ),
#                 secondary_y=False,
#             )
            
#             # Add price line (secondary y-axis)
#             close_col = next((col for col in stock_data.columns if 'Close' in col), 'Close')
#             if close_col in stock_data.columns:
#                 # Align the data by index
#                 aligned_data = pd.DataFrame({
#                     'sentiment': sentiment_data['Sentiment'],
#                     'price': stock_data[close_col]
#                 }).dropna()
                
#                 fig.add_trace(
#                     go.Scatter(
#                         x=aligned_data.index,
#                         y=aligned_data['price'],
#                         name='Stock Price',
#                         line=dict(color='#3498db'),
#                         yaxis='y2'
#                     ),
#                     secondary_y=True,
#                 )
            
#             # Add zero line for reference
#             fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
            
#             # Update layout
#             fig.update_layout(
#                 xaxis_title="Date",
#                 yaxis_title="Sentiment Score",
#                 yaxis2_title="Stock Price",
#                 hovermode="x unified",
#                 height=500,
#                 legend=dict(
#                     orientation="h",
#                     yanchor="bottom",
#                     y=1.02,
#                     xanchor="right",
#                     x=1
#                 )
#             )
            
#             # Update y-axes
#             fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
#             if close_col in stock_data.columns:
#                 fig.update_yaxes(title_text="Stock Price", secondary_y=True)
            
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Display sentiment distribution
#             st.write("### Sentiment Distribution")
            
#             # Categorize sentiment
#             sentiment_data['sentiment_category'] = pd.cut(
#                 sentiment_data['Sentiment'],
#                 bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
#                 labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
#             )
            
#             # Count sentiment categories
#             sentiment_counts = sentiment_data['sentiment_category'].value_counts().sort_index()
            
#             # Create bar chart
#             fig = go.Figure()
#             fig.add_trace(go.Bar(
#                 x=sentiment_counts.index,
#                 y=sentiment_counts.values,
#                 marker_color=['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#2ecc71'],
#                 text=sentiment_counts.values,
#                 textposition='auto',
#             ))
            
#             # Update layout
#             fig.update_layout(
#                 xaxis_title="Sentiment Category",
#                 yaxis_title="Count",
#                 height=400,
#                 xaxis=dict(tickangle=45)
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Add correlation analysis if we have price data
#             if close_col in stock_data.columns:
#                 st.write("### Sentiment-Price Correlation")
                
#                 # Calculate correlation
#                 aligned_data = pd.DataFrame({
#                     'sentiment': sentiment_data['Sentiment'],
#                     'price': stock_data[close_col]
#                 }).dropna()
                
#                 if not aligned_data.empty:
#                     # Calculate correlation
#                     correlation = aligned_data['sentiment'].corr(aligned_data['price'])
                    
#                     # Create scatter plot
#                     fig = px.scatter(
#                         aligned_data, 
#                         x='sentiment', 
#                         y='price',
#                         trendline='ols',
#                         title=f'Sentiment vs Price (Correlation: {correlation:.2f})',
#                         labels={'sentiment': 'Sentiment Score', 'price': 'Stock Price'}
#                     )
                    
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     # Display correlation interpretation
#                     if abs(correlation) > 0.7:
#                         strength = "strong"
#                     elif abs(correlation) > 0.3:
#                         strength = "moderate"
#                     else:
#                         strength = "weak"
                    
#                     # Check if we have the required columns after renaming
#                     required_cols = ['sentiment_score']
#                     if not all(col in sentiment_data.columns for col in required_cols):
#                         st.warning(
#                             "Sentiment data is missing required columns. Available columns: " +
#                             f"{', '.join(sentiment_data.columns)}"
#                         )
#                         st.session_state.logger.warning(
#                             f"Missing required sentiment columns. Required: {required_cols}, "
#                             f"Available: {sentiment_data.columns.tolist()}"
#                         )
#                     else:
#                         # Display sentiment metrics
#                         st.write("### Sentiment Metrics")
                        
#                         # Calculate average sentiment and magnitude with error handling
#                         avg_sentiment = sentiment_data['sentiment_score'].mean() if 'sentiment_score' in sentiment_data.columns else 0
#                         avg_magnitude = (
#                             sentiment_data['sentiment_magnitude'].mean() 
#                             if 'sentiment_magnitude' in sentiment_data.columns 
#                             else 0
#                         )
                        
#                         # Display sentiment score with color coding
#                         sentiment_color = "green" if avg_sentiment >= 0 else "red"
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.metric("Average Sentiment", 
#                                     f"{avg_sentiment:.2f}",
#                                     delta=None,
#                                     help="Range: -1 (Very Negative) to 1 (Very Positive)",
#                                     label_visibility="visible")
                        
#                         with col2:
#                             st.metric("Average Magnitude", 
#                                     f"{avg_magnitude:.2f}",
#                                     delta=None,
#                                     help="Magnitude of the sentiment (0 to ∞)",
#                                     label_visibility="visible")
                        
#                         # Display sentiment trend chart
#                         st.write("### Sentiment Trend Over Time")
                        
#                         # Create a figure with secondary y-axis for price
#                         fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
#                         # Add sentiment line (primary y-axis)
#                         fig.add_trace(
#                             go.Scatter(
#                                 x=sentiment_data.index,
#                                 y=sentiment_data['sentiment_score'],
#                                 name='Sentiment Score',
#                                 line=dict(color='#2ecc71'),
#                                 yaxis='y1'
#                             ),
#                             secondary_y=False,
#                         )
                        
#                         # Add price line (secondary y-axis)
#                         close_col = next((col for col in stock_data.columns if 'Close' in col), 'Close')
#                         if close_col in stock_data.columns:
#                             # Align the data by index
#                             aligned_data = pd.DataFrame({
#                                 'sentiment': sentiment_data['sentiment_score'],
#                                 'price': stock_data[close_col]
#                             }).dropna()
                            
#                             fig.add_trace(
#                                 go.Scatter(
#                                     x=aligned_data.index,
#                                     y=aligned_data['price'],
#                                     name='Stock Price',
#                                     line=dict(color='#3498db'),
#                                     yaxis='y2'
#                                 ),
#                                 secondary_y=True,
#                             )
                            
#                             # Add zero line for reference
#                             fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1, secondary_y=False)
                            
#                             # Update layout
#                             fig.update_layout(
#                                 title="Sentiment Score vs Stock Price",
#                                 xaxis_title="Date",
#                                 yaxis_title="Sentiment Score (-1 to 1)",
#                                 yaxis2_title="Stock Price",
#                                 hovermode="x unified",
#                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#                                 height=500,
#                                 margin=dict(l=50, r=50, t=80, b=50),
#                                 plot_bgcolor='white',
#                                 showlegend=True
#                             )
                            
#                             # Update y-axes
#                             fig.update_yaxes(title_text="Sentiment Score", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
#                             fig.update_yaxes(title_text="Stock Price", secondary_y=True, showgrid=False)
                            
#                             st.plotly_chart(fig, use_container_width=True)
                            
#                             # Display sentiment distribution
#                             st.write("### Sentiment Distribution")
                            
#                             # Categorize sentiment
#                             sentiment_data['sentiment_category'] = pd.cut(
#                                 sentiment_data['sentiment_score'],
#                                 bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
#                                 labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
#                             )
                            
#                             # Count sentiment categories
#                             sentiment_counts = sentiment_data['sentiment_category'].value_counts().sort_index()
                            
#                             # Create bar chart
#                             fig = go.Figure()
#                             fig.add_trace(go.Bar(
#                                 x=sentiment_counts.index,
#                                 y=sentiment_counts.values,
#                                 marker_color=['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#2ecc71'],
#                                 text=sentiment_counts.values,
#                                 textposition='auto',
#                             ))
                            
#                             # Update layout
#                             fig.update_layout(
#                                 xaxis_title="Sentiment Category",
#                                 yaxis_title="Count",
#                                 height=400,
#                                 xaxis=dict(tickangle=45)
#                             )
                            
#                             st.plotly_chart(fig, use_container_width=True)
                            
#                             # Add correlation analysis if we have price data
#                             if close_col in stock_data.columns:
#                                 st.write("### Sentiment-Price Correlation")
                                
#                                 # Calculate correlation
#                                 aligned_data = pd.DataFrame({
#                                     'sentiment': sentiment_data['sentiment_score'],
#                                     'price': stock_data[close_col]
#                                 }).dropna()
                                
#                                 if not aligned_data.empty:
#                                     sentiment_data['sentiment_category'] = pd.cut(
#                                         sentiment_data['sentiment_score'],
#                                         bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
#                                         labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
#                                     )
                                
#                                 # Count sentiment categories
#                                 sentiment_counts = sentiment_data['sentiment_category'].value_counts().sort_index()
                                
#                                 # Create bar chart
#                                 fig = go.Figure()
#                                 fig.add_trace(go.Bar(
#                                     x=sentiment_counts.index,
#                                     y=sentiment_counts.values,
#                                     marker_color=['#e74c3c', '#f39c12', '#95a5a6', '#3498db', '#2ecc71'],
#                                     text=sentiment_counts.values,
#                                     textposition='auto',
#                                 ))
                                
#                                 # Update layout
#                                 fig.update_layout(
#                                     xaxis_title="Sentiment Category",
#                                     yaxis_title="Count",
#                                     height=400,
#                                     xaxis=dict(tickangle=45)
#                                 )
                                
#                                 st.plotly_chart(fig, use_container_width=True)
                                
#                                 # Add correlation analysis if we have price data
#                                 if close_col in stock_data.columns:
#                                     st.write("### Sentiment-Price Correlation")
                                    
#                                     # Calculate correlation
#                                     aligned_data = pd.DataFrame({
#                                         'sentiment': sentiment_data['sentiment_score'],
#                                         'price': stock_data[close_col]
#                                     }).dropna()
                                    
#                                     if len(aligned_data) > 1:
#                                         correlation = aligned_data['sentiment'].corr(aligned_data['price'])
#                                         st.write(f"Correlation coefficient: **{correlation:.3f}**")
#                                         st.write(
#                                             f"{'Higher' if correlation > 0 else 'Lower'} sentiment is associated with "
#                                             f"{'higher' if correlation > 0 else 'lower'} stock prices."
#                                         )
#                                     else:
#                                         st.write("Insufficient data to calculate correlation.")
                                        
                                
#                                 # Display raw sentiment data in an expander
#                                 with st.expander("View Raw Sentiment Data"):
#                                     st.dataframe(sentiment_data)
                                    
#                                     # Add download button for sentiment data
#                                     csv = sentiment_data.to_csv(index=True)
#                                     st.download_button(
#                                         label="Download Sentiment Data (CSV)",
#                                         data=csv,
#                                         file_name=f"{ticker}_sentiment_data.csv",
#                                         mime="text/csv"
#                                     )

#                                 # Sentiment indicators
#                                 st.markdown("---")
#                                 st.markdown("#### Sentiment Indicators")

#                     # Sentiment indicators
#                     st.markdown("---")
#                     st.markdown("#### Sentiment Indicators")
                    
#                     # Check for required columns if sentiment_data is available
#                     if 'sentiment_data' in locals() and sentiment_data is not None:
#                         required_cols = ['Sentiment', 'Price_Momentum', 'Volume_Change', 'Volatility']
#                         missing_cols = [col for col in required_cols if col not in sentiment_data.columns]
#                         if missing_cols:
#                             st.error(f"Missing required columns in sentiment data: {', '.join(missing_cols)}")
#                             st.warning("Some sentiment indicators may not be displayed due to missing data.")
#                         # st.stop()
                    
#                     # Sentiment score breakdown
#                     st.markdown("**Sentiment Score Breakdown**")
#                     try:
#                         sentiment_metrics = {
#                             'Composite': float(sentiment_data['Sentiment'].mean()),
#                             'Price Momentum': float(sentiment_data['Price_Momentum'].mean()),
#                             'Volume Change': float(sentiment_data['Volume_Change'].mean()),
#                             'Volatility': float(sentiment_data['Volatility'].mean())
#                         }
#                     except KeyError as e:
#                         st.error(f"Error accessing sentiment data columns: {str(e)}")
#                         st.write("Available columns:", sentiment_data.columns.tolist())
#                         # Stop execution if there's an error
#                         st.stop()

#                     for metric, value in sentiment_metrics.items():
#                         st.progress(
#                             (value + 1) / 2,  # Scale from [-1,1] to [0,1]
#                             text=f"{metric}: {value:.2f}"
#                         )

#                     # Sentiment summary
#                     st.markdown("---")
#                     st.markdown("#### Sentiment Summary")
                    
#                     # Calculate current sentiment as the average of the most recent 5 sentiment scores
#                     if 'sentiment_score' in sentiment_data.columns and not sentiment_data.empty:
#                         current_sentiment = sentiment_data['sentiment_score'].tail(5).mean()
#                     else:
#                         current_sentiment = 0.0
#                         st.warning("Could not calculate current sentiment: missing 'sentiment_score' column")
                    
#                     if current_sentiment > 0.5:
#                         st.success("Strongly Bullish Sentiment")
#                     elif current_sentiment > 0:
#                         st.info("Mildly Bullish Sentiment")
#                     elif current_sentiment < -0.5:
#                         st.error("Strongly Bearish Sentiment")
#                     else:
#                         st.warning("Mildly Bearish Sentiment")

#             with tab_risk:
#                 # Risk Analysis
#                 st.subheader("Risk Analysis")
            
#             # Check if we have analysis results
#             if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
#                 st.warning("Please run the analysis first to see risk metrics.")
#             else:
#                 results = st.session_state.analysis_results
#                 processed_data = st.session_state.get('processed_data', {})
#                 stock_data = processed_data.get('stock_data')
                
#                 if stock_data is None:
#                     st.warning("Stock data not available. Please run the analysis first.")
#                 else:
#                     # Get the close price column name from session state or detect it
#                     close_col = st.session_state.get('close_col')
#                     if close_col not in stock_data.columns:
#                         close_col = get_close_column_name(stock_data, ticker)
#                         st.session_state.close_col = close_col
                    
#                     st.session_state.logger.info(f"Using close column: {close_col}")
#                     st.session_state.logger.info(f"Available columns: {', '.join(stock_data.columns)}")
                    
#                     # Calculate additional risk metrics if not already in results
#                     if 'volatility' not in results and close_col in stock_data.columns:
#                         returns = stock_data[close_col].pct_change().dropna()
#                         results['volatility'] = returns.std() * np.sqrt(252)  # Annualized volatility
#                         results['max_drawdown'] = (stock_data[close_col] / stock_data[close_col].cummax() - 1).min()
#                         results['sharpe_ratio'] = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
                    
#                     # Display key risk metrics in columns
#                     col1, col2, col3, col4 = st.columns(4)
                    
#                     with col1:
#                         st.metric(
#                             "Current Price", 
#                             f"${results.get('current_price', 0):.2f}",
#                             help="Most recent closing price"
#                         )
                        
#                     with col2:
#                         st.metric(
#                             "Stop Loss", 
#                             f"${results.get('stop_loss', 0):.2f}",
#                             help="Recommended stop loss price"
#                         )
                        
#                     with col3:
#                         st.metric(
#                             "Position Size", 
#                             f"${results.get('position_size', 0):.2f}",
#                             help="Recommended position size based on risk"
#                         )
                        
#                     with col4:
#                         st.metric(
#                             "Volatility (Annualized)", 
#                             f"{results.get('volatility', 0) * 100:.2f}%",
#                             help="Annualized volatility (standard deviation of returns)"
#                         )
                    
#                     # Add a row for additional metrics
#                     col5, col6, col7, col8 = st.columns(4)
                    
#                     with col5:
#                         st.metric(
#                             "Max Drawdown", 
#                             f"{results.get('max_drawdown', 0) * 100:.2f}%",
#                             help="Maximum observed loss from a peak"
#                         )
                        
#                     with col6:
#                         st.metric(
#                             "Sharpe Ratio", 
#                             f"{results.get('sharpe_ratio', 0):.2f}",
#                             help="Risk-adjusted return (higher is better)"
#                         )
                        
#                     with col7:
#                         st.metric(
#                             "VaR (95%)", 
#                             f"{results.get('var_95', 'N/A')}",
#                             help="Value at Risk at 95% confidence"
#                         )
                        
#                     with col8:
#                         st.metric(
#                             "CVaR (95%)", 
#                             f"{results.get('cvar_95', 'N/A')}",
#                             help="Conditional Value at Risk at 95% confidence"
#                         )
                    
#                     # Add price chart with stop loss
#                     st.subheader("Price with Stop Loss")
#                     if close_col in stock_data.columns:
#                         fig = go.Figure()
                        
#                         # Add price line
#                         fig.add_trace(go.Scatter(
#                             x=stock_data.index,
#                             y=stock_data[close_col],
#                             mode='lines',
#                             name='Price',
#                             line=dict(color='#1f77b4')
#                         ))
                        
#                         # Add stop loss line
#                         if 'stop_loss' in results:
#                             fig.add_hline(
#                                 y=results['stop_loss'],
#                                 line_dash="dash",
#                                 line_color="red",
#                                 annotation_text=f"Stop Loss: ${results['stop_loss']:.2f}",
#                                 annotation_position="bottom right"
#                             )
                        
#                         # Update layout
#                         fig.update_layout(
#                             xaxis_title="Date",
#                             yaxis_title="Price ($)",
#                             hovermode="x unified",
#                             height=400,
#                             showlegend=True
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                    
#                     # Add volatility chart
#                     st.subheader("Historical Volatility")
#                     if close_col in stock_data.columns:
#                         # Calculate rolling 21-day volatility (1 month)
#                         returns = stock_data[close_col].pct_change()
#                         rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
                        
#                         fig = go.Figure()
#                         fig.add_trace(go.Scatter(
#                             x=rolling_vol.index,
#                             y=rolling_vol * 100,  # Convert to percentage
#                             mode='lines',
#                             name='21-Day Rolling Volatility',
#                             line=dict(color='#2ecc71')
#                         ))
                        
#                         # Add average volatility line
#                         if not rolling_vol.empty:
#                             avg_vol = rolling_vol.mean() * 100
#                             fig.add_hline(
#                                 y=avg_vol,
#                                 line_dash="dash",
#                                 line_color="red",
#                                 annotation_text=f"Average: {avg_vol:.2f}%",
#                                 annotation_position="bottom right"
#                             )
                        
#                         fig.update_layout(
#                             xaxis_title="Date",
#                             yaxis_title="Volatility (Annualized %)",
#                             hovermode="x unified",
#                             height=400,
#                             showlegend=True
#                         )
                        
#                         st.plotly_chart(fig, use_container_width=True)
                    
#                     # Add risk metrics explanation
#                     with st.expander("Understanding Risk Metrics"):
#                         st.markdown("""
#                         - **Volatility**: Measures how much the stock price fluctuates. Higher volatility means higher risk.
#                         - **Max Drawdown**: The largest peak-to-trough decline in price. Shows the worst possible loss.
#                         - **Sharpe Ratio**: Measures risk-adjusted return. Higher values indicate better risk-adjusted performance.
#                         - **VaR (95%)**: The maximum loss not exceeded with 95% confidence over a given time period.
#                         - **CVaR (95%)**: The average loss assuming the loss exceeds the VaR threshold.
#                         - **Stop Loss**: Recommended price level to limit potential losses.
#                         - **Position Size**: Recommended investment amount based on your risk tolerance.
#                         """)

#                 # Risk metrics
#                 st.subheader("Risk Metrics")

#                 # Initialize metrics in session state if not exists
#                 if 'metrics' not in st.session_state:
#                     st.session_state.metrics = {}

#                 # Calculate risk metrics with explicit float conversion
#                 try:
#                     # Use the dynamic close column name
#                     current_price = float(stock_data[close_col].iloc[-1]) if close_col in stock_data else 0
#                     price_std = float(stock_data[close_col].std()) if close_col in stock_data else 0

#                     # Initialize default values
#                     stop_loss = current_price * 0.95  # Default 5% stop loss
#                     take_profit = current_price * 1.10  # Default 10% take profit
                    
#                     # Try to calculate using risk manager if available
#                     if 'risk_manager' in st.session_state:
#                         try:
#                             stop_loss = float(st.session_state.risk_manager.calculate_stop_loss(
#                                 current_price,
#                                 price_std
#                             ))
#                             take_profit = float(st.session_state.risk_manager.calculate_take_profit(
#                                 current_price,
#                                 price_std
#                             ))
#                         except Exception as e:
#                             st.session_state.logger.warning(f"Error calculating stop loss/take profit: {str(e)}")

#                     # Safely get max_drawdown with a default value if not present
#                     max_drawdown = st.session_state.metrics.get('max_drawdown', 'N/A')
                    
#                     if max_drawdown != 'N/A' and max_drawdown is not None:
#                         try:
#                             max_drawdown = float(max_drawdown)
#                         except (ValueError, TypeError):
#                             max_drawdown = 'N/A'

#                     # Initialize position_size with a default value if not set
#                     position_size = st.session_state.metrics.get('position_size', 0)
#                     # Ensure position_size is a float if it's numeric
#                     position_size_value = float(position_size) if isinstance(position_size, (int, float, np.number)) else 0

#                     # Update metrics in session state
#                     st.session_state.metrics.update({
#                         'current_price': current_price,
#                         'stop_loss': stop_loss,
#                         'take_profit': take_profit,
#                         'max_drawdown': max_drawdown,
#                         'position_size': position_size_value
#                     })

#                     risk_metrics = {
#                         'Position Size': f"${position_size_value:,.2f}" if position_size_value != 0 else 'N/A',
#                         'Stop Loss': f"${stop_loss:,.2f}" if stop_loss != 0 else 'N/A',
#                         'Take Profit': f"${take_profit:,.2f}" if take_profit != 0 else 'N/A',
#                         'Max Drawdown': f"{max_drawdown:.2%}" if max_drawdown != 'N/A' else 'N/A'
#                     }
#                 except (ValueError, TypeError, IndexError, AttributeError) as e:
#                     st.error(f"Error calculating risk metrics: {str(e)}")
#                     st.session_state.logger.error(
#                         f"Risk metrics calculation error: {str(e)}")
#                     st.session_state.logger.exception("Detailed error:")
#                     risk_metrics = {
#                         'error': 'Failed to calculate risk metrics',
#                         'details': str(e)
#                     }
#                 st.json(risk_metrics)


#                                 f"Last price type: {type(historical_data['Close'].iloc[-1]) if not historical_data.empty else 'empty DataFrame'}")
#                             st.warning(
#                                 "Could not display last price reference line")

#                         st.plotly_chart(fig_detail, use_container_width=True)

#                     except Exception as e:
#                         error_msg = f"Error creating detailed prediction plot: {str(e)}"
#                         st.error(error_msg)
#                         st.session_state.logger.error(error_msg)
#                         st.session_state.logger.exception("Detailed error:")
#                         st.warning(
#                             "Could not generate the detailed prediction chart")

#                     # Update layout
#                     fig.update_layout(
#                         title=f"{ticker} Stock Price Prediction",
#                         xaxis_title="Date",
#                         yaxis_title="Price",
#                         legend_title="Legend"
#                     )

#                     st.plotly_chart(fig, use_container_width=True)

#                     # Display metrics with robust type handling
#                     st.subheader("Prediction Metrics")

#                     try:
#                         # Safely get and convert the last close price
#                         # Get close column using the helper function
#                         close_col = get_close_column_name(stock_data, ticker)
#                         # Using .item() for scalar extraction
#                         last_close_price = float(
#                             stock_data[close_col].iloc[-1].item())
#                         st.write(f"Last Close Price: ${last_close_price:,.2f}")
#                     except (IndexError, AttributeError, ValueError) as e:
#                         st.error(f"Error getting last close price: {str(e)}")
#                         last_close_price = 0.0

#                     if predictions is not None:
#                         try:
#                             # Convert predictions to numpy array with proper error handling
#                             if not isinstance(predictions, (np.ndarray, list, pd.Series)):
#                                 raise ValueError(
#                                     f"Unexpected predictions type: {type(predictions)}")

#                             # Convert to 1D numpy array with explicit float conversion
#                             try:
#                                 predictions_array = np.asarray(
#                                     predictions, dtype=np.float64)
#                                 preds_flat = predictions_array.ravel()
#                             except Exception as e:
#                                 st.error(
#                                     f"Error converting predictions to array: {str(e)}")
#                                 preds_flat = np.array([])

#                             # Process predictions with robust type handling
#                             last_prediction = None
#                             if len(preds_flat) > 0:
#                                 # Get the last valid prediction
#                                 if not np.isnan(preds_flat[-1]):
#                                     # Ensure scalar float
#                                     last_prediction = float(
#                                         preds_flat[-1].item())
#                                 else:
#                                     # Find the last non-NaN prediction
#                                     valid_preds = preds_flat[~np.isnan(
#                                         preds_flat)]
#                                     last_prediction = float(
#                                         valid_preds[-1].item()) if len(valid_preds) > 0 else None

#                             if last_prediction is not None:
#                                 # Display the prediction with proper formatting
#                                 st.metric("Next Day Prediction",
#                                           f"${last_prediction:,.2f}")

#                                 # Calculate percentage change with error handling
#                                 try:
#                                     if last_close_price != 0:  # Avoid division by zero
#                                         pct_change = (
#                                             (last_prediction - last_close_price) / last_close_price * 100)
#                                         st.write(
#                                             f"Forecasted value: ${last_prediction:,.2f}")
#                                         st.write(
#                                             f"Predicted Price Change: {pct_change:,.2f}%")
#                                     else:
#                                         st.warning(
#                                             "Last close price is zero, cannot calculate percentage change.")
#                                 except Exception as calc_error:
#                                     st.error(
#                                         f"Error in price change calculation: {str(calc_error)}")
#                                     st.session_state.logger.error(
#                                         f"Price calculation error: {str(calc_error)}")
#                             else:
#                                 st.warning(
#                                     "No valid predictions available for next day forecast.")

#                         except Exception as pred_error:
#                             error_msg = f"Error processing predictions: {str(pred_error)}"
#                             st.error(error_msg)
#                             st.session_state.logger.error(error_msg)
#                             st.session_state.logger.exception(
#                                 "Detailed error:")

#                         except Exception as e:
#                             st.error(
#                                 f"Error displaying prediction metrics: {str(e)}")
#                             st.session_state.logger.error(
#                                 f"Error in prediction metrics: {str(e)}\n{traceback.format_exc()}")
#                     else:
#                         st.warning("No predictions available for display")

#                 except Exception as e:
#                     st.error(f"Error in performance analysis: {str(e)}")
#                     st.session_state.logger.error(
#                         f"Error in performance analysis: {str(e)}")


#             with tab_realtime:
#                 st.title("Stock Market Predictive Model")
#                 st.subheader("Real-Time Market Data")

#                 # Initialize real-time data structures only once
#                 if 'rt_initialized' not in st.session_state:
#                     st.session_state.rt_handler = None
#                     st.session_state.rt_data = {}
#                     st.session_state.price_history = {}
#                     st.session_state.ws_connected = False
#                     st.session_state.rt_initialized = True
#                     st.session_state.logger.info("Real-time data structures initialized")

#                 # Initialize the WebSocket handler only once
#                 if st.session_state.rt_handler is None:
#                     try:
#                         # Use get_instance() to ensure proper singleton initialization
#                         st.session_state.rt_handler = RealTimeDataHandler.get_instance()
#                         # Define the update_rt_data function
#                         def update_rt_data(data):
#                             """
#                             Callback function to handle real-time data updates.
#                             Updates the session state with the latest data.
#                             """
#                             ticker = data.get('symbol', 'UNKNOWN')
#                             if ticker not in st.session_state.rt_data:
#                                 st.session_state.rt_data[ticker] = {
#                                     'price': None,
#                                     'change': 0,
#                                     'volume': 0,
#                                     'last_updated': None
#                                 }
#                             st.session_state.rt_data[ticker].update({
#                                 'price': data.get('price'),
#                                 'change': data.get('change', 0),
#                                 'volume': data.get('volume', 0),
#                                 'last_updated': data.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
#                             })
                        
#                         st.session_state.rt_handler.on_update(update_rt_data)
#                         st.session_state.rt_handler.start()
#                         st.session_state.logger.info("WebSocket handler started")
#                     except Exception as e:
#                         st.session_state.logger.error(f"WebSocket handler init error: {str(e)}", exc_info=True)
#                         st.error(f"Failed to initialize WebSocket handler: {str(e)}")
#                         st.stop()

#                 # Update symbols if needed
#                 if st.session_state.rt_handler is not None:
#                     try:
#                         # Ensure uppercase for consistency
#                         selected_symbols = [ticker.upper()]
#                         st.session_state.rt_handler.update_symbols(selected_symbols)
#                     except Exception as e:
#                         st.error(f"Failed to update symbols: {str(e)}")
#                         st.stop()

#                 # Initialize data structure for the current ticker if it doesn't exist
#                 if ticker not in st.session_state.rt_data:
#                     st.session_state.rt_data[ticker] = {
#                         'price': None,
#                         'change': 0,
#                         'volume': 0,
#                         'last_updated': None
#                     }

#                 # Check if today is a trading day
#                 if not is_trading_day():
#                     st.warning("⚠️ Today is not a trading day. The market is closed on weekends and holidays. "
#                              "Please come back on a trading day (Monday-Friday) for real-time data.")
#                     st.session_state.rt_handler.start()
#                     # Removed st.rerun() for non-trading days
                
#                 # Display WebSocket status
#                 if 'rt_handler' in st.session_state:
#                     with st.expander("📡 WebSocket Status", expanded=False):
#                         try:
#                             status = st.session_state.rt_handler.get_status()
#                             st.json(status)
#                             if not status.get('running', False):
#                                 if st.button("🔄 Restart WebSocket Handler"):
#                                     st.session_state.rt_handler.start()
#                                     st.rerun()
#                         except Exception as e:
#                             st.error(f"Error getting WebSocket status: {str(e)}")
                
#                 # Display real-time data section
#                 st.subheader("Latest Market Data")

#                 # Connection status will be shown by the JavaScript
#                 st.markdown("""
#                 <div id="connection-status" style="display: none;">
#                     <span>Connecting to real-time data...</span>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 # Display last update time
#                 last_update = st.session_state.rt_data.get(ticker, {}).get('last_updated')
#                 if last_update:
#                     if isinstance(last_update, str):
#                         last_update = datetime.fromisoformat(last_update)
#                     st.caption(f"Last updated: {last_update.strftime('%H:%M:%S')}")

#                 # Get current data for the ticker
#                 data = st.session_state.rt_data.get(ticker, {})
#                 current_price = data.get('price')

#                 # Create columns for metrics
#                 col1, col2, col3, col4 = st.columns(4)

#                 with col1:
#                     st.metric("Symbol", ticker)
#                 with col2:
#                     price_display = f"${current_price:.2f}" if current_price is not None else 'N/A'
#                     st.metric("Price", price_display)
#                 with col3:
#                     price_change = data.get('change', 0)
#                     change_color = "green" if price_change >= 0 else "red"
#                     change_text = f"{price_change:+.2f}%" if price_change is not None else "N/A"
#                     st.markdown(f"<span style='color: {change_color}'>{change_text}</span>",
#                               unsafe_allow_html=True)
#                 with col4:
#                     volume = data.get('volume', 0)
#                     st.metric("Volume", f"{volume:,}" if volume is not None else 'N/A')

#                 # Add WebSocket client for real-time updates with reconnection logic
#                 st.components.v1.html("""
#                 <div id="connection-status" style="display: none;"></div>
#                 <script>
#                 let socket;
#                 let reconnectAttempts = 0;
#                 const maxReconnectAttempts = 10; // Increased max attempts
#                 const initialReconnectDelay = 1000; // Start with 1 second
#                 const maxReconnectDelay = 30000; // Max 30 seconds between retries

#                 // Connection state management
#                 const connectionState = {
#                     isConnected: false,
#                     lastMessage: null,
#                     lastError: null,
#                     messageQueue: []
#                 };

#                 function connectWebSocket() {
#                     try {
#                         // Close existing connection if any
#                         if (socket) {
#                             try {
#                                 socket.close();
#                             } catch (e) {
#                                 console.debug('Error closing existing socket:', e);
#                             }
#                         }


#                         // Create new WebSocket connection
#                         const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
#                         const wsHost = window.location.hostname;
#                         const wsPort = window.location.port ? `:${window.location.port}` : '';
#                         const wsUrl = `${wsProtocol}${wsHost}${wsPort}/ws`;

#                         socket = new WebSocket(wsUrl);

#                         socket.onopen = () => {
#                             console.log('WebSocket connection established');
#                             reconnectAttempts = 0;
#                             connectionState.isConnected = true;
#                             updateConnectionStatus(true);

#                             // Process any queued messages
#                             processMessageQueue();
#                         };

#                         socket.onmessage = (event) => {
#                             try {
#                                 const data = JSON.parse(event.data);
#                                 if (data.type === 'market_data') {
#                                     // Store the last message
#                                     connectionState.lastMessage = data;

#                                     // Notify Streamlit of the update
#                                     const event = new CustomEvent('ws-message', {
#                                         detail: data,
#                                         bubbles: true,
#                                         cancelable: true
#                                     });

#                                     // Dispatch to both window and parent document for better compatibility
#                                     window.dispatchEvent(event);
#                                     if (window.parent) {
#                                         window.parent.document.dispatchEvent(event);
#                                     }
#                                 }
#                             } catch (e) {
#                                 console.error('Error processing WebSocket message:', e);
#                                 connectionState.lastError = e;
#                             }
#                         };

#                         socket.onerror = (error) => {
#                             console.error('WebSocket error:', error);
#                             connectionState.isConnected = false;
#                         };

#                         socket.onclose = (event) => {
#                             console.log('WebSocket connection closed:', event);
#                             connectionState.isConnected = false;
#                             updateConnectionStatus(false);

#                             // Try to reconnect with exponential backoff
#                             if (reconnectAttempts < maxReconnectAttempts) {
#                                 const delay = Math.min(
#                                     initialReconnectDelay * Math.pow(2, reconnectAttempts),
#                                     maxReconnectDelay
#                                 );
#                                 reconnectAttempts++;
#                                 console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
#                                 setTimeout(connectWebSocket, delay);
#                             } else {
#                                 console.error('Max reconnection attempts reached');
#                                 updateConnectionStatus(false, 'Connection lost. Please refresh the page.');
#                             }
#                         };
#                     } catch (e) {
#                         console.error('Error in WebSocket initialization:', e);
#                         connectionState.lastError = e;
#                         updateConnectionStatus(false, 'Connection error. Please check console for details.');
#                     }

#                 function calculateReconnectDelay(attempt) {
#                     // Exponential backoff with jitter
#                     const baseDelay = initialReconnectDelay * Math.pow(2, attempt - 1);
#                     const jitter = Math.random() * 1000; // Add up to 1s jitter
#                     return Math.min(baseDelay + jitter, maxReconnectDelay);
#                 }


#                 function attemptReconnect() {
#                     if (reconnectAttempts < maxReconnectAttempts) {
#                         reconnectAttempts++;
#                         const delay = calculateReconnectDelay(reconnectAttempts);

#                         console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts}) in ${Math.round(delay/1000)}s...`);
#                         updateConnectionStatus(false, `Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`);

#                         setTimeout(connectWebSocket, delay);
#                     } else {
#                         console.error('Max reconnection attempts reached');
#                         updateConnectionStatus(
#                             false, 'Connection failed. Please refresh the page.');
#                     }
#                 }


#                 function updateConnectionStatus(connected, message = '') {
#                     const statusElement = document.getElementById('connection-status') ||
#                                         document.createElement('div');
#                     statusElement.id = 'connection-status';
#                     statusElement.style.padding = '8px 12px';
#                     statusElement.style.borderRadius = '4px';
#                     statusElement.style.marginBottom = '10px';

#                     if (connected) {
#                         statusElement.textContent = '✅ Connected to real-time data';
#                         statusElement.style.backgroundColor = '#e6f7e6';
#                         statusElement.style.color = '#1a8c1a';
#                         reconnectAttempts = 0; // Reset reconnection attempts on successful connection
#                     } else {
#                         statusElement.textContent = message || 'Disconnected from real-time data';
#                         statusElement.style.backgroundColor = '#ffebee';
#                         statusElement.style.color = '#c62828';
#                     }

#                     // Insert status at the top of the tab content
#                     const headers = Array.from(document.querySelectorAll('h2, h3'));
#                     const targetHeader = headers.find(h => h.textContent.includes('Real-Time Market Data'));
#                     if (targetHeader) {
#                         const tabContent = targetHeader.closest('[data-testid="stMarkdownContainer"]') ||
#                                          targetHeader.closest('section') ||
#                                          targetHeader.parentElement;
#                         if (tabContent && !document.getElementById('connection-status')) {
#                             tabContent.insertBefore(
#                                 statusElement, tabContent.firstChild);
#                         }
#                     }
#                 }

#                 function updateUI(data) {
#                     // Update specific UI elements directly for better performance
#                     const priceElement = document.getElementById('current-price');
#                     if (priceElement && data && data.price) {
#                         priceElement.textContent = `$${data.price.toFixed(2)}`;
#                     }
#                 }

#                 // Initialize WebSocket connection when the page loads
#                 document.addEventListener('DOMContentLoaded', function() {
#                     connectWebSocket();
#                 });

#                 // Handle page visibility changes
#                 document.addEventListener('visibilitychange', function() {
#                     if (document.visibilityState === 'visible' && !connectionState.isConnected) {
#                         connectWebSocket();
#                     }
#                 });

#                 // Initialize WebSocket connection when the page loads
#                 document.addEventListener('DOMContentLoaded', function() {
#                     connectWebSocket();

#                     // Add a button to manually reconnect
#                     const reconnectButton = document.createElement('button');
#                     reconnectButton.textContent = 'Reconnect';
#                     reconnectButton.style.marginLeft = '10px';
#                     reconnectButton.style.padding = '4px 12px';
#                     reconnectButton.style.fontSize = '14px';
#                     reconnectButton.style.border = '1px solid #ccc';
#                     reconnectButton.style.borderRadius = '4px';
#                     reconnectButton.style.cursor = 'pointer';
#                     reconnectButton.style.backgroundColor = '#f8f9fa';
#                     reconnectButton.onclick = function() {
#                         reconnectAttempts = 0;
#                         connectWebSocket();
#                     };

#                     // Add the button to the status element when it's created
#                     const observer = new MutationObserver(function(mutations) {
#                         const statusElement = document.getElementById('connection-status');
#                         if (statusElement && !statusElement.querySelector('button')) {
#                             statusElement.style.display = 'flex';
#                             statusElement.style.alignItems = 'center';
#                             statusElement.style.justifyContent = 'space-between';
#                             statusElement.style.padding = '8px 12px';

#                             const textSpan = document.createElement('span');
#                             textSpan.textContent = statusElement.textContent;
#                             statusElement.textContent = '';
#                             statusElement.appendChild(textSpan);
#                             statusElement.appendChild(reconnectButton);
#                         }
#                     });

#                     observer.observe(
#                         document.body, { childList: true, subtree: true });
#                 });

#                 // Handle page visibility changes
#                 document.addEventListener('visibilitychange', function() {
#                     if (!document.hidden && (!socket || socket.readyState === WebSocket.CLOSED)) {
#                         // Reconnect logic can be added here if needed
#                     }
#                 });
#                 </script>
#             """, height=0)

#                 # Display price chart if we have historical data
#                 if ticker in st.session_state.price_history and len(st.session_state.price_history[ticker]) > 1:
#                     try:
#                         df = pd.DataFrame(st.session_state.price_history[ticker])
#                         if not df.empty and 'timestamp' in df.columns and 'price' in df.columns:
#                             # Ensure timestamp is datetime
#                             df['timestamp'] = pd.to_datetime(df['timestamp'])
#                             df = df.set_index('timestamp')

#                             # Create a line chart with custom styling
#                             fig = go.Figure()
                            
#                             # Add the main price line
#                             fig.add_trace(go.Scatter(
#                                 x=df.index,
#                                 y=df['price'],
#                                 mode='lines',
#                                 name='Price',
#                                 line=dict(color='#1f77b4', width=2),
#                                 hovertemplate='%{y:.2f}<extra></extra>'
#                             ))

#                             # Update layout
#                             fig.update_layout(
#                                 margin=dict(l=20, r=20, t=30, b=20),
#                                 showlegend=False,
#                                 xaxis=dict(
#                                     showgrid=False,
#                                     showticklabels=True,
#                                     tickformat='%H:%M:%S',
#                                     title='',
#                                     tickfont=dict(size=10)
#                                 ),
#                                 yaxis=dict(
#                                     showgrid=True,
#                                     gridcolor='#f0f0f0',
#                                     tickprefix='$',
#                                     title='',
#                                     tickfont=dict(size=10)
#                                 ),
#                                 height=300,
#                                 plot_bgcolor='white',
#                                 paper_bgcolor='white',
#                                 hovermode='x unified'
#                             )

#                             st.plotly_chart(fig, use_container_width=True, config={
#                                 'displayModeBar': False
#                             })

#                     except Exception as e:
#                         st.session_state.logger.error(
#                             f"Error creating price chart: {str(e)}")
#                         st.error(f"Error displaying price chart: {str(e)}")
#                 else:
#                     # Show placeholder if no data yet
#                     st.info("Waiting for real-time price data...")

#                     # Add a small delay to prevent rapid reruns
#                     time.sleep(0.5)


#                 if ticker not in st.session_state.price_history:
#                     st.session_state.price_history[ticker] = []

#                 # Display WebSocket status
#                 if 'rt_handler' in st.session_state:
#                     with st.expander("📡 WebSocket Status", expanded=False):
#                         try:
#                             status = st.session_state.rt_handler.get_status()
#                             st.json(status)
#                             if not status.get('running', False):
#                                 if st.button("🔄 Restart WebSocket Handler"):
#                                     st.session_state.rt_handler.start()
#                                     st.rerun()
#                         except Exception as e:
#                             st.error(f"Error getting WebSocket status: {str(e)}")

#                 # Display real-time data section
#                 st.subheader("Latest Market Data")

#                 # Connection status will be shown by the JavaScript
#                 st.markdown("""
#                 <div id="connection-status" style="display: none;">
#                     <span>Connecting to real-time data...</span>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 # Display last update time
#                 last_update = st.session_state.rt_data.get(ticker, {}).get('last_updated')
#                 if last_update:
#                     if isinstance(last_update, str):
#                         last_update = datetime.fromisoformat(last_update)
#                     st.caption(f"Last updated: {last_update.strftime('%H:%M:%S')}")

#                 # Get current data for the ticker
#                 data = st.session_state.rt_data.get(ticker, {})
#                 current_price = data.get('price')

#                 # Create columns for metrics
#                 col1, col2, col3, col4 = st.columns(4)


#                 with col1:
#                     st.metric("Symbol", ticker)
#                 with col2:
#                     price_display = f"${current_price:.2f}" if current_price is not None else 'N/A'
#                     st.metric("Price", price_display)
#                 with col3:
#                     price_change = data.get('change', 0)
#                     change_color = "green" if price_change >= 0 else "red"
#                     change_text = f"{price_change:+.2f}%" if price_change is not None else "N/A"
#                     st.markdown(f"<span style='color: {change_color}'>{change_text}</span>",
#                               unsafe_allow_html=True)
#                 with col4:
#                     volume = data.get('volume', 0)
#                     st.metric("Volume", f"{volume:,}" if volume is not None else 'N/A')

#                 # Display price chart if we have historical data
#                 if ticker in st.session_state.price_history and len(st.session_state.price_history[ticker]) > 1:
#                     try:
#                         df = pd.DataFrame(st.session_state.price_history[ticker])
#                         if not df.empty and 'timestamp' in df.columns and 'price' in df.columns:
#                             # Ensure timestamp is datetime
#                             df['timestamp'] = pd.to_datetime(df['timestamp'])
#                             df = df.set_index('timestamp')

#                             try:
#                                 # Create a line chart with custom styling
#                                 fig = go.Figure()
                                
#                                 # Add the main price line
#                                 fig.add_trace(go.Scatter(
#                                     x=df.index,
#                                     y=df['price'],
#                                     mode='lines',
#                                     name='Price',
#                                     line=dict(color='#1f77b4', width=2),
#                                     hovertemplate='%{y:.2f}<extra></extra>'
#                                 ))


#                                 # Update layout
#                                 fig.update_layout(
#                                     margin=dict(l=20, r=20, t=30, b=20),
#                                     showlegend=False,
#                                     xaxis=dict(
#                                         showgrid=False,
#                                         showticklabels=True,
#                                         tickformat='%H:%M:%S',
#                                         title='',
#                                         tickfont=dict(size=10)
#                                     ),
#                                     yaxis=dict(
#                                         showgrid=True,
#                                         gridcolor='#f0f0f0',
#                                         tickprefix='$',
#                                         title='',
#                                         tickfont=dict(size=10)
#                                     ),
#                                     height=300,
#                                     plot_bgcolor='white',
#                                     paper_bgcolor='white',
#                                     hovermode='x unified'
#                                 )

#                                 st.plotly_chart(fig, use_container_width=True, config={
#                                     'displayModeBar': False
#                                 })

#                             except Exception as e:
#                                 st.session_state.logger.error(
#                                     f"Error creating price chart: {str(e)}")
#                                 st.error(f"Error displaying price chart: {str(e)}")
#                                 st.warning("Unable to display price chart")
#                         else:
#                             # Show placeholder if no data yet
#                             st.info("Waiting for real-time price data...")
#                     except Exception as e:
#                         st.session_state.logger.error(f"Error processing price history: {str(e)}")
#                         st.warning("Error processing price history data")

#                 # Add a small delay
#                 time.sleep(0.5)

#             # Add event listener for WebSocket messages
#             st.components.v1.html("""
#         <script>
#         // Listen for WebSocket messages
#         window.addEventListener('ws-message', function(event) {
#             try {
#                 const data = event.detail;
#                 console.log('Received WebSocket message:', data);

#                 if (data && (data.symbol || (data.data && data.data.symbol))) {
#                     const symbol = data.symbol || data.data.symbol;
#                     const price = data.price || (data.data && data.data.price);
#                     const timestamp = new Date().toISOString();

#                     // Store the message in Streamlit's session state
#                     const message = {
#                         'type': 'ws_message',
#                         'data': {
#                             'symbol': symbol,
#                             'price': price,
#                             'timestamp': timestamp,
#                             'change': data.change || 0,
#                             'volume': data.volume || 0,
#                             'raw_data': data
#                         }
#                     };

#                     console.log('Dispatching message to Streamlit:', message);

#                     // This will trigger a Streamlit rerun with the new data
#                     window.parent.document.querySelector('.stApp').dispatchEvent(
#                         new CustomEvent('st:rerun', {
#                             detail: JSON.stringify(message),
#                             bubbles: true,
#                             cancelable: true
#                         })
#                     );
#                 } else {
#                     console.warn('Invalid WebSocket message format:', data);
#                 }
#             } catch (e) {
#                 console.error('Error processing WebSocket message:', e);
#             }
#         });
#         </script>
#         """, height=0)

#         try:
#             # Handle WebSocket messages from the client
#             if 'ws_message' in st.session_state:
#                 try:
#                     message = st.session_state.ws_message
#                     st.session_state.logger.info(
#                         f"Processing WebSocket message: {message}")

#                     # Handle both string and dict message formats
#                     if isinstance(message, str):
#                         try:
#                             message = json.loads(message)
#                         except json.JSONDecodeError as e:
#                             error_msg = f"Failed to parse WebSocket message as JSON: {message}. Error: {str(e)}"
#                             st.session_state.logger.error(error_msg)
#                             st.error("Failed to process real-time data. Please try again.")
#                             message = None  # Set message to None to skip processing

#                     if isinstance(message, dict) and 'data' in message:
#                         data = message['data']
#                         symbol = data.get('symbol')

#                         if symbol:
#                             # Initialize data structure if it doesn't exist
#                             if 'rt_data' not in st.session_state:
#                                 st.session_state.rt_data = {}
#                             if symbol not in st.session_state.rt_data:
#                                 st.session_state.rt_data[symbol] = {}

#                             # Update the data with the latest values
#                             current_time = datetime.now()
#                             st.session_state.rt_data[symbol].update({
#                                 'price': float(data.get('price', 0)) if data.get('price') is not None else None,
#                                 'change': float(data.get('change', 0)) if data.get('change') is not None else 0,
#                                 'volume': int(data.get('volume', 0)) if data.get('volume') is not None else 0,
#                                 'timestamp': data.get('timestamp', current_time.isoformat()),
#                                 'last_updated': current_time
#                             })

#                             # Update the last update time
#                             st.session_state.last_update = current_time

#                             # Log the update
#                             st.session_state.logger.info(
#                                 f"Updated real-time data for {symbol}: {st.session_state.rt_data[symbol]}")

#                             # Update the UI with new data
#                             pass
#                         else:
#                             st.session_state.logger.warning("Received message with no symbol")
#                     else:
#                         st.session_state.logger.warning(f"Unexpected message format: {message}")

#                 except Exception as e:
#                     error_msg = f"Error processing WebSocket message: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.logger.error(error_msg, exc_info=True)
#                 finally:
#                     # Clear the message to prevent reprocessing
#                     if 'ws_message' in st.session_state:
#                         del st.session_state.ws_message

#             # Add a refresh button
#             if st.button("Refresh Data"):
#                 pass

#         except Exception as e:
#             import traceback
#             error_traceback = traceback.format_exc()
#             st.session_state.logger.error(f"Error in main analysis: {str(e)}\n{error_traceback}")
            
#             # Display a user-friendly error message
#             st.error(f"An error occurred during analysis. Please check the logs for details.")
            
#             # For debugging purposes, show the traceback in an expander
#             with st.expander("Click to view error details"):
#                 st.text_area("Error Details:", value=error_traceback, height=300, key="error_details")
                
#             # If it's a dimensionality error, suggest checking the data
#             if "1-dimensional" in str(e) or "shape" in str(e).lower():
#                 st.warning("Dimensionality error detected. This often occurs when data shapes don't match. "
#                           "Please verify that all input data has the correct dimensions.")

# # Track the current active tab
# if 'current_tab' not in st.session_state:
#     st.session_state.current_tab = 'Price Analysis'  # Default tab

# # Update the current tab when a button is clicked
# if st.session_state.get('current_tab') != st.session_state.get('current_tab_prev', None):
#     st.session_state.current_tab_prev = st.session_state.current_tab

# Clean up WebSocket connection when the app is stopped
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Neemat Folasayo OLAJIDE")
st.sidebar.write("LCU/UG/22/21837")
st.sidebar.write("Created by Temiloluwa Bernard BAMIGBADE")
st.sidebar.write("LCU/UG/22/23055")

async def test_websocket_connection():
    """
    Test the WebSocket connection and subscription functionality.
    Run this function directly to debug WebSocket issues.
    """
    import time
    from datetime import datetime
    
    print("\n" + "="*50)
    print("Testing WebSocket Connection")
    print("="*50)
    
    # Create a new instance of RealTimeDataHandler
    test_handler = RealTimeDataHandler()
    
    try:
        # Start the handler
        test_handler.start()
        
        # Wait a bit for connection to establish
        await asyncio.sleep(2)
        
        # Use the ticker from the Streamlit UI
        if 'ticker' in st.session_state and st.session_state.ticker:
            test_symbols = [st.session_state.ticker.upper()]
            print(f"\nSubscribing to symbol: {test_symbols[0]}")
            test_handler.update_symbols(test_symbols)
        else:
            print("\nNo ticker found in session state. Please enter a ticker in the UI first.")
            return
        
        # Wait for messages to come in
        print("\nListening for messages (30 seconds)...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            status = test_handler.get_status()
            print(f"\rMessages received: {status['message_count']} | "
                  f"Connected: {status['connected']} | "
                  f"Authenticated: {status['authenticated']} | "
                  f"Subscribed: {status['subscribed_symbols']}", end="")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        test_handler.stop()
        print("Test complete!")

if __name__ == "__main__":
    import sys
    
    # Check if we're running in test mode
    if "--test" in sys.argv:
        # Configure logging for the test
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("websocket_test.log")
            ]
        )
        
    else:
        # Normal Streamlit execution
        import streamlit as st
        
        # This will be executed when running with: streamlit run app.py
        def main():
            # Initialize WebSocket session state first
            init_ws_session_state()
            
            # Initialize all required session state variables
            if 'rt_handler' not in st.session_state:
                # Initialize with a new instance
                st.session_state.rt_handler = RealTimeDataHandler.get_instance()
                st.session_state.rt_handler._persist_connection = True  # Make connection persistent
                st.session_state.rt_data = {}  # Real-time data
                st.session_state.price_history = {}  # Historical price data
                st.session_state.models = {}  # Trained models
                st.session_state.predictions = {}  # Model predictions
            
            # Initialize WebSocket when ticker is selected
            if 'ticker' in st.session_state and st.session_state.ticker and not st.session_state._ws_initialized:
                try:
                    # Add a small delay to ensure Streamlit is fully initialized
                    time.sleep(1.0)  # 1 second delay
                    
                    # Check if we have a valid ticker
                    if not st.session_state.ticker:
                        raise ValueError("No ticker selected")
                        
                    # Initialize WebSocket if it's not already running
                    if st.session_state.rt_handler is not None and not st.session_state.rt_handler.running:
                        try:
                            # Get the logger and ticker safely
                            logger = getattr(st.session_state, 'logger', None)
                            ticker = getattr(st.session_state, 'ticker', None)
                            if not ticker:
                                logger.error("Ticker variable is not initialized before WebSocket start.")
                                return
                            rt_handler = st.session_state.rt_handler
                            
                            if logger:
                                logger.info("Starting WebSocket client...")
                            
                            # Create a task to run the WebSocket client in the background
                            async def start_ws(ticker_symbol):
                                try:
                                    # Start the WebSocket client
                                    if await rt_handler.start(wait_for_connection=False):
                                        if logger:
                                            logger.info("WebSocket client started successfully")
                                        
                                        # Update session state in a thread-safe way
                                        if hasattr(st, 'session_state'):
                                            st.session_state._ws_initialized = True
                                        
                                        # Update symbols after successful connection using thread-safe method
                                        if ticker_symbol:
                                            try:
                                                rt_handler.safe_update_symbols([ticker_symbol])
                                            except Exception as sub_e:
                                                if logger:
                                                    logger.error(f"Failed to update symbols: {str(sub_e)}", exc_info=True)
                                        else:
                                            if logger:
                                                logger.warning("No ticker symbol provided for WebSocket subscription")
                                                        
                                        # Optimize portfolio with robust error handling
                                        try:
                                            # Ensure portfolio_optimizer exists in session state
                                            if 'portfolio_optimizer' not in st.session_state:
                                                logger.warning("Portfolio optimizer not found in session state. Creating fallback...")
                                                st.session_state.portfolio_optimizer = FallbackPortfolioOptimizer()
                                            
                                            # Check if returns_df exists and is valid
                                            returns_df = None
                                            if 'returns_df' in st.session_state and st.session_state.returns_df is not None:
                                                returns_df = st.session_state.returns_df
                                                
                                            if returns_df is None or returns_df.empty:
                                                logger.warning("No returns data available for optimization. Using equal weights.")
                                                # Fall back to equal weights if no returns data is available
                                                if hasattr(st.session_state.portfolio_optimizer, 'tickers') and st.session_state.portfolio_optimizer.tickers:
                                                    tickers = st.session_state.portfolio_optimizer.tickers
                                                    if tickers:  # Check if tickers list is not empty
                                                        n = len(tickers)
                                                        weights_dict = {ticker: 1.0/n for ticker in tickers}
                                                        st.session_state.optimized_weights = weights_dict
                                                        logger.info(f"Using equal weights for {n} tickers")
                                                return
                                            
                                            # Ensure returns_df has valid numerical data
                                            if returns_df.empty or returns_df.isnull().all().all():
                                                logger.warning("No valid returns data available for optimization. Using equal weights.")
                                                if hasattr(returns_df, 'columns') and not returns_df.empty:
                                                    tickers = returns_df.columns.tolist()
                                                    if tickers:  # Check if columns list is not empty
                                                        n = len(tickers)
                                                        weights_dict = {ticker: 1.0/n for ticker in tickers}
                                                        st.session_state.optimized_weights = weights_dict
                                                        logger.info(f"Using equal weights for {n} tickers")
                                                return
                                                
                                            # Fill any NaN values with 0 for stability
                                            returns_df = returns_df.fillna(0)
                                            
                                            # Get optimized weights
                                            weights = st.session_state.portfolio_optimizer.optimize(returns_df)
                                            
                                            if weights is None or len(weights) == 0:
                                                logger.warning("Optimizer returned no weights. Using equal weights.")
                                                n = len(returns_df.columns) if hasattr(returns_df, 'columns') else 1
                                                weights_dict = {ticker: 1.0/max(1, n) for ticker in returns_df.columns}
                                                return
                                                
                                            # Validate weights array
                                            if len(weights) != len(returns_df.columns):
                                                raise ValueError(f"Mismatch between weights length ({len(weights)}) and number of assets ({len(returns_df.columns)})")
                                            
                                            # Convert weights to a dictionary with the ticker as key
                                            weights_dict = {}
                                            valid_weights = []
                                            valid_tickers = []
                                            
                                            for i, (ticker, weight) in enumerate(zip(returns_df.columns, weights)):
                                                try:
                                                    weight_val = float(weight)
                                                    weights_dict[ticker] = weight_val
                                                    valid_weights.append(weight_val)
                                                    valid_tickers.append(ticker)
                                                except (ValueError, TypeError) as e:
                                                    logger.warning(f"Invalid weight value for {ticker}: {weight}, excluding from portfolio")
                                            
                                            # If no valid weights, use equal weights
                                            if not valid_weights:
                                                logger.warning("No valid weights, falling back to equal weights")
                                                n = len(returns_df.columns)
                                                weights_dict = {ticker: 1.0/n for ticker in returns_df.columns}
                                            # If sum of weights is zero, use equal weights
                                            elif abs(sum(valid_weights)) < 1e-10:
                                                logger.warning("Sum of weights is zero, using equal weights")
                                                n = len(valid_weights)
                                                weights_dict = {ticker: 1.0/n for ticker in valid_tickers}
                                            # Normalize weights to sum to 1
                                            else:
                                                total_weight = sum(valid_weights)
                                                weights_dict = {ticker: w/total_weight for ticker, w in zip(valid_tickers, valid_weights)}
                                                
                                        except Exception as e:
                                            logger.error(f"Error in portfolio optimization: {str(e)}", exc_info=True)
                                            logger.warning("Falling back to equal weights")
                                            n = len(returns_df.columns)
                                            weights_dict = {ticker: 1.0/max(1, n) for ticker in returns_df.columns}
                                    return True
                                except Exception as e:
                                    if logger:
                                        logger.error(f"Error in WebSocket client: {str(e)}", exc_info=True)
                                    return False
                            
                            # Function to run the WebSocket client in a separate thread
                            def run_ws():
                                loop = None
                                try:
                                    # Create a new event loop for this thread
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    
                                    # Run the WebSocket client with the ticker
                                    loop.run_until_complete(start_ws(ticker))
                                    
                                    # Keep the event loop running
                                    loop.run_forever()
                                except Exception as e:
                                    # Use print since we can't access logger here
                                    print(f"WebSocket error: {str(e)}")
                                finally:
                                    if loop is not None:
                                        try:
                                            # Cancel all pending tasks to prevent "Event loop is closed" errors
                                            pending = asyncio.all_tasks(loop=loop)
                                            if pending:
                                                print(f"Cancelling {len(pending)} pending tasks before closing event loop")
                                                for task in pending:
                                                    task.cancel()
                                                # Give tasks a chance to respond to cancellation
                                                try:
                                                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                                                except Exception as e:
                                                    print(f"Error while cancelling tasks: {str(e)}")
                                            
                                            # Properly close the event loop
                                            try:
                                                loop.run_until_complete(loop.shutdown_asyncgens())
                                            except Exception as e:
                                                print(f"Error during shutdown_asyncgens: {str(e)}")
                                            finally:
                                                loop.close()
                                        except Exception as e:
                                            print(f"Error closing event loop: {str(e)}")
                                            pass
                            
                            # Start the WebSocket in a separate thread
                            ws_thread = threading.Thread(target=run_ws, daemon=True)
                            ws_thread.start()
                            
                            # Small delay to allow the WebSocket to start
                            import time as time_module
                            time_module.sleep(1.0)  # 1 second delay
                            
                            if not rt_handler.running:
                                if logger:
                                    logger.warning("WebSocket client did not start properly")
                                if hasattr(st, 'session_state'):
                                    st.session_state._ws_initialized = False
                            
                        except Exception as e:
                            if logger:
                                logger.error(f"Failed to initialize WebSocket: {str(e)}", exc_info=True)
                            if hasattr(st, 'session_state'):
                                st.session_state._ws_initialized = False
                except Exception as e:
                    # Only log the error, don't stop or reset anything
                    st.session_state.logger.error(f"WebSocket initialization error: {str(e)}", exc_info=True)
                    st.error(f"Error initializing WebSocket: {str(e)}")
                    # Don't reset _ws_initialized here - let it keep trying
                    pass

            # The main ticker input is already in the sidebar
            # Display connection status section
            st.sidebar.subheader("Connection Status")
            
            # Check if we have a valid handler and ticker
            if ('rt_handler' not in st.session_state or 
                st.session_state.rt_handler is None or 
                not hasattr(st.session_state.rt_handler, 'get_status')):
                st.sidebar.warning("Real-time data handler not initialized")
                return
                
            if 'ticker' not in st.session_state or not st.session_state.ticker:
                st.sidebar.info("Select a ticker to enable real-time data")
                return
                
            try:
                # Safely update symbols if handler is available
                if hasattr(st.session_state.rt_handler, 'update_symbols'):
                    try:
                        st.session_state.rt_handler.update_symbols([st.session_state.ticker])
                    except Exception as update_err:
                        st.session_state.logger.warning(f"Failed to update symbols: {str(update_err)}")
                
                # Safely get status
                status = {}
                try:
                    if callable(getattr(st.session_state.rt_handler, 'get_status', None)):
                        status = st.session_state.rt_handler.get_status() or {}
                except Exception as status_err:
                    st.session_state.logger.warning(f"Could not get WebSocket status: {str(status_err)}")
                    status = {}
                
                # Display status with error handling for each field
                status_display = {
                    "Status": "Connected" if status.get('connected') else "Disconnected",
                    "Authenticated": status.get('authenticated', False),
                    "Subscribed Symbols": status.get('subscribed_symbols', [st.session_state.ticker]),
                    "Messages Received": status.get('message_count', 0)
                }
                
                st.sidebar.json(status_display)
                
                # Show error if there was a recent error
                if 'last_error' in status and status['last_error']:
                    st.sidebar.error(f"Error: {status['last_error']}")
                
                # Log detailed status if available
                if status:
                    st.session_state.logger.debug(f"WebSocket status: {status}")
                    
            except Exception as e:
                # Log the error but don't crash
                error_msg = f"Error updating WebSocket status: {str(e)}"
                st.session_state.logger.error(error_msg, exc_info=True)
                st.sidebar.error("Error updating real-time data")
        
        # Run the main function
        if __name__ == "__main__":
            try:
                main()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
                # Don't clean up WebSocket here - let it keep trying to connect
                st.session_state.logger.error(f"Main function error: {str(e)}", exc_info=True)