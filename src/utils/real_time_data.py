import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from queue import Queue
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import websocket
import yfinance as yf

# Type aliases
TickerData = Dict[str, Any]
DataFrameDict = Dict[str, pd.DataFrame]
CallbackType = Callable[[str, TickerData], None]


class ConnectionState(Enum):
    """Represents the current state of the WebSocket connection"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


@dataclass
class ConnectionConfig:
    """Configuration for WebSocket connection parameters"""
    url: str = "wss://streamer.finance.yahoo.com"
    ping_interval: int = 30  # seconds
    ping_timeout: int = 10  # seconds
    request_timeout: int = 10  # seconds
    max_queue_size: int = 1000
    max_reconnect_attempts: int = 10
    initial_reconnect_delay: float = 1.0  # seconds
    max_reconnect_delay: float = 30.0  # seconds


@dataclass
class DataValidationConfig:
    """Configuration for data validation parameters"""
    min_data_points: int = 10
    max_null_percentage: float = 0.1  # 10%
    min_volume: int = 0
    min_price: float = 0.01
    max_price_change: float = 0.5  # 50% max change between ticks
    required_columns: List[str] = field(
        default_factory=lambda: [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
        ]
    )


class BaseDataHandler(ABC):
    """Abstract base class for data handlers"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the data handler"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the data handler"""
        pass
    
    @abstractmethod
    def get_latest_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get the latest data for a ticker"""
        pass


class RealTimeData(BaseDataHandler):
    def __init__(self, tickers: List[str], interval: str = "1m", **kwargs):
        """
        Initialize the RealTimeData handler.
        
        Args:
            tickers: List of stock ticker symbols to track
            interval: Data interval (e.g., '1m', '5m', '1d')
            **kwargs: Additional configuration overrides for ConnectionConfig and DataValidationConfig
        """
        # Initialize tickers and interval
        self.tickers = list(set(t.upper() for t in tickers))  # Remove duplicates and standardize case
        self.interval = interval.lower()
        
        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # WebSocket and threading
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self.reconnect_attempts: int = 0
        
        # Configuration
        self.config = ConnectionConfig()
        self.validation_config = DataValidationConfig()
        
        # Apply any configuration overrides from kwargs
        self._apply_config_overrides(**kwargs)
        
        # Thread safety and async processing
        self._lock = RLock()
        self._message_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.callbacks: List[CallbackType] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._init_logger()
        
        # Initialize event loop
        self._init_event_loop()
        
        self.logger.info(
            f"Initialized RealTimeData for {len(self.tickers)} tickers "
            f"with interval {self.interval}"
        )
    
    def _apply_config_overrides(self, **kwargs) -> None:
        """Apply configuration overrides from kwargs"""
        # Apply ConnectionConfig overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.validation_config, key):
                setattr(self.validation_config, key, value)
    
    def _init_event_loop(self) -> None:
        """Initialize the asyncio event loop"""
        try:
            self.loop = asyncio.get_event_loop()
            if self.loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except (RuntimeError, Exception) as e:
            self.logger.warning(
                f"Creating new event loop (reason: {str(e)})"
            )
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def stop(self) -> None:
        """
        Stop the WebSocket connection and clean up resources.
        
        This method will:
        1. Set the running flag to False to stop processing
        2. Close the WebSocket connection if it exists
        3. Stop and clean up the WebSocket thread
        4. Clear any pending messages in the queue
        """
        if not self.running:
            return
            
        self.logger.info("Stopping RealTimeData handler...")
        
        # Set running flag to False to stop processing
        self.running = False
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Close WebSocket if it exists
        if self.ws:
            try:
                self.ws.close()
                self.logger.debug("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")
            finally:
                self.ws = None
        
        # Stop WebSocket thread if it exists
        if self.ws_thread and self.ws_thread.is_alive():
            try:
                self.ws_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                if self.ws_thread.is_alive():
                    self.logger.warning("WebSocket thread did not stop gracefully")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket thread: {str(e)}")
            finally:
                self.ws_thread = None
        
        # Clear the message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
                self._message_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("RealTimeData handler stopped")

    def _init_data(self) -> None:
        """
        Initialize historical data for all tickers.
        
        This method fetches historical data for each ticker and initializes
        the data structures needed for real-time updates.
        """
        self.logger.info(f"Initializing historical data for {len(self.tickers)} tickers...")
        
        for ticker in self.tickers:
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Get historical data with a wider window to ensure we have enough data
                    hist_data = yf.download(
                        ticker,
                        period="5d",  # Get 5 days of data to ensure we have enough
                        interval=self.interval,
                        progress=False,
                        show_errors=False,
                        threads=True
                    )
                    
                    if hist_data is None or hist_data.empty:
                        raise ValueError(f"No data returned from yfinance for {ticker}")
                        
                    # Ensure required columns exist
                    missing_cols = [
                        col for col in self.validation_config.required_columns 
                        if col not in hist_data.columns
                    ]
                    if missing_cols:
                        raise ValueError(f"Missing required columns: {missing_cols}")
                    
                    # Clean and validate data
                    hist_data = self._clean_historical_data(hist_data)
                    
                    # Store the data
                    with self._lock:
                        self.data[ticker] = hist_data
                        self.last_update[ticker] = datetime.now()
                        
                    self.logger.info(
                        f"Initialized data for {ticker} with {len(hist_data)} data points"
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt failed
                        self.logger.error(
                            f"Failed to initialize data for {ticker} after "
                            f"{max_retries} attempts: {str(e)}",
                            exc_info=self.logger.isEnabledFor(logging.DEBUG)
                        )
                        # Initialize with empty DataFrame to avoid KeyErrors
                        with self._lock:
                            self.data[ticker] = pd.DataFrame(
                                columns=self.validation_config.required_columns
                            )
                            self.last_update[ticker] = datetime.now()
                    else:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
    
    def _clean_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate historical data.
        
        Args:
            data: Raw historical data from yfinance
            
        Returns:
            Cleaned and validated DataFrame
            
        Raises:
            ValueError: If data fails validation
        """
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date (oldest first)
            df = df.sort_index()
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Validate data quality
            if not self._validate_historical_data(df):
                raise ValueError("Historical data validation failed")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning historical data: {str(e)}", 
                           exc_info=self.logger.isEnabledFor(logging.DEBUG))
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
            
        Raises:
            ValueError: If too many missing values are present
        """
        # Forward fill missing values if possible
        df = df.ffill()
        
        # If there are still missing values at the beginning, backfill them
        df = df.bfill()
        
        # If any missing values remain, raise an error
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            raise ValueError(
                f"Could not fill all missing values. Missing values:\n{null_counts[null_counts > 0]}"
            )
            
        return df
    
    def _validate_historical_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the quality of historical data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check minimum data points
        if len(df) < self.validation_config.min_data_points:
            self.logger.warning(
                f"Insufficient data points: {len(df)} < "
                f"{self.validation_config.min_data_points}"
            )
            return False
            
        # Check for required columns
        missing_columns = [
            col for col in self.validation_config.required_columns 
            if col not in df.columns
        ]
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for null values
        if df.isnull().any().any():
            self.logger.warning("Data contains null values after cleaning")
            return False
            
        # Check for zero or negative prices
        price_columns = [
            col for col in ['Open', 'High', 'Low', 'Close'] 
            if col in df.columns
        ]
        
        for col in price_columns:
            if (df[col] <= 0).any():
                self.logger.warning(f"Invalid {col} price detected (<= 0)")
                return False
                
        return True
        
    def _init_logger(self):
        """Initialize logger with appropriate handlers"""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def add_callback(self, callback: CallbackType) -> None:
        """
        Add a callback function to be called when new data is received.
        
        Args:
            callback: A function that takes (ticker: str, data: dict) as arguments
            
        Raises:
            ValueError: If the callback is not callable or has an invalid signature
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
            
        # Verify the callback signature
        import inspect
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())
        
        if len(params) < 2:
            raise ValueError(
                f"Callback must accept at least 2 parameters (ticker, data), "
                f"got {len(params)}"
            )
            
        callback_name = (
            getattr(callback, "__name__", "lambda")
            if callback.__name__ != "<lambda>"
            else str(callback)
        )
            
        with self._lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)
                self.logger.debug(f"Added callback: {callback_name}")
            else:
                self.logger.warning(f"Callback already registered: {callback_name}")
    
    def remove_callback(self, callback: CallbackType) -> bool:
        """
        Remove a previously registered callback.
        
        Args:
            callback: The callback function to remove
            
        Returns:
            bool: True if the callback was removed, False if it wasn't found
        """
        with self._lock:
            try:
                self.callbacks.remove(callback)
                self.logger.debug(f"Removed callback: {getattr(callback, '__name__', 'anonymous')}")
                return True
            except ValueError:
                return False
    
    def _validate_ticker_data(self, data: pd.DataFrame) -> bool:
        """
        Validate ticker data against quality thresholds.
        
        Args:
            data: DataFrame containing ticker data to validate
            
        Returns:
            bool: True if data passes all validations, False otherwise
        """
        try:
            # Basic validation
            if data.empty or len(data) == 0:
                self.logger.debug("Validation failed: Empty data")
                return False
                
            # Check for null values
            null_ratio = data.isnull().sum().sum() / data.size
            if null_ratio > self.validation_config.max_null_percentage:
                self.logger.warning(
                    f"Validation failed: Null ratio {null_ratio:.2%} exceeds "
                    f"maximum {self.validation_config.max_null_percentage:.0%}"
                )
                return False
                
            # Check volume if available
            if 'Volume' in data.columns and data['Volume'].iloc[0] < self.validation_config.min_volume:
                self.logger.warning(
                    f"Validation failed: Volume {data['Volume'].iloc[0]} is below "
                    f"minimum {self.validation_config.min_volume}"
                )
                return False
                
            # Check price if available
            if 'Close' in data.columns:
                close_price = data['Close'].iloc[0]
                if close_price < self.validation_config.min_price:
                    self.logger.warning(
                        f"Validation failed: Price {close_price} is below "
                        f"minimum {self.validation_config.min_price}"
                    )
                    return False
                    
                # Check price changes (if we have previous data)
                with self._lock:
                    if len(self.data) > 0 and 'Close' in self.data.iloc[-1:]:
                        last_close = self.data.iloc[-1]['Close']
                        price_change = abs((close_price - last_close) / last_close)
                        if price_change > self.validation_config.max_price_change:
                            self.logger.warning(
                                f"Validation failed: Price change {price_change:.2%} "
                                f"exceeds maximum {self.validation_config.max_price_change:.0%}"
                            )
                            return False
                    
            return True
            
        except Exception as e:
            self.logger.error(
                f"Validation error: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            return False
            
    async def _process_message(self, message: Union[str, bytes]) -> None:
        """
        Process incoming WebSocket message asynchronously.
        
        Args:
            message: Raw message from WebSocket (string or bytes)
        """
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')
                
            data = json.loads(message)
            self.reconnect_attempts = 0  # Reset on successful message
            
            # Process market data
            if 'data' in data and isinstance(data['data'], list):
                for ticker_data in data['data']:
                    ticker = ticker_data.get('symbol')
                    if ticker in self.tickers:
                        await self._process_ticker_data(ticker, ticker_data)
            
            # Process ping/pong
            elif data.get('type') == 'ping':
                await self._send_pong()
                
        except json.JSONDecodeError as je:
            self.logger.error(f"Failed to parse message as JSON: {message[:200]}...")
        except Exception as e:
            self.logger.error(
                f"Error processing message: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
    
    async def _process_ticker_data(self, ticker: str, ticker_data: Dict[str, Any]) -> None:
        """
        Process and validate data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            ticker_data: Raw ticker data from WebSocket
        """
        try:
            # Convert to DataFrame and validate
            new_data = pd.DataFrame([ticker_data])
            if not self._validate_ticker_data(new_data):
                self.logger.warning(f"Invalid data received for {ticker}")
                return
                
            # Update data with thread safety
            with self._lock:
                if ticker not in self.data:
                    self.data[ticker] = new_data
                else:
                    # Append new data and remove duplicates
                    self.data[ticker] = pd.concat([self.data[ticker], new_data])
                    self.data[ticker] = self.data[ticker][~self.data[ticker].index.duplicated(keep='last')]
                    
                    # Keep only recent data (last 24 hours)
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.data[ticker] = self.data[ticker][self.data[ticker].index >= cutoff]
                
                self.last_update[ticker] = datetime.now()
                
            # Notify callbacks
            self._notify_callbacks(ticker, ticker_data)
            
            self.logger.debug(f"Processed update for {ticker}")
            
        except Exception as e:
            self.logger.error(
                f"Error processing {ticker} data: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
    
    async def _send_pong(self) -> None:
        """Send pong response to keep connection alive"""
        try:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self.ws.send(json.dumps({"type": "pong"}))
                self.logger.debug("Sent pong response")
        except Exception as e:
            self.logger.error(
                f"Error sending pong: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            self.connection_state = ConnectionState.ERROR
            self._schedule_reconnect()
            
    def _notify_callbacks(self, ticker: str, data: Dict[str, Any]) -> None:
        """
        Notify all registered callbacks with new data for a ticker.
        
        Args:
            ticker: Ticker symbol
            data: Data dictionary for the ticker
            
        Note:
            Callbacks are executed in the order they were added. If a callback
            raises an exception, it will be logged but will not affect other callbacks.
        """
        if not self.callbacks:
            return
            
        # Create a copy of callbacks to avoid issues if the list is modified during iteration
        callbacks = self.callbacks[:]
        
        for callback in callbacks:
            callback_name = getattr(callback, "__name__", "lambda")
            try:
                # Execute the callback
                callback(ticker, data)
                
                # Log successful callback execution at debug level
                self.logger.debug(
                    f"Notified callback {callback_name} for {ticker}"
                )
                
            except Exception as e:
                # Log the error but continue with other callbacks
                self.logger.error(
                    f"Error in callback {callback_name} for {ticker}: {str(e)}",
                    exc_info=self.logger.isEnabledFor(logging.DEBUG)
                )
                
                # Optionally remove the failing callback
                # with self._lock:
                #     if callback in self.callbacks:
                #         self.callbacks.remove(callback)
                #         self.logger.warning(
                #             f"Removed failing callback {callback_name} after error"
                #         )
    
    def _on_ws_message(self, ws: websocket.WebSocketApp, message: Union[str, bytes]) -> None:
        """
        WebSocket message handler that processes incoming messages in a thread-safe manner.
        
        Args:
            ws: The WebSocket client instance
            message: The incoming message (string or bytes)
        """
        try:
            # Convert bytes to string if needed
            if isinstance(message, bytes):
                message = message.decode('utf-8')
                
            # Schedule the message processing in the event loop
            asyncio.run_coroutine_threadsafe(
                self._process_message(message), 
                self.loop
            )
        except json.JSONDecodeError as je:
            self.logger.error(f"Failed to decode message: {message[:100]}... Error: {str(je)}")
        except UnicodeDecodeError as ude:
            self.logger.error(f"Failed to decode binary message: {str(ude)}")
        except Exception as e:
            self.logger.error(
                f"Unexpected error in WebSocket message handler: {str(e)}", 
                exc_info=True
            )
            self.connection_state = ConnectionState.ERROR
            self._schedule_reconnect()
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        try:
            error_msg = str(error) if error else 'Unknown error'
            self.logger.error(f"WebSocket error: {error_msg}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
            
            # Only attempt reconnect if we're still supposed to be running
            if self.running:
                self._schedule_reconnect()
                
        except Exception as e:
            self.logger.error(f"Error in _on_error handler: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
    
    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """Handle WebSocket close"""
        try:
            close_code = close_status_code if close_status_code is not None else 'No code'
            close_message = close_msg or 'No message'
            self.logger.info(f"WebSocket connection closed (code: {close_code}, message: {close_message})")
            
            # Update connection state
            if self.connection_state != ConnectionState.DISCONNECTED:
                self.connection_state = ConnectionState.DISCONNECTED
                
            # Only try to reconnect if we're supposed to be running
            if self.running:
                self.logger.info("Scheduling reconnection...")
                self._schedule_reconnect()
                
        except Exception as e:
            self.logger.error(f"Error in _on_close handler: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
    
    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt with exponential backoff"""
        if not self.running:
            self.logger.info("Not scheduling reconnect - not running")
            return
            
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.config.max_reconnect_attempts:
            self.logger.error(
                f"Max reconnection attempts ({self.config.max_reconnect_attempts}) reached. Giving up."
            )
            self.running = False
            self.connection_state = ConnectionState.ERROR
            return
            
        # Calculate delay with exponential backoff and jitter
        delay = min(
            self.config.initial_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
            self.config.max_reconnect_delay
        )
        jitter = random.uniform(0.8, 1.2)  # Add random jitter between 0.8x and 1.2x
        delay = min(delay * jitter, self.config.max_reconnect_delay)
        
        self.logger.info(
            f"Scheduling reconnection attempt {self.reconnect_attempts} in {delay:.1f} seconds..."
        )
        
        # Schedule reconnection in a separate thread to avoid blocking
        def delayed_reconnect():
            time.sleep(delay)
            if not self.running:  # Only proceed if still running
                return
                
            try:
                self.logger.info(f"Attempting to reconnect (attempt {self.reconnect_attempts})...")
                success = self._reconnect()
                
                if not success and self.running:
                    # If reconnect failed and we're still running, schedule another attempt
                    self.logger.warning("Reconnect attempt failed, will retry...")
                    self._schedule_reconnect()
                    
            except Exception as e:
                self.logger.error(
                    f"Error during reconnection attempt: {str(e)}",
                    exc_info=True
                )
                if self.running:
                    self._schedule_reconnect()  # Schedule another attempt on error
        
        threading.Thread(target=delayed_reconnect, daemon=True).start()
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to the WebSocket server
        
        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        if not self.running:
            self.logger.info("Not reconnecting - not running")
            return False
            
        self.logger.info("Attempting to reconnect...")
        
        try:
            # Clean up any existing connection
            self._cleanup_websocket()
            
            # Reset connection state
            self.connection_state = ConnectionState.DISCONNECTED
            
            # Start a new connection
            success = self._start_websocket()
            
            if success:
                self.logger.info("Successfully reconnected to WebSocket server")
                self.reconnect_attempts = 0  # Reset reconnect attempts on success
            else:
                self.logger.warning("Failed to reconnect to WebSocket server")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Reconnection failed: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
            return False
    
    def _start_websocket(self) -> bool:
        """Start the WebSocket connection in a separate thread"""
        if not self.running:
            return False
            
        if self.connection_state in [ConnectionState.CONNECTING, ConnectionState.CONNECTED]:
            self.logger.warning("WebSocket connection already in progress")
            return True
            
        self.connection_state = ConnectionState.CONNECTING
        self.logger.info("Starting WebSocket connection...")
        
        try:
            # Clean up any existing connection
            self._cleanup_websocket()
            
            # Initialize WebSocket with enhanced settings
            self.ws = websocket.WebSocketApp(
                self.config.url,
                on_message=self._on_ws_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self._run_websocket,
                name=f"WebSocketThread-{id(self)}",
                daemon=True
            )
            self.ws_thread.start()
            
            # Wait for connection to establish (with timeout)
            max_wait = self.config.request_timeout
            start_time = time.time()
            while self.connection_state == ConnectionState.CONNECTING:
                if time.time() - start_time > max_wait:
                    raise TimeoutError("Timed out waiting for WebSocket connection")
                time.sleep(0.1)
            
            return self.connection_state == ConnectionState.CONNECTED
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
            self._schedule_reconnect()
            return False
            
    def _on_open(self, ws):
        """Handle WebSocket open event"""
        try:
            self.logger.info("WebSocket connection established")
            self.connection_state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            
            # Start message processing task
            self.loop.create_task(self._process_message_queue())
            
            # Subscribe to tickers
            self._subscribe_to_tickers()
            
            self.logger.info("Successfully connected and subscribed to tickers")
            
        except Exception as e:
            self.logger.error(f"Error in _on_open: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
            self._schedule_reconnect()
    
    def _run_websocket(self):
        """Run the WebSocket client with enhanced error handling"""
        try:
            self.ws.run_forever(
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                ping_payload=json.dumps({"type": "ping"}),
                reconnect=0  # We handle reconnection ourselves
            )
        except Exception as e:
            self.logger.error(f"WebSocket error in run_forever: {str(e)}", exc_info=True)
            self.connection_state = ConnectionState.ERROR
        finally:
            self._cleanup_websocket()
            
    async def _process_message_queue(self):
        """Process messages from the queue asynchronously"""
        while self.running and self.connection_state == ConnectionState.CONNECTED:
            try:
                # Get message with timeout to allow for clean shutdown
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=1.0
                    )
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in message queue processing: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _send_ping(self):
        """Send ping to keep connection alive"""
        if self.connection_state == ConnectionState.CONNECTED and self.ws and self.ws.sock:
            try:
                self.ws.send(json.dumps({"type": "ping"}))
                self.logger.debug("Sent ping")
            except Exception as e:
                self.logger.error(f"Error sending ping: {str(e)}")
                self.connection_state = ConnectionState.ERROR
                self._schedule_reconnect()
    
    async def _send_pong(self):
        """Send pong in response to ping"""
        if self.connection_state == ConnectionState.CONNECTED and self.ws and self.ws.sock:
            try:
                self.ws.send(json.dumps({"type": "pong"}))
                self.logger.debug("Sent pong")
            except Exception as e:
                self.logger.error(f"Error sending pong: {str(e)}")
                self.connection_state = ConnectionState.ERROR
                self._schedule_reconnect()
    
    def _on_ping(self, ws, message):
        """Handle ping from server"""
        self.logger.debug("Received ping from server")
        self.loop.create_task(self._send_pong())
    
    def _on_pong(self, ws, message):
        """Handle pong from server"""
        self.logger.debug("Received pong from server")
    
    def _subscribe_to_tickers(self) -> bool:
        """Subscribe to ticker updates with retry logic"""
        if self.connection_state != ConnectionState.CONNECTED or not self.ws:
            self.logger.warning("Cannot subscribe: WebSocket not connected")
            return False
            
        success = True
        for ticker in self.tickers:
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    subscribe_msg = {
                        "type": "subscribe",
                        "symbol": ticker,
                        "interval": self.interval,
                        "request_id": f"sub_{ticker}_{int(time.time())}"
                    }
                    
                    self.ws.send(json.dumps(subscribe_msg))
                    self.logger.info(f"Subscribed to {ticker} (attempt {attempt + 1}/{max_retries})")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt failed
                        self.logger.error(
                            f"Failed to subscribe to {ticker} after {max_retries} attempts: {str(e)}",
                            exc_info=True
                        )
                        success = False
                    else:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        
    def start(self) -> bool:
        """
        Start the real-time data handler.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            self.logger.warning("RealTimeData is already running")
            return True
            
        self.running = True
        self.connection_state = ConnectionState.CONNECTING
        
        try:
            # Initialize historical data
            self._init_data()
            
            # Start WebSocket connection
            self._start_websocket()
            
            # Start the message processing task
            self.loop.create_task(self._process_message_queue())
            
            self.logger.info("RealTimeData started successfully")
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to start RealTimeData: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            self.connection_state = ConnectionState.ERROR
            self.running = False
            self._schedule_reconnect()
            return False
    
    def _cleanup_websocket(self):
        """Clean up WebSocket resources"""
        with self._lock:
            try:
                if hasattr(self, 'ws') and self.ws:
                    self.logger.debug("Cleaning up WebSocket resources...")
                    
                    # Close WebSocket if connected
                    try:
                        if hasattr(self.ws, 'sock') and self.ws.sock:
                            self.ws.sock.close()
                            self.logger.debug("WebSocket socket closed")
                    except Exception as e:
                        self.logger.warning(f"Error closing WebSocket socket: {str(e)}")
                    
                    # Close WebSocketApp
                    try:
                        self.ws.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing WebSocket: {str(e)}")
                    
                    # Clean up thread
                    if hasattr(self, 'ws_thread') and self.ws_thread and self.ws_thread.is_alive():
                        try:
                            self.ws_thread.join(timeout=1.0)
                            if self.ws_thread.is_alive():
                                self.logger.warning("WebSocket thread did not terminate gracefully")
                        except Exception as e:
                            self.logger.warning(f"Error joining WebSocket thread: {str(e)}")
                    
                    self.ws = None
                    self.ws_thread = None
                    self.connection_state = ConnectionState.DISCONNECTED
                    self.logger.info("WebSocket resources cleaned up")
                    
            except Exception as e:
                self.logger.error(f"Error in WebSocket cleanup: {str(e)}", exc_info=True)
                self.connection_state = ConnectionState.ERROR
            finally:
                self.ws = None
                self.ws_thread = None
            
    def close(self, wait: bool = True, timeout: float = 5.0):
        """
        Cleanly close the WebSocket connection and clean up all resources.
        
        Args:
            wait: If True, wait for cleanup to complete
            timeout: Maximum time to wait for cleanup (seconds)
        """
        if not self.running:
            self.logger.debug("RealTimeData is not running, nothing to close")
            return
            
        self.logger.info("Closing RealTimeData...")
        self.running = False
        
        # Clean up WebSocket resources
        self._cleanup_websocket()
        
        # Clear data structures
        with self._lock:
            self.data.clear()
            self.last_update.clear()
        
        # Clear callbacks
        self.callbacks.clear()
        
        # Shutdown the event loop if we created it
        try:
            if hasattr(self, 'loop') and self.loop.is_running():
                if wait:
                    # Give the loop a moment to process pending tasks
                    pending = asyncio.all_tasks(loop=self.loop)
                    if pending:
                        self.logger.debug(f"Waiting for {len(pending)} pending tasks to complete...")
                        done, pending = self.loop.run_until_complete(
                            asyncio.wait(pending, timeout=timeout)
                        )
                        if pending:
                            self.logger.warning(
                                f"{len(pending)} tasks did not complete within timeout"
                            )
                
                # Stop the loop
                self.loop.call_soon_threadsafe(self.loop.stop)
                
                if wait:
                    # Give the loop a moment to stop
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(
                f"Error during event loop shutdown: {str(e)}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
        finally:
            self.connection_state = ConnectionState.DISCONNECTED
            self.logger.info("RealTimeData closed")
        
    def __del__(self):
        """Ensure proper cleanup when the object is garbage collected"""
        try:
            if hasattr(self, 'running') and self.running:
                self.logger.warning("RealTimeData deleted while still running")
                self.close(wait=False)
        except Exception:
            pass  # Avoid exceptions during garbage collection
            
    def get_latest_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Get the latest data for a ticker
        """
        if ticker in self.data:
            return self.data[ticker].copy()
        return None
        
    def get_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest price for a ticker
        """
        data = self.get_latest_data(ticker)
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
        
    def get_volume(self, ticker: str) -> Optional[float]:
        """
        Get the latest volume for a ticker
        """
        data = self.get_latest_data(ticker)
        if data is not None and not data.empty:
            return data['Volume'].iloc[-1]
        return None
        
    def get_price_change(self, ticker: str) -> Optional[float]:
        """
        Get the price change percentage
        """
        data = self.get_latest_data(ticker)
        if data is not None and len(data) > 1:
            current = data['Close'].iloc[-1]
            previous = data['Close'].iloc[-2]
            return ((current - previous) / previous) * 100
        return None
        
    def get_volatility(self, ticker: str) -> Optional[float]:
        """
        Calculate current volatility
        """
        data = self.get_latest_data(ticker)
        if data is not None and len(data) > 5:
            returns = data['Close'].pct_change()
            return returns.std() * np.sqrt(252)  # Annualized volatility
        return None
        
    def get_market_sentiment(self) -> Dict[str, float]:
        """
        Calculate market sentiment based on price movements
        """
        sentiment = {}
        for ticker in self.tickers:
            price_change = self.get_price_change(ticker)
            if price_change is not None:
                sentiment[ticker] = price_change
        return sentiment
        
    def close(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Cleanly close the WebSocket connection and clean up all resources.
        
        Args:
            wait: If True, wait for cleanup to complete
            timeout: Maximum time to wait for cleanup (seconds)
        """
        self.logger.info("Initiating RealTimeData shutdown...")
        
        # Signal all threads to stop
        self.running = False
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Clean up WebSocket resources
        self._cleanup_websocket()
        
        # Clear message queue
        try:
            while not self._message_queue.empty():
                self._message_queue.get_nowait()
                self._message_queue.task_done()
        except Exception as e:
            self.logger.warning(f"Error clearing message queue: {str(e)}")
        
        # Wait for cleanup to complete if requested
        if wait:
            start_time = time.time()
            while (hasattr(self, 'ws_thread') and self.ws_thread and 
                   self.ws_thread.is_alive() and 
                   (time.time() - start_time) < timeout):
                time.sleep(0.1)
                
            if hasattr(self, 'ws_thread') and self.ws_thread and self.ws_thread.is_alive():
                self.logger.warning("Timed out waiting for cleanup to complete")
            else:
                self.logger.info("Cleanup completed successfully")
        
        try:
            # Clean up WebSocket
            self._cleanup_websocket()
            
            # Wait for WebSocket thread to finish with timeout
            if hasattr(self, 'ws_thread') and self.ws_thread and self.ws_thread.is_alive():
                self.logger.info("Waiting for WebSocket thread to terminate...")
                start_time = time.time()
                self.ws_thread.join(timeout=5)
                elapsed = time.time() - start_time
                
                if self.ws_thread.is_alive():
                    self.logger.warning(f"WebSocket thread did not terminate after {elapsed:.2f} seconds")
                    # Try to interrupt the thread if possible
                    if hasattr(self.ws_thread, '_stop'):
                        self.logger.info("Attempting to force thread termination...")
                        try:
                            self.ws_thread._stop()
                        except Exception as e:
                            self.logger.warning(f"Failed to force thread termination: {str(e)}")
                else:
                    self.logger.info(f"WebSocket thread terminated successfully after {elapsed:.2f} seconds")
            
            # Clear data
            self.data.clear()
            self.last_update.clear()
            
            self.logger.info("Successfully closed WebSocket and cleaned up resources")
            
        except Exception as e:
            self.logger.error(f"Error during close: {str(e)}", exc_info=True)
            raise
        finally:
            # Ensure resources are released even if an error occurs
            self.ws = None
            self.ws_thread = None

# Example usage:
if __name__ == "__main__":
    # Initialize real-time data for multiple tickers
    real_time = RealTimeData(tickers=["AAPL", "GOOGL", "MSFT"])
    
    try:
        # Start the real-time data stream
        real_time.start()
        
        # Keep the main thread alive
        while real_time.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutting down real-time data stream...")
        real_time.close()
        print("\nStreaming stopped")
