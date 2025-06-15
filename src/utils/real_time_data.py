import yfinance as yf
import pandas as pd
import numpy as np
import websocket
import json
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Callable

class RealTimeData:
    def __init__(self, tickers: List[str], interval: str = "1m"):
        """
        Initialize real-time data streaming
        
        Args:
            tickers: List of stock tickers to track
            interval: Data interval (default: "1m" for 1 minute)
        """
        self.tickers = list(set(tickers))  # Remove duplicates
        self.interval = interval
        self.data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # Initial delay in seconds
        self.max_reconnect_delay = 60  # Maximum delay in seconds
        self.logger = logging.getLogger(f"{__name__}.RealTimeData")
        self._init_logger()
        
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
        
    def _init_data(self):
        """
        Initialize historical data for real-time updates
        """
        for ticker in self.tickers:
            try:
                # Get last 60 minutes of data
                hist_data = yf.download(
                    ticker,
                    period="60m",
                    interval=self.interval,
                    progress=False
                )
                
                if not hist_data.empty:
                    self.data[ticker] = hist_data
                    self.last_update[ticker] = datetime.now()
                    self.logger.info(f"Initialized data for {ticker}")
                else:
                    self.logger.warning(f"No historical data found for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"Error initializing data for {ticker}: {str(e)}")
                
    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket messages
        """
        try:
            data = json.loads(message)
            self.reconnect_attempts = 0  # Reset reconnect attempts on successful message
            
            for ticker in self.tickers:
                if ticker in data:
                    try:
                        new_data = pd.DataFrame([data[ticker]])
                        new_data.index = pd.to_datetime(new_data.index)
                        
                        if ticker not in self.data:
                            self.data[ticker] = new_data
                        else:
                            # Append new data
                            self.data[ticker] = pd.concat([
                                self.data[ticker],
                                new_data
                            ])
                            
                            # Keep only last 60 minutes
                            cutoff = datetime.now() - timedelta(minutes=60)
                            self.data[ticker] = self.data[ticker][self.data[ticker].index >= cutoff]
                        
                        self.last_update[ticker] = datetime.now()
                        self.logger.debug(f"Updated data for {ticker}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing data for {ticker}: {str(e)}", exc_info=True)
                        
        except json.JSONDecodeError as je:
            self.logger.error(f"Failed to parse message as JSON: {message}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error in _on_message: {str(e)}", exc_info=True)
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {str(error)}", exc_info=True)
        self._schedule_reconnect()
    
    def _on_close(self, ws, close_status_code=None, close_msg=None):
        """Handle WebSocket close"""
        self.logger.info(f"WebSocket connection closed (code: {close_status_code}, message: {close_msg or 'No message'})")
        if self.running:  # Only try to reconnect if we're supposed to be running
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached. Giving up.")
            self.running = False
            return
            
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                   self.max_reconnect_delay)
        
        self.logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} in {delay} seconds...")
        
        # Schedule reconnection in a separate thread to avoid blocking
        def delayed_reconnect():
            time.sleep(delay)
            if self.running:
                self._reconnect()
        
        threading.Thread(target=delayed_reconnect, daemon=True).start()
    
    def _reconnect(self):
        """Attempt to reconnect to the WebSocket server"""
        self.logger.info("Attempting to reconnect...")
        try:
            self._cleanup_websocket()
            self._start_websocket()
        except Exception as e:
            self.logger.error(f"Reconnection failed: {str(e)}", exc_info=True)
            self._schedule_reconnect()
    
    def _start_websocket(self):
        """Start the WebSocket connection in a separate thread"""
        if not self.running:
            return
            
        try:
            self.ws = websocket.WebSocketApp(
                "wss://streamer.finance.yahoo.com",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self._run_websocket,
                name=f"WebSocketThread-{id(self)}",
                daemon=True
            )
            self.ws_thread.start()
            
            # Wait a moment for connection to establish
            time.sleep(1)
            
            # Subscribe to tickers
            self._subscribe_to_tickers()
            
            self.logger.info("WebSocket connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {str(e)}", exc_info=True)
            raise
    
    def _run_websocket(self):
        """Run the WebSocket client"""
        try:
            self.ws.run_forever(
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong response
                ping_payload=json.dumps({"type": "ping"})  # Custom ping message if needed
            )
        except Exception as e:
            self.logger.error(f"WebSocket error in run_forever: {str(e)}", exc_info=True)
        finally:
            self._cleanup_websocket()
    
    def _subscribe_to_tickers(self):
        """Subscribe to ticker updates"""
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            self.logger.warning("Cannot subscribe: WebSocket not connected")
            return
            
        for ticker in self.tickers:
            try:
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "symbol": ticker,
                    "interval": self.interval
                })
                self.ws.send(subscribe_msg)
                self.logger.debug(f"Subscribed to {ticker}")
            except Exception as e:
                self.logger.error(f"Failed to subscribe to {ticker}: {str(e)}", exc_info=True)
    
    def start_streaming(self):
        """Start real-time data streaming"""
        if self.running:
            self.logger.warning("Streaming is already running")
            return
            
        self.running = True
        self.reconnect_attempts = 0
        
        try:
            # Initialize historical data
            self._init_data()
            
            # Start WebSocket connection
            self._start_websocket()
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {str(e)}", exc_info=True)
            self._schedule_reconnect()
    
    def _cleanup_websocket(self):
        """Clean up WebSocket resources"""
        try:
            if hasattr(self, 'ws') and self.ws:
                # Check if the WebSocket is still connected before closing
                if hasattr(self.ws, 'sock') and self.ws.sock:
                    self.logger.info("Closing WebSocket connection...")
                    self.ws.close()
                    # Give it a moment to close gracefully
                    time.sleep(0.5)
                self.ws = None
        except Exception as e:
            self.logger.error(f"Error cleaning up WebSocket: {str(e)}", exc_info=True)
            
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
        
    def close(self):
        """
        Cleanly close the WebSocket connection and clean up all resources
        """
        self.logger.info("Closing RealTimeData and cleaning up resources...")
        self.running = False  # Signal all threads to stop
        
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
        # Start streaming
        real_time.start_streaming()
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        real_time.close()
        print("\nStreaming stopped")
