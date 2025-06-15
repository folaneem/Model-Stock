import os
import logging
import asyncio
import threading
import importlib.util
import sys
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime

# This class is a wrapper around RealTimeDataHandler in app.py
# to maintain backward compatibility while avoiding code duplication
class AlpacaWebSocketClient:
    """WebSocket client for Alpaca Market Data API.
    
    This is a wrapper around the RealTimeDataHandler class in app.py to avoid code duplication.
    It delegates all WebSocket handling to the RealTimeDataHandler class.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, paper: bool = True):
        """
        Initialize Alpaca WebSocket client.
        
        Args:
            api_key: Alpaca API key (default: from ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (default: from ALPACA_SECRET_KEY env var)
            paper: Use paper trading (default: True)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY')
        # Use IEX endpoint which is available for free accounts
        self.ws_url = 'wss://stream.data.alpaca.markets/v2/iex'
        
        self.connected = False
        self.authenticated = False
        self.running = False
        self.subscribed_symbols: Set[str] = set()
        self.data: Dict[str, Dict[str, Any]] = {}
        self.last_update: Dict[str, datetime] = {}
        self.callbacks: List[Callable] = []
        
        self.logger = logging.getLogger('AlpacaWebSocket')
        self._init_logger()
        
        self.loop = None
        self.ws_thread = None
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret must be provided or set in environment variables")
            
        # Import RealTimeDataHandler from app.py
        self._handler = None
        self._import_handler()
    
    def _init_logger(self):
        """Initialize logger with appropriate handlers."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
            
    def _import_handler(self):
        """Import RealTimeDataHandler from app.py"""
        try:
            # Try to import RealTimeDataHandler from app.py
            # First, find the app.py file
            import os
            import sys
            import importlib.util
            
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the src directory
            src_dir = os.path.dirname(current_dir)
            # Add src directory to path if not already there
            if src_dir not in sys.path:
                sys.path.append(src_dir)
                
            # Try different import approaches for maximum compatibility
            try:
                # First try direct import
                from app import RealTimeDataHandler
                self._handler = RealTimeDataHandler.get_instance()
            except ImportError:
                # If that fails, try to load the module from file
                app_path = os.path.join(src_dir, 'app.py')
                if os.path.exists(app_path):
                    spec = importlib.util.spec_from_file_location("app", app_path)
                    app_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(app_module)
                    self._handler = app_module.RealTimeDataHandler.get_instance()
                else:
                    raise ImportError(f"Could not find app.py at {app_path}")
                    
            self.logger.info("Successfully imported RealTimeDataHandler from app.py")
            
            # Update our state based on the handler's state
            self._sync_state_from_handler()
            
        except ImportError as e:
            self.logger.error(f"Failed to import RealTimeDataHandler from app.py: {str(e)}")
            self.logger.warning("AlpacaWebSocketClient will operate in standalone mode (not recommended)")
            self._handler = None
        except Exception as e:
            self.logger.error(f"Error importing RealTimeDataHandler: {str(e)}")
            self._handler = None
            
    def _sync_state_from_handler(self):
        """Sync state from the RealTimeDataHandler"""
        if not self._handler:
            return
            
        # Get status from handler
        status = self._handler.get_status()
        self.connected = status.get('connected', False)
        self.authenticated = status.get('authenticated', False)
        self.running = status.get('running', False)
        self.subscribed_symbols = set(status.get('subscribed_symbols', []))
        
        # Sync data
        data_summary = status.get('data_summary', {})
        for symbol, summary in data_summary.items():
            if symbol not in self.data:
                self.data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            if symbol in self._handler.data:
                self.data[symbol] = self._handler.data[symbol]
                
        # Sync last update times
        for symbol in self.data:
            if symbol in self._handler.last_update:
                self.last_update[symbol] = self._handler.last_update[symbol]
    
    async def connect(self):
        """Establish WebSocket connection and authenticate."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot connect.")
            return False
            
        # Delegate to RealTimeDataHandler
        success = await self._handler._connect()
        if success:
            # Update our state
            self._sync_state_from_handler()
        return success
    
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to market data for given symbols."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot subscribe.")
            return
            
        # Delegate to RealTimeDataHandler
        await self._handler._subscribe_to_symbols(symbols)
        # Update our state
        self._sync_state_from_handler()
    
    async def _unsubscribe(self, symbols: List[str]):
        """Unsubscribe from market data for given symbols."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot unsubscribe.")
            return
            
        # Delegate to RealTimeDataHandler
        await self._handler.unsubscribe(symbols)
        # Update our state
        self._sync_state_from_handler()
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot process message.")
            return
            
        # Delegate to RealTimeDataHandler
        # Note: RealTimeDataHandler processes messages directly in its _run_websocket method
        self.logger.debug("Message processing delegated to RealTimeDataHandler")
    
    async def _notify_callbacks(self, event_type: str, data: dict):
        """Notify all registered callbacks"""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot notify callbacks.")
            return
            
        # Delegate to RealTimeDataHandler
        # Convert event_type and data to the format expected by RealTimeDataHandler
        notification_data = {
            'type': event_type,
            'symbol': data.get('symbol', ''),
            'data': data
        }
        await self._handler._notify_callbacks(notification_data)
    
    async def _handle_trade(self, trade_data: dict):
        """Handle trade data."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot handle trade data.")
            return None
            
        # Delegate to RealTimeDataHandler
        # Note: RealTimeDataHandler handles trades directly in its _run_websocket method
        return None
    
    async def _handle_quote(self, quote_data: dict):
        """Handle quote data."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot handle quote data.")
            return None
            
        # Delegate to RealTimeDataHandler
        # Note: RealTimeDataHandler handles quotes directly in its _run_websocket method
        return None
    
    async def _handle_bar(self, bar_data: dict):
        """Handle bar data."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot handle bar data.")
            return None
            
        # Delegate to RealTimeDataHandler
        # Note: RealTimeDataHandler handles bars directly in its _run_websocket method
        return None
        
    async def _notify_callbacks(self, event_type: str, data: dict):
        """
        Notify all registered callbacks with the given event type and data.
        
        Args:
            event_type: Type of event ('trade', 'quote', or 'bar')
            data: The data associated with the event
        """
        if not hasattr(self, 'callbacks') or not self.callbacks:
            return
            
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    # Run synchronous callbacks in a thread
                    self.executor.submit(callback, event_type, data)
            except Exception as e:
                self.logger.error(f"Error in callback {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}: {str(e)}", 
                               exc_info=True)
    
    async def _run_websocket(self):
        """Main WebSocket message loop."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot run websocket.")
            self.running = False
            return
            
        # Delegate to RealTimeDataHandler
        # Note: RealTimeDataHandler handles the WebSocket message loop in its own _run_websocket method
        self.running = True
        self._sync_state_from_handler()
    
    def start(self):
        """Start the WebSocket client in a background thread."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot start websocket.")
            return False
            
        # Delegate to RealTimeDataHandler
        success = self._handler.start(wait_for_connection=True)
        # Update our state
        self._sync_state_from_handler()
        return success
    
    async def close(self):
        """Close the WebSocket connection."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot close connection.")
            return
                
        # Delegate to RealTimeDataHandler
        await self._handler.close()
        # Update our state
        self._sync_state_from_handler()
    
    def stop(self):
        """Stop the WebSocket client."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot stop client.")
            return False
                
        # Delegate to RealTimeDataHandler
        self._handler.stop()
        # Update our state
        self._sync_state_from_handler()
        return True
        
    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to market data for the given symbols."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot subscribe.")
            return False
                
        # Delegate to RealTimeDataHandler
        self._handler.subscribe(symbols)
        # Update our state
        self._sync_state_from_handler()
        return True
    
    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data for the given symbols."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot unsubscribe.")
            return False
                
        # Delegate to RealTimeDataHandler
        self._handler.unsubscribe(symbols)
        # Update our state
        self._sync_state_from_handler()
        return True
        
    def register_callback(self, callback_func: Callable) -> bool:
        """Register a callback function to be called when a message is received."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot register callback.")
            return False
                
        # Delegate to RealTimeDataHandler
        self._handler.register_callback(callback_func)
        return True
    
    def get_latest_data(self, symbol: str, data_type: str = "trades") -> Optional[Dict[str, Any]]:
        """Get the latest data for a symbol."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot get latest data.")
            return None
                
        # Delegate to RealTimeDataHandler
        return self._handler.get_latest_data(symbol, data_type)
        
    def get_historical_data(self, symbol: str, data_type: str = "trades", limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Get historical data for a symbol."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot get historical data.")
            return None
                
        # Delegate to RealTimeDataHandler
        return self._handler.get_historical_data(symbol, data_type, limit)
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the WebSocket connection."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot get status.")
            return {
                "connected": False,
                "authenticated": False,
                "running": False,
                "subscribed_symbols": [],
                "handler_available": False
            }
                
        # Delegate to RealTimeDataHandler
        handler_status = self._handler.get_status()
        # Combine with our local state
        status = {
            **handler_status,
            "wrapper_connected": self.connected,
            "wrapper_authenticated": self.authenticated,
            "wrapper_running": self.running,
            "wrapper_subscribed_symbols": list(self.subscribed_symbols),
            "handler_available": True
        }
        return status
    
    def add_callback(self, callback: Callable[[str, dict], None]):
        """Add a callback function to be called when new data is received."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot add callback.")
            return self
            
        # Delegate to RealTimeDataHandler
        # Convert callback to format expected by RealTimeDataHandler
        def adapter_callback(data):
            event_type = data.get('type')
            symbol = data.get('symbol')
            callback_data = data.get('data')
            callback(event_type, callback_data)
            
        self._handler.register_callback(adapter_callback)
        return self
        
    def remove_callback(self, callback: Callable):
        """Remove a previously added callback."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot remove callback.")
            return self
            
        # In practice, users should migrate to using RealTimeDataHandler directly
        self.logger.warning("Callback removal not fully supported in delegation mode")
        self.logger.warning("Consider using RealTimeDataHandler directly for better control")
        return self
    
    # The following methods are kept for backward compatibility but are not used
    # They are implemented in RealTimeDataHandler
    
    async def _subscribe(self, symbols: List[str]):
        """Internal method to subscribe to symbols."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available.")
            return
        await self._handler._subscribe_to_symbols(symbols)
            
    async def _process_message(self, message: str):
        """Internal method to process messages."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available.")
            return
        self.logger.debug("Message processing delegated to RealTimeDataHandler")
            
    # Helper methods for internal use
    def _sync_state_from_handler(self):
        """Sync state from the RealTimeDataHandler instance."""
        if not self._handler:
            return
            
        self.connected = self._handler.connected
        self.authenticated = self._handler.authenticated
        self.running = self._handler.running
        self.subscribed_symbols = set(self._handler.subscribed_symbols)
        
    def _get_handler(self) -> Optional[Any]:
        """Get the RealTimeDataHandler instance."""
        return self._handler
    
    def get_latest_data(self, symbol: str, data_type: str = 'trade') -> Optional[dict]:
        """Get the latest data point for a symbol and data type."""
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot get data.")
            return None
            
        # Delegate to RealTimeDataHandler
        return self._handler.get_latest_data(symbol, data_type)
    
    def get_all_data(self, symbol: str = None, data_type: str = None) -> Dict[str, Any]:
        """
        Get all stored data, optionally filtered by symbol and data type.
        
        Args:
            symbol: Optional symbol to filter by
            data_type: Optional data type to filter by ('trade', 'quote', or 'bar')
            
        Returns:
            Dictionary containing the requested data
        """
        if not self._handler:
            self.logger.error("RealTimeDataHandler not available. Cannot get data.")
            return {}
            
        # Delegate to RealTimeDataHandler
        # Convert parameters to match RealTimeDataHandler's expectations
        return self._handler.get_all_data(symbol, data_type).copy()
    
    def is_connected(self) -> bool:
        """Check if the WebSocket is connected and authenticated."""
        if not self._handler:
            return False
            
        # Delegate to RealTimeDataHandler
        return self._handler.connected and self._handler.authenticated


# Singleton instance
_alpaca_ws_client = None

def get_websocket_client() -> AlpacaWebSocketClient:
    """Get or create the global WebSocket client instance."""
    global _alpaca_ws_client
    if _alpaca_ws_client is None:
        try:
            _alpaca_ws_client = AlpacaWebSocketClient()
            _alpaca_ws_client.start()
        except Exception as e:
            logging.error(f"Failed to initialize WebSocket client: {str(e)}")
            raise
    return _alpaca_ws_client

# Add a note about migration
logging.getLogger('AlpacaWebSocket').info(
    "AlpacaWebSocketClient is now a wrapper around RealTimeDataHandler. " +
    "Consider migrating to use RealTimeDataHandler directly for better control and performance."
)