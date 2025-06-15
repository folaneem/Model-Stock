"""
WebSocket Initializer

This module provides functions to initialize and manage the WebSocket connection
using a direct approach based on the successful pattern in test_realtime.py.
"""

import os
import sys
import json
import time
import logging
import asyncio
import concurrent.futures
import threading
import websockets
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

# Import RealTimeDataHandler from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from app import RealTimeDataHandler
    _use_rtdh = True
except ImportError:
    _use_rtdh = False
    logging.getLogger(__name__).warning("Could not import RealTimeDataHandler from app.py. Using local implementation.")

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to track WebSocket state
_ws_client = None
_ws_thread = None
_ws_loop = None
_event_loop_running = False

class AlpacaWebSocketClient:
    """Simple WebSocket client for Alpaca Market Data API."""
    
    def __init__(self):
        """Initialize the WebSocket client with default settings"""
        # WebSocket settings
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"  # Use IEX for free tier data
        self.ws = None
        self.connected = False
        self.authenticated = False
        self.running = False
        
        # API credentials
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        # Connection management
        self.connection_attempts = 0
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_attempts = 10
        self.max_reconnect_delay = 30  # Max 30 seconds between retries
        
        # Data storage
        self.subscribed_symbols = set()
        self.data = {}
        self.last_update = {}
        
        # Message tracking
        self.last_message = None
        self.last_message_time = None
        self.message_count = 0
        
        # Check credentials
        if not self.api_key or not self.api_secret:
            logger.error("API key or secret not found in environment variables")
            logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        else:
            logger.info("API credentials loaded successfully")
            
    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.ws_url}")
        try:
            # Create a simple connection without custom SSL context first
            self.ws = await websockets.connect(self.ws_url)
            self.connected = True
            self.connection_attempts = 0  # Reset connection attempts on success
            logger.info("Connected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            self.connected = False
            return False
    
    async def authenticate(self):
        """Authenticate with the WebSocket server."""
        if not self.connected or not self.ws:
            logger.error("Cannot authenticate: not connected")
            return False
        
        logger.info("Authenticating...")
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        
        try:
            # Send authentication message
            await self.ws.send(json.dumps(auth_msg))
            
            # First message might be a connection success message
            response = await self.ws.recv()
            response_data = json.loads(response)
            logger.info(f"Auth response: {response_data}")
            
            # Check if this is just a connection message and wait for auth response if needed
            if isinstance(response_data, list) and len(response_data) > 0 and response_data[0].get('T') == 'success' and 'connected' in response_data[0].get('msg', ''):
                # This is just the connection message, get the actual auth response
                response = await self.ws.recv()
                response_data = json.loads(response)
                logger.info(f"Second response (auth): {response_data}")
            
            # Check for authentication success
            if isinstance(response_data, list) and len(response_data) > 0:
                for msg in response_data:
                    if msg.get('T') == 'success' and 'authenticated' in msg.get('msg', ''):
                        self.authenticated = True
                        logger.info("Authentication successful")
                        return True
            
            # If we got here, authentication failed
            logger.error(f"Authentication failed: {response_data}")
            return False
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to market data for the given symbols."""
        if not self.connected or not self.authenticated:
            logger.error("Cannot subscribe: not connected or not authenticated")
            return False
        
        # Add symbols to subscription set (uppercase for consistency)
        self.subscribed_symbols.update([s.upper() for s in symbols])
        
        # Create subscription message
        sub_msg = {
            "action": "subscribe",
            "trades": list(self.subscribed_symbols),
            "quotes": list(self.subscribed_symbols),
            "bars": list(self.subscribed_symbols)
        }
        
        try:
            logger.info(f"Subscribing to: {self.subscribed_symbols}")
            await self.ws.send(json.dumps(sub_msg))
            response = await self.ws.recv()
            response_data = json.loads(response)
            logger.info(f"Subscription response: {response_data}")
            return True
        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            return False
    
    async def listen(self, max_messages=None):
        """Listen for messages from the WebSocket server."""
        if not self.connected or not self.authenticated:
            logger.error("Cannot listen: not connected or not authenticated")
            return
        
        logger.info("Listening for messages...")
        
        try:
            message_count = 0
            while self.connected and (max_messages is None or message_count < max_messages):
                try:
                    # Wait for a message with a timeout
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    message_data = json.loads(message)
                    self.last_message = message_data
                    self.last_message_time = datetime.now()
                    self.message_count += 1
                    message_count += 1
                    
                    # Process the message
                    logger.debug(f"Message received: {message_data}")
                    
                    # Store data
                    if isinstance(message_data, list):
                        for item in message_data:
                            msg_type = item.get('T')
                            if msg_type == 't':  # Trade
                                symbol = item.get('S')
                                if symbol:
                                    if symbol not in self.data:
                                        self.data[symbol] = {}
                                    self.data[symbol]['price'] = item.get('p')
                                    self.data[symbol]['size'] = item.get('s')
                                    self.data[symbol]['timestamp'] = item.get('t')
                                    self.last_update[symbol] = datetime.now()
                                    logger.info(f"Trade: {symbol} - ${item.get('p')} x {item.get('s')}")
                            elif msg_type == 'q':  # Quote
                                symbol = item.get('S')
                                if symbol:
                                    if symbol not in self.data:
                                        self.data[symbol] = {}
                                    self.data[symbol]['bid'] = item.get('bp')
                                    self.data[symbol]['ask'] = item.get('ap')
                                    self.data[symbol]['bid_size'] = item.get('bs')
                                    self.data[symbol]['ask_size'] = item.get('as')
                                    self.last_update[symbol] = datetime.now()
                                    logger.debug(f"Quote: {symbol} - Bid: ${item.get('bp')}, Ask: ${item.get('ap')}")
                except asyncio.TimeoutError:
                    # No message received within timeout, check connection
                    logger.debug("No message received within timeout, checking connection...")
                    try:
                        # Simple ping by sending a small message
                        pong_waiter = await self.ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=5.0)
                        logger.debug("Connection is still alive")
                    except Exception as e:
                        logger.warning(f"Connection seems to be dead: {str(e)}")
                        self.connected = False
                        break
                        
        except Exception as e:
            logger.error(f"Listening error: {str(e)}")
            self.connected = False
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
        
        self.connected = False
        self.authenticated = False
        self.running = False

def check_market_status() -> Dict[str, Any]:
    """
    Check if the market is currently open.
    
    Returns:
        Dict[str, Any]: Market status information
    """
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        logger.error("API key or secret not found in environment variables")
        return {"is_open": False, "error": "API credentials not found"}
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    try:
        # Check market status using Alpaca API
        url = "https://api.alpaca.markets/v2/clock"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "is_open": data.get("is_open", False),
                "next_open": data.get("next_open"),
                "next_close": data.get("next_close"),
                "timestamp": data.get("timestamp")
            }
        else:
            logger.error(f"Failed to check market status: {response.status_code} - {response.text}")
            return {"is_open": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error checking market status: {str(e)}")
        return {"is_open": False, "error": str(e)}

async def _run_websocket_client(client: AlpacaWebSocketClient, symbols: List[str]):
    """
    Run the WebSocket client in an async context.
    
    Args:
        client: The AlpacaWebSocketClient instance
        symbols: List of symbols to subscribe to
    """
    global _ws_connected
    
    # Store references to any tasks we create for proper cleanup
    tasks = []
    
    try:
        # Connect to WebSocket
        if not await client.connect():
            logger.error("Failed to connect to WebSocket")
            return
        
        # Authenticate
        if not await client.authenticate():
            logger.error("Failed to authenticate with WebSocket")
            await client.close()
            return
        
        # Subscribe to symbols
        if symbols and not await client.subscribe(symbols):
            logger.error("Failed to subscribe to symbols")
            await client.close()
            return
        
        # Mark as connected and running
        _ws_connected = True
        client.running = True
        
        # Create a separate task for listening to allow for proper cancellation
        listen_task = asyncio.create_task(client.listen())
        tasks.append(listen_task)
        
        # Wait for the listen task to complete or for the connection to be closed
        await listen_task
        
    except asyncio.CancelledError:
        logger.info("WebSocket client task was cancelled")
    except Exception as e:
        logger.error(f"Error in WebSocket client: {str(e)}", exc_info=True)
    finally:
        # Cancel any tasks we created
        for task in tasks:
            if not task.done() and not task.cancelled():
                try:
                    task.cancel()
                    # Wait briefly for task to acknowledge cancellation
                    try:
                        await asyncio.wait_for(task, timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                        pass
                except Exception as e:
                    logger.debug(f"Error cancelling task: {str(e)}")
        
        # Close the WebSocket connection if it exists
        if hasattr(client, 'ws') and client.ws:
            try:
                # First, directly access and close the underlying WebSocket connection
                if hasattr(client.ws, 'recv_messages') and hasattr(client.ws.recv_messages, 'frames'):
                    frames = client.ws.recv_messages.frames
                    if hasattr(frames, 'get_waiter') and frames.get_waiter is not None:
                        try:
                            # Cancel the waiter to prevent it from trying to receive after loop closure
                            frames.get_waiter.cancel()
                            await asyncio.sleep(0.1)  # Brief pause to allow cancellation to process
                        except Exception as e:
                            logger.debug(f"Error cancelling get_waiter: {str(e)}")
                
                # Now close the WebSocket with a timeout
                try:
                    await asyncio.wait_for(client.ws.close(), timeout=1.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"WebSocket close timed out or failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Error during WebSocket cleanup: {str(e)}")
    
    # Wait for the thread to finish if it exists
    if _ws_thread and _ws_thread.is_alive():
        try:
            _ws_thread.join(timeout=3.0)
            logger.info("WebSocket thread joined")
        except Exception as e:
            logger.error(f"Error joining WebSocket thread: {str(e)}")
    
    # Reset global variables
    _ws_client = None
    _ws_thread = None
    _ws_loop = None
    _event_loop_running = False
    
    logger.info("WebSocket cleanup completed")


def _websocket_thread_func(symbols: List[str]):
    """
    Function to run in a separate thread to manage the WebSocket connection.
    
    Args:
        symbols: List of symbols to subscribe to
    """
    global _ws_client, _ws_loop, _event_loop_running
    
    try:
        # Create a new event loop for this thread
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _event_loop_running = True
        
        # Create client instance
        _ws_client = AlpacaWebSocketClient()
        
        # Set up a more robust exception handler to prevent "Event loop is closed" errors
        def exception_handler(loop, context):
            # Extract exception and message
            exception = context.get('exception')
            message = context.get('message', '')
            
            # Handle common shutdown-related errors
            if any(error_text in message for error_text in ['Event loop is closed', 'Task was destroyed but it is pending']):
                logger.debug(f"Ignoring expected shutdown error: {message}")
                return
                
            if exception:
                if isinstance(exception, RuntimeError) and 'Event loop is closed' in str(exception):
                    # Ignore "Event loop is closed" errors during shutdown
                    logger.debug("Ignoring 'Event loop is closed' RuntimeError during shutdown")
                    return
                elif isinstance(exception, asyncio.CancelledError):
                    # Ignore task cancellation during shutdown
                    logger.debug("Ignoring CancelledError during shutdown")
                    return
                elif isinstance(exception, concurrent.futures.CancelledError):
                    # Ignore future cancellation during shutdown
                    logger.debug("Ignoring concurrent.futures.CancelledError during shutdown")
                    return
                    
            # For all other exceptions, use the default handler
            loop.default_exception_handler(context)
        
        # Set the custom exception handler
        _ws_loop.set_exception_handler(exception_handler)
        
        # Run the WebSocket client in this loop
        _ws_loop.run_until_complete(_run_websocket_client(_ws_client, symbols))
        
    except Exception as e:
        logger.error(f"Error in WebSocket thread: {str(e)}")
    finally:
        # Clean up
        try:
            # Mark the event loop as not running to prevent new operations
            _event_loop_running = False
            
            # Safely handle WebSocket shutdown
            if _ws_client:
                # Mark the client as not running
                _ws_client.running = False
                
                # Safely handle the WebSocket connection
                if hasattr(_ws_client, 'ws') and _ws_client.ws:
                    try:
                        # Set flags to indicate we're closing to prevent new operations
                        if hasattr(_ws_client.ws, '_closing'):
                            _ws_client.ws._closing = True
                        
                        # Handle message queues
                        if hasattr(_ws_client.ws, 'recv_messages'):
                            try:
                                # Mark the messages queue as closed
                                if hasattr(_ws_client.ws.recv_messages, 'closed'):
                                    _ws_client.ws.recv_messages.closed = True
                                    
                                # Cancel any pending receive operations
                                if hasattr(_ws_client.ws.recv_messages, 'frames'):
                                    frames = _ws_client.ws.recv_messages.frames
                                    if hasattr(frames, 'get_waiter') and frames.get_waiter is not None:
                                        try:
                                            # Cancel the waiter before closing
                                            frames.get_waiter.cancel()
                                        except Exception as e:
                                            logger.debug(f"Error cancelling get_waiter: {str(e)}")
                            except Exception as e:
                                logger.debug(f"Error handling recv_messages: {str(e)}")
                    except Exception as e:
                        logger.debug(f"Error preparing WebSocket for shutdown: {str(e)}")
            
            # Define a function to safely cancel all tasks
            async def safe_cancel_tasks():
                try:
                    # Get all pending tasks except the current one
                    tasks = [t for t in asyncio.all_tasks(_ws_loop) 
                             if t is not asyncio.current_task(_ws_loop)]
                    
                    if tasks:
                        logger.info(f"Cancelling {len(tasks)} pending tasks before closing event loop")
                        for task in tasks:
                            task.cancel()
                        
                        # Give tasks a chance to respond to cancellation
                        await asyncio.gather(*tasks, return_exceptions=True)
                        
                    # Short delay to allow cancellation to complete
                    await asyncio.sleep(0.1)
                    return True
                except Exception as e:
                    logger.debug(f"Error in safe_cancel_tasks: {str(e)}")
                    return False
            
            # Stop the event loop if it's running
            if _ws_loop and _ws_loop.is_running():
                _ws_loop.stop()
                
            # Cancel all pending tasks if the loop is not closed
            if _ws_loop and not _ws_loop.is_closed():
                try:
                    # Use wait_for to ensure we don't block indefinitely
                    _ws_loop.run_until_complete(
                        asyncio.wait_for(safe_cancel_tasks(), timeout=1.0)
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.debug("Timeout or cancellation during task cleanup")
                except Exception as e:
                    logger.debug(f"Error during task cancellation: {str(e)}")
                
                # Properly shutdown async generators
                try:
                    _ws_loop.run_until_complete(
                        asyncio.wait_for(_ws_loop.shutdown_asyncgens(), timeout=0.5)
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.debug("Timeout during asyncgens shutdown")
                except Exception as e:
                    logger.debug(f"Error during shutdown_asyncgens: {str(e)}")
                
                # Finally close the event loop
                try:
                    _ws_loop.close()
                    logger.info("Event loop closed successfully")
                except Exception as e:
                    logger.debug(f"Error closing event loop: {str(e)}")
        except Exception as e:
            logger.error(f"Error during WebSocket thread cleanup: {str(e)}", exc_info=True)


def stop_websocket():
    """
    Stop the WebSocket client and clean up resources.
    """
    global _ws_client, _ws_thread, _ws_loop, _event_loop_running
    
    logger.info("Stopping WebSocket client...")
    
    # First mark the event loop as not running to prevent new operations
    _event_loop_running = False
    
    # Check if client exists
    if _ws_client:
        # Set a flag to indicate we're closing to prevent new receive operations
        if hasattr(_ws_client, 'ws') and _ws_client.ws:
            try:
                # Mark the connection as closing to prevent new operations
                _ws_client.running = False
                if hasattr(_ws_client.ws, '_closing'):
                    _ws_client.ws._closing = True
                    
                # Safely handle the WebSocket connection
                if hasattr(_ws_client.ws, 'recv_messages'):
                    try:
                        # Mark the messages queue as closed to prevent new operations
                        if hasattr(_ws_client.ws.recv_messages, 'closed'):
                            _ws_client.ws.recv_messages.closed = True
                    except Exception as e:
                        logger.debug(f"Error marking recv_messages as closed: {str(e)}")
            except Exception as e:
                logger.debug(f"Error preparing WebSocket for shutdown: {str(e)}")
        
        # If the event loop is still available, use it to close the client
        if _ws_loop and not (_ws_loop.is_closed()):
            try:
                # Define a function to cancel all pending operations
                async def safe_cancel_all():
                    try:
                        # Cancel any pending WebSocket operations
                        if hasattr(_ws_client, 'ws') and _ws_client.ws:
                            # Cancel any pending receive operations
                            if hasattr(_ws_client.ws, 'recv_messages') and hasattr(_ws_client.ws.recv_messages, 'frames'):
                                frames = _ws_client.ws.recv_messages.frames
                                if hasattr(frames, 'get_waiter') and frames.get_waiter is not None:
                                    frames.get_waiter.cancel()
                            
                            # Close the WebSocket connection
                            await _ws_client.close()
                            
                        # Cancel all pending tasks
                        tasks = [t for t in asyncio.all_tasks(_ws_loop) if t is not asyncio.current_task(_ws_loop)]
                        if tasks:
                            logger.info(f"Cancelling {len(tasks)} pending tasks")
                            for task in tasks:
                                task.cancel()
                            # Wait for tasks to acknowledge cancellation
                            await asyncio.gather(*tasks, return_exceptions=True)
                            
                        # Allow a short delay for cleanup
                        await asyncio.sleep(0.1)
                        return True
                    except Exception as e:
                        logger.debug(f"Error in safe_cancel_all: {str(e)}")
                        return False
                
                # Try to run the cancellation function in the event loop
                try:
                    if _ws_loop.is_running():
                        # If loop is running, use run_coroutine_threadsafe
                        future = asyncio.run_coroutine_threadsafe(safe_cancel_all(), _ws_loop)
                        # Wait with timeout
                        future.result(timeout=2.0)
                    else:
                        # If loop is not running, use run_until_complete
                        _ws_loop.run_until_complete(asyncio.wait_for(safe_cancel_all(), timeout=2.0))
                    logger.info("Successfully cancelled all WebSocket operations")
                except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                    logger.warning("Timeout while cancelling WebSocket operations")
                except Exception as e:
                    logger.warning(f"Error during WebSocket shutdown: {str(e)}")
            except Exception as e:
                logger.error(f"Error stopping WebSocket client: {str(e)}")
        else:
            logger.debug("Event loop is already closed, using direct close")
            # Try direct close as fallback
            try:
                if hasattr(_ws_client, 'ws') and _ws_client.ws:
                    _ws_client.ws.close()
                    logger.info("WebSocket connection closed directly")
            except Exception as e:
                logger.debug(f"Error in direct close: {str(e)}")

    # Wait for the thread to finish if it exists
    if _ws_thread and _ws_thread.is_alive():
        try:
            # Set a reasonable timeout to avoid blocking indefinitely
            _ws_thread.join(timeout=3.0)
            if not _ws_thread.is_alive():
                logger.info("WebSocket thread joined successfully")
            else:
                logger.warning("WebSocket thread did not terminate within timeout")
        except Exception as e:
            logger.error(f"Error joining WebSocket thread: {str(e)}")

    # Reset global variables
    _ws_client = None
    _ws_thread = None
    _ws_loop = None
    _event_loop_running = False
    
    logger.info("WebSocket shutdown complete")
    
    # Force Python's garbage collector to run
    import gc
    gc.collect()
    
    return True