"""
WebSocket Manager for handling real-time data connections.

This module provides a centralized way to manage WebSocket connections
and handle real-time data updates with proper error handling and reconnection logic.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, Optional, Set
import websockets
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections with reconnection and error handling.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.ws_url = "wss://stream.data.alpaca.markets/v2/sip"
            self.ws = None
            self.connected = False
            self.authenticated = False
            self.running = False
            self.callbacks = set()
            self.subscribed_symbols = set()
            self.max_reconnect_attempts = 5
            self.reconnect_delay = 1  # Start with 1 second delay
            self.max_reconnect_delay = 30  # Max 30 seconds between retries
            self._lock = asyncio.Lock()
            self._initialized = True
            self._should_reconnect = True
            self._loop = None
            self._connect_task = None
            self._message_task = None
            self._shutting_down = False
    
    async def _connect(self) -> bool:
        """Establish WebSocket connection with retry logic."""
        attempts = 0
        
        # Close any existing connection
        if self.ws and not self.ws.closed:
            await self.close()
        
        while attempts < self.max_reconnect_attempts and self._should_reconnect:
            try:
                logger.info(f"Connecting to WebSocket (attempt {attempts + 1}/{self.max_reconnect_attempts})")
                # Add close_timeout to prevent hanging
                self.ws = await websockets.connect(
                    self.ws_url, 
                    ping_interval=10, 
                    ping_timeout=5,
                    close_timeout=1
                )
                self.connected = True
                logger.info("WebSocket connection established")
                return True
                
            except Exception as e:
                attempts += 1
                logger.error(f"Connection attempt {attempts} failed: {str(e)}")
                
                if attempts >= self.max_reconnect_attempts:
                    logger.error("Max connection attempts reached")
                    self.connected = False
                    return False
                
                # Exponential backoff with jitter
                delay = min(self.reconnect_delay * (2 ** (attempts - 1)), self.max_reconnect_delay)
                await asyncio.sleep(delay)
        
        return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with the WebSocket server."""
        if not self.connected or not self.ws:
            return False
            
        try:
            auth_msg = {
                "action": "auth",
                "key": os.getenv("ALPACA_API_KEY"),
                "secret": os.getenv("ALPACA_SECRET_KEY")
            }
            await self.ws.send(json.dumps(auth_msg))
            response = await self.ws.recv()
            
            if "authenticated" in response.lower():
                self.authenticated = True
                logger.info("Successfully authenticated with WebSocket server")
                return True
                
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            
        return False
    
    async def subscribe(self, symbols: Set[str]) -> bool:
        """Subscribe to updates for the given symbols."""
        if not self.connected or not self.authenticated or not self.ws:
            return False
            
        try:
            subscribe_msg = {
                "action": "subscribe",
                "trades": list(symbols),
                "quotes": list(symbols),
                "bars": ["AM"]  # Aggregate minute bars
            }
            await self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to updates for symbols: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {str(e)}")
            return False
    
    async def start(self) -> bool:
        """Start the WebSocket connection and message loop."""
        if self.running:
            logger.info("WebSocket is already running")
            return True
            
        try:
            # Ensure any existing connection is properly closed first
            await self.stop()
            
            # Reset state
            self._shutting_down = False
            self.running = True
            
            # Establish new connection
            if not await self._connect():
                logger.error("Failed to establish WebSocket connection")
                self.running = False
                return False
                
            if not await self._authenticate():
                logger.error("Failed to authenticate WebSocket connection")
                self.running = False
                if self.ws and not self.ws.closed:
                    await self.ws.close()
                return False
            
            # Start message loop in a separate task
            self._message_task = asyncio.create_task(self._message_loop())
            logger.info("WebSocket started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting WebSocket: {str(e)}", exc_info=True)
            self.running = False
            if self.ws and not self.ws.closed:
                await self.ws.close()
            return False
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and clean up resources."""
        if self._shutting_down:
            return
            
        self._shutting_down = True
        self.running = False
        self._should_reconnect = False
        
        logger.info("Stopping WebSocket connection...")
        
        # Cancel the message loop task if it exists
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in message task during shutdown: {str(e)}")
            finally:
                self._message_task = None
        
        # Close the WebSocket connection if it exists
        if self.ws and not self.ws.closed:
            try:
                # Use a timeout to ensure we don't hang indefinitely
                close_task = asyncio.create_task(self.ws.close())
                try:
                    # Wait for up to 3 seconds for the connection to close
                    await asyncio.wait_for(close_task, timeout=3.0)
                    logger.info("WebSocket closed successfully")
                except asyncio.TimeoutError:
                    logger.warning("WebSocket close operation timed out")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error during WebSocket close: {str(e)}", exc_info=True)
            finally:
                # Ensure we clear the reference even if close fails
                self.ws = None
        
        # Reset connection state
        self.connected = False
        self.authenticated = False
        self.subscribed_symbols.clear()
        self._shutting_down = False
        
        logger.info("WebSocket connection stopped")
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        logger.info("Starting WebSocket message loop")
        
        while self.running and not self._shutting_down:
            try:
                if not self.ws or self.ws.closed:
                    logger.warning("WebSocket connection not available, attempting to reconnect...")
                    if not await self._reconnect():
                        logger.error("Reconnection failed, stopping message loop")
                        break
                    continue
                
                try:
                    # Use a timeout to prevent hanging on recv()
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    if not message:
                        logger.warning("Received empty message")
                        continue
                        
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.ws and not self.ws.closed:
                        try:
                            await self.ws.ping()
                        except Exception as e:
                            logger.warning(f"Ping failed: {str(e)}")
                            await asyncio.sleep(1)  # Prevent tight loop on ping failure
                    continue
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed: {str(e)}")
                if self._should_reconnect and not self._shutting_down:
                    logger.info("Attempting to reconnect...")
                    if not await self._reconnect():
                        logger.error("Reconnection failed, stopping message loop")
                        break
                else:
                    break
                    
            except asyncio.CancelledError:
                logger.info("Message loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket message loop: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on unexpected errors
                
        logger.info("WebSocket message loop ended")
    
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """Process a single WebSocket message."""
        try:
            # Handle different message types
            if 'T' in data:  # Trade message
                symbol = data.get('S', 'unknown')
                price = data.get('p', 0)
                size = data.get('s', 0)
                timestamp = data.get('t', datetime.utcnow().isoformat())
                
                # Update data for callbacks
                update = {
                    'symbol': symbol,
                    'price': price,
                    'size': size,
                    'timestamp': timestamp,
                    'type': 'trade'
                }
                await self._notify_callbacks(update)
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
    
    async def _notify_callbacks(self, data: Dict[str, Any]) -> None:
        """Notify all registered callbacks with the given data."""
        for callback in list(self.callbacks):
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in WebSocket callback: {str(e)}", exc_info=True)

    async def _reconnect(self) -> bool:
        """Handle reconnection logic.

        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        if not self._should_reconnect or self._shutting_down:
            return False

        logger.info("Attempting to reconnect...")

        try:
            # Close existing connection if any
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.close()
                except Exception as e:
                    logger.warning(f"Error closing connection during reconnect: {str(e)}")

            # Reset connection state
            self.connected = False
            self.authenticated = False

            # Attempt to reconnect
            if not await self._connect():
                logger.error("Failed to re-establish WebSocket connection")
                return False

            if not await self._authenticate():
                logger.error("Failed to re-authenticate WebSocket connection")
                return False

            # Resubscribe to symbols
            if self.subscribed_symbols:
                if not await self.subscribe(self.subscribed_symbols):
                    logger.error("Failed to resubscribe to symbols after reconnection")
                    return False

            logger.info("Successfully reconnected and resubscribed")
            return True

        except Exception as e:
            logger.error(f"Error during reconnection: {str(e)}", exc_info=True)
            return False

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to receive WebSocket updates."""
        self.callbacks.add(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered callback."""
        self.callbacks.discard(callback)

# Global instance with proper cleanup
_websocket_manager_instance = None

def get_websocket_manager():
    """Get or create the WebSocket manager instance with proper cleanup."""
    global _websocket_manager_instance
    
    if _websocket_manager_instance is None or _websocket_manager_instance.ws is None:
        _websocket_manager_instance = WebSocketManager()
        
    return _websocket_manager_instance

# For backward compatibility
websocket_manager = get_websocket_manager()
