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
    
    async def _connect(self) -> bool:
        """Establish WebSocket connection with retry logic."""
        attempts = 0
        
        while attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to WebSocket (attempt {attempts + 1}/{self.max_reconnect_attempts})")
                self.ws = await websockets.connect(self.ws_url, ping_interval=10, ping_timeout=5)
                self.connected = True
                logger.info("WebSocket connection established")
                return True
                
            except Exception as e:
                attempts += 1
                logger.error(f"Connection attempt {attempts} failed: {str(e)}")
                
                if attempts >= self.max_reconnect_attempts:
                    logger.error("Max connection attempts reached")
                    return False
                
                # Exponential backoff with jitter
                delay = min(self.reconnect_delay * (2 ** (attempts - 1)), self.max_reconnect_delay)
                await asyncio.sleep(delay)
    
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
            return True
            
        if not await self._connect():
            return False
            
        if not await self._authenticate():
            return False
            
        self.running = True
        asyncio.create_task(self._message_loop())
        return True
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and clean up resources."""
        logger.info("Stopping WebSocket connection...")
        self.running = False
        self._should_reconnect = False
        
        # First, set flags to prevent reconnection attempts
        self.connected = False
        self.authenticated = False
        
        # Then close the WebSocket connection if it exists
        if self.ws:
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
            finally:
                # Ensure we clear the reference even if close fails
                self.ws = None
                
        # Clear subscribed symbols
        self.subscribed_symbols.clear()
        
        logger.info("WebSocket connection stopped")
        
        # Return control to the event loop to allow pending tasks to complete
        await asyncio.sleep(0)
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        while self.running and self.ws:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                await self._process_message(data)
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed: {str(e)}")
                if self._should_reconnect:
                    await self._handle_reconnect()
                break
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}", exc_info=True)
    
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
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection logic."""
        await self.stop()
        await asyncio.sleep(self.reconnect_delay)
        await self.start()
        await self.subscribe(self.subscribed_symbols)
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to receive WebSocket updates."""
        self.callbacks.add(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a previously registered callback."""
        self.callbacks.discard(callback)

# Global instance
websocket_manager = WebSocketManager()
