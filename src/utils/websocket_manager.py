"""
WebSocket Manager for handling real-time data connections.

This module provides a centralized way to manage WebSocket connections
and handle real-time data updates with proper error handling and reconnection logic.
"""

import asyncio
import logging
import time
from typing import Callable, Dict, Any, Optional, Set, cast, Awaitable
import websockets
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections with reconnection and error handling.
    Implements a singleton pattern to ensure only one active connection exists.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # WebSocket connection settings
            self.ws_url = "wss://stream.data.alpaca.markets/v2/sip"
            self.ws = None
            
            # Connection state tracking
            self.connection_id = None  # Unique ID for each connection attempt
            self.connection_state = "disconnected"  # disconnected, connecting, connected, authenticated
            self.connected = False
            self.authenticated = False
            self.running = False
            self.last_message_time = None
            self.last_ping_time = None
            self.last_error = None
            
            # Callbacks and subscriptions
            self.callbacks = set()
            self.subscribed_symbols = set()
            self.connection_listeners = set()  # For connection state changes
            
            # Reconnection settings
            self.max_reconnect_attempts = 5
            self.reconnect_delay = 1  # Start with 1 second delay
            self.max_reconnect_delay = 30  # Max 30 seconds between retries
            self.consecutive_failures = 0
            
            # Async control
            self._lock = asyncio.Lock()
            self._initialized = True
            self._should_reconnect = True
            self._loop = None
            self._connect_task = None
            self._message_task = None
            self._health_check_task = None
            self._shutting_down = False
            
            # Connection registry to prevent duplicates
            self._active_connections = {}
    
    # Add type annotation for close method to help with type checking
    close: Callable[[], Awaitable[None]]

    def _update_connection_state(self, state: str) -> None:
        """Update the connection state and notify listeners.
        
        Args:
            state: New connection state (disconnected, connecting, connected, authenticated)
        """
        old_state = self.connection_state
        self.connection_state = state
        
        # Update related flags
        if state == "disconnected":
            self.connected = False
            self.authenticated = False
        elif state == "connected":
            self.connected = True
            self.authenticated = False
        elif state == "authenticated":
            self.connected = True
            self.authenticated = True
        
        # Log state change
        if old_state != state:
            logger.info(f"WebSocket connection state changed: {old_state} -> {state}")
            
            # Notify listeners
            for listener in self.connection_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        asyncio.create_task(listener(old_state, state))
                    else:
                        listener(old_state, state)
                except Exception as e:
                    logger.error(f"Error in connection state listener: {e}")

    
    async def close_connection(self) -> None:
        """Close the WebSocket connection and update connection state."""
        async with self._lock:
            ws = cast(Any, getattr(self, 'ws', None))
            if ws is not None and not ws.closed:
                try:
                    await ws.close()
                    logger.info("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self._update_connection_state("disconnected")
                    self.ws = None
                    # Remove from active connections
                    if self.connection_id in self._active_connections:
                        del self._active_connections[self.connection_id]

    # Alias for backward compatibility
    close = close_connection

    async def _connect(self) -> bool:
        """Establish WebSocket connection with retry logic and prevent duplicates."""
        attempts = 0

        # Generate a unique connection ID
        self.connection_id = f"conn-{int(time.time())}-{id(self)}"
        self._update_connection_state("connecting")

        # Close any existing connection
        if hasattr(self, 'ws') and self.ws and not self.ws.closed:
            await self.close()

        while attempts < self.max_reconnect_attempts and self._should_reconnect:
            try:
                logger.info(f"Connecting to WebSocket (attempt {attempts + 1}/{self.max_reconnect_attempts})")
                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=10,
                    ping_timeout=5,
                    close_timeout=1
                )
                # Register this connection
                self._active_connections[self.connection_id] = {
                    "ws": self.ws,
                    "created_at": datetime.utcnow().isoformat(),
                    "last_activity": datetime.utcnow().isoformat()
                }
                # Update state
                self._update_connection_state("connected")
                self.last_message_time = time.time()
                self.last_ping_time = time.time()
                self.consecutive_failures = 0
                logger.info(f"WebSocket connection established (ID: {self.connection_id})")
                return True
            except Exception as e:
                attempts += 1
                self.last_error = str(e)
                logger.error(f"Connection attempt {attempts} failed: {str(e)}")
                if attempts >= self.max_reconnect_attempts:
                    logger.error("Max connection attempts reached")
                    self._update_connection_state("disconnected")
                    return False
                delay = min(self.reconnect_delay * (2 ** (attempts - 1)), self.max_reconnect_delay)
                await asyncio.sleep(delay)
        self._update_connection_state("disconnected")
        return False

    
    async def _connect(self) -> bool:
        """Establish WebSocket connection with retry logic."""
        attempts = 0
        
        # Close any existing connection
        if hasattr(self, 'ws') and self.ws and not self.ws.closed:
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
        """Authenticate with the WebSocket server using improved state tracking and error handling."""
        if not self.connected or not self.ws:
            logger.error("Cannot authenticate: No active connection")
            return False
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            api_secret = os.getenv("ALPACA_SECRET_KEY")
            if not api_key or not api_secret:
                logger.error("Authentication failed: Missing API credentials")
                return False
            auth_msg = {
                "action": "auth",
                "key": api_key,
                "secret": api_secret
            }
            await self.ws.send(json.dumps(auth_msg))
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            self.last_message_time = time.time()
            try:
                resp_data = json.loads(response)
                if isinstance(resp_data, list) and len(resp_data) > 0:
                    msg_type = resp_data[0].get('T', '')
                    if msg_type == 'success' and 'authenticated' in response.lower():
                        self._update_connection_state("authenticated")
                        logger.info("Successfully authenticated with WebSocket server")
                        return True
            except json.JSONDecodeError:
                if "authenticated" in response.lower():
                    self._update_connection_state("authenticated")
                    logger.info("Successfully authenticated with WebSocket server")
                    return True
            logger.error(f"Authentication failed: Unexpected response: {response}")
        except asyncio.TimeoutError:
            logger.error("Authentication failed: Timed out waiting for response")
            self.last_error = "Authentication timeout"
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self.last_error = str(e)
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
        """Start the WebSocket connection, message loop, and health check."""
        async with self._lock:
            if self.running:
                logger.info("WebSocket is already running")
                return True
            try:
                # Ensure any existing connection is properly closed first
                await self.stop()
                # Reset state
                self._shutting_down = False
                self.running = True
                self._should_reconnect = True
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
                # Start health check in a separate task
                self._health_check_task = asyncio.create_task(self._health_check())
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
    
    async def _health_check(self) -> None:
        """Periodically check the health of the WebSocket connection."""
        while self.running and not self._shutting_down:
            try:
                # Check if connection is healthy
                if self.connected and self.ws and not self.ws.closed:
                    current_time = time.time()
                    if self.last_message_time and (current_time - self.last_message_time) > 60:
                        logger.warning("No messages received in over 60 seconds, sending ping")
                        try:
                            await self.ws.ping()
                            self.last_ping_time = current_time
                        except Exception as e:
                            logger.error(f"Ping failed: {str(e)}")
                            # Trigger reconnection
                            await self._reconnect()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("Health check task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check: {str(e)}")
                await asyncio.sleep(10)

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages with improved reconnection handling."""
        logger.info("Starting WebSocket message loop")
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.running and not self._shutting_down:
            try:
                # Check if connection is healthy
                if not self.ws or self.ws.closed or not self.connected or not self.authenticated:
                    logger.warning("WebSocket connection not healthy, attempting to reconnect...")
                    if not await self._reconnect():
                        consecutive_failures += 1
                        logger.warning(f"Reconnection failed ({consecutive_failures}/{max_consecutive_failures} attempts)")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Max reconnection attempts reached, giving up")
                            break
                            
                        # Wait before next reconnection attempt
                        await asyncio.sleep(min(5 * consecutive_failures, 30))  # Max 30s between retries
                        continue
                    
                    # Reset failure counter on successful reconnection
                    consecutive_failures = 0
                
                try:
                    # Use a timeout to prevent hanging on recv()
                    message = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    if not message:
                        logger.warning("Received empty message")
                        continue
                        
                    # Reset failure counter on successful message
                    consecutive_failures = 0
                    
                    # Process the message
                    try:
                        data = json.loads(message)
                        await self._process_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message as JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}", exc_info=True)
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.ws and not self.ws.closed:
                        try:
                            await self.ws.ping()
                            # Reset failure counter on successful ping
                            consecutive_failures = 0
                        except Exception as e:
                            logger.warning(f"Ping failed: {str(e)}")
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                logger.error("Max ping failures reached, reconnecting...")
                                continue
                            
                            # Wait before next attempt
                            await asyncio.sleep(min(5 * consecutive_failures, 30))
                    continue
                    
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed: {str(e)}")
                if not self._should_reconnect or self._shutting_down:
                    break
                    
                consecutive_failures += 1
                logger.info(f"Attempting to reconnect (attempt {consecutive_failures}/{max_consecutive_failures})...")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Max reconnection attempts reached, giving up")
                    break
                    
                # Wait before next reconnection attempt with backoff
                await asyncio.sleep(min(5 * consecutive_failures, 30))
                
            except asyncio.CancelledError:
                logger.info("Message loop cancelled")
                break
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Unexpected error in WebSocket message loop (attempt {consecutive_failures}/{max_consecutive_failures}): {str(e)}", 
                           exc_info=consecutive_failures >= max_consecutive_failures)
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Max error threshold reached, stopping message loop")
                    break
                    
                # Wait before next attempt with backoff
                await asyncio.sleep(min(5 * consecutive_failures, 30))
                
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
        """Handle reconnection logic with exponential backoff and improved state tracking."""
        if not self._should_reconnect or self._shutting_down:
            return False
        # Reset the reconnect delay if we had a successful connection
        if self.connected and self.authenticated:
            self.reconnect_delay = 1
            return True
        logger.info(f"Attempting to reconnect (next attempt in {self.reconnect_delay}s)...")
        try:
            # Close existing connection if any
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.close()
                except Exception as e:
                    logger.debug(f"Error closing WebSocket during reconnect: {str(e)}")
                finally:
                    self._update_connection_state("disconnected")
                    self.ws = None
                    if self.connection_id in self._active_connections:
                        del self._active_connections[self.connection_id]
            # Reconnect with backoff
            if not await self._connect():
                logger.error("Failed to re-establish WebSocket connection")
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay) * (0.8 + 0.4 * (time.time() % 1))
                await asyncio.sleep(self.reconnect_delay)
                return False
            # Re-authenticate
            if not await self._authenticate():
                logger.error("Failed to re-authenticate WebSocket connection")
                self._update_connection_state("disconnected")
                if self.ws and not self.ws.closed:
                    await self.ws.close()
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay) * (0.8 + 0.4 * (time.time() % 1))
                await asyncio.sleep(self.reconnect_delay)
                return False
            # Reset reconnect delay on successful connection
            self.reconnect_delay = 1
            # Re-subscribe to symbols
            if self.subscribed_symbols:
                try:
                    await self.subscribe(self.subscribed_symbols)
                except Exception as e:
                    logger.error(f"Failed to re-subscribe to symbols: {str(e)}")
            logger.info("Successfully reconnected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {str(e)}", exc_info=True)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay) * (0.8 + 0.4 * (time.time() % 1))
            await asyncio.sleep(self.reconnect_delay)
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
