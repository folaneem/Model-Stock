#!/usr/bin/env python
"""
Test script for real-time data connection with Alpaca API.
Run this script directly to test WebSocket connection and data streaming.

Usage:
    python test_realtime.py [ticker]
    
Example:
    python test_realtime.py AAPL
"""

import os
import sys
import json
import time
import logging
import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('realtime_test')

class RealTimeTest:
    """Simple test class for Alpaca WebSocket connection."""
    
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
        
        # Message tracking
        self.last_message = None
        self.last_message_time = None
        self.message_count = 0
        
        # Check credentials
        if not self.api_key or not self.api_secret:
            logger.error("API key or secret not found in environment variables")
            logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
            sys.exit(1)
        
        logger.info("RealTimeTest initialized")
    
    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.ws_url}")
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.connected = True
            logger.info("Connected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
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
            await self.ws.send(json.dumps(auth_msg))
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            logger.info(f"Auth response: {response_data}")
            
            # First message is usually a connection success message
            # We need to receive the actual auth response
            if isinstance(response_data, list) and len(response_data) > 0 and response_data[0].get('T') == 'success' and 'connected' in response_data[0].get('msg', ''):
                # This is just the connection message, get the actual auth response
                response = await self.ws.recv()
                response_data = json.loads(response)
                logger.info(f"Second response (auth): {response_data}")
            
            # Check for authentication success
            if isinstance(response_data, list) and len(response_data) > 0:
                for msg in response_data:
                    if msg.get('T') == 'success' and msg.get('msg') == 'authenticated':
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
        
        # Add symbols to subscription set
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
            logger.info(f"Subscription response: {response}")
            return True
        except Exception as e:
            logger.error(f"Subscription error: {str(e)}")
            return False
    
    async def listen(self, max_messages=10):
        """Listen for messages from the WebSocket server."""
        if not self.connected or not self.authenticated:
            logger.error("Cannot listen: not connected or not authenticated")
            return
        
        logger.info(f"Listening for up to {max_messages} messages...")
        
        try:
            message_count = 0
            # Increase timeout to 20 seconds to allow more time for messages during market hours
            while message_count < max_messages:
                message = await asyncio.wait_for(self.ws.recv(), timeout=20.0)
                message_data = json.loads(message)
                self.last_message = message_data
                self.last_message_time = datetime.now()
                self.message_count += 1
                message_count += 1
                
                # Process the message
                logger.info(f"Message {message_count}: {message_data}")
                
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
                                logger.info(f"Quote: {symbol} - Bid: ${item.get('bp')} x {item.get('bs')}, Ask: ${item.get('ap')} x {item.get('as')}")
                
        except asyncio.TimeoutError:
            logger.warning("Listening timeout - no messages received in 5 seconds")
        except Exception as e:
            logger.error(f"Listening error: {str(e)}")
        
        # Print summary
        logger.info(f"Received {message_count} messages")
        logger.info(f"Current data: {json.dumps(self.data, indent=2)}")
    
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
    
    async def run_test(self, symbols: List[str], duration: int = 30):
        """Run a complete test of the WebSocket connection."""
        # Connect
        if not await self.connect():
            logger.error("Test failed: Could not connect")
            return False
        
        # Authenticate
        if not await self.authenticate():
            logger.error("Test failed: Could not authenticate")
            await self.close()
            return False
        
        # Subscribe
        if not await self.subscribe(symbols):
            logger.error("Test failed: Could not subscribe")
            await self.close()
            return False
        
        # Listen for messages
        await self.listen(max_messages=duration)
        
        # Close connection
        await self.close()
        
        return True

# Check market status
async def check_market_status():
    """Check if the market is currently open."""
    import requests
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        logger.error("API key or secret not found in environment variables")
        return None
    
    # Log the first few characters of the credentials for debugging
    logger.info(f"Using API key starting with: {api_key[:4]}...")
    
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret
    }
    
    try:
        logger.info("Checking market status via Alpaca API...")
        # Use the paper trading API endpoint for free tier access
        response = requests.get('https://paper-api.alpaca.markets/v2/clock', headers=headers)
        if response.status_code == 200:
            data = response.json()
            is_open = data.get('is_open', False)
            next_open = data.get('next_open')
            next_close = data.get('next_close')
            
            if is_open:
                logger.info("🟢 Market is OPEN")
                if next_close:
                    next_close_time = datetime.fromisoformat(next_close.replace('Z', '+00:00'))
                    logger.info(f"Market will close at: {next_close_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            else:
                logger.info("🔴 Market is CLOSED")
                if next_open:
                    next_open_time = datetime.fromisoformat(next_open.replace('Z', '+00:00'))
                    logger.info(f"Market will open at: {next_open_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            return data
        else:
            logger.error(f"Failed to get market status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error checking market status: {str(e)}")
        return None

async def main():
    """Main function."""
    # Get ticker from command line argument or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    
    logger.info(f"Testing real-time data for {ticker}")
    
    # Check market status
    market_status = await check_market_status()
    if market_status:
        is_open = market_status.get('is_open', False)
        if not is_open:
            logger.warning("⚠️ Market is currently CLOSED. You may not receive real-time data.")
            logger.warning("⚠️ The WebSocket connection will still work, but you might not get any messages.")
            logger.warning("⚠️ Consider running this test during market hours (9:30 AM - 4:00 PM ET, Monday-Friday).")
            
            # Ask if the user wants to continue
            print("\nMarket is closed. Do you want to continue anyway? (y/n): ", end='')
            response = input().lower()
            if response != 'y':
                logger.info("Test aborted by user.")
                return
    
    # Check if API keys are set
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        logger.error("API key or secret not found in environment variables")
        logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    # Print masked credentials for debugging
    if api_key and len(api_key) > 8:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        logger.info(f"Using API key: {masked_key}")
    if api_secret and len(api_secret) > 8:
        masked_secret = f"{api_secret[:4]}...{api_secret[-4:]}"
        logger.info(f"Using API secret: {masked_secret}")
    
    # Check market status
    await check_market_status()
    
    # Create and run test
    test = RealTimeTest()
    await test.run_test([ticker], duration=20)
    
    logger.info("Test completed")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
