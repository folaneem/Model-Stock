import os
import asyncio
import json
import logging
from dotenv import load_dotenv
import websockets
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Alpaca WebSocket URL
ALPACA_WEBSOCKET_URL = "wss://stream.data.alpaca.markets/v2/iex"

class AlpacaWebSocketTest:
    def __init__(self):
        """Initialize the Alpaca WebSocket test client."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.websocket = None
        self.running = False

    async def connect(self):
        """Connect to Alpaca WebSocket."""
        try:
            logger.info(f"Connecting to Alpaca WebSocket at {ALPACA_WEBSOCKET_URL}")
            self.websocket = await websockets.connect(
                ALPACA_WEBSOCKET_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=20,   # Wait 20 seconds for pong
                close_timeout=10,  # Wait 10 seconds when closing
                max_size=2**23     # 8MB max message size
            )
            self.running = True
            logger.info("Successfully connected to Alpaca WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {str(e)}", exc_info=True)
            return False

    async def authenticate(self):
        """Authenticate with Alpaca WebSocket."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return False

        try:
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self.websocket.send(json.dumps(auth_msg))
            logger.info("Authentication request sent")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False

    async def subscribe(self, symbols):
        """Subscribe to real-time data for given symbols."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return False

        try:
            # Start with a minimal subscription to trades for AAPL only
            subscribe_msgs = [
                {
                    "action": "subscribe",
                    "trades": symbols,
                    "quotes": symbols
                }
            ]
            
            # We'll keep it simple with just trades and quotes for now
            
            for msg in subscribe_msgs:
                await self.websocket.send(json.dumps(msg))
                logger.info(f"Sent subscription: {msg}")
                await asyncio.sleep(1)  # Small delay between subscriptions
            
            logger.info(f"Successfully subscribed to {symbols}")
            return True
        except Exception as e:
            logger.error(f"Subscription failed: {str(e)}", exc_info=True)
            return False

    async def listen(self):
        """Listen for incoming WebSocket messages."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return

        logger.info("Listening for messages...")
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                    data = json.loads(message)
                    self._handle_message(data)
                except asyncio.TimeoutError:
                    logger.warning("No messages received for 30 seconds. Sending ping...")
                    await self.websocket.ping()
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}")
                    break
        except Exception as e:
            logger.error(f"Listener error: {str(e)}")
        finally:
            await self.close()

    def _handle_message(self, data):
        """Handle incoming WebSocket messages."""
        try:
            if isinstance(data, list):
                for msg in data:
                    self._process_single_message(msg)
            else:
                self._process_single_message(data)
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

    def _process_single_message(self, msg):
        """Process a single WebSocket message."""
        try:
            # Log the raw message for debugging
            logger.debug(f"Raw message: {json.dumps(msg, indent=2)}")
            
            if 'msg' in msg:
                if msg['msg'] == 'connected':
                    logger.info("Connected to Alpaca WebSocket")
                    return
                elif msg['msg'] == 'authenticated':
                    logger.info("Successfully authenticated with Alpaca WebSocket")
                    return
                elif msg['msg'] == 'listening':
                    logger.info(f"Listening to channels: {msg.get('listen', {}).get('channels', [])}")
                    return
            
            # Handle different message types
            if 'T' in msg:  # Trade update
                self._handle_trade_update(msg)
            elif 'Q' in msg:  # Quote update
                self._handle_quote_update(msg)
            elif 'b' in msg:  # Bar update
                self._handle_bar_update(msg)
            elif 't' in msg:  # Another trade format
                self._handle_trade_update(msg)
            elif 'data' in msg:  # Nested data
                self._process_single_message(msg['data'])
            elif isinstance(msg, list):
                for m in msg:
                    self._process_single_message(m)
            else:
                logger.debug(f"Unhandled message type: {msg}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    def _handle_trade_update(self, trade):
        """Handle trade update message."""
        symbol = trade.get('S', 'N/A')
        price = trade.get('p', 0)
        size = trade.get('s', 0)
        timestamp = trade.get('t', '')
        
        if timestamp:
            try:
                # Convert nanoseconds to datetime
                dt = datetime.fromtimestamp(int(timestamp) / 1e9)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except (ValueError, TypeError):
                pass
        
        logger.info(f"TRADE - {symbol}: ${price:.2f} x {size} @ {timestamp}")

    def _handle_quote_update(self, quote):
        """Handle quote update message."""
        symbol = quote.get('S', 'N/A')
        bid_price = quote.get('p', 0)
        bid_size = quote.get('s', 0)
        ask_price = quote.get('P', 0)
        ask_size = quote.get('S', 0)
        timestamp = quote.get('t', '')
        
        if timestamp:
            try:
                # Convert nanoseconds to datetime
                dt = datetime.fromtimestamp(int(timestamp) / 1e9)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except (ValueError, TypeError):
                pass
        
        logger.info(f"QUOTE - {symbol}: ${bid_price:.2f} x {bid_size} / ${ask_price:.2f} x {ask_size} @ {timestamp}")

    def _handle_bar_update(self, bar):
        """Handle bar update message."""
        symbol = bar.get('S', 'N/A')
        open_price = bar.get('o', 0)
        high = bar.get('h', 0)
        low = bar.get('l', 0)
        close = bar.get('c', 0)
        volume = bar.get('v', 0)
        timestamp = bar.get('t', '')
        
        if timestamp:
            try:
                # Convert to datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                pass
        
        logger.info(f"BAR - {symbol}: O:{open_price:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} V:{volume} @ {timestamp}")

    async def close(self):
        """Close the WebSocket connection."""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

async def run_test(symbols):
    """Run the WebSocket test."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API keys are set
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        logger.error("Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
        return

    client = AlpacaWebSocketTest()
    
    try:
        # Connect to WebSocket
        if not await client.connect():
            return
        
        # Authenticate
        if not await client.authenticate():
            return
        
        # Subscribe to symbols
        if not await client.subscribe(symbols):
            return
        
        # Listen for messages
        await client.listen()
        
    except KeyboardInterrupt:
        logger.info("\nTest stopped by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    finally:
        await client.close()

def main():
    """Main function to run the test."""
    # Enable debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enable websockets debug logging
    logger = logging.getLogger('websockets')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    
    # Test with just AAPL to minimize connection issues
    symbols = ["AAPL"]  # Focus on Apple stock only
    
    print("=== Alpaca WebSocket Test ===")
    print(f"Testing real-time data for: {', '.join(symbols)}")
    print("Press Ctrl+C to stop the test\n")
    
    # Run the test with a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_test(symbols))
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the loop
        loop.close()

if __name__ == "__main__":
    main()
