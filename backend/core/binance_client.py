"""
Binance Free API Client - No API Key Required
Uses public endpoints for market data
"""

import aiohttp
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class BinanceClient:
    """Free Binance API client - no authentication needed"""
    
    BASE_URL = "https://api.binance.com/api"
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make async request to Binance API"""
        try:
            url = f"{self.BASE_URL}{endpoint}"
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"API Error: {response.status} - {await response.text()}")
                    return None
        except asyncio.TimeoutError:
            print(f"Request timeout: {endpoint}")
            return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[List]:
        """
        Get candlestick data
        intervals: 1m, 5m, 15m, 1h, 4h, 1d, etc.
        """
        endpoint = "/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        data = await self._request(endpoint, params)
        if data:
            return data
        return []
    
    async def get_ticker_24h(self, symbol: str) -> Dict:
        """Get 24h ticker data"""
        endpoint = "/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        data = await self._request(endpoint, params)
        return data if data else {}
    
    async def get_price(self, symbol: str) -> float:
        """Get current price of a symbol"""
        endpoint = "/v3/ticker/price"
        params = {"symbol": symbol}
        
        data = await self._request(endpoint, params)
        if data and "price" in data:
            return float(data["price"])
        return 0.0
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange info with all trading pairs"""
        endpoint = "/v3/exchangeInfo"
        data = await self._request(endpoint)
        return data if data else {}
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book snapshot"""
        endpoint = "/v3/depth"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        data = await self._request(endpoint, params)
        return data if data else {}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        endpoint = "/v3/trades"
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000)
        }
        
        data = await self._request(endpoint, params)
        return data if data else []
    
    async def get_multi_timeframe_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for multiple timeframes"""
        data = {}
        
        for timeframe in timeframes:
            klines = await self.get_klines(symbol, timeframe, limit=500)
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore'
                ])
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                data[timeframe] = df
            
            await asyncio.sleep(0.1)  # Rate limit
        
        return data
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists on Binance"""
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            symbols = [s["symbol"] for s in exchange_info["symbols"]]
            return symbol in symbols
        return False
    
    async def get_all_symbols(self) -> List[str]:
        """Get list of all trading symbols"""
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            return [s["symbol"] for s in exchange_info["symbols"] if s["status"] == "TRADING"]
        return []
    
    async def get_24h_volume(self, symbol: str) -> float:
        """Get 24h trading volume"""
        ticker = await self.get_ticker_24h(symbol)
        if ticker and "quoteVolume" in ticker:
            return float(ticker["quoteVolume"])
        return 0.0
    
    async def get_price_change_percent(self, symbol: str) -> float:
        """Get 24h price change percentage"""
        ticker = await self.get_ticker_24h(symbol)
        if ticker and "priceChangePercent" in ticker:
            return float(ticker["priceChangePercent"])
        return 0.0


async def get_binance_client() -> BinanceClient:
    """Factory function to create Binance client"""
    return BinanceClient()
