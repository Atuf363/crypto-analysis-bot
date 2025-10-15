
"""
Binance API Client - Free API (Read-only)
Handles all communication with Binance API
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


class BinanceClient:
    """Async Binance API client for market data (free tier)"""
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session: Optional[aiohttp.ClientSession] = None
        self.weight_used = 0
        self.weight_reset_time = time.time() + 60
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, weight: int = 1):
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time >= self.weight_reset_time:
            self.weight_used = 0
            self.weight_reset_time = current_time + 60
        
        # Check if adding this request exceeds limit
        if self.weight_used + weight > 1200:
            sleep_time = self.weight_reset_time - current_time
            raise Exception(f"Rate limit reached. Wait {sleep_time:.0f}s")
        
        self.weight_used += weight
    
    async def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request to Binance API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                # Track rate limit from headers
                weight = int(response.headers.get('x-mbx-used-weight-1m', 1))
                self.weight_used = weight
                
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Binance API error {response.status}: {error}")
                    
        except Exception as e:
            print(f"API request failed: {e}")
            raise
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        self._check_rate_limit(1)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        data = await self._get("/api/v3/klines", params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only necessary columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def get_ticker_24h(self, symbol: str) -> Dict:
        """Get 24h ticker statistics"""
        self._check_rate_limit(1)
        
        params = {"symbol": symbol}
        return await self._get("/api/v3/ticker/24hr", params)
    
    async def get_all_tickers(self) -> List[Dict]:
        """Get all ticker prices"""
        self._check_rate_limit(2)
        
        return await self._get("/api/v3/ticker/price")
    
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange trading rules and symbol info"""
        self._check_rate_limit(10)
        
        params = {"symbol": symbol} if symbol else {}
        return await self._get("/api/v3/exchangeInfo", params)
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book depth
        
        Args:
            symbol: Trading pair
            limit: Depth (5, 10, 20, 50, 100, 500, 1000, 5000)
        """
        weight_map = {5: 1, 10: 1, 20: 1, 50: 1, 100: 1, 500: 5, 1000: 10, 5000: 50}
        weight = weight_map.get(limit, 1)
        self._check_rate_limit(weight)
        
        params = {"symbol": symbol, "limit": limit}
        return await self._get("/api/v3/depth", params)
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for futures (if available)"""
        try:
            # Futures endpoint
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('lastFundingRate', 0)) * 100  # Convert to percentage
        except:
            pass
        
        return None
    
    async def get_usdt_pairs(self, min_volume_24h: float = 50000000) -> List[str]:
        """
        Get all USDT trading pairs with sufficient volume
        
        Args:
            min_volume_24h: Minimum 24h volume in USD
            
        Returns:
            List of symbols like ['BTCUSDT', 'ETHUSDT', ...]
        """
        exchange_info = await self.get_exchange_info()
        all_tickers = await self.get_all_tickers()
        
        # Get trading symbols
        usdt_symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
        ]
        
        # Get 24h volumes
        ticker_map = {t['symbol']: t for t in all_tickers}
        
        # Filter by volume
        valid_pairs = []
        for symbol in usdt_symbols:
            try:
                ticker = await self.get_ticker_24h(symbol)
                volume_usd = float(ticker['quoteVolume'])
                
                if volume_usd >= min_volume_24h:
                    valid_pairs.append(symbol)
                    
                # Rate limit protection
                await asyncio.sleep(0.1)
                
            except:
                continue
        
        return sorted(valid_pairs)
    
    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str] = ['1d', '4h', '1h', '15m']
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes at once
        
        Returns:
            Dict with timeframe as key, DataFrame as value
        """
        tasks = []
        for tf in timeframes:
            tasks.append(self.get_klines(symbol, tf, limit=200))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                print(f"Error fetching {symbol} {tf}: {result}")
                continue
            data[tf] = result
        
        return data
    
    def get_rate_limit_info(self) -> Dict:
        """Get current rate limit usage"""
        return {
            "weight_used": self.weight_used,
            "weight_limit": 1200,
            "weight_remaining": 1200 - self.weight_used,
            "resets_in": max(0, self.weight_reset_time - time.time())
        }


# Singleton instance
_client: Optional[BinanceClient] = None

def get_binance_client(api_key: str, api_secret: str) -> BinanceClient:
    """Get or create Binance client instance"""
    global _client
    if _client is None:
        _client = BinanceClient(api_key, api_secret)
    return _client
