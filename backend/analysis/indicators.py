"""
Technical Indicators Calculator
EMA, RSI, MACD, OBV, ATR, Volume Profile
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class IndicatorCalculator:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2):
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            'middle': sma,
            'upper': upper,
            'lower': lower
        }
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for a DataFrame"""
        df = df.copy()
        
        # Convert to numeric if needed
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN rows
        df = df.dropna()
        
        # EMA
        df['ema_50'] = IndicatorCalculator.calculate_ema(df['close'], 50)
        df['ema_200'] = IndicatorCalculator.calculate_ema(df['close'], 200)
        
        # SMA
        df['sma_20'] = IndicatorCalculator.calculate_sma(df['close'], 20)
        
        # RSI
        df['rsi'] = IndicatorCalculator.calculate_rsi(df['close'], 14)
        
        # MACD
        macd_data = IndicatorCalculator.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        # ATR
        df['atr'] = IndicatorCalculator.calculate_atr(df, 14)
        
        # OBV
        df['obv'] = IndicatorCalculator.calculate_obv(df)
        df['obv_ema'] = IndicatorCalculator.calculate_ema(df['obv'], 20)
        
        # Bollinger Bands
        bb = IndicatorCalculator.calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        
        return df
    
    @staticmethod
    def get_current_values(df: pd.DataFrame) -> Dict:
        """Get latest indicator values"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'price': float(latest['close']),
            'ema_50': float(latest.get('ema_50', 0) or 0),
            'ema_200': float(latest.get('ema_200', 0) or 0),
            'rsi': float(latest.get('rsi', 50) or 50),
            'macd': float(latest.get('macd', 0) or 0),
            'macd_signal': float(latest.get('macd_signal', 0) or 0),
            'macd_histogram': float(latest.get('macd_histogram', 0) or 0),
            'atr': float(latest.get('atr', 0) or 0),
            'obv': float(latest.get('obv', 0) or 0),
            'obv_ema': float(latest.get('obv_ema', 0) or 0),
            'bb_upper': float(latest.get('bb_upper', 0) or 0),
            'bb_middle': float(latest.get('bb_middle', 0) or 0),
            'bb_lower': float(latest.get('bb_lower', 0) or 0),
        }
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
        """Calculate volume profile and POC (Point of Control)"""
        if df.empty or len(df) < 10:
            return {'poc': df['close'].iloc[-1] if not df.empty else 0}
        
        price_range = df['close'].max() - df['close'].min()
        bin_size = price_range / bins if price_range > 0 else 1
        
        bins_list = np.arange(df['close'].min(), df['close'].max() + bin_size, bin_size)
        volume_at_price = []
        
        for i in range(len(bins_list) - 1):
            mask = (df['close'] >= bins_list[i]) & (df['close'] < bins_list[i+1])
            vol = df[mask]['volume'].sum()
            volume_at_price.append({'price': bins_list[i], 'volume': vol})
        
        if volume_at_price:
            poc = max(volume_at_price, key=lambda x: x['volume'])['price']
        else:
            poc = df['close'].iloc[-1]
        
        return {
            'poc': float(poc),
            'profile': volume_at_price
        }
