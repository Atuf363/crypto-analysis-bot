"""
Structure Detection & Zone Identification
High/Low pattern detection, Order Blocks, FVG, Supply/Demand
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class StructureDetector:
    """Detect market structure (HH/HL/LH/LL)"""
    
    @staticmethod
    def detect_structure(df: pd.DataFrame) -> Dict:
        """
        Detect current market structure
        Returns: HH_HL (Uptrend), LH_LL (Downtrend), RANGING (Sideways)
        """
        if len(df) < 4:
            return {"type": "NEUTRAL", "pattern": "RANGING"}
        
        # Get last 3 significant swing points
        closes = df['close'].tail(20).values
        
        # Simplified detection
        if len(closes) < 3:
            return {"type": "NEUTRAL", "pattern": "RANGING"}
        
        # Compare recent highs and lows
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        prior_high = df['high'].tail(20).head(10).max()
        prior_low = df['low'].tail(20).head(10).min()
        
        current_price = df['close'].iloc[-1]
        
        # Uptrend: Higher Highs and Higher Lows
        if recent_high > prior_high and recent_low > prior_low:
            pattern = "HH_HL"
            trend_type = "BULLISH"
        
        # Downtrend: Lower Highs and Lower Lows
        elif recent_high < prior_high and recent_low < prior_low:
            pattern = "LH_LL"
            trend_type = "BEARISH"
        
        # Ranging: No clear direction
        else:
            pattern = "RANGING"
            trend_type = "NEUTRAL"
        
        return {
            "type": trend_type,
            "pattern": pattern,
            "recent_high": float(recent_high),
            "recent_low": float(recent_low),
            "prior_high": float(prior_high),
            "prior_low": float(prior_low)
        }
    
    @staticmethod
    def detect_trend_strength(df: pd.DataFrame) -> int:
        """Rate trend strength 0-100"""
        if len(df) < 20:
            return 50
        
        recent = df.tail(20)
        closes = recent['close'].values
        
        # Calculate trend consistency
        ups = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        trend_ratio = (ups / len(closes)) * 100
        
        return int(trend_ratio)


class ZoneDetector:
    """Detect support/resistance zones, order blocks, FVGs"""
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> List[Dict]:
        """
        Detect Order Blocks - strong consolidation areas
        """
        zones = []
        
        if len(df) < 10:
            return zones
        
        # Look for strong reversals (impulse + consolidation)
        for i in range(10, len(df) - 5):
            # Impulse move
            if i >= 10:
                prior_range = df.iloc[i-10:i]
                current_range = df.iloc[i:i+5]
                
                impulse_high = prior_range['high'].max()
                impulse_low = prior_range['low'].min()
                consolidation_high = current_range['high'].max()
                consolidation_low = current_range['low'].min()
                
                # Bullish OB: Impulse up then consolidation
                if (df['close'].iloc[i] > df['close'].iloc[i-10] and
                    consolidation_high > impulse_high * 1.005):
                    
                    zones.append({
                        "type": "ORDER_BLOCK_BULLISH",
                        "top": float(consolidation_high),
                        "bottom": float(consolidation_low),
                        "timeframe": "4h",
                        "strength": "MEDIUM",
                        "created_at": datetime.now().isoformat()
                    })
                
                # Bearish OB: Impulse down then consolidation
                if (df['close'].iloc[i] < df['close'].iloc[i-10] and
                    consolidation_low < impulse_low * 0.995):
                    
                    zones.append({
                        "type": "ORDER_BLOCK_BEARISH",
                        "top": float(consolidation_high),
                        "bottom": float(consolidation_low),
                        "timeframe": "4h",
                        "strength": "MEDIUM",
                        "created_at": datetime.now().isoformat()
                    })
        
        return zones[-3:] if zones else []  # Return last 3
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG)
        Bullish FVG: Low of candle > High of candle 2 bars ago
        """
        zones = []
        
        if len(df) < 3:
            return zones
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            two_bars_ago = df.iloc[i-2]
            
            # Bullish FVG
            if current['low'] > two_bars_ago['high']:
                zones.append({
                    "type": "FVG_BULLISH",
                    "top": float(current['low']),
                    "bottom": float(two_bars_ago['high']),
                    "timeframe": "4h",
                    "strength": "HIGH",
                    "created_at": datetime.now().isoformat()
                })
            
            # Bearish FVG
            elif current['high'] < two_bars_ago['low']:
                zones.append({
                    "type": "FVG_BEARISH",
                    "top": float(two_bars_ago['low']),
                    "bottom": float(current['high']),
                    "timeframe": "4h",
                    "strength": "HIGH",
                    "created_at": datetime.now().isoformat()
                })
        
        return zones[-3:] if zones else []
    
    @staticmethod
    def detect_supply_demand(df: pd.DataFrame) -> List[Dict]:
        """
        Detect Supply (Resistance) and Demand (Support) zones
        """
        zones = []
        
        if len(df) < 20:
            return zones
        
        # Find recent swing highs (supply)
        for i in range(5, len(df) - 5):
            window = df.iloc[i-5:i+6]
            
            if df['high'].iloc[i] == window['high'].max():
                zones.append({
                    "type": "SUPPLY",
                    "top": float(df['high'].iloc[i] * 1.001),
                    "bottom": float(df['high'].iloc[i] * 0.999),
                    "timeframe": "4h",
                    "strength": "MEDIUM",
                    "created_at": datetime.now().isoformat()
                })
            
            # Find recent swing lows (demand)
            if df['low'].iloc[i] == window['low'].min():
                zones.append({
                    "type": "DEMAND",
                    "top": float(df['low'].iloc[i] * 1.001),
                    "bottom": float(df['low'].iloc[i] * 0.999),
                    "timeframe": "4h",
                    "strength": "MEDIUM",
                    "created_at": datetime.now().isoformat()
                })
        
        return zones[-4:] if zones else []
    
    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame) -> List[Dict]:
        """
        Detect liquidity sweep events
        Price breaking below support/above resistance then reversing
        """
        sweeps = []
        
        if len(df) < 10:
            return sweeps
        
        recent = df.tail(20)
        
        # Recent swing low
        swing_low = recent['low'].min()
        swing_low_idx = recent['low'].idxmin()
        
        # Check if price broke below then reversed
        lowest_point = df['low'].tail(5).min()
        if lowest_point < swing_low * 0.998:
            sweeps.append({
                "type": "SWEEP_BELOW_SUPPORT",
                "price": float(lowest_point),
                "description": "Liquidity sweep below support"
            })
        
        # Recent swing high
        swing_high = recent['high'].max()
        highest_point = df['high'].tail(5).max()
        if highest_point > swing_high * 1.002:
            sweeps.append({
                "type": "SWEEP_ABOVE_RESISTANCE",
                "price": float(highest_point),
                "description": "Liquidity sweep above resistance"
            })
        
        return sweeps


class ConfluenceAnalyzer:
    """Analyze confluence of multiple factors for signals"""
    
    @staticmethod
    def calculate_confluence_score(
        structure_1d: Dict,
        structure_4h: Dict,
        structure_1h: Dict,
        zones: List[Dict],
        indicators: Dict,
        sweeps: List[Dict]
    ) -> tuple:
        """
        Calculate overall confluence score (0-100)
        Returns: (score, factors, trend)
        """
        
        score = 0
        factors = []
        
        # Structure alignment (25 points max)
        structure_alignment = 0
        if structure_1d['type'] == structure_4h['type'] == structure_1h['type']:
            structure_alignment = 25
            factors.append({
                "name": "Structure Alignment",
                "weight": 25,
                "met": True,
                "description": "All timeframes aligned"
            })
        elif structure_1d['type'] == structure_4h['type']:
            structure_alignment = 15
            factors.append({
                "name": "Structure Alignment",
                "weight": 15,
                "met": True,
                "description": "Daily and 4H aligned"
            })
        else:
            factors.append({
                "name": "Structure Alignment",
                "weight": 0,
                "met": False,
                "description": "Misaligned structures"
            })
        
        score += structure_alignment
        
        # Zone confluence (20 points max)
        if zones:
            zone_score = min(20, len(zones) * 5)
            score += zone_score
            factors.append({
                "name": "Zone Confluence",
                "weight": zone_score,
                "met": True,
                "description": f"{len(zones)} zones identified"
            })
        else:
            factors.append({
                "name": "Zone Confluence",
                "weight": 0,
                "met": False,
                "description": "No zones found"
            })
        
        # Indicator confluence (25 points max)
        indicator_score = 0
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        ema_aligned = False
        
        # RSI extreme levels
        if rsi > 70 and structure_1h['type'] == 'BEARISH':
            indicator_score += 8
        elif rsi < 30 and structure_1h['type'] == 'BULLISH':
            indicator_score += 8
        
        # MACD positive histogram
        if macd_hist > 0 and structure_1h['type'] == 'BULLISH':
            indicator_score += 8
        elif macd_hist < 0 and structure_1h['type'] == 'BEARISH':
            indicator_score += 8
        
        # EMA alignment
        price = indicators.get('price', 0)
        ema_50 = indicators.get('ema_50', price)
        ema_200 = indicators.get('ema_200', price)
        
        if price > ema_50 > ema_200 and structure_1h['type'] == 'BULLISH':
            indicator_score += 9
            ema_aligned = True
        elif price < ema_50 < ema_200 and structure_1h['type'] == 'BEARISH':
            indicator_score += 9
            ema_aligned = True
        
        score += indicator_score
        factors.append({
            "name": "Indicator Confluence",
            "weight": indicator_score,
            "met": indicator_score > 10,
            "description": f"RSI: {rsi:.0f}, MACD: {macd_hist:.2f}, EMA: {ema_aligned}"
        })
        
        # Liquidity sweep (15 points max)
        if sweeps:
            sweep_score = min(15, len(sweeps) * 7)
            score += sweep_score
            factors.append({
                "name": "Liquidity Sweep",
                "weight": sweep_score,
                "met": True,
                "description": f"{len(sweeps)} sweep(s) detected"
            })
        else:
            factors.append({
                "name": "Liquidity Sweep",
                "weight": 0,
                "met": False,
                "description": "No sweeps"
            })
        
        # Trend determination
        trend = "BULLISH" if structure_1h['type'] == "BULLISH" else \
                "BEARISH" if structure_1h['type'] == "BEARISH" else "NEUTRAL"
        
        # Cap score at 100
        final_score = min(100, score)
        
        return final_score, factors, trend
