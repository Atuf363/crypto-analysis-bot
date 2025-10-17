"""
FastAPI Main Application - Crypto Analysis Bot
Real Binance FREE API (No API Key Required)
"""

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our real modules
from core.binance_client import BinanceClient
from ..analysis.indicators import IndicatorCalculator
from analysis.structure_detector import StructureDetector, ZoneDetector, ConfluenceAnalyzer

load_dotenv()

app = FastAPI(
    title="Crypto Analysis Bot",
    description="Multi-timeframe technical analysis for crypto trading",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_connections: List[WebSocket] = []
analysis_cache: Dict[str, Dict] = {}
scan_interval = int(os.getenv("SCAN_INTERVAL", 300))
is_scanning = False


def load_coins_config() -> Dict:
    """Load coins configuration"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config", "coins_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Config load error: {e}")
    
    return {
        "watchlists": {
            "tier_a": {
                "name": "Tier A - Primary",
                "coins": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                "enabled": True
            },
            "tier_b": {
                "name": "Tier B - Secondary",
                "coins": ["LINKUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT"],
                "enabled": True
            }
        }
    }


def get_enabled_coins() -> List[str]:
    """Get list of enabled coins"""
    config = load_coins_config()
    coins = []
    
    for tier, data in config["watchlists"].items():
        if data.get("enabled", True):
            coins.extend(data["coins"])
    
    return list(set(coins))


async def analyze_coin(client: BinanceClient, symbol: str) -> Dict:
    """Perform complete multi-timeframe analysis on a coin using REAL data"""
    try:
        # Get real OHLCV data from Binance FREE API
        timeframes = ['1d', '4h', '1h']
        data = await client.get_multi_timeframe_data(symbol, timeframes)
        
        if not data or len(data) < 2:
            raise Exception(f"Insufficient data for {symbol}")
        
        # Get real ticker info
        ticker = await client.get_ticker_24h(symbol)
        if not ticker:
            raise Exception(f"No ticker data for {symbol}")
        
        # Calculate indicators for each timeframe
        for tf in timeframes:
            if tf in data:
                data[tf] = IndicatorCalculator.calculate_all_indicators(data[tf])
        
        # Detect structure on each timeframe
        structure_1d = StructureDetector.detect_structure(data['1d']) if '1d' in data else {"type": "NEUTRAL", "pattern": "RANGING"}
        structure_4h = StructureDetector.detect_structure(data['4h']) if '4h' in data else {"type": "NEUTRAL", "pattern": "RANGING"}
        structure_1h = StructureDetector.detect_structure(data['1h']) if '1h' in data else {"type": "NEUTRAL", "pattern": "RANGING"}
        
        # Detect zones
        primary_df = data['4h'] if '4h' in data else data['1h']
        order_blocks = ZoneDetector.detect_order_blocks(primary_df)
        fvgs = ZoneDetector.detect_fvg(primary_df)
        supply_demand = ZoneDetector.detect_supply_demand(primary_df)
        sweeps = ZoneDetector.detect_liquidity_sweeps(data['1h'] if '1h' in data else primary_df)
        
        # Combine all zones
        all_zones = order_blocks + fvgs + supply_demand
        
        # Get current indicator values
        indicators = IndicatorCalculator.get_current_values(data['1h'] if '1h' in data else primary_df)
        
        # Calculate confluence score
        confluence_score, confluence_factors, trend = ConfluenceAnalyzer.calculate_confluence_score(
            structure_1d,
            structure_4h,
            structure_1h,
            all_zones,
            indicators,
            sweeps
        )
        
        # Generate trading signal if confluence is high
        has_signal = confluence_score >= 70
        signal_direction = None
        entry_price = None
        stop_loss = None
        targets = []
        risk_reward = None
        
        if has_signal:
            current_price = float(ticker['lastPrice'])
            atr = indicators.get('atr', current_price * 0.02)
            
            if trend == "BULLISH":
                signal_direction = "LONG"
                entry_price = current_price
                stop_loss = current_price - (1.5 * atr)
                targets = [
                    current_price + (2 * atr),
                    current_price + (3 * atr),
                    current_price + (4 * atr)
                ]
                if stop_loss > 0:
                    risk_reward = (targets[0] - entry_price) / (entry_price - stop_loss)
            
            elif trend == "BEARISH":
                signal_direction = "SHORT"
                entry_price = current_price
                stop_loss = current_price + (1.5 * atr)
                targets = [
                    current_price - (2 * atr),
                    current_price - (3 * atr),
                    current_price - (4 * atr)
                ]
                if stop_loss > entry_price:
                    risk_reward = (entry_price - targets[0]) / (stop_loss - entry_price)
        
        # Get volume profile
        volume_profile = IndicatorCalculator.calculate_volume_profile(primary_df)
        
        # Build analysis result
        analysis = {
            "symbol": symbol,
            "trend": trend,
            "strength": round(confluence_score, 1),
            "structure_1d": structure_1d.get('pattern', 'RANGING'),
            "structure_4h": structure_4h.get('pattern', 'RANGING'),
            "structure_1h": structure_1h.get('pattern', 'RANGING'),
            "zones": all_zones,
            "nearest_demand": min([z['bottom'] for z in all_zones if z['type'] in ['DEMAND', 'ORDER_BLOCK_BULLISH']], default=None),
            "nearest_supply": max([z['top'] for z in all_zones if z['type'] in ['SUPPLY', 'ORDER_BLOCK_BEARISH']], default=None),
            "indicators": indicators,
            "has_signal": has_signal,
            "signal_direction": signal_direction,
            "entry_price": round(entry_price, 8) if entry_price else None,
            "stop_loss": round(stop_loss, 8) if stop_loss else None,
            "targets": [round(t, 8) for t in targets] if targets else None,
            "risk_reward": round(risk_reward, 2) if risk_reward else None,
            "confluence_factors": confluence_factors,
            "confluence_score": int(confluence_score),
            "current_price": float(ticker['lastPrice']),
            "volume_24h": float(ticker.get('quoteVolume', 0)),
            "price_change_24h": float(ticker.get('priceChangePercent', 0)),
            "sweeps": sweeps,
            "poc": round(volume_profile['poc'], 8),
            "last_updated": datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "trend": "NEUTRAL",
            "strength": 0,
            "has_signal": False,
            "last_updated": datetime.now().isoformat()
        }


async def scan_all_coins():
    """Scan all enabled coins periodically using real Binance data"""
    global is_scanning, analysis_cache
    
    while is_scanning:
        try:
            coins = get_enabled_coins()
            print(f"[REAL DATA] Scanning {len(coins)} coins from Binance...")
            
            async with BinanceClient() as client:
                for symbol in coins:
                    try:
                        analysis = await analyze_coin(client, symbol)
                        analysis_cache[symbol] = analysis
                        await broadcast_update(analysis)
                        await asyncio.sleep(0.3)  # Rate limiting
                    except Exception as e:
                        print(f"Error scanning {symbol}: {e}")
                        await asyncio.sleep(0.5)
            
            print(f"[REAL DATA] Scan complete. Next scan in {scan_interval}s")
            await asyncio.sleep(scan_interval)
            
        except Exception as e:
            print(f"Scan error: {e}")
            await asyncio.sleep(60)


async def broadcast_update(data: Dict):
    """Broadcast update to WebSocket clients"""
    if active_connections:
        message = json.dumps(data)
        disconnected = []
        
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            active_connections.remove(conn)


@app.on_event("startup")
async def startup_event():
    global is_scanning
    is_scanning = True
    asyncio.create_task(scan_all_coins())
    print("🚀 Bot started - using REAL Binance FREE API (no key required)...")


@app.on_event("shutdown")
async def shutdown_event():
    global is_scanning
    is_scanning = False
    print("🛑 Bot stopped")


@app.get("/")
async def root():
    return {
        "status": "online",
        "name": "Crypto Analysis Bot",
        "version": "1.0.0",
        "mode": "REAL_BINANCE_FREE_API",
        "coins_monitored": len(get_enabled_coins()),
        "active_connections": len(active_connections),
        "cached_analyses": len(analysis_cache)
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "mode": "REAL_BINANCE_FREE_API",
        "coins_cached": len(analysis_cache)
    }


@app.get("/api/coins")
async def get_coins():
    config = load_coins_config()
    return {
        "watchlists": config["watchlists"],
        "total_coins": len(get_enabled_coins()),
        "mode": "REAL_BINANCE_FREE_API"
    }


@app.get("/api/analysis")
async def get_all_analysis():
    if not analysis_cache:
        return {"message": "Analyzing coins... check back soon", "coins": [], "mode": "REAL_BINANCE_FREE_API"}
    
    return {
        "coins": list(analysis_cache.values()),
        "total": len(analysis_cache),
        "last_updated": datetime.now().isoformat(),
        "mode": "REAL_BINANCE_FREE_API"
    }


@app.get("/api/analysis/{symbol}")
async def get_coin_analysis(symbol: str):
    symbol = symbol.upper()
    if symbol in analysis_cache:
        return analysis_cache[symbol]
    else:
        available = list(analysis_cache.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Analysis for {symbol} not found. Available: {available[:10]}"
        )


@app.get("/api/signals")
async def get_signals():
    signals = [
        analysis for analysis in analysis_cache.values()
        if analysis.get('has_signal', False) and 'error' not in analysis
    ]
    signals.sort(key=lambda x: x.get('confluence_score', 0), reverse=True)
    return {
        "signals": signals,
        "count": len(signals),
        "mode": "REAL_BINANCE_FREE_API"
    }


@app.post("/api/coins/select")
async def select_coins(coins: List[str]):
    try:
        config = load_coins_config()
        config['watchlists']['custom'] = {
            "name": "Custom Watchlist",
            "coins": coins,
            "enabled": True
        }
        
        config_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, "coins_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return {
            "message": f"Updated watchlist with {len(coins)} coins",
            "coins": coins,
            "mode": "REAL_BINANCE_FREE_API"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        if analysis_cache:
            await websocket.send_text(json.dumps({
                "type": "initial",
                "data": list(analysis_cache.values()),
                "mode": "REAL_BINANCE_FREE_API"
            }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
