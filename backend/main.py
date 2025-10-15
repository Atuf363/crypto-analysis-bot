"""
FastAPI Main Application
Crypto Analysis Bot - Dr. Shahid Strategy
"""

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Import our modules
from core.binance_client import BinanceClient, get_binance_client
from analysis.indicators import IndicatorCalculator
from analysis.structure_detector import StructureDetector, ZoneDetector, ConfluenceAnalyzer

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Crypto Analysis Bot",
    description="Multi-timeframe technical analysis for crypto trading",
    version="1.0.0"
)

# CORS configuration
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
selected_coins: List[str] = []
scan_interval = int(os.getenv("SCAN_INTERVAL", 300))
is_scanning = False


def load_coins_config() -> Dict:
    """Load coins configuration from JSON file"""
    try:
        with open("backend/config/coins_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default configuration
        return {
            "watchlists": {
                "tier_a": {
                    "coins": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                    "enabled": True
                },
                "tier_b": {
                    "coins": ["LINKUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT"],
                    "enabled": True
                }
            }
        }


def get_enabled_coins() -> List[str]:
    """Get list of enabled coins from config"""
    config = load_coins_config()
    coins = []
    
    for tier, data in config["watchlists"].items():
        if data.get("enabled", True):
            coins.extend(data["coins"])
    
    return list(set(coins))  # Remove duplicates


async def analyze_coin(client: BinanceClient, symbol: str) -> Dict:
    """Perform complete multi-timeframe analysis on a coin"""
    try:
        # Get multi-timeframe data
        timeframes = ['1d', '4h', '1h', '15m']
        data = await client.get_multi_timeframe_data(symbol, timeframes)
        
        if not data or len(data) < 3:
            raise Exception(f"Insufficient data for {symbol}")
        
        # Get 24h ticker
        ticker = await client.get_ticker_24h(symbol)
        
        # Calculate indicators
        for tf in timeframes:
            if tf in data:
                data[tf] = IndicatorCalculator.calculate_all_indicators(data[tf])
        
        # Detect structure
        structure_1d = StructureDetector.detect_structure(data['1d']) if '1d' in data else {"type": "NEUTRAL"}
        structure_4h = StructureDetector.detect_structure(data['4h']) if '4h' in data else {"type": "NEUTRAL"}
        structure_1h = StructureDetector.detect_structure(data['1h']) if '1h' in data else {"type": "NEUTRAL"}
        
        # Detect zones
        primary_df = data['4h'] if '4h' in data else data['1h']
        order_blocks = ZoneDetector.detect_order_blocks(primary_df)
        fvgs = ZoneDetector.detect_fvg(primary_df)
        supply_demand = ZoneDetector.detect_supply_demand(primary_df)
        sweeps = ZoneDetector.detect_liquidity_sweeps(data['1h'] if '1h' in data else primary_df)
        
        # Combine zones
        all_zones = order_blocks + fvgs + supply_demand
        
        # Get indicators
        indicators = IndicatorCalculator.get_current_values(data['1h'] if '1h' in data else primary_df)
        
        # Calculate confluence
        confluence_score, confluence_factors, trend = ConfluenceAnalyzer.calculate_confluence_score(
            structure_1d,
            structure_4h,
            structure_1h,
            all_zones,
            indicators,
            sweeps
        )
        
        # Determine signal
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
                risk_reward = (entry_price - targets[0]) / (stop_loss - entry_price)
        
        funding_rate = await client.get_funding_rate(symbol)
        volume_profile = IndicatorCalculator.calculate_volume_profile(primary_df)
        
        analysis = {
            "symbol": symbol,
            "trend": trend,
            "strength": round(confluence_score, 1),
            "structure_1d": structure_1d['pattern'],
            "structure_4h": structure_4h['pattern'],
            "structure_1h": structure_1h['pattern'],
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
            "confluence_score": confluence_score,
            "current_price": float(ticker['lastPrice']),
            "volume_24h": float(ticker['quoteVolume']),
            "funding_rate": funding_rate,
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
    """Scan all enabled coins periodically"""
    global is_scanning, analysis_cache
    
    # Using public Binance API - no credentials needed
async with BinanceClient("", "") as client:
        while is_scanning:
            try:
                coins = get_enabled_coins()
                print(f"Scanning {len(coins)} coins...")
                
                for symbol in coins:
                    try:
                        analysis = await analyze_coin(client, symbol)
                        analysis_cache[symbol] = analysis
                        await broadcast_update(analysis)
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        print(f"Error scanning {symbol}: {e}")
                
                print(f"Scan complete. Next scan in {scan_interval}s")
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                print(f"Scan error: {e}")
                await asyncio.sleep(60)


async def broadcast_update(data: Dict):
    """Broadcast update to all WebSocket clients"""
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
    print("🚀 Bot started - scanning coins...")


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
        "coins_monitored": len(get_enabled_coins()),
        "active_connections": len(active_connections)
    }


@app.get("/api/coins")
async def get_coins():
    config = load_coins_config()
    return {
        "watchlists": config["watchlists"],
        "total_coins": len(get_enabled_coins())
    }


@app.get("/api/analysis")
async def get_all_analysis():
    if not analysis_cache:
        return {"message": "No analysis data yet", "coins": []}
    
    return {
        "coins": list(analysis_cache.values()),
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/analysis/{symbol}")
async def get_coin_analysis(symbol: str):
    symbol = symbol.upper()
    if symbol in analysis_cache:
        return analysis_cache[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"Analysis for {symbol} not found")


@app.get("/api/signals")
async def get_signals():
    signals = [
        analysis for analysis in analysis_cache.values()
        if analysis.get('has_signal', False)
    ]
    signals.sort(key=lambda x: x.get('confluence_score', 0), reverse=True)
    return {
        "signals": signals,
        "count": len(signals)
    }


@app.post("/api/coins/select")
async def select_coins(coins: List[str]):
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    async with BinanceClient(api_key, api_secret) as client:
        try:
            exchange_info = await client.get_exchange_info()
            valid_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            
            invalid_coins = [c for c in coins if c not in valid_symbols]
            if invalid_coins:
                raise HTTPException(status_code=400, detail=f"Invalid coins: {invalid_coins}")
            
            config = load_coins_config()
            config['watchlists']['custom'] = {
                "coins": coins,
                "enabled": True
            }
            
            with open("backend/config/coins_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return {"message": f"Updated watchlist with {len(coins)} coins", "coins": coins}
            
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
                "data": list(analysis_cache.values())
            }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

