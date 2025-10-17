"""
FastAPI Main Application - Crypto Analysis Bot
Simplified version with mock data for testing
"""
from core.binance_client import BinanceClient
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
import random

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


def generate_mock_analysis(symbol: str) -> Dict:
    """Generate realistic mock analysis data"""
    base_price = random.uniform(100, 50000)
    trend = random.choice(["BULLISH", "BEARISH", "NEUTRAL"])
    confidence = random.randint(40, 95)
    
    atr = base_price * random.uniform(0.01, 0.05)
    
    analysis = {
        "symbol": symbol,
        "trend": trend,
        "strength": confidence,
        "structure_1d": random.choice(["HH_HL", "LH_LL", "RANGING"]),
        "structure_4h": random.choice(["HH_HL", "LH_LL", "RANGING"]),
        "structure_1h": random.choice(["HH_HL", "LH_LL", "RANGING"]),
        "zones": [
            {
                "type": random.choice(["DEMAND", "SUPPLY", "ORDER_BLOCK"]),
                "top": base_price * 1.02,
                "bottom": base_price * 0.98,
                "timeframe": "4h",
                "strength": random.choice(["HIGH", "MEDIUM", "LOW"]),
                "created_at": datetime.now().isoformat()
            }
        ],
        "nearest_demand": base_price * 0.95,
        "nearest_supply": base_price * 1.05,
        "indicators": {
            "price": base_price,
            "ema_50": base_price * random.uniform(0.98, 1.02),
            "ema_200": base_price * random.uniform(0.95, 1.05),
            "rsi": random.randint(20, 80),
            "macd": random.uniform(-100, 100),
            "macd_signal": random.uniform(-100, 100),
            "obv": random.uniform(1000000, 10000000),
            "obv_ema": random.uniform(1000000, 10000000),
            "atr": atr
        },
        "has_signal": confidence >= 70,
        "signal_direction": trend if confidence >= 70 else None,
        "entry_price": base_price if confidence >= 70 else None,
        "stop_loss": (base_price - atr * 1.5) if confidence >= 70 and trend == "BULLISH" else (base_price + atr * 1.5) if confidence >= 70 else None,
        "targets": [
            base_price + (atr * 2),
            base_price + (atr * 3),
            base_price + (atr * 4)
        ] if trend == "BULLISH" and confidence >= 70 else [
            base_price - (atr * 2),
            base_price - (atr * 3),
            base_price - (atr * 4)
        ] if trend == "BEARISH" and confidence >= 70 else None,
        "risk_reward": round(random.uniform(1.5, 3.0), 2) if confidence >= 70 else None,
        "confluence_factors": [
            {"name": "EMA Alignment", "weight": 20, "met": random.choice([True, False]), "description": "Price above EMA 200"},
            {"name": "RSI Divergence", "weight": 15, "met": random.choice([True, False]), "description": "RSI shows divergence"},
            {"name": "Zone Confluence", "weight": 25, "met": random.choice([True, False]), "description": "Multiple zones align"},
            {"name": "Order Block Break", "weight": 20, "met": random.choice([True, False]), "description": "Recent OB break"},
            {"name": "Liquidity Sweep", "weight": 20, "met": random.choice([True, False]), "description": "Recent sweep detected"},
        ],
        "confluence_score": confidence,
        "current_price": base_price,
        "volume_24h": random.uniform(10000000, 500000000),
        "funding_rate": random.uniform(-0.001, 0.001),
        "poc": base_price * random.uniform(0.99, 1.01),
        "last_updated": datetime.now().isoformat()
    }
    
    return analysis


async def scan_all_coins():
    """Scan all enabled coins with mock data"""
    global is_scanning, analysis_cache
    
    while is_scanning:
        try:
            coins = get_enabled_coins()
            print(f"[MOCK] Scanning {len(coins)} coins...")
            
            for symbol in coins:
                analysis = generate_mock_analysis(symbol)
                analysis_cache[symbol] = analysis
                await broadcast_update(analysis)
                await asyncio.sleep(0.2)
            
            print(f"[MOCK] Scan complete. Next scan in {scan_interval}s")
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
    print("🚀 Bot started - scanning coins (MOCK DATA)...")


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
        "mode": "MOCK_DATA",
        "coins_monitored": len(get_enabled_coins()),
        "active_connections": len(active_connections)
    }


@app.get("/api/health")
async def health():
    return {"status": "healthy", "mode": "MOCK_DATA"}


@app.get("/api/coins")
async def get_coins():
    config = load_coins_config()
    return {
        "watchlists": config["watchlists"],
        "total_coins": len(get_enabled_coins()),
        "mode": "MOCK_DATA"
    }


@app.get("/api/analysis")
async def get_all_analysis():
    if not analysis_cache:
        return {"message": "No analysis data yet", "coins": [], "mode": "MOCK_DATA"}
    
    return {
        "coins": list(analysis_cache.values()),
        "total": len(analysis_cache),
        "last_updated": datetime.now().isoformat(),
        "mode": "MOCK_DATA"
    }


@app.get("/api/analysis/{symbol}")
async def get_coin_analysis(symbol: str):
    symbol = symbol.upper()
    if symbol in analysis_cache:
        return analysis_cache[symbol]
    else:
        raise HTTPException(status_code=404, detail=f"Analysis for {symbol} not found. Try: {list(analysis_cache.keys())}")


@app.get("/api/signals")
async def get_signals():
    signals = [
        analysis for analysis in analysis_cache.values()
        if analysis.get('has_signal', False)
    ]
    signals.sort(key=lambda x: x.get('confluence_score', 0), reverse=True)
    return {
        "signals": signals,
        "count": len(signals),
        "mode": "MOCK_DATA"
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
            "mode": "MOCK_DATA"
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
                "mode": "MOCK_DATA"
            }))
        
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

