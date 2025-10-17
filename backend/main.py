from fastapi import FastAPI
import os

app = FastAPI(title="Crypto Analysis Bot", version="1.0.0")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Bot is running"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.get("/api/coins")
def get_coins():
    return {"coins": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]}

@app.get("/api/routes")
def get_routes():
    routes = []
    for route in app.routes:
        routes.append(str(route.path))
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting bot on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
