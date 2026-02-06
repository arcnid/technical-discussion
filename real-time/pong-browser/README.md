# MQTT Pong - Browser Edition

A real-time multiplayer Pong game demonstrating MQTT communication over WebSockets.

**🌐 Works from ANYWHERE** - Connects to public Raptor MQTT broker

## Quick Start

### 1. Clone and Run

```bash
git clone https://github.com/arcnid/pong.git
cd pong-browser
npm run dev
```

That's it! The server will automatically find an available port (tries 3000-3010) and show you the URLs.

### 2. Play Locally (Testing)

**Terminal 1 - Player 1:**
```bash
npm run p1
```

**Terminal 2 - Player 2:**
```bash
npm run p2
```

OR manually open:
- Player 1: http://localhost:3000?player=1
- Player 2: http://localhost:3000?player=2

### 3. Play with a Friend (Different Computers)

**You:**
```bash
npm run dev
# Open: http://localhost:3000?player=1
```

**Your Friend** (on their computer):
```bash
git clone https://github.com/arcnid/pong.git
cd pong-browser
npm run dev
# Open: http://localhost:3000?player=2
```

**Both enter the SAME Game ID** (e.g., "my-game") and click "Connect & Play"!

## How It Works

- **Broker:** Raptor public server (3.141.116.27:9001)
- **Protocol:** MQTT over WebSocket
- **Source of Truth:** Game server running on Raptor
- **Clients:** Browser connects from anywhere

Both players connect to the same public broker, so it works across the internet!

## Controls

- **Arrow Up / W** - Move paddle up
- **Arrow Down / S** - Move paddle down
- **SPACEBAR** - Serve the ball
- **R** - Restart game (when ended)

## Connection

The game connects to the Raptor MQTT broker at:
- **Broker:** 3.141.116.27
- **WebSocket Port:** 9001 (standard Mosquitto)
- **Protocol:** MQTT over WebSocket

## Troubleshooting

### "Connection failed" error?

The broker needs WebSocket support. If port 9001 doesn't work, try:

1. **Check if WebSocket is enabled on broker**
2. **Try alternative ports** (edit index.html, line 228):
   - Port 8083
   - Port 8080
   - Port 80

3. **Run local Mosquitto with WebSocket:**
```bash
# Install Mosquitto
brew install mosquitto

# Start with WebSocket support
mosquitto -p 1883 -c /opt/homebrew/etc/mosquitto/mosquitto.conf
```

Add to `/opt/homebrew/etc/mosquitto/mosquitto.conf`:
```
listener 9001
protocol websockets
```

Then update the server to use localhost:
```bash
cd ~/technical-discussion/real-time/pong-server
MQTT_BROKER="mqtt://localhost:1883" npm start
```

And in `index.html`, change line 228 to:
```javascript
const brokerUrl = 'ws://localhost:9001';
```

## Architecture

**Browser Client:**
- Canvas rendering at 60fps
- MQTT over WebSocket
- Instant client-side paddle prediction
- Server-authoritative ball position

**Game Server** (Node.js):
- Physics simulation at 60fps
- MQTT pub/sub
- Handles collisions, scoring, game state

**MQTT Broker:**
- Routes messages between clients and server
- Supports both TCP (server) and WebSocket (browser)

## Features

✅ Smooth 60fps Canvas rendering
✅ Real-time multiplayer over MQTT
✅ Works from anywhere (public broker)
✅ Ball serve mechanic
✅ Client-side prediction (your paddle)
✅ Server-authoritative physics
✅ Clean, professional UI

## Next Steps

- Open in multiple browsers to test
- Share with teammates (same game ID)
- Try from different computers on same network
- Great for demonstrating real-time IoT concepts!
