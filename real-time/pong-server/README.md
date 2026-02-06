# Pong Game Server - Authoritative MQTT Server

This is the authoritative game server for MQTT Pong. It runs physics simulations and publishes game state to all connected clients over MQTT.

## Architecture

```
Client 1                Server (Authority)              Client 2
├─ Send paddle    →    ├─ Receive inputs           ←   ├─ Send paddle
├─ Receive state  ←    ├─ Run physics (60fps)      →   ├─ Receive state
└─ Render              ├─ Publish ball/state          └─ Render
                       └─ Handle scoring
                            │
                       MQTT Broker
```

## Why Authoritative Server?

**Before (P2P):** Player 1 ran physics → jittery, can cheat, inconsistent
**After (Server):** Server runs physics → smooth, fair, single source of truth

This matches real IoT architecture:
- **Raptor:** Server processes sensor data, sends commands
- **Pong:** Server processes paddle positions, sends ball state

## Features

- ✅ Authoritative physics simulation
- ✅ Multiple concurrent games (game IDs)
- ✅ Automatic game cleanup
- ✅ Reconnection handling
- ✅ Runs as systemd service
- ✅ Detailed logging

## Deployment

### Quick Deploy to Raptor Server

```bash
./deploy.sh
```

This will:
1. Build the TypeScript code
2. Upload to server
3. Install Node.js (if needed)
4. Set up systemd service
5. Start the server

### Manual Deployment

```bash
# Build
npm run build

# Copy to server
scp -r dist package.json server:/home/ec2-user/pong-server/

# On server
npm install --production
node dist/index.js
```

## Local Development

```bash
# Install dependencies
npm install

# Run in dev mode (with auto-reload)
npm run dev

# Build for production
npm run build

# Run built version
npm start
```

## Environment Variables

- `MQTT_BROKER` - MQTT broker URL (default: `mqtt://localhost:1883`)
- `MQTT_USER` - MQTT username (default: `raptor`)
- `MQTT_PASS` - MQTT password (default: `raptorMQTT2025`)

## MQTT Topics

### Subscribe (Input from clients):
- `pong/game/{gameId}/p1/paddle` - Player 1 paddle position
- `pong/game/{gameId}/p2/paddle` - Player 2 paddle position
- `pong/game/{gameId}/join` - Player join notifications

### Publish (Output to clients):
- `pong/game/{gameId}/ball` - Authoritative ball position (30/sec)
- `pong/game/{gameId}/state` - Game state (scores, status)

## Server Management

### Check Status
```bash
ssh -i ~/raptor-server/raptor.pem ec2-user@3.141.116.27 'sudo systemctl status pong-server'
```

### View Logs
```bash
ssh -i ~/raptor-server/raptor.pem ec2-user@3.141.116.27 'sudo journalctl -u pong-server -f'
```

### Restart
```bash
ssh -i ~/raptor-server/raptor.pem ec2-user@3.141.116.27 'sudo systemctl restart pong-server'
```

### Stop
```bash
ssh -i ~/raptor-server/raptor.pem ec2-user@3.141.116.27 'sudo systemctl stop pong-server'
```

## Game Lifecycle

1. **Game Creation:** Server detects first paddle message for a game ID
2. **Game Start:** Server receives join message → starts physics loop
3. **Game Play:** 60fps physics, 30fps ball updates published
4. **Game End:** First to 5 points → publishes end state
5. **Cleanup:** Game removed from memory after 30 seconds

## Why This Matters for IoT

This server architecture demonstrates:

1. **Authoritative Control:** Just like Raptor server controls motors
2. **State Management:** Single source of truth for distributed clients
3. **Real-Time Publishing:** MQTT broadcasts state to all subscribers
4. **Input Aggregation:** Multiple clients send data, server decides action
5. **Scalability:** Can run multiple games concurrently

**This is exactly how industrial IoT systems work!**

## Technical Details

- **Language:** TypeScript (Node.js runtime)
- **MQTT Client:** mqtt.js v5
- **Physics Rate:** 60 FPS
- **Publish Rate:** 30 FPS (ball), on-change (state)
- **QoS:** 0 (at-most-once) - appropriate for real-time game data

## Performance

- **CPU:** ~1-2% per active game
- **Memory:** ~50MB base + ~1MB per active game
- **Network:** ~1KB/sec per game (30 ball updates/sec @ ~30 bytes each)

Multiple games can run concurrently without performance degradation.
