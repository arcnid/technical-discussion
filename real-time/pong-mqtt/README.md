# MQTT Pong - Real-Time Multiplayer Demo

A terminal-based multiplayer Pong game that uses MQTT for real-time communication. This demo shows how IoT devices communicate using pub/sub messaging.

## Quick Start

### Prerequisites

- **Node.js 18+** - [Download](https://nodejs.org/)
- **MQTT Broker** - By default, uses the Raptor cloud broker (no setup needed!)
  - For local testing: Add `--local` flag or start mosquitto locally

### Installation

```bash
# From the pong-mqtt directory
npm install
```

### Running the Game

You need **two terminal windows** - one for each player.

#### Terminal 1 (Player 1 - Host):
```bash
npm run dev -- --player 1 --game abc123
```

#### Terminal 2 (Player 2):
```bash
npm run dev -- --player 2 --game abc123
```

**Important:** Both players must use the **same game ID** (`abc123` in this example).

**Note:** The game uses the Raptor cloud MQTT broker by default, so it works from anywhere! For local testing, add the `--local` flag.

### Controls

- **Arrow Up** - Move paddle up
- **Arrow Down** - Move paddle down
- **Q** - Quit game

## How It Works

### MQTT Topics

The game uses the following MQTT topic structure:

```
pong/game/{gameId}/
  ├── p1/paddle    - Player 1 paddle position
  ├── p2/paddle    - Player 2 paddle position
  ├── ball         - Ball position (published by host)
  ├── state        - Game state (scores, status)
  └── join         - Player join notifications
```

### Message Flow

```
Player 1 (Host)                          Player 2
     │                                       │
     ├─── Publish: p1/paddle (on keypress) ─┤
     │                                       │
     ├─── Publish: ball (60fps game loop) ──┤
     │                                       │
     ├─── Publish: state (on score) ────────┤
     │                                       │
     └─── Subscribe: p2/paddle ──────────────┘
          Subscribe: all game topics ────────┘

              [All messages route through MQTT Broker]
```

### Game Logic

- **Player 1 is the host** - runs game physics and ball simulation
- **Player 2 is a client** - renders game state from MQTT messages
- Each player publishes their paddle position on keypress
- Host publishes ball position 60 times per second
- Both players subscribe to all game messages
- First to 5 points wins

## Watch the Messages!

Want to see the MQTT magic? Open a **third terminal** and subscribe to all game messages:

**Cloud broker (default):**
```bash
mosquitto_sub -h 3.141.116.27 -u raptor -P raptorMQTT2025 -t "pong/game/abc123/#"
```

**Local broker:**
```bash
mosquitto_sub -h localhost -t "pong/game/abc123/#"
```

You'll see every message flying between the players in real-time:

```
pong/game/abc123/p1/paddle {"y":8.5,"timestamp":1234567890}
pong/game/abc123/ball {"x":40,"y":10,"dx":0.5,"dy":0.5,"timestamp":1234567891}
pong/game/abc123/p2/paddle {"y":12.0,"timestamp":1234567892}
```

This is **exactly** how IoT devices communicate!

## Configuration Options

### Cloud vs Local Broker

**Default (Cloud):** Uses the Raptor cloud broker - works from anywhere!
```bash
npm run dev -- --player 1 --game demo
```

**Local Broker:** Connect to localhost instead
```bash
npm run dev -- --player 1 --game demo --local
```

**Custom Broker:** Specify any MQTT broker
```bash
npm run dev -- --player 1 --game demo --broker mqtt://192.168.1.100:1883 --username user --password pass
```

### Different Game ID

```bash
# Player 1
npm run dev -- --player 1 --game my-game

# Player 2
npm run dev -- --player 2 --game my-game
```

You can have multiple games running simultaneously with different game IDs!

## Troubleshooting

### "Error connecting to MQTT broker"

**Problem:** Can't connect to MQTT broker

**Solutions:**
1. Make sure mosquitto is running: `docker ps` or `ps aux | grep mosquitto`
2. Check the port is correct (default: 1883)
3. Try specifying the broker URL: `--broker mqtt://localhost:1883`

### "Waiting for players..." forever

**Problem:** Game stays in waiting state

**Solutions:**
1. Make sure both players use the **same game ID**
2. Check that both players are connected to the **same broker**
3. Try restarting both players

### Paddle doesn't move

**Problem:** Arrow keys don't work

**Solutions:**
1. Make sure the terminal window is focused
2. Some terminals may need special key handling
3. Try a different terminal (iTerm2, Terminal.app, etc.)

## Code Structure

```
src/
├── index.tsx           - Entry point, CLI arg parsing
├── game/
│   ├── Game.tsx        - Main game component (Ink UI)
│   └── types.ts        - TypeScript interfaces
└── mqtt/
    ├── client.ts       - MQTT connection wrapper
    └── topics.ts       - Topic name helpers
```

## The IoT Connection

This game demonstrates the **same patterns** used in real IoT systems:

| Pong Game | Industrial IoT |
|-----------|----------------|
| Paddle position messages | Sensor readings |
| Ball position updates | Motor speed telemetry |
| Score updates | Alert notifications |
| QoS 0 (fire-and-forget) | Temperature readings |
| Player topics (p1, p2) | Device topics (device-1, device-2) |
| Game state synchronization | Equipment state monitoring |

**The protocol is identical.** The only difference is what you're measuring and controlling.

## Learning Exercises

### 1. Watch the Message Rate

Monitor how many messages flow in 1 second:

```bash
mosquitto_sub -h localhost -t "pong/game/abc123/#" | pv -l -i 1 > /dev/null
```

You'll see ~60 ball updates per second from the host!

### 2. Add a Third "Spectator"

Create a third terminal that subscribes to all game topics but doesn't play. This shows how MQTT's pub/sub allows unlimited subscribers without affecting the publishers.

### 3. Simulate Network Issues

Use QoS 1 for paddle messages to see guaranteed delivery:

Edit `src/mqtt/client.ts` and change `qos: 0` to `qos: 1` in the subscribe call.

### 4. Message Inspection

Parse and pretty-print game messages:

```bash
mosquitto_sub -h localhost -t "pong/game/#" | jq .
```

(Requires `jq` for JSON formatting)

## Next Steps

Now that you've seen MQTT in action:

1. Read [MQTT Basics](../docs/mqtt-basics.md) for deeper protocol understanding
2. Explore [Real-Time Concepts](../docs/concepts.md) to compare different approaches
3. Check out the Raptor system code to see MQTT controlling real hardware
4. Try modifying the game:
   - Add a third player
   - Change QoS levels
   - Add retained messages for game state
   - Implement Last Will and Testament for disconnect detection

## Production Considerations

This demo uses QoS 0 for simplicity. In production systems:

- **Sensor data** (high frequency): QoS 0
- **Commands** (must arrive): QoS 1
- **Critical operations** (emergency stop): QoS 2
- **Status messages**: Use retained messages
- **Connection monitoring**: Implement Last Will and Testament

See the Raptor gateway code for production examples!

## Questions?

This demo is meant to spark curiosity. If something doesn't make sense or you want to explore deeper, check out:

- [Parent README](../README.md) - Overview and motivation
- [MQTT Basics](../docs/mqtt-basics.md) - Protocol deep dive
- [Real-Time Concepts](../docs/concepts.md) - When to use what

---

**Have fun!** You're now seeing the same technology that powers billions of IoT devices worldwide.
