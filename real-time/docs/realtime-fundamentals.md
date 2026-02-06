# Real-Time Communication Fundamentals

A comprehensive guide to understanding WebSockets, MQTT, and event-based systems through practical examples from our hardware and software projects.

---

## Table of Contents

1. [The Problem with HTTP](#the-problem-with-http)
2. [Real-Time Solutions](#real-time-solutions)
3. [WebSockets: Direct Two-Way Channels](#websockets-direct-two-way-channels)
4. [MQTT: Publish/Subscribe for Hardware](#mqtt-publishsubscribe-for-hardware)
5. [Real-World Examples from Our Projects](#real-world-examples-from-our-projects)
6. [Choosing the Right Protocol](#choosing-the-right-protocol)
7. [Learning Through Pong](#learning-through-pong)

---

## The Problem with HTTP

HTTP is a **request-response** protocol. The client asks, the server answers, then the connection closes. This is perfect for loading web pages, but terrible for real-time communication.

### Traditional HTTP Polling (The Bad Way)

```javascript
// Check for updates every 5 seconds
setInterval(function() {
  fetch('/api/temperature')
    .then(res => res.json())
    .then(data => updateDisplay(data));
}, 5000);
```

**Problems:**
- **Latency**: Up to 5 seconds before you see changes
- **Bandwidth waste**: Sends requests even when nothing changed
- **Doesn't scale**: 100 clients = 1,200 requests/minute
- **Server load**: Constant connection churn

### The Real-Time Approach

Instead of asking "got anything new?", the server **pushes** updates the moment they happen:

```
Traditional HTTP:    Client → "Any updates?" → Server
                     Server → "Nope"          → Client
                     (repeat every 5 seconds...)

Real-Time:           Client ← "Temp changed!" ← Server
                     (only when it actually changes)
```

This fundamental shift from **pulling** to **pushing** is what makes real-time communication possible.

---

## Real-Time Solutions

There are two main approaches for real-time communication:

### 1. WebSockets
**Best for:** Web applications, dashboards, chat, live updates

A WebSocket **upgrades** an HTTP connection to a persistent, bidirectional channel:

```
                 HTTP Handshake
Browser  ──────────────────────────→  Server
         ←──────────────────────────
              "Upgrade: websocket"

         ═══════════════════════════  ← Persistent connection
         ↕ Data flows both ways ↕
         ═══════════════════════════
```

### 2. MQTT (Message Queue Telemetry Transport)
**Best for:** IoT devices, hardware control, distributed systems

MQTT uses a **broker** (message router) with a publish/subscribe pattern:

```
Temperature Sensor  ──→  [MQTT Broker]  ──→  Dashboard
                     publish            subscribe
                  "sensors/temp"     "sensors/temp"
```

**Key difference:** In MQTT, devices don't talk directly. They all talk to a broker using **topics**.

---

## WebSockets: Direct Two-Way Channels

### How WebSockets Work

1. **Handshake**: Client sends HTTP request with `Upgrade: websocket`
2. **Connection**: Server agrees, connection stays open
3. **Bidirectional**: Both sides can send messages anytime
4. **Frames**: Messages sent as lightweight frames (not full HTTP requests)

### WebSocket Example: Pong Game Server

From our `pong-server` project (TypeScript with WebSockets):

```typescript
// Server: Accept WebSocket connections
const wss = new WebSocketServer({ port: 8080 });

wss.on('connection', (ws) => {
  console.log('Player connected');

  // Receive paddle position from client
  ws.on('message', (data) => {
    const update = JSON.parse(data.toString());
    // Update game state
    gameState.paddles[update.player] = update.y;

    // Broadcast to all clients
    wss.clients.forEach(client => {
      client.send(JSON.stringify(gameState));
    });
  });
});
```

**Characteristics:**
- Direct connection between client and server
- Low latency (perfect for gaming)
- Simple for point-to-point communication
- Built into browsers (no special libraries needed)

**When to use WebSockets:**
- Real-time dashboards
- Chat applications
- Live notifications
- Games with few players
- When you control both client and server

---

## MQTT: Publish/Subscribe for Hardware

### The Broker Pattern

MQTT decouples publishers from subscribers using a central **broker**:

```
                    ┌─────────────┐
   Temperature  ──→ │             │ ──→ Dashboard 1
   Sensor           │    MQTT     │
                    │   Broker    │ ──→ Dashboard 2
   Humidity     ──→ │             │
   Sensor           │             │ ──→ Alert System
                    └─────────────┘
```

**Key concepts:**
- **Topics**: Hierarchical message channels (e.g., `raptor/shop/device-1/temp`)
- **Publish**: Send a message to a topic
- **Subscribe**: Receive all messages from a topic
- **QoS**: Quality of Service levels (0=fire-and-forget, 1=at-least-once, 2=exactly-once)

### Why MQTT for IoT?

1. **Lightweight**: Designed for constrained devices (low CPU, low bandwidth)
2. **Resilient**: Handles intermittent connections gracefully
3. **One-to-Many**: Publish once, unlimited subscribers
4. **No Direct Connection**: Devices don't need to know about each other
5. **QoS Levels**: Critical messages guaranteed to arrive

---

## Real-World Examples from Our Projects

Let's see how we use these protocols across different languages and platforms:

### Example 1: raptor-core (Go + MQTT + Modbus)

**Purpose:** Industrial motor control system - reads VFD data via Modbus and publishes to MQTT

**Location:** `/Users/kalebtringale/raptor-core/main.go`

```go
// Connect to MQTT broker
opts := mqtt.NewClientOptions().
    AddBroker("tcp://10.0.106.26:1883").
    SetClientID("raptor-core-" + device).
    SetAutoReconnect(true).
    SetUsername(mqUser).
    SetPassword(mqPass)

mc := mqtt.NewClient(opts)
if tok := mc.Connect(); !tok.WaitTimeout(10*time.Second) || tok.Error() != nil {
    log.Fatalf("mqtt connect: %v", tok.Error())
}

// Publish motor telemetry every 2 seconds
stateTopic := fmt.Sprintf("raptor/%s/%s/state", site, device)
mc.Publish(stateTopic, 1, false, telemetryJSON)

// Subscribe to control commands
cmdTopic := fmt.Sprintf("raptor/%s/%s/cmd", site, device)
mc.Subscribe(cmdTopic, 1, func(_ mqtt.Client, msg mqtt.Message) {
    var cmd CmdPayload
    json.Unmarshal(msg.Payload(), &cmd)
    // Execute motor control via Modbus
})
```

**What's happening:**
1. Go program runs on Raspberry Pi in grain bin
2. Reads motor RPM, voltage, amps from VFD via Modbus
3. **Publishes** state to `raptor/shop/revpi-135593/state` every 2 seconds
4. **Subscribes** to `raptor/shop/revpi-135593/cmd` for control commands
5. When command arrives, writes to Modbus to control motors

**Why MQTT here:**
- Raspberry Pi has unreliable internet (grain bin in rural area)
- MQTT's auto-reconnect handles network drops
- Multiple dashboards can monitor same equipment (pub/sub)
- Commands work even from multiple locations

**Topic Structure:**
```
raptor/
  shop/              ← site
    revpi-135593/    ← device
      state          ← telemetry (published by device)
      cmd            ← commands (subscribed by device)
```

---

### Example 2: raptor-frontend (React + MQTT over WebSocket)

**Purpose:** Web dashboard for monitoring and controlling Raptor systems

**Location:** `/Users/kalebtringale/raptor-frontend/packages/mqtt/hooks/use-mqtt.ts`

```typescript
// MQTT from browser using WebSocket transport
const mqttOptions = {
  reconnectPeriod: 1000,
  connectTimeout: 10000,
  keepalive: 30,
  clean: true,
  resubscribe: true,
  // Custom WebSocket factory - forces 'mqtt' subprotocol
  createWebsocket: (wsUrl: string) => {
    const ws = new WebSocket(wsUrl, 'mqtt');
    ws.binaryType = 'arraybuffer';
    return ws;
  },
};

// Connect to MQTT broker via WebSocket
const client = mqtt.connect('wss://3.141.116.27:8083', mqttOptions);

client.on("connect", () => {
  console.log(`Connected to MQTT broker`);

  // Subscribe to motor state
  client.subscribe("raptor/shop/+/state", (err) => {
    if (!err) console.log("Subscribed to all device states");
  });
});

// Receive state updates
client.on("message", (topic, payload) => {
  const data = JSON.parse(payload.toString());
  // Update React state → UI updates automatically
  setMotorState(data);
});

// Send control command
function startMotor() {
  client.publish(
    "raptor/shop/revpi-135593/cmd",
    JSON.stringify({ wheels_running: true }),
    { qos: 1 }
  );
}
```

**What's happening:**
1. React app runs in browser (web dashboard)
2. Connects to MQTT broker using **WebSocket transport** (not TCP)
3. Subscribes to state from all devices: `raptor/shop/+/state`
4. Real-time updates flow from hardware → broker → browser
5. User clicks button → command flows browser → broker → hardware

**Why WebSocket transport:**
- Browsers can't make raw TCP connections (security)
- MQTT brokers support WebSocket on port 8083 (we use `wss://` for TLS)
- Same MQTT protocol, just over WebSocket instead of TCP

**Network-aware switching:**
The frontend is smart - it detects if you're on the local network:

```typescript
// Check if we can reach local Pi
const onPi = await fetch('http://raptor3.local:3000/health')
  .then(() => true)
  .catch(() => false);

if (onPi) {
  // Connect to local broker (lower latency, works offline)
  connectToBroker('ws://raptor3.local:1883', 'local');
} else {
  // Connect to cloud broker (works from anywhere)
  connectToBroker('wss://3.141.116.27:8083', 'cloud');
}
```

---

### Example 3: guardian-1pm (ESP32 + MQTT)

**Purpose:** WiFi relay controller for garage door monitoring (embedded C++)

**Location:** `/Users/kalebtringale/guardian-1pm/src/main.cpp`

```cpp
#include <PubSubClient.h>  // MQTT library for Arduino

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// Configure MQTT broker
mqttClient.setServer("3.141.116.27", 1883);
mqttClient.setCallback(mqttCallback);

// Connect to broker
bool connectToMQTT() {
  String clientId = "guardian-" + String(deviceId);

  if (mqttClient.connect(clientId.c_str(), mqttUser, mqttPass)) {
    Serial.println("MQTT Connected");

    // Subscribe to relay control commands
    String cmdTopic = "guardian/" + String(deviceId) + "/cmd";
    mqttClient.subscribe(cmdTopic.c_str());

    // Publish relay state
    String stateTopic = "guardian/" + String(deviceId) + "/state";
    String payload = "{\"relay\":\"" + String(relayState ? "on" : "off") + "\"}";
    mqttClient.publish(stateTopic.c_str(), payload.c_str());

    return true;
  }
  return false;
}

// Handle incoming commands
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  // Parse command and control relay
  if (message.indexOf("\"relay\":\"on\"") > 0) {
    digitalWrite(RELAY_PIN, HIGH);
    relayState = true;
  } else if (message.indexOf("\"relay\":\"off\"") > 0) {
    digitalWrite(RELAY_PIN, LOW);
    relayState = false;
  }

  // Publish updated state
  publishRelayState();
}

void loop() {
  // Keep MQTT connection alive
  mqttClient.loop();

  // Heartbeat - publish state every 60 seconds
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat >= 60000) {
    lastHeartbeat = millis();
    publishRelayState();
  }
}
```

**What's happening:**
1. ESP32 microcontroller connects to WiFi
2. Connects to MQTT broker over TCP (port 1883)
3. Subscribes to commands: `guardian/device-123/cmd`
4. Publishes state every 60s: `guardian/device-123/state`
5. When command arrives, toggles GPIO pin to control relay

**Why MQTT for embedded:**
- **PubSubClient** library is only 6KB (fits in ESP32 flash)
- Auto-reconnect handles power cycles and WiFi drops
- Multiple apps can control same device
- Publish state = all subscribers get updates instantly

---

### Example 4: pong-mqtt (Node.js Terminal Game)

**Purpose:** Interactive multiplayer Pong game using MQTT (learning demo)

**Location:** `/Users/kalebtringale/technical-discussion/real-time/pong-mqtt/`

```typescript
// Game Server (Node.js)
import mqtt from 'mqtt';

const client = mqtt.connect('mqtt://localhost:1883', {
  clientId: `pong-server-${Date.now()}`,
  clean: true,
  keepalive: 60,
});

client.on('connect', () => {
  // Subscribe to paddle updates from both players
  client.subscribe('pong/game/+/p1/paddle', { qos: 0 });
  client.subscribe('pong/game/+/p2/paddle', { qos: 0 });
});

client.on('message', (topic, message) => {
  const parts = topic.split('/');
  const gameId = parts[2];
  const player = parts[3]; // 'p1' or 'p2'

  const data = JSON.parse(message.toString());

  // Update game state
  gameState[gameId].paddles[player] = data.y;

  // Publish ball position to all clients (60 fps)
  const ballUpdate = {
    x: ball.x,
    y: ball.y,
    dx: ball.dx,
    dy: ball.dy,
  };
  client.publish(
    `pong/game/${gameId}/ball`,
    JSON.stringify(ballUpdate),
    { qos: 0 }
  );
});
```

**Game Client (Terminal UI with Ink)**
```typescript
// Player terminal
const client = getMqttClient({ brokerUrl: 'mqtt://3.141.116.27:1883' });

// Subscribe to ball position
client.subscribe(`pong/game/${gameId}/ball`);

// Publish paddle position when player moves
function movePaddle(y: number) {
  client.publish(
    `pong/game/${gameId}/p${playerNumber}/paddle`,
    JSON.stringify({ y, timestamp: Date.now() }),
    { qos: 0 }
  );
}

// Receive ball updates and render
client.on('message', (topic, payload) => {
  if (topic.includes('/ball')) {
    const ball = JSON.parse(payload.toString());
    renderGame(ball); // Update terminal display
  }
});
```

**Topic Structure:**
```
pong/
  game/
    demo/              ← game ID
      p1/
        paddle         ← player 1 paddle position
      p2/
        paddle         ← player 2 paddle position
      ball             ← ball position (60 updates/sec)
      state            ← game state (score, status)
      join             ← player join messages
      serve            ← serve ball command
      restart          ← restart game command
```

**Why this is brilliant for learning:**
1. **See the messages:** Run `mosquitto_sub -h localhost -t "pong/game/#"` to watch every message
2. **Same as hardware:** Exact same pub/sub pattern as industrial control
3. **Interactive:** Play the game, feel the real-time communication
4. **Multiple clients:** Three terminals can all watch the same game

**The "Aha!" Moment:**
```bash
# Terminal 1: Player 1
npm run dev -- --player 1 --game demo

# Terminal 2: Player 2
npm run dev -- --player 2 --game demo

# Terminal 3: Spectator (just listen to messages)
mosquitto_sub -h localhost -t "pong/game/demo/#"
```

You'll see every paddle movement, every ball update, every score change **exactly as they flow through MQTT**. This is precisely how we monitor grain bins, except instead of paddle positions, it's motor RPM. Instead of ball positions, it's temperatures.

---

## Choosing the Right Protocol

### Use WebSockets when:
- ✅ Building web dashboards with real-time updates
- ✅ You control both client and server
- ✅ Direct, low-latency connection is needed
- ✅ Point-to-point communication (not broadcast)
- ✅ Client is a web browser or mobile app

**Examples:** Chat apps, live dashboards, game lobbies

### Use MQTT when:
- ✅ Connecting IoT devices and hardware
- ✅ One-to-many communication (sensors → dashboards)
- ✅ Unreliable networks (auto-reconnect needed)
- ✅ Low bandwidth / constrained devices
- ✅ Devices should be decoupled (don't know about each other)
- ✅ Message delivery guarantees needed (QoS)

**Examples:** Industrial control, home automation, sensor networks, telemetry

### Side-by-side Comparison

| Feature | WebSockets | MQTT |
|---------|-----------|------|
| **Connection** | Direct client-server | Via broker |
| **Pattern** | Point-to-point | Pub/Sub |
| **Message routing** | Application handles it | Broker routes by topic |
| **One-to-many** | Must broadcast manually | Built-in (subscribe) |
| **Overhead** | Low (after handshake) | Very low (2-byte header) |
| **Browser support** | Native | Via WebSocket transport |
| **QoS levels** | No (TCP reliability only) | Yes (0, 1, 2) |
| **Offline messages** | No | Yes (retained messages) |
| **Best for** | Web apps, direct channels | IoT, sensors, hardware |

### Hybrid Approach (What we do!)

In the Raptor system, we use **both**:

```
┌─────────────┐   WebSocket/MQTT   ┌─────────────┐   MQTT/TCP   ┌─────────────┐
│   Browser   │ ←───────────────→  │    MQTT     │ ←─────────→  │ Raspberry   │
│  Dashboard  │    (port 8083)     │   Broker    │  (port 1883) │     Pi      │
└─────────────┘                     └─────────────┘              └─────────────┘
                                           ↕                            ↕
                                       Messages                     Modbus
                                     Topic-based                    RS-485
                                       Routing                      VFD (Motors)
```

- **Frontend uses MQTT over WebSocket:** Browser → Broker (port 8083 with TLS)
- **Backend uses MQTT over TCP:** Raspberry Pi → Broker (port 1883)
- **Broker handles routing:** Same topics, different transports

This gives us:
- ✅ Browser compatibility (WebSocket)
- ✅ Efficient hardware communication (TCP)
- ✅ Single message broker for everything
- ✅ Pub/sub decoupling

---

## Learning Through Pong

The Pong game demos are designed to make these concepts **tangible**:

### 1. pong-server (WebSockets)
- Simple, direct connections
- See how WebSockets work
- Low-level control

### 2. pong-mqtt (MQTT)
- **Same game, different protocol**
- Pub/sub pattern
- Broker-based routing
- **Identical to industrial systems**

### Try This Exercise

1. **Play the game:**
   ```bash
   cd ~/technical-discussion/real-time/pong-mqtt
   npm run dev -- --player 1 --game demo
   ```

2. **Watch the messages:**
   ```bash
   mosquitto_sub -h 3.141.116.27 -t "pong/game/#" -u raptor -P raptorMQTT2025
   ```

3. **Compare to real hardware:**
   ```bash
   mosquitto_sub -h 3.141.116.27 -t "raptor/shop/+/state" -u raptor -P raptorMQTT2025
   ```

**You'll notice:** The message patterns are **identical**.

| Pong Game | Industrial System |
|-----------|-------------------|
| `pong/game/demo/p1/paddle` → paddle Y position | `raptor/shop/device-1/state` → motor RPM |
| `pong/game/demo/ball` → ball X, Y | `raptor/shop/device-1/state` → voltage, amps |
| `pong/game/demo/state` → score | `raptor/shop/device-1/state` → run/stop status |
| `pong/game/demo/p1/paddle` updates 60/sec | `raptor/shop/device-1/state` updates 2/sec |
| Server publishes ball position | Hardware publishes telemetry |
| Client publishes paddle position | Dashboard publishes commands |

**The protocol is the same. The pattern is the same. Only the data is different.**

---

## From Games to Grain Bins

Once you play Pong over MQTT, you instantly understand our industrial systems:

### Pong Architecture
```
Terminal 1 (Player)  ─→  [MQTT Broker]  ─→  Terminal 2 (Player)
   publishes              routes by            subscribes
   paddle position        topic pattern        all game data
```

### Raptor Architecture
```
Raspberry Pi (VFD)  ─→  [MQTT Broker]  ─→  Web Dashboard
   publishes             routes by           subscribes
   motor telemetry       topic pattern       all device data
```

### Guardian Architecture
```
ESP32 (Relay)       ─→  [MQTT Broker]  ─→  Mobile App
   publishes             routes by           subscribes
   relay state           topic pattern       device status
```

**It's all the same system.** Pub/sub. Topics. Broker. The only difference is:
- Pong sends paddle positions
- Raptor sends motor RPM
- Guardian sends relay states

---

## Key Takeaways

### 1. Real-time means PUSH, not PULL
Stop polling. Let the server tell you when things change.

### 2. Two main approaches
- **WebSockets:** Direct channel, great for web apps
- **MQTT:** Broker-based pub/sub, perfect for IoT

### 3. Same protocols, different languages
- Go: Native MQTT over TCP for hardware control
- TypeScript: MQTT over WebSocket for browsers
- C++: Lightweight MQTT for embedded devices

### 4. Topics organize everything
Hierarchical topic structure makes routing scalable:
```
<project>/<site>/<device>/<data-type>
  raptor/shop/device-1/state
  raptor/shop/device-1/cmd
  guardian/home/garage-1/state
  pong/game/demo/ball
```

### 5. Pub/sub decouples systems
Devices don't know about dashboards. Dashboards don't know about devices. Broker routes messages. This is powerful.

### 6. Learn by playing
The Pong game isn't just fun—it's a perfect analog for real hardware. Play it, watch the messages, understand the pattern. Then apply it to motors, sensors, and relays.

---

## Next Steps

**For hands-on learning:**
1. Run the MQTT Pong game: `cd pong-mqtt && npm run dev`
2. Watch messages with `mosquitto_sub -h localhost -t "pong/game/#"`
3. Connect to our live systems: `mosquitto_sub -h 3.141.116.27 -t "raptor/shop/+/state" -u raptor -P raptorMQTT2025`
4. Build your own: Try creating a simple temperature sensor simulator

**Explore our projects:**
- `raptor-core`: Industrial Go + MQTT + Modbus
- `raptor-frontend`: React dashboard with MQTT
- `guardian-1pm`: ESP32 embedded MQTT
- `pong-mqtt`: Interactive learning demo

**Dive deeper:**
- MQTT specification: [mqtt.org](https://mqtt.org)
- WebSocket API: [MDN WebSocket docs](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- Mosquitto broker: [mosquitto.org](https://mosquitto.org)

---

Real-time communication isn't complicated. It's just pushing data instead of polling. Master these fundamentals, and you'll understand how modern IoT, industrial control, and collaborative systems work.

**Now go play Pong. Watch the messages. See the pattern. Then build something amazing.**
