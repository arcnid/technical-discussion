# Real-Time Communication & IoT Fundamentals

Introduction to real-time communication protocols through interactive examples. This demo shows the progression from traditional server-side rendering to modern push-based protocols used in IoT systems.

## Evolution of Data Communication Patterns

### 1. Server-Side Rendering (PHP)

Traditional approach: server generates HTML on every request.

```php
// page.php
<?php
$temperature = getTemperature();
?>
<html>
  <body>
    <h1>Temperature: <?= $temperature ?>°F</h1>
  </body>
</html>
```

**How it works:**
- User clicks link or submits form
- Browser sends HTTP request
- Server queries database, generates HTML
- Browser receives and displays new page
- Connection closes

**Limitations:**
- Full page reload required for any update
- No automatic updates when data changes
- High server load (generates HTML on every request)

---

### 2. AJAX Request/Response

Client-side JavaScript makes requests without reloading the page.

```javascript
// jQuery AJAX
$("#refresh-button").click(function () {
  $.ajax({
    url: "/api/temperature",
    success: function (data) {
      $("#temperature").text(data.temp + "°F");
    },
  });
});
```

**How it works:**
- User clicks button
- JavaScript sends HTTP request in background
- Server returns JSON data (not full HTML)
- JavaScript updates part of page
- Connection closes

**Limitations:**
- Still requires user action (button click)
- No automatic updates
- Each request creates new HTTP connection

---

### 3. Polling (setInterval + AJAX)

Automatic repeated requests at fixed intervals.

```javascript
// Poll every 5 seconds
setInterval(function () {
  $.ajax({
    url: "/api/temperature",
    success: function (data) {
      $("#temperature").text(data.temp + "°F");
    },
  });
}, 5000);
```

**How it works:**
- JavaScript automatically sends request every N seconds
- Server responds with current data
- Connection closes after each response
- Repeat indefinitely

**Limitations:**
- Wastes bandwidth (requests even when data unchanged)
- Fixed latency (up to N seconds before seeing updates)
- Scalability issues: 100 clients polling every 5s = 1,200 requests/minute
- Server creates new connection for each poll

---

### 4. Long Polling

Server holds request open until data changes.

```javascript
function poll() {
  $.ajax({
    url: "/api/temperature/wait",
    timeout: 30000, // 30 second timeout
    success: function (data) {
      $("#temperature").text(data.temp + "°F");
      poll(); // Immediately start next request
    },
    error: function () {
      setTimeout(poll, 5000); // Retry after error
    },
  });
}
poll();
```

```php
// Server-side (simplified)
<?php
$lastTemp = getTemperature();
$timeout = time() + 30;

while (time() < $timeout) {
  $currentTemp = getTemperature();
  if ($currentTemp !== $lastTemp) {
    echo json_encode(['temp' => $currentTemp]);
    exit;
  }
  sleep(1); // Check every second
}

// Timeout - return current value
echo json_encode(['temp' => $currentTemp]);
```

**How it works:**
- Client sends request
- Server holds connection open (doesn't respond immediately)
- When data changes, server responds and closes connection
- Client immediately sends new request
- If timeout expires, server responds with current data anyway

**Improvements:**
- Lower latency (responds immediately when data changes)
- Fewer requests than polling (only when data changes)

**Limitations:**
- Still uses HTTP request/response model
- Server resources held during wait (one thread per connection)
- Doesn't scale well (100 concurrent connections = 100 server threads)
- Complex error handling and reconnection logic

---

### 5. WebSockets

Persistent bidirectional connection.

```javascript
// Client
const ws = new WebSocket("ws://localhost:8080");

ws.onopen = function () {
  console.log("Connected");
};

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  $("#temperature").text(data.temp + "°F");
};

// Can also send messages to server
ws.send(JSON.stringify({ command: "subscribe", sensor: "temp1" }));
```

```javascript
// Server (Node.js)
const WebSocket = require("ws");
const wss = new WebSocket.Server({ port: 8080 });

wss.on("connection", (ws) => {
  // Connection stays open
  setInterval(() => {
    const temp = getTemperature();
    ws.send(JSON.stringify({ temp }));
  }, 1000);
});
```

**How it works:**
- HTTP handshake upgrades connection to WebSocket
- Connection stays open indefinitely
- Both client and server can send messages anytime
- No connection overhead after initial handshake

**Improvements:**
- True bidirectional communication
- Server pushes updates instantly
- Low overhead (no HTTP headers on each message)
- One connection handles all messages

**Use cases:**
- Real-time dashboards
- Chat applications
- Live notifications
- Multiplayer games

---

### 6. MQTT (Message Queue Telemetry Transport)

Publish/subscribe pattern with central broker.

```javascript
// Client 1: Temperature sensor publishes
const mqtt = require("mqtt");
const client = mqtt.connect("mqtt://localhost:1883");

client.on("connect", () => {
  setInterval(() => {
    const temp = getTemperature();
    client.publish("sensors/temp", JSON.stringify({ temp }));
  }, 1000);
});
```

```javascript
// Client 2: Dashboard subscribes
const mqtt = require("mqtt");
const client = mqtt.connect("mqtt://localhost:1883");

client.on("connect", () => {
  client.subscribe("sensors/temp");
});

client.on("message", (topic, message) => {
  const data = JSON.parse(message.toString());
  console.log(`Temperature: ${data.temp}°F`);
});
```

**How it works:**
- All clients connect to central broker (not to each other)
- Publishers send messages to topics
- Subscribers receive messages from topics they subscribe to
- Broker routes messages based on topic patterns

**Advantages over WebSockets:**
- Decoupled: publishers don't know about subscribers
- One-to-many: publish once, unlimited subscribers
- Quality of Service (QoS): guaranteed message delivery
- Retained messages: new subscribers get last value
- Lightweight: minimal overhead for IoT devices

**Use cases:**
- IoT sensor networks
- Industrial control systems
- Home automation
- Any scenario with multiple publishers/subscribers

---

## Comparison Summary

| Method            | Connection Type | Latency      | Bandwidth     | Scalability | Complexity |
| ----------------- | --------------- | ------------ | ------------- | ----------- | ---------- |
| Server-Side       | Per request     | High         | High          | Low         | Low        |
| AJAX              | Per request     | High         | Medium        | Low         | Low        |
| Polling           | Per request     | Medium-High  | High          | Low         | Medium     |
| Long Polling      | Held open       | Low          | Medium        | Medium      | High       |
| WebSockets        | Persistent      | Very Low     | Low           | High        | Medium     |
| MQTT              | Persistent      | Very Low     | Very Low      | Very High   | Medium     |

---

## MQTT for Industrial Systems

MQTT is the protocol used in the Raptor grain bin control system. Same concepts as the game demo below, but controlling real hardware.

**Example: Grain bin temperature monitoring**

Traditional polling approach:
- Dashboard polls sensor every 10 seconds
- Sensor responds with temperature
- If temperature unchanged: 8,640 wasted requests per day
- Each new dashboard multiplies the requests

MQTT approach:
- Sensor publishes temperature changes to topic: `raptor/site1/bin3/temp`
- Dashboards subscribe to topic
- Temperature unchanged for 1 hour: zero messages
- 10 dashboards subscribe: still zero extra sensor load
- Temperature spikes: all dashboards notified within 1 second

**Raptor System Topics:**
```
raptor/
  shop/
    revpi-135593/
      state    ← Device publishes motor RPM, voltage, amps
      cmd      ← Dashboard publishes control commands
```

Same pattern as the Pong game, just different data.

---

## Demo: MQTT Pong

Multiplayer Pong game using MQTT. Same pub/sub pattern as industrial control.

```
Terminal 1              MQTT Broker              Terminal 2
┌────────┐              ┌────────┐              ┌────────┐
│   █    │              │        │              │    █   │
│   █    │─paddle pos─>│        │─paddle pos─>│    █   │
│   █    │              │        │              │    █   │
│        │<─ball pos────│        │<─ball pos───│        │
└────────┘              └────────┘              └────────┘
```

**Message Flow:**

1. Player 1 publishes paddle position to `pong/game/demo/p1/paddle`
2. Player 2 publishes paddle position to `pong/game/demo/p2/paddle`
3. Server publishes ball position to `pong/game/demo/ball` (60 fps)
4. Both players subscribe to all topics
5. Broker routes messages to subscribers

**Topic Structure:**
```
pong/
  game/
    demo/              ← game ID
      p1/paddle        ← player 1 position
      p2/paddle        ← player 2 position
      ball             ← ball position (60 updates/sec)
      state            ← score, game status
```

**Watch Messages in Real-Time:**

```bash
# Open third terminal to see all messages
mosquitto_sub -h localhost -t "pong/game/#"
```

This shows every message flowing through the broker. Same pattern used to monitor industrial equipment.

**Pattern Comparison:**

| Pong Game                  | Raptor System                 |
| -------------------------- | ----------------------------- |
| `pong/game/demo/p1/paddle` | `raptor/shop/device-1/state`  |
| Paddle Y position (int)    | Motor RPM (int)               |
| Ball position updates      | Temperature readings          |
| Score change               | Alarm trigger                 |
| 60 messages/second         | 0.5 messages/second           |

Same protocol. Same pattern. Different data.

---

## Running the Demo

**Prerequisites:**
- Node.js 18+
- MQTT broker access (we'll use the Raptor cloud broker)
- Two terminals

**Start Game:**

```bash
cd pong-mqtt
npm install

# Terminal 1 (Player 1)
npm run dev -- --player 1 --game demo

# Terminal 2 (Player 2)
npm run dev -- --player 2 --game demo

# Terminal 3 (Watch messages - optional)
mosquitto_sub -h 3.141.116.27 -t "pong/game/#" -u raptor -P raptorMQTT2025
```

See [pong-mqtt/README.md](./pong-mqtt/README.md) for full instructions.

---

## Key Concepts

**Publish/Subscribe:**
- Publishers send messages to topics
- Subscribers receive messages from topics
- Publisher doesn't know who's subscribed
- Subscriber doesn't know who published

**Topics:**
- Hierarchical structure: `domain/location/device/metric`
- Wildcards: `+` (single level), `#` (multi-level)
- Example: `raptor/+/+/state` subscribes to all device states

**Quality of Service (QoS):**
- QoS 0: Fire and forget (best effort)
- QoS 1: At least once (guaranteed delivery)
- QoS 2: Exactly once (highest overhead)

**Retained Messages:**
- Broker stores last message for topic
- New subscribers immediately receive last value
- Useful for state (e.g., "motor running")

---

## Further Reading

- [Real-Time Communication Deep Dive](./docs/realtime-fundamentals.md) - Complete technical reference with code examples from our projects
- [MQTT Specification](https://mqtt.org/mqtt-specification/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

---

## Summary

**Evolution:**
1. Server-side rendering: Full page reload per update
2. AJAX: Partial updates, still manual
3. Polling: Automatic but wasteful
4. Long polling: Less wasteful but complex
5. WebSockets: Bidirectional, efficient
6. MQTT: Pub/sub, decoupled, scalable

**When to use MQTT:**
- IoT devices with intermittent connectivity
- One-to-many communication
- Decoupled systems
- Guaranteed message delivery needed

**When to use WebSockets:**
- Browser-based applications
- Direct client-server communication
- Lower latency requirements
- Simpler infrastructure (no broker needed)
