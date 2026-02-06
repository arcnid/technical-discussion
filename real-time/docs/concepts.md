# Real-Time Communication Concepts

This guide explains the fundamentals of real-time communication, comparing different approaches and when to use each.

## The Evolution: From Polling to Push

### 1. Traditional Polling (The Old Way)

Client repeatedly asks server "got anything new?"

```javascript
// Check every 5 seconds
setInterval(() => {
  fetch('/api/status')
    .then(res => res.json())
    .then(data => updateUI(data));
}, 5000);
```

**Pros:**
- Simple to implement
- Works with any HTTP server
- Client controls update frequency

**Cons:**
- Wastes bandwidth (requests even when nothing changed)
- High latency (wait up to polling interval)
- Server load increases with clients (100 clients = 20 req/sec)
- Battery drain on mobile devices

**Use when:** Updates are infrequent and latency doesn't matter

### 2. Long-Polling (The Bridge)

Client asks, server waits to respond until it has new data.

```javascript
function longPoll() {
  fetch('/api/updates')  // Server holds this open
    .then(res => res.json())
    .then(data => {
      updateUI(data);
      longPoll();  // Immediately reconnect
    });
}
```

**Pros:**
- Lower latency than polling
- Updates arrive as they happen
- Works with existing HTTP infrastructure

**Cons:**
- Still wastes connections (ties up a server connection per client)
- Doesn't truly scale
- Complicated timeout handling

**Use when:** You need real-time but can't use WebSockets

### 3. WebSockets (The Modern Web Solution)

Persistent bidirectional connection - both sides can send anytime.

```javascript
const ws = new WebSocket('ws://server.com');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateUI(data);
};

// Send data anytime
ws.send(JSON.stringify({ action: 'subscribe' }));
```

**Pros:**
- True bidirectional communication
- Low latency (no connection overhead)
- Efficient (single persistent connection)
- Native browser support

**Cons:**
- Requires special server support
- Connection management (reconnection logic)
- Doesn't handle offline well
- Point-to-point (each client needs direct connection)

**Use when:** Building real-time web applications (chat, dashboards, live updates)

### 4. MQTT (The IoT Champion)

Pub/Sub with a broker - clients don't connect to each other, they connect to a central router.

```javascript
const mqtt = require('mqtt');
const client = mqtt.connect('mqtt://broker.local');

// Subscribe to topics
client.subscribe('sensors/temperature');

// Receive messages
client.on('message', (topic, message) => {
  console.log(`${topic}: ${message.toString()}`);
});

// Publish data
client.publish('sensors/temperature', '72.5');
```

**Pros:**
- Extremely lightweight (designed for constrained devices)
- Broker handles routing (devices don't need to know about each other)
- Quality of Service levels (guarantee delivery)
- Handles intermittent connections gracefully
- One-to-many: publish once, unlimited subscribers
- Perfect for distributed systems

**Cons:**
- Requires MQTT broker
- Not built into browsers (needs library)
- Slightly more complex setup

**Use when:** IoT, hardware integration, industrial systems, multi-device coordination

## Quality of Service (QoS) in MQTT

One of MQTT's killer features is QoS levels that let you balance reliability vs performance:

### QoS 0: At Most Once (Fire and Forget)
- Message sent once, no confirmation
- Fastest, but might lose messages
- **Use for:** Sensor readings where next reading comes soon (temperature every second)

### QoS 1: At Least Once (Acknowledged Delivery)
- Message confirmed delivered
- Might get duplicates
- **Use for:** Important data where duplicates are OK (logging events)

### QoS 2: Exactly Once (Guaranteed Delivery)
- Four-way handshake ensures single delivery
- Slowest, but no duplicates
- **Use for:** Critical commands (start motor, unlock door)

**Example:** In the Raptor system:
- Temperature telemetry: QoS 0 (reading every 2 seconds, losing one is fine)
- Motor commands: QoS 1 (must arrive, duplicate start command is safe)
- Emergency stop: QoS 2 (absolutely must arrive exactly once)

## Connection Patterns

### Point-to-Point (WebSocket)

```
Client A  ←→  Server  ←→  Client B
     Direct connections
```

- Server knows each client directly
- Server must route messages between clients
- Scales linearly with connections

### Pub/Sub (MQTT)

```
Client A  →  Broker  ←  Client B
              ↓
        Client C subscribes
```

- Clients don't know about each other
- Broker handles routing via topics
- Scales logarithmically

## Topic Naming in MQTT

Topics use hierarchical structure with `/` separators:

```
factory/
  building1/
    machine1/
      temperature
      speed
      status
    machine2/
      temperature
      speed
  building2/
    machine1/
      temperature
```

**Wildcards:**
- `+` - Single level: `factory/building1/+/temperature` (all machines' temps in building1)
- `#` - Multiple levels: `factory/building1/#` (everything in building1)

**Best Practices:**
- Start general, end specific: `location/device/metric` not `metric/device/location`
- Use consistent naming: `temperature` not `temp` sometimes, `temperature` other times
- Avoid spaces and special chars
- Keep it readable: `grain-bin/temp` not `gb/t`

## Reconnection and Resilience

Real-world connections drop. Your code must handle it:

### Naive Approach (Don't Do This)
```javascript
const ws = new WebSocket('ws://server');
// Connection drops → app breaks
```

### Robust Approach
```javascript
class ReconnectingClient {
  connect() {
    this.ws = new WebSocket('ws://server');

    this.ws.onclose = () => {
      setTimeout(() => this.connect(), 5000); // Retry after 5s
    };

    this.ws.onerror = (err) => {
      console.error('Connection error:', err);
    };
  }
}
```

**MQTT handles this better with built-in reconnection:**
```javascript
const client = mqtt.connect('mqtt://broker', {
  reconnectPeriod: 5000,  // Auto-reconnect
  clean: false,            // Resume subscriptions
  will: {                  // Tell others if we disconnect
    topic: 'devices/status',
    payload: 'offline'
  }
});
```

## When to Use What

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Live sports scores | WebSocket | Browser native, low latency, direct updates |
| Chat application | WebSocket | Bidirectional, user-to-user messaging |
| Stock ticker | WebSocket or SSE | Continuous stream, web-based |
| Temperature sensors | MQTT | Hundreds of devices, need offline resilience |
| Industrial control | MQTT | QoS guarantees, hierarchical topics |
| Door sensors | MQTT | Lightweight, intermittent connections |
| Multi-room dashboard | MQTT | Many subscribers to same data |
| Mobile app notifications | MQTT | Battery efficient, handles spotty connections |

## Latency Considerations

**Traditional Polling:**
- Average latency: Half the polling interval
- 10-second polling = 5-second average delay

**Long-Polling:**
- Latency: Network round-trip time
- Typically 50-200ms

**WebSocket:**
- Latency: 10-50ms
- Best for user-facing real-time needs

**MQTT:**
- Latency: 20-100ms depending on QoS
- QoS 0 fastest, QoS 2 slowest
- Still excellent for most IoT applications

## Security Basics

### WebSocket
- Use WSS (WebSocket Secure) over TLS
- Authenticate during HTTP upgrade
- Same-origin policy helps protect

### MQTT
- Use MQTTS (MQTT over TLS) for encryption
- Username/password authentication
- ACLs (Access Control Lists) on broker for topic permissions
- Example: Client A can publish to `devices/a/#` but not `devices/b/#`

## The Pong Demo Connection

In our MQTT Pong game, you'll see these concepts in action:

- **Pub/Sub**: Each player publishes paddle position, both subscribe to ball
- **Topics**: `pong/game/{id}/p1/paddle`, `pong/game/{id}/ball`
- **QoS 0**: We use "fire and forget" since missing one frame is OK
- **Broker**: The MQTT broker routes messages between players
- **Reconnection**: If connection drops, game pauses and reconnects

When you watch with `mosquitto_sub`, you're seeing the exact message flow that happens in industrial systems. The only difference is the payload content.

## Summary

Real-time communication is about **pushing data when it happens** instead of constantly asking "anything new?"

- **WebSockets** for web apps that need instant updates
- **MQTT** for IoT, hardware, and distributed systems
- **QoS** lets you balance speed vs reliability
- **Pub/Sub** scales better than point-to-point
- **Resilience** is critical - connections will drop

The best part? Once you understand one real-time protocol, the concepts transfer to all of them. They're all solving the same problem: get data from A to B as fast as possible, reliably.
