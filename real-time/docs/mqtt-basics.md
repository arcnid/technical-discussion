# MQTT Basics: The IoT Protocol Explained

MQTT (Message Queue Telemetry Transport) is the protocol that powers most IoT systems. It's lightweight, reliable, and perfect for connecting hardware to software.

## The Mental Model: Radio Stations

Think of MQTT like radio broadcasting:

- **Radio Station** = Device publishing data
- **Radio Frequency** = MQTT topic
- **Radio** = Device subscribing to data
- **Radio Tower** = MQTT broker (routes signals)

Just like you tune your radio to 101.5 FM to hear that station, you subscribe to `sensors/temperature` to receive temperature readings.

**Key insight:** The radio station doesn't need to know who's listening. It just broadcasts. This is pub/sub.

## The Three Players

### 1. Publisher (The Data Source)

A device or application that **publishes messages** to topics.

```javascript
// A temperature sensor publishing data
client.publish("sensors/temperature", "72.5");
```

Examples:

- Temperature sensor publishing every 5 seconds
- Motor controller publishing RPM
- Door sensor publishing open/close events
- Your Pong paddle publishing position

### 2. Subscriber (The Data Consumer)

A device or application that **subscribes to topics** to receive messages.

```javascript
// A dashboard subscribing to temperature
client.subscribe("sensors/temperature");

client.on("message", (topic, message) => {
  console.log(`Temp: ${message.toString()}°F`);
});
```

Examples:

- Dashboard displaying sensor data
- Alert system watching for critical values
- Logger recording all events
- Your Pong game receiving opponent's paddle position

### 3. Broker (Traffic Handler)

The central server that routes messages from publishers to subscribers.

```
Sensor A ──publish──> Broker ──route──> Dashboard
Sensor B ──publish──> Broker ──route──> Logger
                                     └──> Alert System
```

**Popular brokers:**

- Eclipse Mosquitto (what we use in Raptor)
- HiveMQ
- EMQX
- AWS IoT Core

## Topics: The Addressing System

Topics are hierarchical paths separated by forward slashes:

```
building/floor/room/device/metric
```

### Real Examples

**Factory Setup:**

```
factory/
  building-1/
    line-a/
      motor-1/
        rpm
        temperature
        status
      motor-2/
        rpm
        temperature
    line-b/
      conveyor/
        speed
        load
  building-2/
    hvac/
      temperature
      humidity
```

**Subscribe Examples:**

```javascript
// Specific: Just motor-1 RPM
client.subscribe("factory/building-1/line-a/motor-1/rpm");

// Single wildcard (+): All motors on line-a
client.subscribe("factory/building-1/line-a/+/rpm");

// Multi wildcard (#): Everything in building-1
client.subscribe("factory/building-1/#");

// Everything from all motors
client.subscribe("factory/+/+/+/rpm");
```

### Topic Best Practices

**DO:**

- Use lowercase with hyphens: `grain-bin/temp`
- Start general, end specific: `site/device/metric`
- Be consistent: always `temperature` not sometimes `temp`
- Use meaningful names: `motor-rpm` not `mr`

**DON'T:**

- Use spaces: `grain bin` (use `grain-bin`)
- Mix conventions: `GrainBin/temp` and `grain-bin/Temp`
- Start with `/`: `/sensors/temp` (just `sensors/temp`)
- Put dynamic IDs early: `12345/sensor` (use `sensor/12345`)

## Quality of Service (QoS)

MQTT lets you choose reliability level per message:

### QoS 0: At Most Once (Fire and Forget)

```javascript
client.publish("sensors/temp", "72", { qos: 0 });
```

**Flow:**

1. Publisher sends message
2. Done (no confirmation)

**Use for:**

- High-frequency sensor data (temp every second)
- Status updates where latest value is all that matters
- Non-critical notifications

**Example:** Our Pong game uses QoS 0 because if we miss one paddle position, the next one comes in 16ms.

### QoS 1: At Least Once (Acknowledged)

```javascript
client.publish("commands/start-motor", "START", { qos: 1 });
```

**Flow:**

1. Publisher sends message
2. Broker acknowledges receipt
3. If no ack, publisher resends

**Use for:**

- Important data that must arrive
- Commands where duplicates are safe
- Logging events

**Example:** Raptor motor commands use QoS 1 - must arrive, but duplicate "start" is harmless.

### QoS 2: Exactly Once (Guaranteed)

```javascript
client.publish("critical/emergency-stop", "STOP", { qos: 2 });
```

**Flow:**

1. Publisher sends message
2. Broker acknowledges
3. Publisher confirms acknowledgment
4. Broker confirms confirmation

**Use for:**

- Critical commands (emergency stop)
- Financial transactions
- State changes that can't duplicate

**Example:** An emergency stop must happen exactly once - not zero times (dangerous) and not twice (might cause issues).

## Retained Messages

The broker can **remember** the last message on a topic:

```javascript
// Publish with retain flag
client.publish("devices/status", "online", { retain: true });
```

**Why this matters:**

Without retain:

```
11:00 AM - Device publishes "online"
11:30 AM - Dashboard connects → sees nothing
```

With retain:

```
11:00 AM - Device publishes "online" (retained)
11:30 AM - Dashboard connects → immediately gets "online"
```

**Use for:**

- Status messages (online/offline)
- Configuration values
- Last known state
- Any "what's the current value?" data

## Last Will and Testament (LWT)

Tell the broker "if I disconnect unexpectedly, publish this message":

```javascript
const client = mqtt.connect("mqtt://broker", {
  will: {
    topic: "devices/status",
    payload: "offline",
    qos: 1,
    retain: true,
  },
});
```

**Flow:**

1. Device connects with LWT configured
2. Everything works normally
3. Device suddenly loses power
4. Broker detects disconnect
5. Broker publishes the LWT message

**Use for:**

- Device online/offline status
- Heartbeat monitoring
- Detecting failures

**Example:** In Raptor, gateways use LWT so the cloud knows immediately if a site goes offline.

## Clean Session vs Persistent Session

### Clean Session (true)

```javascript
const client = mqtt.connect("mqtt://broker", {
  clean: true,
});
```

- Fresh start each connection
- Subscriptions lost on disconnect
- Queued messages discarded
- **Use for:** Temporary clients, testing, high-frequency data

### Persistent Session (false)

```javascript
const client = mqtt.connect("mqtt://broker", {
  clean: false,
  clientId: "dashboard-main", // Must have unique ID
});
```

- Broker remembers subscriptions
- Queued messages saved while offline
- Receive missed messages on reconnect
- **Use for:** Critical data collection, offline devices, guaranteed delivery

## Connection Example (Putting It All Together)

```javascript
const mqtt = require("mqtt");

// Connect with full options
const client = mqtt.connect("mqtt://broker.local:1883", {
  clientId: "sensor-" + Math.random().toString(16).substr(2, 8),
  clean: false, // Persistent session
  keepalive: 60, // Ping every 60 seconds
  reconnectPeriod: 5000, // Auto-reconnect after 5s
  username: "device1", // Optional authentication
  password: "secret123",
  will: {
    // Last Will and Testament
    topic: "devices/sensor-1/status",
    payload: "offline",
    qos: 1,
    retain: true,
  },
});

// Connection successful
client.on("connect", () => {
  console.log("Connected to broker");

  // Publish status
  client.publish("devices/sensor-1/status", "online", {
    qos: 1,
    retain: true,
  });

  // Subscribe to commands
  client.subscribe("devices/sensor-1/commands", { qos: 1 });
});

// Receive messages
client.on("message", (topic, message) => {
  console.log(`${topic}: ${message.toString()}`);

  if (topic === "devices/sensor-1/commands") {
    handleCommand(message.toString());
  }
});

// Connection lost
client.on("offline", () => {
  console.log("Disconnected, will auto-reconnect");
});

// Publish sensor data every 5 seconds
setInterval(() => {
  const temp = readTemperature();
  client.publish("sensors/temperature", temp.toString(), { qos: 0 });
}, 5000);
```

## Wildcards Deep Dive

### Single-Level Wildcard (+)

Matches exactly one level:

```javascript
// Subscribe to all motors on line-a
client.subscribe("factory/building-1/line-a/+/rpm");
```

**Matches:**

- `factory/building-1/line-a/motor-1/rpm` ✅
- `factory/building-1/line-a/motor-2/rpm` ✅

**Doesn't match:**

- `factory/building-1/line-a/rpm` ❌ (no device level)
- `factory/building-1/line-a/motor-1/temp` ❌ (not rpm)
- `factory/building-1/line-b/motor-1/rpm` ❌ (wrong line)

### Multi-Level Wildcard (#)

Matches zero or more levels (must be last):

```javascript
// Subscribe to everything from motor-1
client.subscribe("factory/building-1/line-a/motor-1/#");
```

**Matches:**

- `factory/building-1/line-a/motor-1/rpm` ✅
- `factory/building-1/line-a/motor-1/temp` ✅
- `factory/building-1/line-a/motor-1/status/alarm` ✅

**Common Patterns:**

```javascript
// Everything (caution: high volume!)
client.subscribe("#");

// Everything in a building
client.subscribe("factory/building-1/#");

// All metrics from all devices
client.subscribe("factory/+/+/+/#");
```

## MQTT vs HTTP: When to Use What

| Feature    | MQTT               | HTTP/REST           |
| ---------- | ------------------ | ------------------- |
| Connection | Persistent         | Request/response    |
| Direction  | Bidirectional      | Client-initiated    |
| Overhead   | Very low (2 bytes) | High (headers)      |
| Real-time  | Excellent          | Poor (polling)      |
| Pub/Sub    | Native             | Requires extra work |
| Caching    | Retained messages  | HTTP caches         |
| Use case   | IoT, real-time     | Web APIs, CRUD      |

**Use MQTT when:**

- Connecting hardware devices
- Real-time updates needed
- Many devices publishing to many subscribers
- Intermittent connections expected
- Bandwidth is limited

**Use HTTP when:**

- Traditional web API
- Request/response pattern fits naturally
- Existing infrastructure required
- One-off queries ("what's the temperature right now?")

## Common Patterns

### Pattern 1: Telemetry Collection

```javascript
// 100 sensors publishing
sensor.publish("farm/bin-1/temp", "72");
sensor.publish("farm/bin-2/temp", "68");
// ...

// One dashboard subscribing to all
dashboard.subscribe("farm/+/temp");
```

### Pattern 2: Command & Control

```javascript
// Control system publishes commands
control.publish("motors/motor-1/cmd", "START");

// Motor subscribes to its commands
motor.subscribe("motors/motor-1/cmd");
motor.on("message", (topic, msg) => {
  if (msg.toString() === "START") {
    startMotor();
  }
});
```

### Pattern 3: Status Monitoring

```javascript
// Device publishes heartbeat
setInterval(() => {
  device.publish("devices/device-1/heartbeat", Date.now(), {
    qos: 1,
    retain: true,
  });
}, 10000);

// Monitor subscribes
monitor.subscribe("devices/+/heartbeat");
```

### Pattern 4: Event Logging

```javascript
// Multiple sources publish events
motor.publish(
  "events/motor/fault",
  JSON.stringify({
    timestamp: Date.now(),
    error: "overcurrent",
    value: 15.2,
  }),
  { qos: 1 },
);

// Logger captures all
logger.subscribe("events/#");
```

## Try It Yourself

Want to see MQTT in action? Use the `mosquitto_pub` and `mosquitto_sub` command-line tools:

```bash
# Terminal 1: Subscribe to a topic
mosquitto_sub -h localhost -t "test/demo"

# Terminal 2: Publish a message
mosquitto_pub -h localhost -t "test/demo" -m "Hello MQTT!"

# Subscribe with wildcards
mosquitto_sub -h localhost -t "test/#"

# Publish with retain
mosquitto_pub -h localhost -t "test/status" -m "online" -r

# Watch the Pong game messages
mosquitto_sub -h localhost -t "pong/game/#"
```

## Common Gotchas

1. **Topic names are case-sensitive:** `Sensor/Temp` ≠ `sensor/temp`

2. **Leading/trailing slashes matter:** `a/b/c` ≠ `/a/b/c` ≠ `a/b/c/`

3. **QoS is "requested" not "guaranteed":** Subscriber gets min(publisher QoS, subscriber QoS)

4. **Wildcard subscriptions can overwhelm:** `#` might be millions of messages

5. **Client IDs must be unique:** Two clients with same ID will fight (connect/disconnect loop)

## The Raptor Connection

In the Raptor system, we use MQTT for:

- **State telemetry:** VFD publishes motor speed every 2 seconds (QoS 0)
- **Commands:** Dashboard publishes start/stop commands (QoS 1)
- **Status:** Gateway publishes online/offline with LWT (QoS 1, retained)
- **Faults:** VFD publishes alarms (QoS 1 to ensure delivery)

**Topics:**

```
raptor/
  shop/              # Site
    revpi-135593/    # Device
      state          # Telemetry
      cmd            # Commands
      status         # Online/offline
      faults         # Alarms
```

When you play the Pong demo and subscribe to `pong/game/#`, you're seeing exactly what we see when we monitor Raptor with `mosquitto_sub -t "raptor/#"`.

## Summary

MQTT is:

- **Lightweight** - Perfect for constrained devices
- **Pub/Sub** - Publishers and subscribers are decoupled
- **Reliable** - QoS levels ensure delivery guarantees
- **Scalable** - One publisher, unlimited subscribers
- **Resilient** - Handles disconnections gracefully

Once you understand topics, pub/sub, and QoS, you understand 90% of what makes IoT systems work. The Pong demo shows these concepts in action - same protocol, same patterns, just different data.

Next: [WebSocket Basics](./websockets-basics.md) to see the alternative approach.
