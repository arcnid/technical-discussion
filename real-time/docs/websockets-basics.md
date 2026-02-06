# WebSocket Basics: Real-Time for the Web

WebSockets provide full-duplex communication over a single TCP connection. They're perfect for web applications that need instant updates.

## The Upgrade Process

WebSockets start as HTTP requests, then "upgrade" to a persistent connection:

```
Client:  GET /chat HTTP/1.1
         Upgrade: websocket
         Connection: Upgrade

Server:  HTTP/1.1 101 Switching Protocols
         Upgrade: websocket
         Connection: Upgrade

         [Now using WebSocket protocol]
```

After the upgrade, data flows freely in both directions without HTTP overhead.

## Browser API (Client-Side)

WebSockets are built into browsers - no library needed:

```javascript
// Create connection
const ws = new WebSocket('ws://localhost:8080');

// Connection opened
ws.addEventListener('open', (event) => {
  console.log('Connected to server');
  ws.send('Hello Server!');
});

// Receive messages
ws.addEventListener('message', (event) => {
  console.log('Server says:', event.data);
});

// Connection closed
ws.addEventListener('close', (event) => {
  console.log('Disconnected');
});

// Error handling
ws.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## Server-Side (Node.js)

Popular libraries: `ws`, `socket.io`, `uWebSockets.js`

### Basic Server (ws library)

```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  console.log('Client connected');

  // Receive messages from this client
  ws.on('message', (message) => {
    console.log('Received:', message.toString());

    // Echo back
    ws.send(`You said: ${message}`);
  });

  // Send to this client
  ws.send('Welcome!');

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

// Broadcast to all clients
function broadcast(data) {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  });
}
```

## Chat Room Example

Here's a complete chat application:

### Server
```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

const clients = new Set();

wss.on('connection', (ws) => {
  clients.add(ws);
  console.log(`Client connected. Total: ${clients.size}`);

  ws.on('message', (message) => {
    const data = JSON.parse(message);

    // Broadcast to all clients
    clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({
          user: data.user,
          message: data.message,
          timestamp: Date.now()
        }));
      }
    });
  });

  ws.on('close', () => {
    clients.delete(ws);
    console.log(`Client left. Total: ${clients.size}`);
  });
});
```

### Client
```javascript
const ws = new WebSocket('ws://localhost:8080');
const username = prompt('Enter your name:');

ws.addEventListener('open', () => {
  console.log('Connected to chat');
});

ws.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  displayMessage(data.user, data.message);
});

function sendMessage(message) {
  ws.send(JSON.stringify({
    user: username,
    message: message
  }));
}

// Hook up to your UI
document.querySelector('#send').addEventListener('click', () => {
  const msg = document.querySelector('#input').value;
  sendMessage(msg);
});
```

## Handling Disconnections

Connections drop. Build resilience:

```javascript
class ReconnectingWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectDelay = 5000;
    this.connect();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.addEventListener('open', () => {
      console.log('Connected');
      this.reconnectDelay = 5000; // Reset delay
    });

    this.ws.addEventListener('close', () => {
      console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay *= 2; // Exponential backoff
      this.reconnectDelay = Math.min(this.reconnectDelay, 30000); // Max 30s
    });

    this.ws.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  send(data) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      console.warn('WebSocket not open, message not sent');
    }
  }

  on(event, callback) {
    this.ws.addEventListener(event, callback);
  }
}

// Usage
const ws = new ReconnectingWebSocket('ws://localhost:8080');
```

## Heartbeats / Ping-Pong

Keep connections alive and detect failures:

### Server
```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
  ws.isAlive = true;

  ws.on('pong', () => {
    ws.isAlive = true;
  });

  ws.on('message', (message) => {
    // Handle messages
  });
});

// Check every 30 seconds
const interval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (ws.isAlive === false) {
      return ws.terminate(); // Dead connection
    }

    ws.isAlive = false;
    ws.ping(); // Send ping, expect pong
  });
}, 30000);

wss.on('close', () => {
  clearInterval(interval);
});
```

### Client
```javascript
// Browsers auto-respond to pings with pongs
// No extra code needed!

// If you want explicit heartbeats:
const ws = new WebSocket('ws://localhost:8080');

setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'heartbeat' }));
  }
}, 30000);
```

## Message Formats

WebSockets support text and binary data:

### Text (JSON)
```javascript
// Send
ws.send(JSON.stringify({ type: 'chat', message: 'Hello' }));

// Receive
ws.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.message);
});
```

### Binary (ArrayBuffer)
```javascript
// Send
const buffer = new Uint8Array([1, 2, 3, 4]);
ws.send(buffer);

// Receive
ws.addEventListener('message', (event) => {
  if (event.data instanceof Blob) {
    event.data.arrayBuffer().then(buffer => {
      const view = new Uint8Array(buffer);
      console.log(view);
    });
  }
});
```

## Security: WSS (WebSocket Secure)

Always use `wss://` (WebSocket over TLS) in production:

```javascript
// Client
const ws = new WebSocket('wss://secure.example.com');

// Server (with HTTPS server)
const https = require('https');
const fs = require('fs');
const WebSocket = require('ws');

const server = https.createServer({
  cert: fs.readFileSync('cert.pem'),
  key: fs.readFileSync('key.pem')
});

const wss = new WebSocket.Server({ server });

server.listen(443);
```

## Authentication

Authenticate during initial HTTP upgrade:

### Option 1: Token in URL
```javascript
// Client
const token = 'user-auth-token';
const ws = new WebSocket(`ws://localhost:8080?token=${token}`);

// Server
const url = require('url');

wss.on('connection', (ws, req) => {
  const params = url.parse(req.url, true).query;

  if (!validateToken(params.token)) {
    ws.close(1008, 'Invalid token');
    return;
  }

  // Connection authenticated
});
```

### Option 2: Headers (requires custom request)
```javascript
// Using a library that supports custom headers
const ws = new WebSocket('ws://localhost:8080', {
  headers: {
    'Authorization': 'Bearer token123'
  }
});
```

## WebSocket vs HTTP: The Difference

### HTTP Request/Response
```
Client:  GET /data HTTP/1.1
Server:  HTTP/1.1 200 OK
         { "data": "value" }

[Connection closed]

Client:  GET /data HTTP/1.1  [New connection]
Server:  HTTP/1.1 200 OK
         { "data": "value" }
```

Every request: TCP handshake + TLS negotiation + HTTP headers = ~1-2KB overhead

### WebSocket
```
Client:  [Upgrade to WebSocket]
Server:  [Upgrade confirmed]

Client:  { "data": "value" }  [2-14 bytes overhead]
Server:  { "data": "value" }
Client:  { "data": "value" }
Server:  { "data": "value" }

[Same connection, minimal overhead]
```

## WebSocket vs MQTT

| Feature | WebSocket | MQTT |
|---------|-----------|------|
| **Protocol Layer** | Transport | Application |
| **Message Routing** | None (direct) | Topics/Broker |
| **Built-in Pub/Sub** | No | Yes |
| **QoS Guarantees** | No | Yes (0, 1, 2) |
| **Browser Support** | Native | Via library |
| **Server Required** | Custom logic | MQTT broker |
| **Use Case** | Web apps | IoT devices |
| **Connection** | Point-to-point | Many-to-many |
| **Offline Messages** | No | Yes (persistent sessions) |

### When to Use WebSocket
- Building a web application (native browser support)
- Direct client-to-server communication
- Custom protocol on top
- Real-time dashboard
- Chat application
- Live notifications

### When to Use MQTT
- Connecting hardware devices
- Need pub/sub pattern
- Want delivery guarantees (QoS)
- Intermittent connections
- Many publishers, many subscribers
- IoT / Industrial systems

## Real-World Example: Live Dashboard

```javascript
// Server: Broadcast sensor data to all dashboards
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

// Simulated sensor
setInterval(() => {
  const sensorData = {
    temperature: 70 + Math.random() * 10,
    humidity: 50 + Math.random() * 20,
    timestamp: Date.now()
  };

  // Broadcast to all connected dashboards
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(sensorData));
    }
  });
}, 1000);

// Client: Dashboard displays live data
const ws = new WebSocket('ws://localhost:8080');

ws.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);

  // Update UI
  document.getElementById('temp').textContent = data.temperature.toFixed(1);
  document.getElementById('humidity').textContent = data.humidity.toFixed(1);
});
```

## Common Patterns

### Pattern 1: Request-Response
```javascript
// Client
ws.send(JSON.stringify({ id: 123, action: 'getData' }));

ws.addEventListener('message', (event) => {
  const response = JSON.parse(event.data);
  if (response.id === 123) {
    console.log('Got response:', response.data);
  }
});
```

### Pattern 2: Subscriptions
```javascript
// Client subscribes to specific data
ws.send(JSON.stringify({
  action: 'subscribe',
  topic: 'sensor-1'
}));

// Server tracks subscriptions
const subscriptions = new Map();

ws.on('message', (message) => {
  const msg = JSON.parse(message);

  if (msg.action === 'subscribe') {
    if (!subscriptions.has(msg.topic)) {
      subscriptions.set(msg.topic, new Set());
    }
    subscriptions.get(msg.topic).add(ws);
  }
});
```

### Pattern 3: Rooms/Channels
```javascript
// Server manages rooms
const rooms = new Map();

function joinRoom(ws, roomId) {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, new Set());
  }
  rooms.get(roomId).add(ws);
}

function broadcast(roomId, message) {
  rooms.get(roomId)?.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}
```

## Debugging Tips

### Chrome DevTools
1. Open DevTools (F12)
2. Network tab
3. Filter: "WS" (WebSocket)
4. Click connection to see frames (messages)

### Command-Line Tools
```bash
# wscat (npm install -g wscat)
wscat -c ws://localhost:8080

# Send messages interactively
> Hello server!
< Server response

# websocat (Rust tool)
websocat ws://localhost:8080
```

## Common Gotchas

1. **Browser limits:** Most browsers limit ~255 WebSocket connections per domain
2. **Message size:** Some proxies limit message size, split large messages
3. **Buffering:** `ws.bufferedAmount` shows queued bytes, pause sending if high
4. **Same-origin:** WebSockets aren't restricted by CORS, validate origin on server
5. **Load balancers:** Need sticky sessions or Redis for multi-server setups

## Summary

WebSockets provide:
- **Full-duplex** communication
- **Low latency** (no HTTP overhead after upgrade)
- **Native browser** support
- **Simple** direct connections

They're perfect for web applications that need instant updates: dashboards, chat, notifications, live feeds.

For IoT and hardware, MQTT is usually better due to its pub/sub model and reliability guarantees. But for web-first applications, WebSockets are the natural choice.

Next: Try the [MQTT Pong Demo](../pong-mqtt) to see pub/sub in action!
