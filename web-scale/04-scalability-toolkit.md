# Part 4: The Scalability Toolkit

## Introduction: One Tool, One Problem

Each scalability tool solves a **specific bottleneck**. Using them without understanding the problem is premature optimization.

**Golden Rule:** Only add complexity when you have a specific problem to solve.

---

## 1. CDN (Content Delivery Network)

### Problem It Solves

**Symptom:** Users far from your server experience slow page loads
**Root cause:** Geographic latency + bandwidth costs for static files

### How It Works

**Without CDN:**

```
User in Sydney → Your server in Ohio

GET /app.js (500 KB)
- Latency: 200ms
- Download: 2 seconds (on slow connection)
- Total: 2.2 seconds

All static files served from Ohio
```

**With CDN:**

```
User in Sydney → CDN edge server in Sydney

GET /app.js (500 KB)
- Latency: 5ms (local!)
- Download: 2 seconds
- Total: 2.005 seconds

Static files cached at edge, close to user
```

### Architecture

```
┌─────────┐
│  User   │
│ (Sydney)│
└────┬────┘
     │ GET /app.js
     ▼
┌──────────────────┐
│ CDN Edge Server  │
│   (Sydney)       │  Cache HIT ✅ → Return file (fast!)
└──────────────────┘

Cache MISS ❌ ↓

┌──────────────────┐
│  Origin Server   │  Fetch file once, cache at edge
│    (Ohio)        │
└──────────────────┘
```

**First user from Sydney:** Slow (fetches from Ohio)
**Subsequent users from Sydney:** Fast (served from Sydney CDN)

### What to Cache on CDN

**✅ Static assets (never change without deploy):**
- JavaScript bundles (`app.js`, `vendor.js`)
- CSS files (`styles.css`)
- Images (`logo.png`, `hero.jpg`)
- Fonts (`roboto.woff2`)

**❌ Dynamic data (changes frequently):**
- API responses (`/api/products`)
- User-specific data (`/api/cart`)
- Real-time data (`/api/notifications`)

### Implementation

#### 1. Simple: Use a CDN provider

```javascript
// Before (self-hosted)
<script src="https://yourdomain.com/app.js"></script>
<img src="https://yourdomain.com/logo.png">

// After (Cloudflare CDN)
<script src="https://cdn.yourdomain.com/app.js"></script>
<img src="https://cdn.yourdomain.com/logo.png">
```

**CDN provider fetches from your origin, caches globally.**

#### 2. Advanced: Set cache headers

```javascript
// Express server - tell CDN how long to cache
app.use('/static', express.static('public', {
    maxAge: '1y', // Cache for 1 year
    immutable: true // File will never change (use versioned filenames)
}));

// Results in HTTP header:
// Cache-Control: public, max-age=31536000, immutable
```

**Versioned filenames for cache busting:**

```html
<!-- When you deploy, filename changes -->
<script src="/app.abc123.js"></script>

<!-- After deploy -->
<script src="/app.def456.js"></script>
```

**Old version cached? Doesn't matter, new version has new filename.**

### Cost Comparison

**Without CDN (self-hosted):**

```
100,000 users/day × 3 MB per user = 300 GB/day = 9 TB/month
VPS bandwidth: $0.10/GB
Cost: 9,000 GB × $0.10 = $900/month
```

**With CDN:**

```
CDN bandwidth: $0.01-0.05/GB (Cloudflare: free tier available!)
Cost: 9,000 GB × $0.02 = $180/month
Savings: $720/month (80% reduction)
```

**Plus:** Users get faster load times.

### When to Use

✅ Users in multiple geographic regions
✅ Significant static assets (images, videos, JS/CSS)
✅ Bandwidth costs becoming significant
✅ Want to improve global page load times

### When NOT to Use

❌ All your users are in one geographic area
❌ Purely API server (no static files)
❌ Under 1,000 daily users (cost > benefit)

---

## 2. Caching (Redis / Memcached)

### Problem It Solves

**Symptom:** Database getting hammered with repeated queries for same data
**Root cause:** Every request re-fetches data that rarely changes

### How It Works

**Without caching:**

```
Every request:
1. Query database
2. Process data
3. Return result

100 requests for /api/products:
- 100 database queries
- Database CPU: high
- Response time: 100ms each
```

**With caching:**

```
First request:
1. Check cache (MISS)
2. Query database
3. Store in cache
4. Return result

Next 99 requests:
1. Check cache (HIT)
2. Return cached result (no database query!)

100 requests for /api/products:
- 1 database query
- 99 cache hits
- Database CPU: minimal
- Response time: 5ms (cached)
```

### Architecture

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ GET /api/products
     ▼
┌─────────────┐
│ API Server  │
│             │
│ 1. Check   │─────→ ┌────────────┐
│    Redis   │       │   Redis    │
│             │←─────│ (In-memory)│
│ 2. If MISS:│       └────────────┘
│    Query DB│         (Fast!)
│             │─────→ ┌────────────┐
│ 3. Cache   │       │  Database  │
│    Result  │←─────│ (Slower)   │
└─────────────┘       └────────────┘
```

### Implementation

```javascript
const redis = require('redis');
const client = redis.createClient();

app.get('/api/products', async (req, res) => {
    const cacheKey = 'products:all';

    // 1. Check cache
    const cached = await client.get(cacheKey);
    if (cached) {
        console.log('Cache HIT');
        return res.json(JSON.parse(cached));
    }

    console.log('Cache MISS - querying database');

    // 2. Query database
    const products = await db.query('SELECT * FROM products');

    // 3. Store in cache (expire after 5 minutes)
    await client.setex(cacheKey, 300, JSON.stringify(products));

    res.json(products);
});
```

### Cache Invalidation Strategies

**The hard problem:** When do you clear the cache?

#### Strategy 1: Time-based (TTL - Time To Live)

```javascript
// Cache for 5 minutes
await client.setex('products:all', 300, data);

// After 5 minutes, cache automatically expires
```

**Pros:** Simple
**Cons:** Data might be stale for up to 5 minutes

#### Strategy 2: Event-based (invalidate on change)

```javascript
// When product is updated
app.put('/api/products/:id', async (req, res) => {
    // 1. Update database
    await db.query('UPDATE products SET ... WHERE id = ?', [req.params.id]);

    // 2. Invalidate cache
    await client.del('products:all');
    await client.del(`products:${req.params.id}`);

    res.json({ success: true });
});
```

**Pros:** Always fresh data
**Cons:** More complex, must invalidate everywhere data changes

#### Strategy 3: Write-through cache

```javascript
// When product is updated
app.put('/api/products/:id', async (req, res) => {
    // 1. Update database
    await db.query('UPDATE products SET ... WHERE id = ?', [req.params.id]);

    // 2. Update cache immediately (not just delete)
    const updated = await db.query('SELECT * FROM products WHERE id = ?', [req.params.id]);
    await client.setex(`products:${req.params.id}`, 300, JSON.stringify(updated));

    res.json({ success: true });
});
```

**Pros:** Cache always warm, no MISS after writes
**Cons:** Most complex

### What to Cache

**✅ Good cache candidates:**

```javascript
// Product catalog (changes rarely)
cache('products:all', () => db.query('SELECT * FROM products'));

// User profile (changes occasionally)
cache(`user:${userId}`, () => db.query('SELECT * FROM users WHERE id = ?', [userId]));

// Homepage content (static)
cache('homepage:hero', () => db.query('SELECT * FROM hero_section'));

// Expensive calculations
cache('analytics:daily', () => calculateDailyStats());
```

**❌ Bad cache candidates:**

```javascript
// Real-time data (defeats purpose)
cache('stock:AAPL', () => getStockPrice()); // Changes every second!

// User-specific dynamic data
cache(`cart:${userId}`, () => getCart(userId)); // Frequently updated

// Large datasets (memory waste)
cache('logs:all', () => db.query('SELECT * FROM logs')); // Millions of rows
```

### Performance Impact

**Example: Product list endpoint**

```
Without cache:
- 100 requests/second
- Each query takes 50ms
- Database: 100 queries/sec
- Response time: 50ms average

With cache (80% hit rate):
- 100 requests/second
- 80 cache hits (5ms each)
- 20 cache misses (50ms each)
- Database: 20 queries/sec (80% reduction!)
- Response time: 14ms average (72% faster)
```

### Cost

**Redis/Memcached hosting:**

```
Managed Redis (AWS ElastiCache, etc.):
- $15-50/month (small instance)
- $100-500/month (production instance)

Self-hosted:
- Free (run on same server)
- But uses server RAM
```

### When to Use

✅ High read-to-write ratio (read 10x more than write)
✅ Expensive database queries
✅ Data that doesn't change often
✅ Database becoming bottleneck

### When NOT to Use

❌ Data changes constantly (real-time stock prices)
❌ Cache invalidation too complex
❌ Database not under load yet
❌ Every user has unique data (caching won't help)

---

## 3. Load Balancer

### Problem It Solves

**Symptom:** Single server can't handle traffic, need to scale horizontally
**Root cause:** All requests going to one server

### How It Works

**Without load balancer:**

```
┌─────────┐
│ Client  │
└────┬────┘
     │ All requests
     ▼
┌─────────────┐
│  Server     │  ← Overloaded at 1000 req/sec
│ (Max 1000   │
│  req/sec)   │
└─────────────┘
```

**With load balancer:**

```
┌─────────┐
│ Client  │
└────┬────┘
     │
     ▼
┌─────────────────┐
│ Load Balancer   │  Distributes requests
└────┬───────┬────┘
     │       │
     ▼       ▼
┌─────┐   ┌─────┐
│Srv 1│   │Srv 2│  ← Each handles 500 req/sec
│Max  │   │Max  │     Total: 1000 req/sec
│500  │   │500  │
└─────┘   └─────┘
```

**Can handle 2x traffic with 2 servers.**

### Load Balancing Strategies

#### Round Robin (simplest)

```
Request 1 → Server 1
Request 2 → Server 2
Request 3 → Server 1
Request 4 → Server 2
...
```

**Pros:** Simple, even distribution
**Cons:** Doesn't account for server load

#### Least Connections

```
Server 1: 10 active connections
Server 2: 5 active connections
New request → Server 2 (fewer connections)
```

**Pros:** Better for long-running requests
**Cons:** More complex

#### IP Hash (sticky sessions)

```
Client IP: 192.168.1.100
Hash(192.168.1.100) % 2 = 0 → Always Server 1
```

**Pros:** Same user always hits same server (for session consistency)
**Cons:** Uneven distribution if users not evenly distributed

### Implementation

#### nginx as Load Balancer

```nginx
# nginx.conf
upstream api_servers {
    server 10.0.0.1:3000;  # Server 1
    server 10.0.0.2:3000;  # Server 2
    server 10.0.0.3:3000;  # Server 3
}

server {
    listen 80;

    location /api {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Requests to /api are distributed across 3 servers.**

#### Health Checks

```nginx
upstream api_servers {
    server 10.0.0.1:3000 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:3000 max_fails=3 fail_timeout=30s;

    # If server fails 3 times, marked as down for 30 seconds
}
```

**If server crashes, load balancer stops sending requests to it.**

### Required: Stateless Servers

**Problem with stateful servers:**

```
User logs in → Server 1 (session in memory)
Next request → Server 2 (user appears logged out!)
```

**Solutions:**

#### 1. Shared session storage (Redis)

```javascript
// Store sessions in Redis, not server memory
const session = require('express-session');
const RedisStore = require('connect-redis')(session);

app.use(session({
    store: new RedisStore({ client: redisClient }),
    secret: 'secret',
    resave: false,
    saveUninitialized: false
}));

// Now sessions work across all servers
```

#### 2. Token-based auth (JWT)

```javascript
// Login returns token
const token = jwt.sign({ userId: 123 }, 'secret');
res.json({ token });

// Client sends token with every request
// Any server can verify token (stateless!)
```

### When to Use

✅ Single server maxing out (CPU > 80%)
✅ Need high availability (if one server dies, others handle traffic)
✅ Want zero-downtime deployments (deploy one server at a time)
✅ Traffic exceeds single server capacity

### When NOT to Use

❌ Single server handling load fine
❌ Adds complexity (now managing multiple servers)
❌ Database is the bottleneck (more app servers won't help)

---

## 4. Database Indexing

### Problem It Solves

**Symptom:** Queries getting slower as table grows
**Root cause:** Database scanning entire table to find rows

### How It Works

**Without index:**

```sql
-- Find user by email
SELECT * FROM users WHERE email = 'user@example.com';

-- Database must check EVERY row:
Row 1: email = 'alice@example.com'? No
Row 2: email = 'bob@example.com'? No
Row 3: email = 'carol@example.com'? No
...
Row 47,293: email = 'user@example.com'? YES!

Time: O(n) - linear scan
1M rows = ~500ms
```

**With index:**

```sql
-- Create index
CREATE INDEX idx_users_email ON users(email);

-- Now query uses index (B-tree):
SELECT * FROM users WHERE email = 'user@example.com';

-- Database uses index (like a book index):
Check index B-tree → Points to row 47,293
Fetch row 47,293 directly

Time: O(log n) - binary search
1M rows = ~5ms (100x faster!)
```

### Index Types

#### Single Column Index

```sql
CREATE INDEX idx_users_email ON users(email);

-- Fast queries:
SELECT * FROM users WHERE email = 'user@example.com';
```

#### Composite Index (multiple columns)

```sql
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- Fast queries:
SELECT * FROM orders WHERE user_id = 123 ORDER BY created_at DESC;
SELECT * FROM orders WHERE user_id = 123 AND created_at > '2025-01-01';

-- Doesn't help:
SELECT * FROM orders WHERE created_at > '2025-01-01'; -- user_id not in WHERE
```

**Index column order matters!**

#### Unique Index

```sql
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Ensures no duplicate usernames
-- Also speeds up lookups
```

### When to Add Indexes

**✅ Add index when:**

- Column used in WHERE clause frequently
- Column used in JOIN condition
- Column used in ORDER BY
- Query doing full table scan (check with EXPLAIN)

**Example:**

```sql
-- Slow query (found via monitoring)
SELECT * FROM orders
WHERE user_id = 123
  AND status = 'pending'
ORDER BY created_at DESC;

-- Check execution plan
EXPLAIN SELECT * FROM orders WHERE user_id = 123 AND status = 'pending';

-- Result: "Seq Scan on orders" (bad! scanning entire table)

-- Add index
CREATE INDEX idx_orders_user_status_date ON orders(user_id, status, created_at);

-- Check again
EXPLAIN SELECT * FROM orders WHERE user_id = 123 AND status = 'pending';

-- Result: "Index Scan using idx_orders_user_status_date" (good!)
```

### Index Costs

**Indexes are not free:**

1. **Storage:** Index takes disk space
   ```
   1M row table: ~500 MB
   Index on 2 columns: ~100 MB
   3 indexes: ~300 MB extra storage
   ```

2. **Write performance:** Every INSERT/UPDATE must update indexes
   ```
   No indexes: INSERT takes 5ms
   3 indexes: INSERT takes 15ms (3x slower writes)
   ```

3. **Maintenance:** Indexes need periodic rebuilding

**Trade-off:** Faster reads, slower writes.

### When to Use

✅ Read-heavy tables (more SELECTs than INSERTs)
✅ WHERE clauses on same columns repeatedly
✅ Queries getting slower as table grows
✅ Identified slow queries via profiling

### When NOT to Use

❌ Write-heavy tables (more INSERTs than SELECTs)
❌ Small tables (< 10K rows - full scan is fast enough)
❌ Column values not selective (e.g., `status` with only 2 values)
❌ Indexing every column "just in case" (wasteful)

---

## 5. Read Replicas

### Problem It Solves

**Symptom:** Database CPU maxed out, mostly from read queries
**Root cause:** Too many SELECT queries hitting single database

### How It Works

```
┌──────────────┐
│   Primary    │  ← Handles all writes (INSERT, UPDATE, DELETE)
│  Database    │
└──────┬───────┘
       │ Replicates data
       ├─────────┬─────────┐
       ▼         ▼         ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │Replica1│ │Replica2│ │Replica3│  ← Handle reads (SELECT)
   └────────┘ └────────┘ └────────┘
```

**Writes:** Go to primary
**Reads:** Distributed across replicas

### Implementation

```javascript
const { Pool } = require('pg');

// Primary database (for writes)
const primary = new Pool({
    host: 'primary.db.example.com',
    database: 'myapp',
    max: 20
});

// Read replica (for reads)
const replica = new Pool({
    host: 'replica.db.example.com',
    database: 'myapp',
    max: 50 // More connections for read-heavy load
});

// Write operations → Primary
app.post('/api/products', async (req, res) => {
    const result = await primary.query(
        'INSERT INTO products (name, price) VALUES ($1, $2)',
        [req.body.name, req.body.price]
    );
    res.json(result.rows[0]);
});

// Read operations → Replica
app.get('/api/products', async (req, res) => {
    const result = await replica.query('SELECT * FROM products');
    res.json(result.rows);
});
```

### Replication Lag

**Problem:** Replica is slightly behind primary (usually < 1 second)

```
Time 0: User creates product → Write to primary
Time 1ms: User fetches products → Read from replica
          Replica hasn't replicated yet → New product not visible!
```

**Solutions:**

#### 1. Read-after-write from primary

```javascript
app.post('/api/products', async (req, res) => {
    // Write to primary
    await primary.query('INSERT INTO products ...');

    // Read from primary immediately after write
    const result = await primary.query('SELECT * FROM products WHERE id = $1', [id]);
    res.json(result.rows[0]);
});
```

#### 2. Accept eventual consistency

```javascript
// For less critical data, accept that replica might be slightly behind
app.get('/api/products', async (req, res) => {
    // This might not show products created < 1 second ago
    const result = await replica.query('SELECT * FROM products');
    res.json(result.rows);
});
```

### When to Use

✅ Read-heavy workload (90%+ reads, <10% writes)
✅ Primary database CPU high from read queries
✅ Can tolerate slight replication lag
✅ Want to scale read capacity

### When NOT to Use

❌ Write-heavy workload (replicas don't help with writes)
❌ Need strong consistency (replicas have lag)
❌ Database not under load yet
❌ Cost-sensitive (replicas cost same as primary)

---

## 6. Connection Pooling

### Problem It Solves

**Symptom:** "Too many connections" database errors
**Root cause:** Creating new database connection for every request

### How It Works

**Without pooling:**

```javascript
app.get('/api/products', async (req, res) => {
    // Create new connection (expensive: ~50ms)
    const db = await createConnection();

    // Execute query (fast: 5ms)
    const products = await db.query('SELECT * FROM products');

    // Close connection
    await db.close();

    res.json(products);
});

// 100 requests = 100 connections created/destroyed
```

**With pooling:**

```javascript
// Create pool once at startup
const pool = new Pool({
    max: 20, // Maximum 20 connections
    min: 5,  // Keep 5 connections always open
    idleTimeoutMillis: 30000
});

app.get('/api/products', async (req, res) => {
    // Get connection from pool (fast: <1ms)
    const client = await pool.connect();

    // Execute query
    const products = await client.query('SELECT * FROM products');

    // Return connection to pool (not destroyed!)
    client.release();

    res.json(products);
});

// 100 requests = reuse same 20 connections
```

### Performance Impact

```
Without pooling:
- Connection creation: 50ms
- Query execution: 5ms
- Total per request: 55ms

With pooling:
- Get from pool: <1ms
- Query execution: 5ms
- Total per request: 6ms

9x faster!
```

### When to Use

✅ Always use connection pooling
✅ Default choice for database access
✅ Required for production applications

### When NOT to Use

❌ Never don't use it (always use connection pooling)

---

## 7. Message Queues

### Problem It Solves

**Symptom:** Slow API responses for long-running tasks
**Root cause:** Blocking request while processing heavy work

### Example Problem

```javascript
// Slow endpoint (blocks for 30 seconds)
app.post('/api/export-report', async (req, res) => {
    // 1. Query large dataset (5 seconds)
    const data = await db.query('SELECT * FROM orders WHERE ...');

    // 2. Generate PDF (20 seconds)
    const pdf = await generatePDF(data);

    // 3. Upload to S3 (5 seconds)
    await s3.upload(pdf);

    // 30 seconds later...
    res.json({ url: 'https://s3.../report.pdf' });
});

// User waits 30 seconds (bad UX!)
```

### Solution: Queue the Work

```javascript
const Bull = require('bull');
const reportQueue = new Bull('report-generation');

// Endpoint returns immediately
app.post('/api/export-report', async (req, res) => {
    // Add job to queue
    const job = await reportQueue.add({
        userId: req.userId,
        params: req.body
    });

    // Return immediately (< 100ms)
    res.json({
        jobId: job.id,
        status: 'processing',
        message: 'Report is being generated. Check back in a few minutes.'
    });
});

// Worker processes jobs in background
reportQueue.process(async (job) => {
    const { userId, params } = job.data;

    // Do the heavy work (30 seconds)
    const data = await db.query('SELECT * FROM orders WHERE ...');
    const pdf = await generatePDF(data);
    const url = await s3.upload(pdf);

    // Notify user (email, websocket, etc.)
    await notifyUser(userId, url);
});
```

**User experience:**

```
Before: Click "Export" → Wait 30 seconds → Get file
After: Click "Export" → Get "Processing" response instantly → Email when ready
```

### Use Cases

**✅ Perfect for message queues:**

- Email sending
- Image processing (resize, thumbnails)
- Video transcoding
- Report generation
- Data imports/exports
- Webhooks to third-party APIs

**❌ Not suitable:**

- Real-time data (use WebSockets/MQTT)
- Immediate responses required
- Simple, fast operations (< 1 second)

### When to Use

✅ Long-running tasks (> 5 seconds)
✅ Tasks that can be asynchronous
✅ Need retry logic for failed tasks
✅ Want to limit concurrency (process N jobs at a time)

### When NOT to Use

❌ Tasks that must complete before response
❌ Simple, fast operations
❌ Adds complexity for no benefit

---

## Summary: Decision Matrix

| Problem | Solution | When to Use | Cost |
|---------|----------|-------------|------|
| **Geographic latency** | CDN | Global userbase | $0-200/month |
| **Repeated DB queries** | Caching (Redis) | High read/write ratio | $15-500/month |
| **Single server overload** | Load Balancer | CPU > 80% consistently | $10-50/month + servers |
| **Slow queries** | Database Indexing | Queries > 100ms | Free (but slower writes) |
| **DB read overload** | Read Replicas | 90%+ reads, primary CPU high | $50-500/month |
| **Too many connections** | Connection Pooling | Always | Free |
| **Slow async tasks** | Message Queue | Tasks > 5 seconds | $0-100/month |

---

## The Right Order to Add Tools

**Don't add everything at once!** Add tools as specific problems emerge:

### Stage 1: Single Server (< 1K users)
- ✅ Connection pooling (always)
- ✅ Basic monitoring
- ⏸️ Everything else can wait

### Stage 2: Growing (1K-10K users)
- ✅ Database indexing (identify slow queries)
- ✅ CDN for static files
- ⏸️ Hold off on load balancer

### Stage 3: Medium Scale (10K-50K users)
- ✅ Redis caching
- ✅ Message queue for async tasks
- ✅ Consider read replica if DB CPU high

### Stage 4: Large Scale (50K+ users)
- ✅ Load balancer + multiple app servers
- ✅ Read replicas
- ✅ Full caching strategy

---

**Next:** [Part 5: Modern Shortcuts →](./05-modern-shortcuts.md)

What if you don't want to manage all this yourself? Enter BaaS and JAMstack...
