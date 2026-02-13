# Part 5: Modern Shortcuts (BaaS & JAMstack)

## The Premise

**From Part 4, you learned you might need:**
- Database with connection pooling and indexing
- CDN for static files
- Caching layer (Redis)
- Load balancer
- Read replicas
- Message queue
- Authentication system
- File storage

**Question:** What if someone else managed all of this for you?

**Answer:** That's BaaS (Backend-as-a-Service) and JAMstack.

---

## Backend-as-a-Service (BaaS)

### What It Is

**BaaS provides a complete backend through APIs:**
- Database (with auto-scaling)
- Authentication
- File storage
- Real-time subscriptions
- Edge functions (serverless)
- All the scalability tools from Part 4, managed for you

**You focus on:** Frontend code
**They handle:** Everything else

### Major Players

1. **Firebase (Google)**
   - NoSQL database (Firestore)
   - Real-time sync
   - Authentication (Google, Facebook, etc.)
   - File storage
   - Hosting

2. **Supabase (Open Source)**
   - PostgreSQL database
   - Real-time subscriptions
   - Authentication
   - File storage
   - Row Level Security (RLS)

3. **AWS Amplify (Amazon)**
   - Multiple AWS services packaged together
   - GraphQL API
   - Authentication
   - Storage

---

## Supabase Deep Dive

### Architecture

```
Your Frontend (React, Next.js, Mobile)
         ↓
Supabase Client Library
         ↓
┌────────────────────────────────┐
│        Supabase Cloud          │
│                                │
│  ┌──────────────────────────┐ │
│  │  PostgreSQL Database     │ │  ← Auto-scaling, connection pooling
│  │  + Row Level Security    │ │
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │  Authentication (JWT)    │ │  ← Social logins, magic links
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │  File Storage            │ │  ← CDN included
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │  Real-time (WebSocket)   │ │  ← Database changes → push
│  └──────────────────────────┘ │
│                                │
│  ┌──────────────────────────┐ │
│  │  Edge Functions (Deno)   │ │  ← Custom serverless functions
│  └──────────────────────────┘ │
└────────────────────────────────┘
```

**You get:** All the tools from Part 4, pre-configured.

### Code Comparison: Traditional vs Supabase

#### Traditional Stack: User Authentication

**Backend (Node.js):**

```javascript
// Database schema
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

// Registration endpoint
app.post('/api/register', async (req, res) => {
    const { email, password } = req.body;

    // Hash password
    const hash = await bcrypt.hash(password, 10);

    // Insert user
    await db.query(
        'INSERT INTO users (email, password_hash) VALUES ($1, $2)',
        [email, hash]
    );

    res.json({ success: true });
});

// Login endpoint
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;

    // Find user
    const user = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    if (!user) return res.status(401).json({ error: 'Invalid credentials' });

    // Verify password
    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) return res.status(401).json({ error: 'Invalid credentials' });

    // Generate JWT
    const token = jwt.sign({ userId: user.id }, 'secret', { expiresIn: '7d' });

    res.json({ token });
});

// Protected endpoint
app.get('/api/profile', authenticateToken, async (req, res) => {
    const user = await db.query('SELECT * FROM users WHERE id = $1', [req.userId]);
    res.json(user);
});
```

**Frontend (React):**

```javascript
// Registration
async function register(email, password) {
    const res = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    return res.json();
}

// Login
async function login(email, password) {
    const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    const { token } = await res.json();
    localStorage.setItem('token', token);
}

// Fetch profile
async function getProfile() {
    const token = localStorage.getItem('token');
    const res = await fetch('/api/profile', {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    return res.json();
}
```

**Lines of code:** ~100 (backend) + ~30 (frontend) = **130 lines**

---

#### Supabase: Same Functionality

**Backend:** None needed (Supabase handles it)

**Frontend (React):**

```javascript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient('https://your-project.supabase.co', 'your-anon-key');

// Registration
async function register(email, password) {
    const { data, error } = await supabase.auth.signUp({ email, password });
    return { data, error };
}

// Login
async function login(email, password) {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    return { data, error };
}

// Fetch profile (automatic auth)
async function getProfile() {
    const { data: { user } } = await supabase.auth.getUser();
    return user;
}
```

**Lines of code:** **~15 lines** (87% less code!)

**What you get for free:**
- Password hashing (bcrypt)
- JWT generation and verification
- Token refresh
- Email verification
- Password reset
- Social logins (Google, GitHub, etc.)
- Magic links (passwordless)

---

### Database: Row Level Security (RLS)

**Traditional approach:**

```javascript
// Must check authorization in every endpoint
app.get('/api/orders', authenticateToken, async (req, res) => {
    // Manual auth check
    const orders = await db.query(
        'SELECT * FROM orders WHERE user_id = $1',
        [req.userId] // ← Must remember to filter by user!
    );
    res.json(orders);
});

// Easy to forget and leak data:
app.get('/api/orders/:id', authenticateToken, async (req, res) => {
    // BUG: Forgot to check if order belongs to user!
    const order = await db.query('SELECT * FROM orders WHERE id = $1', [req.params.id]);
    res.json(order); // ← Other users can see this order!
});
```

**Supabase approach (Row Level Security):**

```sql
-- Define security policy at DATABASE level
CREATE POLICY "Users can only see their own orders"
ON orders
FOR SELECT
USING (auth.uid() = user_id);

-- Now ANY query automatically filters:
-- SELECT * FROM orders → Only returns current user's orders
-- SELECT * FROM orders WHERE id = 123 → Returns only if belongs to current user
```

**Frontend code:**

```javascript
// Query ALL orders - RLS automatically filters to current user's
const { data: orders } = await supabase
    .from('orders')
    .select('*');

// Try to access other user's order - RLS blocks it
const { data: order } = await supabase
    .from('orders')
    .select('*')
    .eq('id', 123)
    .single();
// Returns null if order doesn't belong to current user
```

**Security enforced at database level** → Can't accidentally leak data

---

### Real-time: Database Changes → Live Updates

**Traditional approach (polling):**

```javascript
// Poll for new orders every 5 seconds
setInterval(async () => {
    const orders = await fetch('/api/orders').then(r => r.json());
    setOrders(orders);
}, 5000);

// Wasteful: Sends request even if nothing changed
```

**Supabase approach (WebSocket subscriptions):**

```javascript
// Subscribe to real-time changes
const subscription = supabase
    .channel('orders')
    .on('postgres_changes', {
        event: 'INSERT',
        schema: 'public',
        table: 'orders'
    }, (payload) => {
        // New order inserted → receive instantly
        setOrders(prev => [...prev, payload.new]);
    })
    .subscribe();

// No polling! Updates pushed from database.
```

**How it works (under the hood):**

```
PostgreSQL → Logical Replication Slot → Realtime Server → WebSocket → Client

Any INSERT/UPDATE/DELETE → Broadcasted to subscribers
```

---

### File Storage

**Traditional approach:**

```javascript
const multer = require('multer');
const AWS = require('aws-sdk');
const s3 = new AWS.S3();

// Configure multer for file upload
const upload = multer({ dest: '/tmp' });

app.post('/api/upload', upload.single('file'), async (req, res) => {
    // Upload to S3
    const result = await s3.upload({
        Bucket: 'my-bucket',
        Key: req.file.originalname,
        Body: fs.createReadStream(req.file.path),
        ACL: 'public-read'
    }).promise();

    res.json({ url: result.Location });
});
```

**Supabase approach:**

```javascript
// Upload file
const { data, error } = await supabase.storage
    .from('avatars')
    .upload('public/avatar.png', file);

// Get public URL (CDN included)
const { data: { publicUrl } } = supabase.storage
    .from('avatars')
    .getPublicUrl('public/avatar.png');
```

**Includes:**
- CDN for fast delivery
- Image transformations (resize, crop)
- Access control policies

---

## JAMstack Architecture

### What It Is

**JAM = JavaScript + APIs + Markup**

**Traditional:**
```
User request → Server generates HTML → Send to browser
```

**JAMstack:**
```
Build time: Generate all HTML (static site)
User request → CDN serves static HTML (instant)
Dynamic data: Fetch from APIs (Supabase, etc.)
```

### Example: Blog

**Traditional WordPress:**

```
User visits /blog/my-post
    ↓
PHP server:
1. Connects to MySQL
2. Queries for post
3. Queries for comments
4. Queries for sidebar widgets
5. Generates HTML
6. Sends to user (500ms)
```

**JAMstack with Next.js + Supabase:**

```
Build time (once, when you publish):
- Fetch all blog posts
- Generate HTML for each
- Deploy to CDN

User visits /blog/my-post
    ↓
CDN serves pre-generated HTML (50ms)
    ↓
Browser fetches comments from Supabase API (100ms)
    ↓
Total: 150ms (3x faster)
```

### Implementation

```javascript
// pages/blog/[slug].js (Next.js)

// Generate static pages at build time
export async function getStaticProps({ params }) {
    // Fetch post from Supabase
    const { data: post } = await supabase
        .from('posts')
        .select('*')
        .eq('slug', params.slug)
        .single();

    return {
        props: { post },
        revalidate: 60 // Regenerate every 60 seconds
    };
}

export async function getStaticPaths() {
    // Get all post slugs
    const { data: posts } = await supabase.from('posts').select('slug');

    return {
        paths: posts.map(p => ({ params: { slug: p.slug } })),
        fallback: 'blocking'
    };
}

function BlogPost({ post }) {
    const [comments, setComments] = useState([]);

    // Fetch dynamic data (comments) on client
    useEffect(() => {
        supabase
            .from('comments')
            .select('*')
            .eq('post_id', post.id)
            .then(({ data }) => setComments(data));
    }, [post.id]);

    return (
        <div>
            <h1>{post.title}</h1>
            <div>{post.content}</div>

            <h2>Comments</h2>
            {comments.map(c => <Comment key={c.id} {...c} />)}
        </div>
    );
}
```

**Result:**
- Post content: Static HTML (fast, SEO-friendly)
- Comments: Dynamic (real-time updates possible)

---

## Platform-as-a-Service (PaaS)

### Vercel / Netlify

**What they provide:**

1. **Automatic deployments** from Git
   ```bash
   git push origin main
   # → Automatic build and deploy
   # → Live in ~2 minutes
   ```

2. **Global CDN** (free)
   - Your site served from 100+ edge locations
   - No configuration needed

3. **Serverless functions**
   ```javascript
   // api/hello.js
   export default function handler(req, res) {
       res.json({ message: 'Hello World' });
   }
   // Deployed as: https://yoursite.com/api/hello
   ```

4. **Preview deployments**
   - Every PR gets its own URL
   - Test before merging

5. **Edge functions** (runs code close to user)
   ```javascript
   // middleware.js
   export function middleware(req) {
       // Runs at CDN edge, before request hits origin
       if (req.geo.country === 'US') {
           return NextResponse.redirect('/us-version');
       }
   }
   ```

---

## Cost Comparison: DIY vs BaaS

### Scenario: Medium-sized app (10K daily users)

#### DIY (Self-hosted)

```
VPS (2 servers + load balancer): $150/month
Database (managed Postgres): $50/month
Redis (managed): $30/month
CDN (Cloudflare Pro): $20/month
S3 storage: $10/month
Monitoring (DataDog): $50/month
SSL certificates: Free (Let's Encrypt)
Time to manage: 10 hours/month @ $100/hour = $1,000/month
---
Total: $1,310/month
```

#### BaaS (Supabase + Vercel)

```
Supabase Pro: $25/month
  - Database (8GB, auto-scaling)
  - Auth
  - Storage (100GB)
  - Real-time
Vercel Pro: $20/month
  - Hosting
  - CDN
  - Serverless functions
Time to manage: 1 hour/month @ $100/hour = $100/month
---
Total: $145/month
```

**Savings: $1,165/month (89% cheaper)**

---

### Scenario: Large app (100K daily users)

#### DIY

```
Servers (10x with load balancing): $1,000/month
Database cluster (primary + 2 replicas): $500/month
Redis cluster: $200/month
CDN: $200/month
S3: $100/month
Monitoring + logging: $200/month
DevOps engineer (part-time): $3,000/month
---
Total: $5,200/month
```

#### BaaS

```
Supabase Pro + Add-ons: $500/month
  - Larger database
  - More bandwidth
Vercel Enterprise: $500/month
---
Total: $1,000/month
```

**Savings: $4,200/month (81% cheaper)**

---

## What You Give Up with BaaS

### 1. Vendor Lock-in

**Supabase:** Less severe (it's Postgres - can export)
**Firebase:** Severe (Firestore is proprietary)

**Migration difficulty:**
```
Supabase → Self-hosted Postgres: Easy (pg_dump)
Firebase → Anything else: Hard (custom export scripts)
```

### 2. Customization Limits

**Can't:**
- Use specific Postgres extensions (only what Supabase enables)
- Run custom database migrations in production
- SSH into servers (it's managed)
- Fine-tune server configuration

**Workaround:** Edge Functions for custom logic

### 3. Cost at Hyperscale

**At 1M+ daily users:**

```
Supabase Enterprise: $2,000-10,000/month (estimate)
DIY with dedicated team: $15,000/month (servers + staff)
```

**At some scale, DIY becomes cheaper** (but requires team)

### 4. Performance Ceiling

**Can't:**
- Use multiple database clusters
- Implement custom caching strategies
- Use specialized databases (graph DB, time-series DB)

**For 99% of apps, this doesn't matter.**

---

## When to Use BaaS

### ✅ Use BaaS When:

1. **Startup / MVP** - Need to move fast
2. **Small team** (< 10 developers)
3. **Standard use case** (CRUD app, not specialized)
4. **Want to focus on product, not infrastructure**
5. **Cost-conscious** (until hyperscale)

### ❌ Build Custom When:

1. **Very specialized needs** (custom protocols like MQTT broker)
2. **Large team** that can manage infrastructure
3. **Need specific tech** not offered by BaaS (e.g., Kafka, Cassandra)
4. **Hyperscale** where DIY is cheaper
5. **Regulatory requirements** (must run in specific data center)

---

## Hybrid Approach: Best of Both Worlds

**This is what you do with Raptor:**

```
Frontend: Vercel (Next.js)
    ↓
Database + Auth: Supabase
    ↓
Custom Backend Services: Self-hosted (Go + MQTT)
    ↓
Hardware: Raptor devices
```

**Why hybrid?**

- ✅ Supabase for standard CRUD (products, users, orders)
- ✅ Custom Go service for MQTT (Supabase doesn't do MQTT)
- ✅ Vercel for frontend (fast, easy deploys)

**Use BaaS for 80% of your stack, custom for specialized 20%.**

---

## Example: Todo App

### Full Stack (Traditional)

**Backend:**
- Node.js/Express server
- PostgreSQL database
- JWT auth
- WebSocket server for real-time

**Lines of code:** ~500

---

### JAMstack + BaaS (Modern)

**Backend:** None (Supabase)

**Frontend:**

```javascript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(URL, KEY);

// Login
const { data } = await supabase.auth.signInWithPassword({ email, password });

// Fetch todos
const { data: todos } = await supabase.from('todos').select('*');

// Add todo
await supabase.from('todos').insert({ title: 'New todo' });

// Real-time updates
supabase
    .channel('todos')
    .on('postgres_changes', { event: '*', schema: 'public', table: 'todos' }, (payload) => {
        console.log('Change received!', payload);
    })
    .subscribe();
```

**Lines of code:** ~50 (90% less!)

**Database schema:**

```sql
CREATE TABLE todos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id),
    title TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Row Level Security
ALTER TABLE todos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can only access their own todos"
ON todos
FOR ALL
USING (auth.uid() = user_id);
```

**Done.** Auth, real-time, database, all working.

---

## Summary

### Modern Stack = BaaS + JAMstack + PaaS

**BaaS (Supabase):** Backend without code
**JAMstack (Next.js):** Pre-rendered static sites + APIs
**PaaS (Vercel):** Deploy without DevOps

### What You Get

✅ 90% less code to maintain
✅ Auto-scaling (handle traffic spikes)
✅ Global CDN (fast worldwide)
✅ Security best practices (out of the box)
✅ Lower cost (until hyperscale)
✅ Focus on product, not infrastructure

### What You Give Up

❌ Some customization
❌ Vendor lock-in risk
❌ Can't use specialized tech
❌ May hit performance ceiling (at extreme scale)

### The Decision

**For most apps:** Modern stack is the right choice.
**For specialized needs:** Hybrid (BaaS + custom services).
**For tech giants:** Full custom (you have the team and scale).

---

**Next:** [Part 6: Hyperscale Reality →](./06-hyperscale.md)

When BaaS isn't enough, how do tech giants build for millions of concurrent users?
