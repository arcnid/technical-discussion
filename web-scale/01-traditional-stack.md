# Part 1: The Traditional Stack

## The LAMP Architecture

**L**inux + **A**pache + **M**ySQL + **P**HP

This is where most web applications started, and where many still run successfully today. There's nothing wrong with this architecture for many use cases.

```
┌──────────┐
│  Client  │
│ (Browser)│
└────┬─────┘
     │ HTTP Request
     ▼
┌─────────────────┐
│   Web Server    │
│   (Apache)      │
│                 │
│  ┌───────────┐  │
│  │    PHP    │  │
│  │  Engine   │  │
│  └─────┬─────┘  │
│        │        │
└────────┼────────┘
         │ SQL Query
         ▼
┌─────────────────┐
│     MySQL       │
│   Database      │
└─────────────────┘
```

**Everything happens on the server.** The client (browser) is just a display device.

---

## How It Works

### Request Lifecycle

1. **User clicks a link** or submits a form
2. **Browser sends HTTP request** to server
3. **Apache receives request**, routes to PHP file
4. **PHP executes:**
   - Connects to database
   - Queries for data
   - Processes business logic
   - Generates HTML
5. **Server sends complete HTML page** back to browser
6. **Browser displays the page**
7. **Connection closes**

Every action requires this full cycle.

---

## Code Example: Simple E-Commerce Product List

### Database Schema

```sql
CREATE TABLE products (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  price DECIMAL(10, 2) NOT NULL,
  description TEXT,
  stock INT DEFAULT 0
);

CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL
);

CREATE TABLE sessions (
  session_id VARCHAR(128) PRIMARY KEY,
  user_id INT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Login Page (login.php)

```php
<?php
session_start();

// Check if already logged in
if (isset($_SESSION['user_id'])) {
    header('Location: products.php');
    exit;
}

// Handle form submission
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // Connect to database
    $db = new PDO('mysql:host=localhost;dbname=shop', 'root', 'password');

    // Query user
    $stmt = $db->prepare('SELECT id, password_hash FROM users WHERE username = ?');
    $stmt->execute([$username]);
    $user = $stmt->fetch();

    // Verify password
    if ($user && password_verify($password, $user['password_hash'])) {
        // Create session
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['username'] = $username;

        header('Location: products.php');
        exit;
    } else {
        $error = 'Invalid username or password';
    }
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Login</title>
</head>
<body>
    <h1>Login</h1>

    <?php if (isset($error)): ?>
        <p style="color: red;"><?= htmlspecialchars($error) ?></p>
    <?php endif; ?>

    <form method="POST">
        <label>Username: <input type="text" name="username" required></label><br>
        <label>Password: <input type="password" name="password" required></label><br>
        <button type="submit">Login</button>
    </form>

    <p><a href="register.php">Create an account</a></p>
</body>
</html>
```

### Product List Page (products.php)

```php
<?php
session_start();

// Require authentication
if (!isset($_SESSION['user_id'])) {
    header('Location: login.php');
    exit;
}

// Connect to database
$db = new PDO('mysql:host=localhost;dbname=shop', 'root', 'password');

// Get filter from query string
$filter = $_GET['filter'] ?? 'all';

// Build query based on filter
if ($filter === 'in_stock') {
    $stmt = $db->query('SELECT * FROM products WHERE stock > 0 ORDER BY name');
} else {
    $stmt = $db->query('SELECT * FROM products ORDER BY name');
}

$products = $stmt->fetchAll();
?>

<!DOCTYPE html>
<html>
<head>
    <title>Products</title>
    <style>
        .product { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        .out-of-stock { background: #fee; }
    </style>
</head>
<body>
    <header>
        <h1>Products</h1>
        <p>Welcome, <?= htmlspecialchars($_SESSION['username']) ?>!
           <a href="logout.php">Logout</a>
        </p>
    </header>

    <nav>
        <a href="?filter=all">All Products</a> |
        <a href="?filter=in_stock">In Stock Only</a>
    </nav>

    <main>
        <?php foreach ($products as $product): ?>
            <div class="product <?= $product['stock'] === 0 ? 'out-of-stock' : '' ?>">
                <h2><?= htmlspecialchars($product['name']) ?></h2>
                <p>Price: $<?= number_format($product['price'], 2) ?></p>
                <p><?= htmlspecialchars($product['description']) ?></p>
                <p>
                    <?php if ($product['stock'] > 0): ?>
                        Stock: <?= $product['stock'] ?> available
                        <form method="POST" action="add_to_cart.php" style="display:inline;">
                            <input type="hidden" name="product_id" value="<?= $product['id'] ?>">
                            <button type="submit">Add to Cart</button>
                        </form>
                    <?php else: ?>
                        <strong>OUT OF STOCK</strong>
                    <?php endif; ?>
                </p>
            </div>
        <?php endforeach; ?>
    </main>
</body>
</html>
```

### Add to Cart (add_to_cart.php)

```php
<?php
session_start();

if (!isset($_SESSION['user_id'])) {
    header('Location: login.php');
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $productId = (int)$_POST['product_id'];

    // Initialize cart in session if it doesn't exist
    if (!isset($_SESSION['cart'])) {
        $_SESSION['cart'] = [];
    }

    // Add or increment quantity
    if (isset($_SESSION['cart'][$productId])) {
        $_SESSION['cart'][$productId]++;
    } else {
        $_SESSION['cart'][$productId] = 1;
    }

    // Redirect back to products
    header('Location: products.php');
    exit;
}
?>
```

---

## What's Happening Here

### Server Does Everything

1. **Authentication:** Session checks on every page
2. **Authorization:** Redirect if not logged in
3. **Data Fetching:** SQL queries on every request
4. **Business Logic:** Filter logic, cart management
5. **HTML Generation:** Complete page markup
6. **State Management:** Sessions stored server-side

### Request Flow for Viewing Products

```
1. GET /products.php
2. Apache routes to PHP
3. PHP starts session (reads from disk or database)
4. PHP checks authentication
5. PHP connects to MySQL
6. PHP executes SELECT query
7. MySQL returns rows
8. PHP loops through results, building HTML string
9. PHP sends complete HTML (20-50 KB)
10. Browser receives and renders
11. Connection closes
```

**Time:** ~100-300ms on localhost, ~500ms+ over network

---

## Why This Works

### For Small to Medium Applications

**Simplicity:**
- One codebase, one deployment
- No API layer needed
- No build step
- Direct database access

**Developer Experience:**
- Easy to reason about (request → process → response)
- Familiar mental model
- Quick to build CRUD apps
- Tools are mature (phpMyAdmin, etc.)

**Cost:**
- One server handles everything
- Shared hosting is cheap ($5-20/month)

**Performance (at small scale):**
- 100-500 concurrent users: works fine
- Database on same server: fast queries
- No network hops between services

---

## Where It Starts to Break

### Problem 1: Server CPU Bottleneck

Every request regenerates HTML, even if data hasn't changed.

```
10 users viewing products page:
- 10 separate PHP processes
- 10 database queries
- 10 HTML generation cycles
- ~300ms each

100 users:
- 100 simultaneous processes
- Server CPU maxes out
- Requests queue up
- Response time degrades
```

### Problem 2: Database Connections

```php
// Each request creates a new connection
$db = new PDO('mysql:host=localhost;dbname=shop', 'root', 'password');
```

**MySQL default:** 151 max connections

At 200 concurrent requests, you'll get:
```
SQLSTATE[HY000] [1040] Too many connections
```

### Problem 3: Session Management at Scale

Sessions stored on disk:
```
/tmp/sess_a3f2c1b4d5...
/tmp/sess_z9x8c7v6b5...
```

**Problems:**
- Disk I/O becomes bottleneck
- Can't load balance (sessions tied to one server)
- Session files pile up

### Problem 4: No Interactivity Without Reload

Want to update product stock in real-time? **Can't.**

User adds item to cart, sees "15 in stock". Another user buys 10. First user tries to checkout... error.

Only solution: Refresh the page.

---

## Performance Characteristics

### Single Server LAMP Stack

| Metric | Value |
|--------|-------|
| Max concurrent connections | ~500 (with tuning) |
| Request latency | 100-500ms |
| Requests per second | ~200-500 |
| Database queries per request | 1-10 |
| HTML generation per request | ~10-50 KB |

### Scaling Up (Vertical Scaling)

You can get a bigger server:
- More CPU cores
- More RAM
- Faster disk

**Limits:**
- Expensive (exponential cost increase)
- Eventually hits physical limits
- Single point of failure

### What Happens at Scale

**1,000 concurrent users:**
- Server struggles
- Response time: 2-5 seconds
- Some requests timeout

**5,000 concurrent users:**
- Server crashes
- Database overloaded
- Site goes down

---

## The .NET Equivalent

For reference, here's the same pattern in ASP.NET:

```csharp
// ProductsController.cs
public class ProductsController : Controller
{
    private readonly ApplicationDbContext _db;

    public ProductsController(ApplicationDbContext db)
    {
        _db = db;
    }

    [Authorize]
    public IActionResult Index(string filter = "all")
    {
        IQueryable<Product> query = _db.Products;

        if (filter == "in_stock")
        {
            query = query.Where(p => p.Stock > 0);
        }

        var products = query.OrderBy(p => p.Name).ToList();

        return View(products); // Renders Products.cshtml
    }
}
```

**Same architecture:**
- Server-side rendering (Razor views)
- Session management (ASP.NET Session State)
- Database queries on every request
- HTML generation on server

**Same limitations:**
- Server CPU bottleneck
- Connection pooling limits
- No real-time updates without reload

---

## Summary

### The Traditional Stack is Good When:

✅ Building internal tools (< 100 concurrent users)
✅ CRUD applications with simple requirements
✅ Team familiar with server-side rendering
✅ Budget-conscious projects
✅ Rapid prototyping

### Problems Emerge When:

❌ Need to scale beyond single server
❌ Want real-time features
❌ Building mobile apps (can't render server-side HTML)
❌ Geographic distribution (users worldwide)
❌ High interactivity requirements

---

**The solution?** Move work from the server to the client.

**Next:** [Part 2: The Client-Side Rendering Revolution →](./02-client-rendering.md)
