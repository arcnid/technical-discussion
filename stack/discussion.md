# Application Architecture and The Stack

A tour of how a real, working web app is put together — using our own quote program (NewCIC) as the "this is how we do it" example, and contrasting each piece with how a modern application would do the same job today.

The goal isn't to convince you our app is bad or that modern apps are better. The goal is to give you a **map of the territory**. Once you can see the layers, you can pick which one you want to specialize in.

---

## What is "the stack"?

When engineers say "the stack," they mean **the layered set of technologies that work together to serve a user's request**.

A typical web app stack, top to bottom:

```
┌─────────────────────────────────────────┐
│  Browser / Client                       │  ← HTML, CSS, JavaScript
├─────────────────────────────────────────┤
│  Network                                │  ← HTTP, HTTPS, WebSockets
├─────────────────────────────────────────┤
│  Web Server                             │  ← IIS, Nginx, Apache
├─────────────────────────────────────────┤
│  Application Code                       │  ← PHP, Node.js, Python, Go
├─────────────────────────────────────────┤
│  Data Layer                             │  ← SQL Server, Postgres, Redis
├─────────────────────────────────────────┤
│  Operating System                       │  ← Windows Server, Linux
└─────────────────────────────────────────┘
```

A request from a salesperson clicking a button in the quote program travels **down** through every one of these layers, gets processed, and the response travels **back up**.

A "full stack engineer" is someone who can work confidently at every layer. Not equally expert in all of them — nobody is — but able to read, debug, and write code that crosses from the browser down into the database and back.

---

## What it means to be full stack

It usually breaks into two halves:

- **Front end** — what the user sees and clicks. HTML, CSS, JavaScript. In NewCIC: everything in `assets/js/*.js` and `application/views/*.php`.
- **Back end** — what runs on the server. Routing, business logic, database calls. In NewCIC: everything in `application/controllers/` and `application/models/`.

"Full stack" means you can move between them without getting stuck. The salesperson reports a bug — "the dealer dropdown is empty." A full-stack engineer can:

1. Open the browser dev tools, see the AJAX call failing.
2. Trace it to the controller method that handles the request.
3. Read the SQL the model is running.
4. Fix the bug wherever it actually lives — front end, back end, or database — without needing to hand it off.

---

## A tour through NewCIC's stack (with the modern equivalent)

For each layer, we'll look at **how we do it** in NewCIC, then **how a modern application typically does the same thing**.

---

### 1. Authentication — "who are you?"

#### How we do it in NewCIC

The quote program uses **server-side sessions**. When you log in, the server checks your username and password against the `Users` table and, if they match, stuffs your identity into a **session** — a blob of data the server keeps in memory (or on disk) tied to a cookie in your browser.

Login handler (`application/controllers/LogIn.php`):

```php
public function index()
{
    $Username = strtolower($this->input->post("Username"));
    $Password = $this->input->post("Password");

    if ($this->LogInModel->Login($Username, $Password)) {
        $Userdata = $this->CommonFunctionsModel->GetUserData($Username);

        $SessionData = array(
            'userid'    => $Userdata->user_id,
            'username'  => $Userdata->username,
            'group_id'  => $Userdata->group_id,
            'dealerID'  => $Userdata->dealerID,
            'LoggedIn'  => TRUE,
            'Roles'     => $UserRoles
        );

        $this->session->set_userdata($SessionData);
        redirect(site_url("Dashboard"));
    }
}
```

Password check (`application/models/LogInModel.php`):

```php
$sql = "SELECT ... FROM UserRoles
        JOIN Users ON UserRoles.user_id = Users.user_id
        WHERE Users.user_id = ?
          AND Users.pswd = ?";

$query = $this->db->query($sql, array($Username, md5($Password . SALT)));
```

And every subsequent request runs through a guard library (`application/libraries/Authentication.php`) that gets auto-loaded on every page:

```php
if ($this->CI->session->userdata('LoggedIn') != TRUE) {
    $this->CI->session->set_userdata('referred_from', urlencode(current_url()));
    redirect("LogIn");
}
```

Translation: every page load, before the controller does anything, the system checks "do you have a valid session cookie that says `LoggedIn = TRUE`?" If not, kick you back to the login screen.

#### What's worth noting

- Passwords are hashed with **MD5 + a salt**. MD5 has been considered broken for password hashing for over a decade — modern apps use bcrypt, scrypt, or Argon2.
- Sessions are **server-side state**. The server has to remember every logged-in user. That's fine for ~100 dealers, awkward at internet scale.
- Identity lives in **one big session blob**, mutated freely.

#### How a modern app does it

Two dominant patterns today:

**Pattern A: JWT (JSON Web Tokens)**

The server signs a token containing the user's identity and hands it to the browser. The browser sends that token on every subsequent request. The server doesn't have to remember anything — it just verifies the signature.

```js
// Modern Node.js / Express
app.post('/login', async (req, res) => {
    const user = await db.users.findOne({ username: req.body.username });
    const ok = await bcrypt.compare(req.body.password, user.passwordHash);

    if (!ok) return res.status(401).json({ error: 'Invalid credentials' });

    const token = jwt.sign(
        { userId: user.id, role: user.role },
        process.env.JWT_SECRET,
        { expiresIn: '8h' }
    );

    res.json({ token });
});
```

**Pattern B: OAuth / OpenID Connect** (most common in 2026)

You don't write your own login at all. You hand off to a dedicated identity provider — Auth0, Clerk, AWS Cognito, Okta, "Sign in with Google." They handle password storage, MFA, password resets, account recovery, and you just receive a verified identity token back.

```tsx
// Modern Next.js with Clerk
import { auth } from '@clerk/nextjs/server';

export default async function Dashboard() {
    const { userId } = await auth();
    if (!userId) redirect('/sign-in');
    // ...
}
```

| Concern | NewCIC | Modern app |
|---|---|---|
| Password hash | MD5 + salt | bcrypt / argon2 / outsourced |
| State | Server-side session | Stateless token, or outsourced |
| Who writes the login form? | We do | Often a third party (Clerk, Auth0) |
| MFA, password reset, SSO | We'd build it | Free with the identity provider |

---

### 2. Web requests — "how does a click become a server action?"

#### How we do it in NewCIC

Two flavors of request, both old-school:

**Flavor 1: Full page navigation.** Click a link, the browser throws away everything and asks the server for an entire new HTML page.

```
GET /Dashboard
```

CodeIgniter's router (`application/config/routes.php`) maps that URL to a controller method, which assembles data and renders a view:

```php
$this->load->view('template/DashboardHeader', $data);
$this->load->view('Dashboard', $data);
$this->load->view('template/DashboardFooter', $data);
```

The server builds the entire HTML page on the server and ships it down. The browser displays it.

**Flavor 2: jQuery AJAX.** Once a page is loaded, JavaScript can fire off a background request for just a piece of data — no full page reload.

From `assets/js/Dashboard.js`:

```js
$.ajax({
    type: "POST",
    url: "Dashboard/GetOpenOrders",
    data: {
        custForeignOrDomestic: custForeignOrDomestic,
        custLastFive: custLastFive,
        po_no: po_no,
        dateFrom: dateFrom,
        dateTo: dateTo,
    },
    dataType: "json",
    success: callback,
});
```

That hits a method on the `Dashboard` controller that returns JSON, and the JS callback updates the page.

#### How a modern app does it

The modern default is the **Single Page Application (SPA)**. The browser loads the app **once** as a big JavaScript bundle (React, Vue, Svelte, etc.). After that, the app never does a full page reload. All subsequent server communication is JSON over `fetch()`, and the UI re-renders itself in response.

```tsx
// Modern React + fetch
const { data, error } = useQuery({
    queryKey: ['openOrders', filters],
    queryFn: () => fetch('/api/orders/open', {
        method: 'POST',
        body: JSON.stringify(filters),
    }).then(r => r.json()),
});
```

Newer frameworks (Next.js App Router, Remix, SvelteKit) blend the two — they pre-render the first page on the server for speed, then hydrate into a SPA in the browser. Best of both worlds.

| Concern | NewCIC | Modern app |
|---|---|---|
| Page navigation | Server re-renders entire page | Client-side router, no reload |
| Data fetching | `$.ajax` from jQuery | `fetch` / TanStack Query / RTK Query |
| Rendering location | Server | Client (or both) |
| Result | Many full-page round trips | One initial load, then JSON |

---

### 3. API endpoints — "how does the server expose data?"

#### How we do it in NewCIC

We don't really have a separate "API." Our controllers do double duty: some methods return HTML pages, some methods return JSON. They live side-by-side in the same controller.

From `application/controllers/Dashboard.php`:

```php
// Renders an HTML page
public function index()
{
    $data['title'] = 'Dashboard';
    // ... gather data ...
    $this->load->view('template/DashboardHeader', $data);
    $this->load->view('Dashboard', $data);
    $this->load->view('template/DashboardFooter', $data);
}

// Returns JSON
public function GetChildDealers()
{
    $foreignOrDomestic = $this->input->post("ForeignOrDomestic");
    $lastFive          = $this->input->post("LastFive");

    print json_encode($this->DashboardModel->GetChildDealers($foreignOrDomestic, $lastFive));
}

// Also returns JSON
public function GetOpenQuotes()
{
    $ForeignOrDomestic = $this->input->post("ForeignOrDomestic");
    $LastFive          = $this->input->post("LastFive");
    // ...
    print json_encode($rows);
}
```

A few things to notice:

- Everything is `POST`, even for things that are really just reads (`GetOpenQuotes`).
- The URL is verb-shaped: `Dashboard/GetOpenQuotes`, `Dashboard/GetChildDealers`. The action is in the URL.
- There's no clear separation between "the page" and "the API."

#### How a modern app does it

Modern APIs follow one of two dominant patterns:

**REST** — resources, not verbs. The HTTP method (`GET`, `POST`, `PUT`, `DELETE`) is the verb, and the URL is the resource.

```
GET    /api/quotes              ← list quotes
GET    /api/quotes/12345        ← get one quote
POST   /api/quotes              ← create a quote
PUT    /api/quotes/12345        ← update a quote
DELETE /api/quotes/12345        ← delete a quote
```

```ts
// Modern Express REST endpoint
app.get('/api/quotes/:id', async (req, res) => {
    const quote = await db.quotes.findById(req.params.id);
    if (!quote) return res.status(404).json({ error: 'Not found' });
    res.json(quote);
});
```

**GraphQL** — the client asks for exactly the fields it wants, in one query, no matter how many resources are involved.

```graphql
query {
  quote(id: "12345") {
    quoteNumber
    customer { name region }
    lineItems { partNumber quantity price }
  }
}
```

The API layer is usually its own thing — a separate set of files, separate from the page renderer, often a separate deployment. The front end (which could be a web app, a mobile app, or a partner integration) just talks JSON to it.

| Concern | NewCIC | Modern app |
|---|---|---|
| Style | Action-style URLs, `POST` for everything | REST resources or GraphQL |
| Separation | Pages and API in the same controller | Distinct API layer |
| Documentation | None — read the source | OpenAPI / Swagger / GraphQL schema |
| Consumers | Our jQuery only | Web, mobile, partners, scripts |

---

### 4. Database access — "how does the code read and write data?"

#### How we do it in NewCIC

Models talk to SQL Server using **hand-written SQL** with CodeIgniter's thin query helper:

```php
public function GetStiffenersPerSheet($BinNumber)
{
    $sql = "select numberOfStiffeners as NumberOfStiffeners
            from OptionsAssistant.dbo.ss_bin_specs_new
            where item_no = ?";

    $StiffenerParams = array($BinNumber);
    $query = $this->db->query($sql, $StiffenerParams);
    return $query->result_object();
}
```

Connection settings live in `application/config/database.php`:

```php
$db['default'] = array(
    'hostname' => 'ssdsvm06',
    'username' => 'web_user',
    'password' => '...',
    'database' => 'CIC',
    'dbdriver' => 'sqlsrv',
);
```

Some of it is good (parameterized queries with `?` placeholders — safe from SQL injection). Some of it is rough — look at `LogInModel::LogTracking`:

```php
$sql = "INSERT INTO ".CIC.".[dbo].[ss_login_tracking_new]
       ([UserID], [LogInDateTime], [IPAddress])
        VALUES ('$Username', GETDATE(), '$IPAddress')";
```

That's string-concatenating user input directly into SQL. Classic SQL injection vector. The login form rejects bad logins, but a username of `' OR 1=1 --` would still be a problem here if it ever reached this line.

#### How a modern app does it

Modern apps usually use an **ORM** (Object-Relational Mapper) — a library that lets you write database queries as method calls on typed objects.

**Prisma** (TypeScript):

```ts
const stiffeners = await prisma.binSpecs.findUnique({
    where: { itemNo: binNumber },
    select: { numberOfStiffeners: true },
});
```

**SQLAlchemy** (Python):

```python
stiffeners = (
    session.query(BinSpec.number_of_stiffeners)
           .filter(BinSpec.item_no == bin_number)
           .one()
)
```

**Entity Framework** (.NET):

```csharp
var stiffeners = await db.BinSpecs
    .Where(b => b.ItemNo == binNumber)
    .Select(b => b.NumberOfStiffeners)
    .SingleAsync();
```

Why ORMs caught on:

- **Type safety.** Your editor knows `BinSpec.numberOfStiffeners` exists. Misspelling it is a compile-time error, not a 2 a.m. runtime crash.
- **Injection-proof.** The ORM never concatenates strings. Filters are values, not SQL fragments.
- **Migrations.** Schema changes are tracked in code, version-controlled, and applied in the same way to dev, staging, and prod.
- **Multi-database.** Swap Postgres for MySQL without rewriting your queries.

The tradeoff: ORMs hide what's actually happening on the database. A two-line ORM call can generate a 50-line SQL query that scans every row in your table. Engineers who only know the ORM and never look at the generated SQL are a real performance liability.

| Concern | NewCIC | Modern app |
|---|---|---|
| Query style | Raw SQL strings | ORM method chains |
| Safety from injection | Parameterized (mostly) | Built in |
| Schema changes | Run a SQL script by hand | Versioned migrations |
| Type checking | None | Full IDE support |
| Performance visibility | Direct — you see the SQL | Indirect — you trust the ORM |

---

## Architecture patterns — the shape of the whole system

Beyond the per-layer choices, there are **whole-system patterns** that change how the pieces talk to each other.

### Request / response (what NewCIC is)

The classic shape. The browser asks for something, the server answers, and that's the end of the conversation. The next time anything happens, the browser has to ask again.

```
Browser  ──── "give me dashboard data" ───▶  Server
Browser  ◀──────── JSON payload ──────────   Server
```

Simple. Predictable. Everything we do is this.

**Limitation:** the server can never *initiate* — it can only respond. If a new quote comes in while you're staring at the dashboard, you won't know until you refresh.

### Real-time

For UIs that need to update the instant something happens — chat apps, multiplayer games, live dashboards, stock tickers, collaborative editors (Google Docs).

Two main mechanisms:

- **WebSockets** — a persistent two-way connection between browser and server. Either side can push data at any time.
- **Server-Sent Events (SSE)** — one-way push from server to browser over a long-lived HTTP connection.

```ts
// Modern WebSocket on the client
const socket = new WebSocket('wss://api.example.com/quotes');
socket.onmessage = (event) => {
    const update = JSON.parse(event.data);
    updateDashboard(update);
};
```

If we wanted the quote program's dashboard to show new quotes the second they're created — without refresh — real-time is the pattern. Today, every refresh is a manual round trip.

### Event-driven

Instead of services calling each other directly ("hey order service, please update inventory"), they **publish events** to a message broker, and other services **subscribe** to the events they care about.

```
[Quote Service] ──"QuoteCreated"──▶ [Message Broker] ──▶ [Email Service]
                                                     ──▶ [Analytics Service]
                                                     ──▶ [PDF Service]
```

Brokers you'll hear about: **Kafka**, **RabbitMQ**, **AWS SNS/SQS**, **Google Pub/Sub**, **MQTT** (used in industrial IoT — including the Raptor Sweep program).

Why teams choose this:

- **Loose coupling.** The quote service doesn't know who cares about quotes. New consumers can be added without touching the quote service.
- **Resilience.** If the email service is down, events queue up and process when it comes back.
- **Audit log.** The event stream itself is a complete history of what happened.

The tradeoff: debugging becomes much harder. A single business action triggers a cascade of events across services, and reconstructing "what actually happened" requires distributed tracing.

NewCIC is **not** event-driven. When you create a quote, the quote controller directly calls everything that needs to happen, all in one synchronous request. That's simpler — but it also means the user waits for every side effect to finish before the page responds.

---

## Low-level patterns — building blocks

Underneath the architecture, code itself follows recurring patterns. A few of the most common:

### Singleton

**One instance, accessible from everywhere.** Useful when you genuinely only ever want one of something — a database connection pool, a logger, a config object.

CodeIgniter actually uses this pattern under the hood. There's exactly one CodeIgniter "super object," and every model and library reaches it via:

```php
$this->CI =& get_instance();   // from Authentication.php
```

That `get_instance()` is a textbook singleton accessor.

**Why it's controversial:** singletons are global state in disguise. They make testing harder (you can't swap in a fake) and they make dependencies invisible (a function that uses a singleton doesn't declare it in its signature). Modern code prefers **dependency injection** — pass the thing in explicitly.

```ts
// Modern: dependency injected
class QuoteService {
    constructor(private db: Database, private logger: Logger) {}
    // ...
}
```

### Factory

**A function whose job is to build other objects.** You don't `new` the thing directly — you ask a factory for it, and the factory decides what concrete class to give you based on inputs.

```ts
function buildQuoteCalculator(productLine: string): QuoteCalculator {
    switch (productLine) {
        case 'farm-bin':       return new FarmBinCalculator();
        case 'commercial-bin': return new CommercialBinCalculator();
        case 'sweep':          return new SweepCalculator();
        default: throw new Error(`Unknown product line: ${productLine}`);
    }
}
```

NewCIC essentially does this with controllers — `BinQuoteInit`, `FarmBinQuote`, `CommercialBinQuote` are all separate controllers, and the router picks the right one based on the URL. It's a factory pattern spread across the router config.

### Observer (pub/sub)

**Things subscribe to events; the publisher doesn't need to know who's listening.** Event-driven architecture (above) is observer pattern at the system scale. At the code scale, every time you write a DOM event listener:

```js
button.addEventListener('click', () => save());
```

…you're using observer pattern. The button doesn't know what `save` does. It just announces "I was clicked," and anyone who registered gets called.

### Other patterns worth knowing the names of

- **Repository** — a class that hides "how to read/write this kind of thing from the database." Our `*Model.php` files are repositories.
- **Strategy** — swap out an algorithm at runtime. (e.g. "for this customer, use the foreign-pricing strategy.")
- **Adapter** — wrap an awkward API to make it look like the API your code expects. (e.g. wrap an old SOAP service to look like a REST client.)
- **Middleware** — a chain of functions that each get a chance to inspect/modify a request before it reaches the handler. The CodeIgniter `Authentication` library is essentially auth middleware.

You don't need to memorize patterns. You need to recognize them when you see them — and recognize when reaching for one would untangle code you're stuck on.

---

## So why does any of this matter?

Two things to take away.

### Know your environment

You can't debug what you don't understand. When something breaks in NewCIC, the answer is somewhere in this stack: maybe the browser sent the wrong AJAX payload, maybe the session expired, maybe a SQL query is returning the wrong shape, maybe IIS is misconfigured. The better your mental model of the layers, the faster you find it.

Every layer in our stack has a knob you can turn. Knowing where the knobs are is most of the job.

### Know other environments — for contrast

If NewCIC is the only thing you've ever seen, you'll think *all* web apps look like NewCIC. They don't. Modern apps make different tradeoffs — sometimes better, sometimes worse, almost always *different*.

Knowing what a modern app does for auth tells you why our MD5 password hashing should make us uncomfortable. Knowing what real-time architectures look like tells you what we'd need to change to get a live dashboard. Knowing what ORMs do tells you why our hand-written SQL is both a liability (maintenance) and an asset (we can actually see and tune what's running).

You can't make good engineering tradeoffs without knowing the menu.

### A roadmap for specialization

Different people are drawn to different layers. Some examples of where you could go deep:

| If you like… | Specialize in… | What you'd own |
|---|---|---|
| Pixel-perfect UI, design, animation | **Front end** | React / Vue, CSS, accessibility, perf |
| Business logic, APIs, services | **Back end** | Node, Go, Python, .NET, system design |
| Data, queries, reporting | **Database / Data Engineering** | SQL, query plans, warehouses, ETL |
| Infrastructure, deploys, uptime | **DevOps / SRE / Platform** | Cloud, CI/CD, Kubernetes, monitoring |
| Phones, native apps | **Mobile** | iOS (Swift), Android (Kotlin), React Native |
| Models, predictions, ML | **AI / ML** | PyTorch, TensorFlow, model serving |
| Security, threat modeling | **Security / AppSec** | OWASP top 10, pentest, threat modeling |
| Sensors, hardware, embedded | **Edge / IoT** | MQTT, Pi, microcontrollers, low-power |

None of these are dead ends. All of them are growth tracks. Pick the one that energizes you, and the rest of the stack becomes context you pick up along the way.

Being **full stack** isn't being expert at all of them. It's having a working model of every layer so you can collaborate with the people who *are* expert in each, and so you can ship something end-to-end when nobody else is available.

That's the goal.
