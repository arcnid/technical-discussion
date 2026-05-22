# Git CLI: Command Guide

A practical reference for the Git commands you'll actually use. For each command: **what it does**, **the syntax**, and **how it's typically used**.

---

## Core Concepts (Quick Refresher)

A few terms used throughout this guide:

- **Repository ("repo")** — a folder Git is tracking, identified by a hidden `.git` directory inside it.
- **Commit** — a saved snapshot of the project at a moment in time, with a message describing what changed.
- **Branch** — an independent line of development. The default branch is usually called `main` (or `master` on older repos — see [Branch Conventions](#branch-conventions)).
- **Remote** — a copy of the repo hosted elsewhere (e.g. GitHub). Default name is `origin`.

Files move through three areas as you work:

```
   edit files         git add          git commit
┌────────────┐     ┌──────────┐     ┌──────────────┐
│  Working   │ ──▶ │ Staging  │ ──▶ │  Repository  │
│ Directory  │     │  Area    │     │  (committed) │
└────────────┘     └──────────┘     └──────────────┘
  Files on disk    "Ready to be     Permanent
  you're editing    committed"      project history
```

The staging area lets you pick *which* of your changes to bundle into a single commit — you don't have to commit everything you've changed at once.

---

## Navigating to Your Repository (Windows)

Git commands always run **inside** a repository's folder — so the first step in any Git session is making sure your terminal is pointed at the right directory. On Windows, open **PowerShell** (or **Git Bash**) and use the following commands.

### `pwd` — Where am I?

```powershell
pwd
```

**Typical use:** Prints the full path of the directory you're currently in. When you open PowerShell, you usually start in your user folder, e.g. `C:\Users\YourName`.

### `ls` — What's in this folder?

```powershell
ls
```

**Typical use:** Lists the files and folders in the current directory. Useful for checking what's around before you `cd` into something. (In Windows PowerShell, `ls` and `dir` both work.)

### `cd` — Change directory

```powershell
cd projects                   # move into a subfolder called "projects"
cd C:\Users\YourName\code     # jump to an absolute path
cd ..                         # go up one level (to the parent folder)
cd ~                          # go to your home folder (C:\Users\YourName)
```

**Typical use:** Move into the folder that contains (or will contain) your Git repo. Path separators in PowerShell can be either `\` or `/` — both work.

**Tip:** Press **Tab** to auto-complete folder names. Type `cd pro` and press Tab — PowerShell will fill in `projects\` for you.

### Putting It Together

A typical session of getting to a repo:

```powershell
pwd                                   # C:\Users\YourName
cd code                               # move into your code folder
ls                                    # see what projects are there
cd my-project                         # move into the repo folder
ls                                    # should see a .git folder, README, etc.
git status                            # now Git commands will work here
```

If `git status` prints `fatal: not a git repository`, you're not inside a repo folder yet — `cd` into the right place (or run `git init` / `git clone` to create one here).

### Opening PowerShell Directly in a Folder

You can skip the navigation entirely by opening PowerShell already pointed at the folder you want:

- In **File Explorer**, navigate to the folder
- Right-click in an empty area inside the folder
- Choose **Open in Terminal** (Windows 11) or **Open PowerShell window here** (Windows 10, hold Shift while right-clicking)

This drops you into a PowerShell window with that folder already as the current directory.

---

## Setup

### `git config` — Set your identity and preferences

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --list
```

**Typical use:** Run the first two commands once on a new machine. Every commit you make from then on is tagged with this name and email. `--global` applies to every repo on the machine; drop the flag to set it for just the current repo.

---

## Starting a Repo

### `git init` — Create a new repo

```bash
git init
```

**Typical use:** Run inside an existing folder to start tracking it with Git. Creates the hidden `.git` directory. Nothing is committed yet — you still need to `git add` and `git commit`.

### `git clone` — Copy an existing repo

```bash
git clone https://github.com/someuser/some-project.git
git clone https://github.com/someuser/some-project.git my-folder
```

**Typical use:** Download a project that already exists on GitHub (or another remote). Creates a new folder containing the full project and its entire history. The second form clones into a folder of your choosing instead of the default name.

---

## The Everyday Loop

### `git status` — Show the current state

```bash
git status
```

**Typical use:** Run constantly. Shows which branch you're on, which files have unstaged changes, which are staged, and which are untracked. Changes nothing — safe to run any time. When you're confused about what Git thinks is going on, this is the first command to reach for.

---

### `git add` — Stage changes

```bash
git add file.txt              # stage one file
git add folder/               # stage everything in a folder
git add .                     # stage everything changed in the current directory
```

**Typical use:** After editing files, pick which changes you want in your next commit. You can edit ten files and stage only two — Git will commit just those.

---

### `git commit` — Save a snapshot

```bash
git commit -m "Add login form"
git commit                    # opens a text editor for the message
```

**Typical use:** Records whatever's currently staged as a new commit. The `-m` flag lets you write the message inline. Without it, Git opens your default editor.

**Conventions for commit messages:**

- Short (~50 characters in the title)
- Imperative mood: "Add login form" not "Added login form"
- Describes *why*, not just *what*

---

### `git log` — Show commit history

```bash
git log
git log --oneline             # one line per commit
git log --graph --oneline     # show branches as an ASCII graph
git log -n 5                  # only the last 5 commits
git log -- file.txt           # only commits that touched file.txt
```

**Typical use:** Browse the history of the repo. Newest commits first. Press `q` to exit. `--oneline` is the most useful day-to-day variant.

---

### `git diff` — Show line-by-line changes

```bash
git diff                      # working dir vs. staging area
git diff --staged             # staged changes vs. last commit
git diff main feature-login   # differences between two branches
```

**Typical use:** Before committing, run `git diff` to double-check what you actually changed. Run `git diff --staged` to confirm what's about to go into the next commit. Lines starting with `-` were removed; lines starting with `+` were added.

---

## Branching and Merging

### `git branch` — List, create, or delete branches

```bash
git branch                    # list local branches; * marks current
git branch feature-login      # create a new branch (doesn't switch to it)
git branch -d feature-login   # delete a branch (refuses if unmerged)
git branch -D feature-login   # force-delete, even if unmerged
```

**Typical use:** Most often run with no arguments to see where you are. The `-d` variant is the safe way to clean up branches after merging; `-D` is the "I really mean it" version.

---

### `git switch` — Change branches

```bash
git switch main               # switch to an existing branch
git switch -c feature-login   # create AND switch to a new branch
git switch -                  # switch back to the previous branch
```

**Typical use:** Switching to a new branch is the standard way to start work on a separate feature or fix. The `-c` form is the most common — it combines create-and-switch in one step.

> You may also see `git checkout` used for switching branches — it's the older command and still works, but `switch` is preferred for new code.

---

### `git merge` — Combine branches

```bash
git switch main               # go to the branch that will RECEIVE the changes
git merge feature-login       # bring feature-login's commits into main
```

**Typical use:** After finishing work on a feature branch, merge it back into `main`. If Git can combine the changes automatically, you're done. If both branches changed the same lines, you get a **merge conflict**:

```
<<<<<<< HEAD
your version of the line
=======
their version of the line
>>>>>>> feature-login
```

Edit the file to keep what you want, remove the conflict markers, then `git add` and `git commit` to finalize the merge.

---

## Branch Conventions

### Branch off `main` for new work

New feature branches and bug-fix branches should almost always be created from an up-to-date copy of `main`:

```bash
git switch main
git pull
git switch -c my-new-feature
```

Starting from the current `main` ensures your branch has the latest code and won't have to merge in months of unrelated changes when it's time to integrate.

**Exception:** if you're building directly on top of an in-progress feature branch (for example, while it's still under review), branching off that feature branch is fine. Branching off unrelated branches, however, almost always causes pain later.

### `main` vs. `master`

The default branch in a Git repo is just a name. Historically it was called `master`. Since 2020, GitHub creates new repos with `main` instead, and many existing projects have migrated.

**They behave identically** — the name is just a label. Anywhere this guide says `main`, substitute `master` if that's what your project uses.

---

## Working with Remotes

### `git remote` — Manage remote connections

```bash
git remote -v                                       # list remotes and their URLs
git remote add origin https://github.com/user/repo.git
git remote remove origin
```

**Typical use:** When you clone a repo, the remote is set up for you automatically. You only need `git remote add` if you created the repo locally with `git init` and now want to connect it to GitHub.

---

### `git push` — Upload commits to the remote

```bash
git push                      # push current branch to its remote counterpart
git push -u origin main       # first push of a branch: also set the upstream link
```

**Typical use:** Share your local commits with the remote. The `-u` flag links your local branch to a remote branch of the same name — you only need it the first time you push a new branch. After that, plain `git push` works.

---

### `git pull` — Download and merge remote changes

```bash
git pull
```

**Typical use:** Shorthand for "fetch the latest commits from the remote, then merge them into my current branch." Run this before starting new work to avoid building on an outdated copy.

---

### `git fetch` — Download without merging

```bash
git fetch                     # all remotes
git fetch origin              # just origin
```

**Typical use:** When you want to see what's on the remote before integrating it. Useful for inspecting changes others have pushed without disturbing your working state.

**Pull vs. fetch:**

| Command | Downloads from remote? | Merges into your branch? |
|---|---|---|
| `git fetch` | ✅ | ❌ |
| `git pull` | ✅ | ✅ |

---

## Working with GitHub and Pull Requests

GitHub is the most common place to host Git remotes, and it adds a workflow on top of Git called **pull requests**. Pull requests don't exist in Git itself — they're a GitHub feature that wraps around `git push` and `git merge`.

### What is a Pull Request?

A **pull request** (PR) is a proposal to merge the commits on one branch into another (usually your branch into `main`). It gives teammates a place to:

- Review your code line by line
- Leave comments and request changes
- Run automated checks (tests, linters, CI)
- Discuss the change before it lands

### Creating a Pull Request

After pushing a branch for the first time:

```bash
git push -u origin my-branch
```

Then open the repository on GitHub in your browser and follow these steps:

1. Go to the repository's main page on GitHub (e.g. `https://github.com/your-org/your-repo`).
2. GitHub usually shows a yellow banner near the top that says **"my-branch had recent pushes — Compare & pull request"**. Click that button.
   - If the banner isn't there, click the **Pull requests** tab, then click **New pull request**, and pick your branch from the **compare** dropdown.
3. Confirm the **base** branch (usually `main`) and the **compare** branch (your branch) at the top of the page.
4. Write a **title** that describes the change in one short sentence.
5. Write a **description** explaining **what changed and why**. Include screenshots, reproduction steps, or links to related issues if relevant.
6. Optionally use the right sidebar to:
   - Assign **Reviewers**
   - Add **Labels** (e.g. `bug`, `feature`)
   - Link the PR to an **Issue** it resolves
7. Click **Create pull request**.

### Reviewing a Pull Request on GitHub

When it's your turn to review a teammate's PR, the GitHub UI walks you through the same diff view you'd use to check your own work:

1. Open the PR from the **Pull requests** tab on the repository page.
2. Read the **Conversation** tab first — the title, description, and any prior discussion give you the context for the change.
3. Click the **Files changed** tab to see the full diff.
4. Walk through each file:
   - Hover over a line number and click the blue **+** to leave an inline comment on that exact line.
   - Click and drag across multiple line numbers to comment on a range.
   - Click **Viewed** in the top-right of each file once you've finished checking it — this collapses the file and helps you track progress on large PRs.
5. When you're done, click **Review changes** at the top right and choose one of:
   - **Comment** — leave feedback without approving or blocking
   - **Approve** — sign off on the change
   - **Request changes** — block the merge until specific issues are addressed
6. Add an overall summary comment if useful, then click **Submit review**.

**What to look for as a reviewer:**

- Does the diff actually match what the title and description claim?
- Are there obvious bugs, typos, or unhandled edge cases?
- Is anything unclear that should have a comment explaining *why*?
- Are tests included or updated for the new behavior?
- Are there unrelated changes that should be in a separate PR?

### Updating a Pull Request

A PR automatically reflects new commits pushed to its branch — no special command needed:

```bash
# fix code based on review feedback
git add .
git commit -m "Address review feedback"
git push                              # the PR updates automatically
```

### Reviewing the Diff on GitHub

Before you ask anyone else to review your PR, **review your own diff first**. This is the single most effective habit for catching mistakes.

On the pull request page, click the **Files changed** tab. GitHub shows every change in the branch:

```
┌──────────────────────────────────────────────────────┐
│  Conversation   Commits   Checks   Files changed     │
├──────────────────────────────────────────────────────┤
│  src/login.js                                        │
│                                                      │
│  - function login(user, pass) {                      │
│  + function login(user, password) {                  │
│        const hash = hashPassword(password);          │
│        ...                                           │
└──────────────────────────────────────────────────────┘
```

Lines starting with `-` (red) were removed; lines starting with `+` (green) were added.

**What to look for when reviewing your own diff:**

- Debug code, `console.log`s, or commented-out experiments you forgot to remove
- Files you didn't mean to commit (config files, build outputs, `.env`)
- Unrelated changes that crept in (an accidental save in a file you weren't working on)
- Whitespace-only edits that bloat the diff without changing behavior

You can leave comments on your own PR — useful for flagging things you want reviewers to look at closely.

### Keep Diffs Small and Focused

A pull request should do **one thing**, and the title should describe that thing accurately. The diff is what proves the title is honest.

✅ **Good PR:** Title says "Fix off-by-one in pagination" — the diff touches the pagination function and its test, nothing else.

❌ **Bad PR:** Title says "Fix off-by-one in pagination" — the diff also reformats three unrelated files, renames a variable in the auth module, and bumps a dependency version.

**Why small diffs matter:**

- Reviewers can actually read them carefully. A 50-line diff gets a thorough review; a 2,000-line diff gets a rubber stamp.
- Bugs are easier to spot when changes are isolated.
- If the change causes a problem later, reverting one focused PR is safe; reverting a sprawling PR also undoes unrelated work.
- Conflicts are smaller and easier to resolve.

**Rules of thumb:**

- If you find yourself writing "and also" in the PR description, it's probably two PRs.
- Unrelated cleanup (formatting, renames, dependency bumps) belongs in its own PR.
- If a PR grows past a few hundred changed lines, consider whether it can be split.

The commit message and PR title are a **promise** to the reviewer about what's in the diff. Keep that promise.

### Merging a Pull Request

Once reviewers approve and any automated checks pass, click **Merge pull request** on GitHub. This merges your branch into `main` on the remote.

**After merging, clean up locally:**

```bash
git switch main                       # back to main
git pull                              # pull in the just-merged changes
git branch -d my-branch               # delete your finished branch
```

---

## Undoing Things

### `git restore --staged` — Unstage a file

```bash
git restore --staged file.txt
```

**Typical use:** You staged a file with `git add` but realized you don't want it in the next commit. The file's edits stay in your working directory — they just leave the staging area.

---

### `git restore` — Discard uncommitted edits

```bash
git restore file.txt
```

**Typical use:** Throw away your local edits to a file and reset it back to whatever the last commit looked like.

⚠️ Destructive — there is no undo for these edits.

---

### `git commit --amend` — Fix the last commit

```bash
git commit --amend -m "Better message"
git commit --amend            # also add newly staged changes to the last commit
```

**Typical use:** Fix a typo in your most recent commit message, or sneak a forgotten file into the last commit.

⚠️ Only safe on commits you haven't pushed yet. Amending a pushed commit rewrites history that other people may already have.

---

### `git reset` — Move the branch pointer

```bash
git reset --soft HEAD~1       # undo last commit, keep changes staged
git reset --mixed HEAD~1      # undo last commit, keep changes unstaged (default)
git reset --hard HEAD~1       # undo last commit, DISCARD changes entirely
```

**Typical use:** "I committed too early." `--soft` is the friendliest variant — your changes are still there, just no longer committed. `HEAD~1` means "one commit back"; use `HEAD~2` for two, and so on.

⚠️ `--hard` is destructive — it throws away your uncommitted work. Never use it on pushed commits.

---

### `git revert` — Undo a commit safely

```bash
git revert <commit-hash>
```

**Typical use:** Undo a commit that's already been pushed. Instead of erasing the commit, this creates a *new* commit that reverses its changes. Safe to use even when other people already have the original commit.

**Which undo command to use:**

| Situation | Command | Safe if pushed? |
|---|---|---|
| Unstage a file | `git restore --staged <file>` | ✅ |
| Discard uncommitted edits | `git restore <file>` | ✅ (local only) |
| Fix the last commit message | `git commit --amend` | ❌ |
| Undo last commit, keep work | `git reset --soft HEAD~1` | ❌ |
| Erase last commit and work | `git reset --hard HEAD~1` | ❌ |
| Undo a pushed commit | `git revert <hash>` | ✅ |

---

## Ignoring Files

### `.gitignore` — Tell Git which files to skip

Create a file called `.gitignore` in the repo root, one pattern per line:

```gitignore
node_modules/
*.log
.env
build/
```

**Typical use:** Keep build outputs, secrets, and editor scratch files out of the repo. Commit `.gitignore` itself so everyone on the team shares the same rules.

**Pattern syntax:**

- `node_modules/` — entire folder
- `*.log` — any file ending in `.log`
- `.env` — specific filename
- `!important.log` — exception: include this file even though `*.log` excludes it

---

## Getting Help

### `git help` — Built-in documentation

```bash
git help commit               # full manual page
git commit --help             # same thing
git commit -h                 # short summary
```

**Typical use:** When you forget the exact flag for a command. `-h` is the quickest — fits on one screen. `--help` opens the full manual.

When something goes wrong, `git status` usually tells you what state you're in and suggests the next command.

---

## Working with Others

A few habits prevent most of the problems that come up when collaborating in a shared repo.

### Always check for new commits before starting work

```bash
git switch main
git fetch
git pull
```

Starting new work from an outdated copy of `main` is the most common source of avoidable merge conflicts. Make these three commands the first thing you run before creating a branch.

### Pull regularly while working

If your branch lives for more than a day or two, periodically bring in the latest changes from `main`:

```bash
git switch my-branch
git fetch
git merge main                        # bring main's new commits into your branch
```

This keeps your branch close to the current state of the codebase and surfaces conflicts early, while they're small and easy to resolve.

### Don't rewrite shared history

Once a commit has been pushed and other people may have pulled it, treat it as permanent. Certain Git commands can *rewrite* history — replacing or removing commits that already exist on the remote — and using them on shared branches can erase teammates' work or leave their local repos in a broken state.

If you need to undo a pushed commit, use **`git revert`** (covered in the [Undoing Things](#undoing-things) section). It safely adds a *new* commit that reverses the original, leaving history intact for everyone else.

If you ever find yourself reaching for a `--force` flag or a "hard" variant of a command on a shared branch, stop and ask a teammate first.

### Keep secrets and large files out of the repo

API keys, passwords, `.env` files, and large binaries should be excluded with `.gitignore`. Once committed, secrets are difficult to fully purge from history.

### Write descriptive commit messages and PR descriptions

Your future self and your teammates will read these to understand *why* a change was made. `"fix"` tells nobody anything; `"Fix race condition in payment retry logic"` tells the whole story.

---

## A Typical Workflow

Putting the everyday commands together:

```bash
git switch main                       # 1. Start from main          (older: git checkout main)
git fetch                             # 2. Check the remote for new commits
git pull                              # 3. Bring those new commits into local main
git switch -c add-login-form          # 4. Create a branch for your work   (older: git checkout -b add-login-form)
# ...edit files in your editor...
git add .                             # 5. Stage your changes
git commit -m "Add login form"        # 6. Save the snapshot
git push -u origin add-login-form     # 7. Push to GitHub
```

The commands in parentheses are the **older equivalents** you'll see in tutorials and documentation written before `git switch` was introduced. They still work and do the same thing — `git checkout` historically handled both switching branches and restoring files, while the newer `git switch` and `git restore` split those responsibilities into clearer commands.

From there, open a **Pull Request** on GitHub so a teammate can review the branch and merge it into `main`.

---

## Quick Reference

| Command | What it does |
|---|---|
| `git init` | Start a new repo in the current folder |
| `git clone <url>` | Copy a remote repo to your machine |
| `git status` | Show what's changed and staged |
| `git add <file>` | Stage changes for the next commit |
| `git commit -m "msg"` | Save staged changes as a new commit |
| `git log` | View commit history |
| `git diff` | Show line-by-line changes |
| `git branch` | List or create branches |
| `git switch <branch>` | Move to a different branch |
| `git merge <branch>` | Combine another branch into this one |
| `git pull` | Fetch and merge remote changes |
| `git push` | Upload local commits to the remote |
| `git restore <file>` | Discard changes to a file |
| `git reset` | Move the branch pointer to an earlier commit |
| `git revert <hash>` | Make a new commit that undoes an old one |
| `git help <command>` | Open documentation for a command |
