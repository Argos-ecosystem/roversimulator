# ◈ Rover Navigator

A grid-world rover navigated by an LLM, with a BFS safety net. Built to explore how a language model behaves as a real-time planning agent under partial observability and a changing environment.

The rover starts blind. Its sensor reveals nearby cells. It builds a mental map. The LLM plans a path from that mental map. If the LLM gets stuck, a basic BFS algorithm steps in as ground truth.

---

## Features

- **Configurable grid (8×8 to 25×25)** with manual or randomized obstacle placement
- **Two obstacle types:**
  - **Fixed** — placed by you, never move
  - **Hidden / mobile** — random initial positions, drift to adjacent cells each tick at a configurable probability
- **Sensor radius (1–4 cells)** — defines how far the rover sees
- **Memory model** — rover only knows what it has scanned. Forgets cells re-scanned as empty (obstacles moved away).
- **LLM planner (OpenAI)** — gets the ASCII map of rover's memory and returns a sequence of moves
- **BFS fallback** — when the LLM produces 3+ failed plans within 8 steps, BFS computes a guaranteed-correct path through known walls
- **Stuck mode** — when BFS finds no path either, LLM is re-prompted with explicit context about what to consider (exploring `?` cells, backtracking, accepting risk)
- **Sensor-before-move** — the rover scans BEFORE committing a move, so it aborts gracefully when an obstacle appears in its path instead of crashing into it
- **Decision timeline** — every plan (LLM or BFS) is logged with its source, reasoning, and the exact ASCII map the model saw at that moment. Click any entry to inspect.
- **Mission export** — download full JSON of decisions + log for analysis

---

## Setup

Requires Python 3.9+ and an OpenAI API key.

```bash
cd ~/rover-nav
pip3 install flask openai
export OPENAI_API_KEY="sk-..."          # required
export ROVER_MODEL="gpt-4o-mini"        # optional (default)
python3 app.py
```

Then open <http://localhost:5050>.

---

## How to use

### Setup mode

1. Adjust sliders: grid size, fixed obstacles count, hidden obstacles count, sensor radius, mobile-obstacle movement probability
2. Click cells to toggle fixed walls, or hit `RANDOMIZE` to scatter walls
3. Click `PLACE ROVER` / `PLACE TARGET` then click a cell to relocate them (default: top-left and bottom-right corners)
4. Click `LAUNCH MISSION`

Your grid configuration is auto-saved to `localStorage`. Refreshing the page restores it.

### Mission mode

- **▶ GO** — single button. Plans with the LLM, executes the plan step by step, and replans automatically when the route fails
- **⏸ PAUSE / ▶ RESUME** — same button, toggles
- **Step speed** — delay between moves (LLM latency dominates anyway)
- **💾 EXPORT JSON** — download full mission state
- **↩ SETUP** — back to setup

### What the colors mean

| Symbol | Meaning |
|---|---|
| `◈` blue | Rover |
| `✦` yellow | Target |
| `▪` red | Known wall (in rover's memory) |
| `↑ ↓ ← →` green | Active plan (with direction) |
| `·` dim green | Trail — cells already visited |
| `✕` dim red | Last failed plan (visible during replan) |
| empty dark | Unscanned (rover has no info) |
| green outline | Current sensor range |

### Decision timeline

The right panel shows every plan the rover has made. Click an entry to see the exact ASCII grid that was given to the LLM (or used by BFS) at that moment. Color codes:

- 🧠 green — normal LLM plan
- ⚙ amber — BFS fallback
- ✦ pink — LLM stuck mode

---

## How it works

### One step of simulation

```
1. Hidden obstacles move (world tick)
2. Rover sensor scans surroundings → updates known_walls
3. Pop next move from current plan
4. If destination is now in known_walls → smart abort, replan (no crash)
5. If destination is wall outside prior sensor range → true crash, replan
6. Otherwise move the rover and scan from new position
```

This sequence is what makes the rover "smart enough" to abort moves when its sensor catches an obstacle in its path before commit. True crashes are rare and only happen when an obstacle moves into a cell outside the rover's sensor range from its current position.

### Memory model

- `revealed` — every cell the rover has ever scanned
- `known_walls` — cells the rover currently believes hold an obstacle. Updated each scan: adds walls seen, removes walls now confirmed empty (obstacles drifted away).
- The LLM only ever sees `known_walls` + `revealed`, never the real world. It can be wrong if hidden obstacles moved out of cells it once saw but hasn't re-scanned.

### BFS fallback trigger

The system counts how many replans happened in the last 8 steps. When that count reaches 3, BFS runs over `known_walls` and returns the shortest known path. If BFS also finds nothing (the rover's memory has it boxed in), the LLM is re-prompted with a "stuck" hint to consider exploring unknowns or backtracking.

This prevents the rover from looping forever on bad LLM advice.

### LLM contract

Prompt is a plain ASCII grid plus a small legend. The model must return JSON of the form:

```json
{"moves": ["N", "E", "E", "S", ...], "reasoning": "brief"}
```

Invalid moves (`"UP"`, strings instead of arrays, etc.) are filtered out. If the response has zero valid moves, one retry is attempted before giving up.

---

## Architecture

```
rover-nav/
├── app.py              # Flask backend + simulation + LLM/BFS planners
├── templates/
│   └── index.html      # Single-page UI (vanilla JS + CSS)
└── README.md
```

The backend keeps a single `game` dict in-memory — this is **single-tenant**. Multiple browser tabs will collide on state. Treat it as a local debugging tool, not a multi-user service.

### Endpoints

| Endpoint | Verb | Purpose |
|---|---|---|
| `/api/start` | POST | Initialize a new mission with the provided grid + config |
| `/api/plan`  | POST | Ask the planner (LLM or BFS) for the next sequence of moves |
| `/api/step`  | POST | Execute the next move in the current plan; world advances one tick |
| `/api/export`| GET  | Dump full mission JSON (decisions, log, state) |

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API credential. |
| `ROVER_MODEL` | `gpt-4o-mini` | Any OpenAI chat model with JSON mode (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.) |

OpenAI calls have a 15-second timeout.

---

## Why this exists

To get an honest feel for how a frontier LLM behaves as an embodied agent:

- Does it remember what it has seen?
- Does it explore or rush?
- How does it react when its assumptions break?
- How often does it produce invalid output?

The decision timeline + ASCII-grid inspector make the model's failure modes visible, instead of letting them hide behind a "smart agent" black box. The BFS fallback is there to keep the rover making progress while you watch what the LLM actually does.

---

## Known limitations

- Single-user (global `game` state in process memory)
- No persistence between server restarts (only `localStorage` for the setup config)
- No undo for placed obstacles
- Movement is 4-directional only (no diagonals)
- The LLM occasionally returns plans containing immediate U-turns; the BFS fallback usually catches these via the replan-window mechanism
