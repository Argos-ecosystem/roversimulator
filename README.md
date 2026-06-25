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
- **LLM planner (OpenAI)** — gets the ASCII map of rover's memory and returns a sequence of relative commands
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
3. Pop next command from current plan
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

Each planning call sends the model the current mission context, the rover's known map, and the relative command contract. The model does not receive the full hidden world; it receives the rover's memory.

The prompt includes:

- The operator mission text.
- The labeled reference points `A`, `B`, and `C`, with coordinates and visited/not-visited status.
- The rover's current coordinate and heading:
  `Rover at (row X, col Y), facing N/S/E/W`.
- The rover's start coordinate, used as `origin` / `home` / `start` if the mission mentions it.
- Recent decision history from the last few plans, so the model can continue multi-step missions.
- The model's previous mission-state fields: `phase`, `current_goal`, `next_goal`, and `notes`.
- An ASCII map of the rover's memory.
- The sensor shape/range: configured cells ahead, `2` behind, and `1` to each side.
- The list of legal commands and the required JSON response shape.

The ASCII map is the main sensor/obstacle representation:

| Symbol | Meaning |
|---|---|
| `R` | Rover current position |
| `A/B/C` | Labeled reference points |
| `#` | Known obstacle detected by sensor memory |
| `.` | Confirmed clear cell |
| `*` | Painted clear cell |
| `?` | Unscanned / unknown cell |

Sensor data is currently implicit in this ASCII map. For example, if the rover has scanned an obstacle, the model sees it as `#` at that coordinate. If a cell has not been scanned yet, the model sees `?`. The prompt does not currently include a separate list like `obstacles detected this turn: [(r,c)]`.

Legal commands are relative to the rover's current heading:

| Command | Meaning |
|---|---|
| `F` | Move 1 cell forward |
| `B` | Move 1 cell backward without changing heading |
| `L` | Rotate 90 degrees left in place |
| `R` | Rotate 90 degrees right in place |
| `P` | Paint the current cell only |

`P` can also be returned as `PINTA`, `PINTAR`, or `PAINT`; the server normalizes those aliases to `P`. Painting is intentionally side-effect free: it does not move, rotate, scan, advance obstacles, or complete a phase.

The model must return JSON of the form:

```json
{
  "moves": ["F", "R", "F", "P"],
  "reasoning": "what the rover is doing this turn and why",
  "phase": "1/2",
  "current_goal": "reach A",
  "next_goal": "return to origin",
  "notes": "anything else worth remembering",
  "done": false
}
```

Invalid commands (`"UP"`, unknown strings, non-arrays, etc.) are filtered out. If the response has zero valid commands and `done` is not true, one retry is attempted before giving up.

When `plan_iterations` is greater than `1`, later passes also receive the previous draft plan and reasoning for the same turn, plus the simulated endpoint of that draft. The model is asked to critique and improve its own command sequence.

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
| `/api/plan`  | POST | Ask the planner (LLM or BFS) for the next sequence of commands |
| `/api/step`  | POST | Execute the next command in the current plan; world advances one tick |
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
