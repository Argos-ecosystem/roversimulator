"""Rover navigator backed by an LLM with a BFS safety net."""

from flask import Flask, render_template, jsonify, request, session, redirect, render_template_string
from openai import OpenAI
from collections import deque
from functools import wraps
import random
import json
import os
import time
import secrets
from pathlib import Path


# ─── Auto-load .env if present (no python-dotenv dependency) ──────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

app = Flask(__name__)
app.secret_key = os.environ.get("ROVER_SECRET", secrets.token_hex(16))
MODEL = os.environ.get("ROVER_MODEL", "gpt-4o-mini")
PASSWORD = os.environ.get("ROVER_PASSWORD")  # if unset → auth disabled
ENV_FILE = Path(__file__).parent / ".env"


def get_client(model_name):
    """Return an OpenAI-compatible client for the given model.
    Gemini models use Google's OpenAI-compatible endpoint."""
    if model_name.startswith("gemini"):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set. Configure it in Settings (⚙).")
        return OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=15.0,
        )
    # OpenAI default
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Configure it in Settings (⚙).")
    return OpenAI(timeout=15.0)


def update_env_file(updates):
    """Update or insert keys in .env file. Also updates os.environ at runtime."""
    existing = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            existing[k.strip()] = v.strip()
    for k, v in updates.items():
        if v is None or v == "":
            existing.pop(k, None)
            os.environ.pop(k, None)
        else:
            existing[k] = v
            os.environ[k] = v
    content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    ENV_FILE.write_text(content)
    try:
        os.chmod(ENV_FILE, 0o600)
    except Exception:
        pass


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if PASSWORD and not session.get("authed"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapped


LOGIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Rover Navigator — login</title>
<style>
  body { background:#080c18; color:#c8d8c0; font-family:'Courier New',monospace;
         display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; }
  .box { background:#0c1020; border:1px solid #1a3a2a; border-radius:10px;
         padding:32px 36px; min-width:300px; box-shadow:0 0 60px #00ff8822; }
  h1 { color:#00ff88; font-size:18px; letter-spacing:4px; margin-bottom:18px;
       text-transform:uppercase; text-shadow:0 0 18px #00ff8855; }
  label { display:block; font-size:11px; color:#889988; margin-bottom:6px; letter-spacing:2px; }
  input { width:100%; background:#070a14; color:#c8d8c0; border:1px solid #1a3a2a;
          padding:10px; border-radius:4px; font-family:inherit; font-size:14px; margin-bottom:14px; }
  input:focus { outline:none; border-color:#00ff88; }
  button { width:100%; background:#0c1e14; color:#00ff88; border:1px solid #00ff88;
           padding:10px; border-radius:4px; cursor:pointer; font-family:inherit; font-size:13px;
           letter-spacing:2px; text-transform:uppercase; }
  button:hover { background:#00ff88; color:#080c18; }
  .err { color:#ff6644; font-size:12px; margin-bottom:10px; text-align:center; }
</style></head><body>
<form class="box" method="POST">
  <h1>◈ Rover Navigator</h1>
  {% if error %}<div class="err">⚠ {{ error }}</div>{% endif %}
  <label>Access password</label>
  <input type="password" name="password" autofocus required>
  <button type="submit">▶ Enter</button>
</form></body></html>"""

# Cell types in the real grid
EMPTY, FIXED, HIDDEN = 0, 1, 2

# Discoverable landmarks (letters that the rover learns about by scanning)
LANDMARK_LABELS = list("DEFGHIJKLM")  # 10 letters, won't collide with A/B/C markers
DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

# Relative rover commands
VALID_MOVES = ("F", "B", "L", "R")
TURN_LEFT  = {"N": "W", "W": "S", "S": "E", "E": "N"}
TURN_RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}
OPPOSITE   = {"N": "S", "S": "N", "E": "W", "W": "E"}


def apply_command(heading, command):
    """Return (new_heading, dr, dc). (0,0) means pure rotation in place."""
    if command == "F":
        return heading, *DIRS[heading]
    if command == "B":
        dr, dc = DIRS[OPPOSITE[heading]]
        return heading, dr, dc  # B doesn't rotate
    if command == "L":
        return TURN_LEFT[heading], 0, 0
    if command == "R":
        return TURN_RIGHT[heading], 0, 0
    raise ValueError(f"unknown command {command!r}")


def simulate_plan(rover, heading, plan):
    """Run the plan virtually and return (final_pos, final_heading)."""
    r, c = rover
    h = heading
    for cmd in plan:
        if cmd == "L":
            h = TURN_LEFT[h]
        elif cmd == "R":
            h = TURN_RIGHT[h]
        elif cmd == "F":
            dr, dc = DIRS[h]
            r, c = r + dr, c + dc
        elif cmd == "B":
            dr, dc = DIRS[OPPOSITE[h]]
            r, c = r + dr, c + dc
    return (r, c), h


def absolute_to_relative(abs_moves, start_heading):
    """Convert a BFS path [N,E,S,W,...] into rover commands [F,L,R,B,...]."""
    commands = []
    h = start_heading
    for m in abs_moves:
        if m == h:
            commands.append("F")
        elif m == OPPOSITE[h]:
            commands.append("B")  # heading unchanged
        elif TURN_LEFT[h] == m:
            commands.append("L")
            commands.append("F")
            h = m
        else:  # TURN_RIGHT[h] == m
            commands.append("R")
            commands.append("F")
            h = m
    return commands
BACK_RANGE = 2              # fixed: rover sees 2 cells behind it
SIDE_RANGE = 1              # fixed: rover sees 1 cell to each side
REPLAN_WINDOW = 8
REPLAN_THRESHOLD = 3


def initial_heading(rover, target):
    dr = target[0] - rover[0]
    dc = target[1] - rover[1]
    if abs(dr) >= abs(dc):
        return "S" if dr >= 0 else "N"
    return "E" if dc >= 0 else "W"


def sensor_cells(rover_pos, heading, gs, forward_range):
    """Cross-shaped sensor: forward_range ahead, 2 back, 1 each side."""
    r, c = rover_pos
    cells = {(r, c)}
    fwd = DIRS[heading]
    back = (-fwd[0], -fwd[1])
    if heading in ("N", "S"):
        sides = [(0, -1), (0, 1)]
    else:
        sides = [(-1, 0), (1, 0)]

    for vec, rng in [(fwd, forward_range), (back, BACK_RANGE)]:
        for i in range(1, rng + 1):
            nr, nc = r + vec[0] * i, c + vec[1] * i
            if 0 <= nr < gs and 0 <= nc < gs:
                cells.add((nr, nc))
            else:
                break

    for side in sides:
        for i in range(1, SIDE_RANGE + 1):
            nr, nc = r + side[0] * i, c + side[1] * i
            if 0 <= nr < gs and 0 <= nc < gs:
                cells.add((nr, nc))
            else:
                break
    return cells

game = {}


# ─── World / state init ───────────────────────────────────────────────────
def init_game(forward_range, grid_size, hidden_count, move_prob,
              move_prob_fixed=0.0, manual_grid=None, rover=None, target=None,
              targets=None, model=None, mission=None, plan_iterations=1):
    rover = list(rover) if rover else [0, 0]

    # Normalize targets: prefer multi-target list, fallback to single target
    if targets:
        target_list = []
        for t in targets:
            if isinstance(t, dict):
                target_list.append({"label": t.get("label", "?"), "pos": list(t["pos"]), "visited": False})
            else:
                target_list.append({"label": "A", "pos": list(t), "visited": False})
    else:
        legacy = list(target) if target else [grid_size - 1, grid_size - 1]
        target_list = [{"label": "A", "pos": legacy, "visited": False}]

    # Primary target (for BFS rescue + back-compat): first unvisited
    primary_target = target_list[0]["pos"]

    if manual_grid:
        grid = [row[:] for row in manual_grid]
    else:
        grid = [[EMPTY] * grid_size for _ in range(grid_size)]

    placed = set()
    has_hidden_already = False
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r][c] == FIXED:
                placed.add((r, c))
            elif grid[r][c] == HIDDEN:
                placed.add((r, c))
                has_hidden_already = True
    placed.add(tuple(rover))
    for t in target_list:
        placed.add(tuple(t["pos"]))
        grid[t["pos"][0]][t["pos"][1]] = EMPTY
    grid[rover[0]][rover[1]] = EMPTY

    # Only randomize hidden if the imported grid doesn't already include them
    if not has_hidden_already:
        for _ in range(hidden_count):
            for _ in range(300):
                r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                if (r, c) not in placed:
                    grid[r][c] = HIDDEN
                    placed.add((r, c))
                    break

    # Place discoverable landmarks (D-M) on empty cells. They don't block movement.
    landmarks = []
    for label in LANDMARK_LABELS:
        for _ in range(300):
            r = random.randint(0, grid_size - 1)
            c = random.randint(0, grid_size - 1)
            if (r, c) not in placed and grid[r][c] == EMPTY:
                landmarks.append({"label": label, "pos": [r, c], "discovered": False})
                placed.add((r, c))
                break

    state = {
        "grid": grid,
        "rover": rover,
        "target": primary_target,
        "targets": target_list,
        "heading": initial_heading(rover, primary_target),
        "forward_range": forward_range,
        "back_range": BACK_RANGE,
        "side_range": SIDE_RANGE,
        "grid_size": grid_size,
        "move_prob": move_prob,
        "move_prob_fixed": move_prob_fixed,
        "model": model or MODEL,
        "mission": (mission or "Reach the target efficiently.").strip(),
        "plan_iterations": max(1, min(int(plan_iterations or 1), 6)),
        "landmarks": landmarks,
        "revealed": [],
        "known_walls": [],
        "plan": [],
        "log": [],
        "decisions": [],          # full timeline for LLM/BFS calls
        "replan_steps": [],       # step numbers when replans happened
        "done": False,
        "recalculations": 0,
        "steps": 0,
        "started_at": time.time(),
        "start_pos": list(rover),
        "initial_grid": [row[:] for row in grid],   # snapshot for export/replay
        "metrics": {
            "llm_calls": 0, "llm_total_ms": 0.0, "llm_last_ms": 0.0,
            "bfs_calls": 0, "stuck_calls": 0,
            "crashes": 0, "aborts": 0, "boundary_hits": 0,
            "rotations": 0, "forwards": 0, "backwards": 0,
            "last_plan_total": 0, "last_plan_used": 0,
            "plan_completion_rates": [],
        },
    }
    reveal_sensor(state, log_events=False)
    return state


# ─── Sensor / memory model ────────────────────────────────────────────────
def reveal_sensor(state, log_events=True):
    """Directional cone-cross scan. Updates memory based on rover heading."""
    gs = state["grid_size"]
    grid = state["grid"]
    revealed = set(map(tuple, state["revealed"]))
    known = set(map(tuple, state["known_walls"]))
    new_walls, cleared = [], []
    scan = sensor_cells(state["rover"], state["heading"], gs, state["forward_range"])

    for (nr, nc) in scan:
        first_scan = (nr, nc) not in revealed
        revealed.add((nr, nc))
        cell = grid[nr][nc]
        if cell in (FIXED, HIDDEN):
            if (nr, nc) not in known:
                new_walls.append((nr, nc))
            known.add((nr, nc))
        else:
            if (nr, nc) in known and not first_scan:
                cleared.append((nr, nc))
            known.discard((nr, nc))

    state["revealed"] = [list(x) for x in revealed]
    state["known_walls"] = [list(x) for x in known]

    # Discover landmarks within sensor scan
    new_landmarks = []
    for lm in state.get("landmarks", []):
        if not lm["discovered"] and tuple(lm["pos"]) in scan:
            lm["discovered"] = True
            new_landmarks.append(lm)

    if log_events:
        if new_walls:
            coords = ", ".join(f"({r},{c})" for r, c in new_walls[:4])
            extra = f" +{len(new_walls)-4} more" if len(new_walls) > 4 else ""
            state["log"].append(f"📡 Sensor detected wall at {coords}{extra}")
        if cleared:
            coords = ", ".join(f"({r},{c})" for r, c in cleared[:3])
            state["log"].append(f"💨 Memory cleared at {coords} (obstacle moved)")
        if new_landmarks:
            txt = ", ".join(f"{lm['label']} at ({lm['pos'][0]},{lm['pos'][1]})" for lm in new_landmarks)
            state["log"].append(f"🔎 Discovered landmark: {txt}")


def move_obstacles(state, prob_hidden, prob_fixed):
    """World tick: each obstacle may walk to a random empty neighbour.
    Hidden and fixed obstacles can move with independent probabilities."""
    gs = state["grid_size"]
    grid = state["grid"]
    rover = tuple(state["rover"])
    target = tuple(state["target"])
    forbidden = {rover, target}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    movers = [
        (r, c, grid[r][c])
        for r in range(gs) for c in range(gs)
        if grid[r][c] in (FIXED, HIDDEN)
    ]
    for r, c, kind in movers:
        prob = prob_fixed if kind == FIXED else prob_hidden
        if prob <= 0 or random.random() > prob:
            continue
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (0 <= nr < gs and 0 <= nc < gs
                    and grid[nr][nc] == EMPTY
                    and (nr, nc) not in forbidden):
                grid[r][c] = EMPTY
                grid[nr][nc] = kind
                break


# ─── BFS fallback (uses rover memory only) ────────────────────────────────
def next_unvisited_target(state):
    """Pick the closest unvisited target (by Manhattan from rover)."""
    rover = state["rover"]
    pending = [t for t in state.get("targets", []) if not t.get("visited")]
    if not pending:
        return None
    return min(pending, key=lambda t: abs(t["pos"][0]-rover[0]) + abs(t["pos"][1]-rover[1]))


def update_target_visits(state):
    """Mark targets visited if rover stands on them. Refresh primary target."""
    rover_pos = list(state["rover"])
    newly_visited = []
    for t in state.get("targets", []):
        if not t.get("visited") and t["pos"] == rover_pos:
            t["visited"] = True
            newly_visited.append(t["label"])
    nxt = next_unvisited_target(state)
    if nxt:
        state["target"] = nxt["pos"]
    return newly_visited


def bfs_path(state):
    gs = state["grid_size"]
    start = tuple(state["rover"])
    target = tuple(state["target"])
    walls = set(map(tuple, state["known_walls"]))
    if start == target:
        return []
    visited = {start}
    q = deque([(start, [])])
    while q:
        (r, c), path = q.popleft()
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if (0 <= nr < gs and 0 <= nc < gs
                    and (nr, nc) not in walls
                    and (nr, nc) not in visited):
                new_path = path + [d]
                if (nr, nc) == target:
                    return new_path
                visited.add((nr, nc))
                q.append(((nr, nc), new_path))
    return None


# ─── LLM helpers ──────────────────────────────────────────────────────────
def build_llm_ascii(state):
    gs = state["grid_size"]
    rover = tuple(state["rover"])
    target = tuple(state["target"])
    revealed = set(map(tuple, state["revealed"]))
    walls = set(map(tuple, state["known_walls"]))

    rows = []
    for r in range(gs):
        row = []
        for c in range(gs):
            if (r, c) == rover:
                row.append("R")
            elif (r, c) == target:
                row.append("T")
            elif (r, c) in walls:
                row.append("#")
            elif (r, c) in revealed:
                row.append(".")
            else:
                row.append("?")
        rows.append(" ".join(row))
    return "\n".join(rows)


def validate_moves(raw):
    """Return (clean_moves, dropped_items). Coerces strings/CSV to list."""
    if isinstance(raw, str):
        raw = [x.strip().upper() for x in raw.replace(",", " ").split()]
    if not isinstance(raw, list):
        return [], [raw]
    clean = [m for m in raw if isinstance(m, str) and m.upper() in VALID_MOVES]
    clean = [m.upper() for m in clean]
    dropped = [m for m in raw if not (isinstance(m, str) and m.upper() in VALID_MOVES)]
    return clean, dropped


def ask_llm(state, stuck_hint=None, retries=1, refine_context=None, tactical_phase=None):
    ascii_grid = build_llm_ascii(state)
    rover = state["rover"]
    target = state["target"]
    gs = state["grid_size"]

    heading = state.get("heading", "S")
    fwd_r = state.get("forward_range", 3)
    mission = state.get("mission", "Reach all targets efficiently.")
    targets = state.get("targets", [])

    marker_lines = []
    for t in targets:
        seen = "visited" if t.get("visited") else "not visited yet"
        marker_lines.append(f"  {t['label']} → (row {t['pos'][0]}, col {t['pos'][1]})  [{seen}]")
    marker_block = "\n".join(marker_lines) if marker_lines else "  (no markers placed)"

    discovered = [lm for lm in state.get("landmarks", []) if lm.get("discovered")]
    if discovered:
        lm_lines = "\n".join(f"  {lm['label']} at (row {lm['pos'][0]}, col {lm['pos'][1]})" for lm in discovered)
        landmark_block = lm_lines
    else:
        landmark_block = "  (none discovered yet — they appear as the rover scans them)"

    # Recent decisions for continuity across plan calls
    recent = state.get("decisions", [])[-3:]
    if recent:
        hist = "\n".join(f"  • step {d['step']} from ({d['rover'][0]},{d['rover'][1]}) [{d['source']}]: {d['reasoning'][:140]}" for d in recent)
    else:
        hist = "  (this is the first decision of the mission)"

    notes = state.get("mission_notes", "")
    phase = state.get("mission_phase", "")
    current_goal = state.get("current_goal", "")
    next_goal = state.get("next_goal", "")

    state_block_lines = []
    if phase:        state_block_lines.append(f"  Phase: {phase}")
    if current_goal: state_block_lines.append(f"  Current goal: {current_goal}")
    if next_goal:    state_block_lines.append(f"  Next goal: {next_goal}")
    if notes:        state_block_lines.append(f"  Free notes: {notes}")
    mission_state_block = "\n".join(state_block_lines) if state_block_lines else "  (first decision of the mission — define the phase plan and start)"

    # Detect "achievement": current_goal mentions a marker that's now visited.
    # This catches the common bug where the LLM keeps planning toward the old goal.
    achievement_hint = ""
    if current_goal:
        achieved = []
        for t in targets:
            if t.get("visited") and t["label"] in current_goal:
                achieved.append(t["label"])
        if achieved:
            achievement_hint = (
                f"\n⚡ ACHIEVEMENT JUST UNLOCKED: marker(s) {', '.join(achieved)} are NOW VISITED. "
                f"Your previous current_goal ('{current_goal}') is COMPLETE. "
                f"Switch to next_goal ('{next_goal or 'declare done if mission is finished'}') THIS turn."
            )

    start_pos = state.get("start_pos", rover)
    at_origin = list(rover) == list(start_pos)
    at_origin_hint = ""
    if at_origin and state.get("steps", 0) > 0:
        at_origin_hint = (
            "\n⚡ YOU ARE BACK AT ORIGIN (rover position == start position). "
            "If your mission says to return home / origin / start, this is the moment to set \"done\": true."
        )

    base = f"""You are the navigation AI for an autonomous rover on a {gs}x{gs} grid.

MISSION (operator instruction — HIGHEST PRIORITY, this defines what success means):
  {mission or "(no mission given — explore at your discretion)"}

LABELED REFERENCE POINTS on the map (you decide what to do with them based on the mission):
{marker_block}

DISCOVERED LANDMARKS (random letters the rover has scanned and now remembers their position — useful for navigation reference):
{landmark_block}

CURRENT STATE
  Rover at (row {rover[0]}, col {rover[1]}), facing {heading}.
  Rover STARTED at (row {start_pos[0]}, col {start_pos[1]})  ← this is "origin" / "home" / "start point" if the mission mentions it.

RECENT DECISIONS (your own past plans — use these for continuity across multi-step missions):
{hist}

MISSION STATE (what YOU declared last call — read this before deciding what to do next):
{mission_state_block}

GRID (row 0 = top, col 0 = left, N=up S=down E=right W=left):
{ascii_grid}

Legend:
  R = rover (facing {heading})        A/B/C = labeled reference points (NOT mandatory destinations — see mission)
  # = known obstacle (impassable)     . = confirmed clear     ? = unscanned

RELATIVE COMMANDS (interpreted from current heading):
  F = move 1 cell FORWARD in heading direction
  B = move 1 cell BACKWARD (opposite of heading; heading does NOT change)
  L = rotate 90° LEFT in place (no movement)
  R = rotate 90° RIGHT in place (no movement)

SENSOR (cross-shape, rotates with heading):
  {fwd_r} cells ahead · 2 behind · 1 each side

INTERPRETATION RULES:
  • The MISSION text is your contract. A/B/C are NOT automatic goals — they are just points you can reference if the mission says so.
  • If the mission says "visit A then B" → visit them.
  • If the mission says "ignore C" or "stay away from B" → respect that.
  • If the mission gives a free instruction (patrol, explore, etc.) → use your judgment.
  • You may use up to {gs * 6} commands per plan. Plan partial routes — you'll be called again as you progress.
  • Avoid # cells. Prefer . over ? but cross ? if necessary.

MULTI-STEP MISSIONS — STATE MACHINE PROTOCOL:
  Every call you MUST update these structured fields so the next-you can pick up coherently:
    - "phase":         e.g. "1/2", "2/3", "single" — which step of the mission you're on
    - "current_goal":  short description of THIS turn's objective (e.g. "reach A")
    - "next_goal":     what comes after this is done (e.g. "return to origin (0,0)") — empty if mission ends here
    - "notes":         free scratchpad (anything else worth remembering)
  Read the MISSION STATE block above carefully. If phase/current_goal already exist, you are CONTINUING — don't restart planning from scratch.
  When the FULL mission is complete (all phases done), signal with "done": true.{at_origin_hint}{achievement_hint}

Respond ONLY with valid JSON (no markdown):
{{
  "moves": ["F","R","F",...],
  "reasoning": "what you're doing this turn and why",
  "phase": "1/2",
  "current_goal": "reach A",
  "next_goal": "return to origin",
  "notes": "anything else worth remembering",
  "done": false
}}"""
    if stuck_hint:
        base += f"\n\n⚠ HINT (rover seems stuck):\n{stuck_hint}"

    # Tactical mode: focus on a single phase, server handles transitions
    if tactical_phase:
        phase = tactical_phase
        base += f"""

═══ TACTICAL EXECUTION MODE ═══
You are NOT planning the whole mission. The server already decomposed it into phases and is tracking which one you're on. You only need to plan moves for THIS phase:

  Phase goal: {phase['goal']}
  End condition: rover must reach position (row {phase['end_when_pos'][0]}, col {phase['end_when_pos'][1]})

Ignore previous mission state — focus purely on getting from your current position to the phase target. The server will auto-advance to the next phase when you arrive.

Set "done": false (server decides when whole mission is done).
You can leave phase/current_goal/next_goal empty — server tracks those now."""

    # Iterative refinement: feed the model its own previous draft to critique + improve
    if refine_context:
        rc = refine_context
        sim_pos, _ = simulate_plan(rover, heading, rc["prev_plan"])
        base += f"""

═══ REFINEMENT PASS {rc['pass']}/{rc['total']} ═══
This is NOT a fresh plan. You already drafted one for THIS SAME turn. Critique and improve it.

Your previous draft:
  moves: {rc['prev_plan']}
  reasoning: {rc['prev_reasoning']}
  → if executed, the rover would end at row {sim_pos[0]}, col {sim_pos[1]}

Critique it honestly:
  • Does it actually reach this turn's goal? (it ends at {sim_pos})
  • Does any step cross a # obstacle?
  • Are there wasteful rotations or back-and-forth moves?
  • Does it respect the mission?

Output an IMPROVED plan in the same JSON format. If the draft was already optimal, return it unchanged."""

    model_name = state.get("model", MODEL)
    client = get_client(model_name)

    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                max_tokens=800,
                response_format={"type": "json_object"},
                timeout=15,
                messages=[
                    {"role": "system",
                     "content": "You are a precise rover navigation AI. Reply with valid JSON only."},
                    {"role": "user", "content": base},
                ],
            )
            elapsed_ms = (time.time() - t0) * 1000
            m = state.get("metrics", {})
            m["llm_calls"] = m.get("llm_calls", 0) + 1
            m["llm_total_ms"] = m.get("llm_total_ms", 0.0) + elapsed_ms
            m["llm_last_ms"] = elapsed_ms
            data = json.loads(resp.choices[0].message.content)
            done_flag = bool(data.get("done"))
            clean, dropped = validate_moves(data.get("moves", []))
            if not clean and not done_flag:
                last_err = f"no valid moves in response (got {data.get('moves')!r})"
                continue
            reasoning = data.get("reasoning", "").strip() or "(no reasoning)"
            if dropped:
                reasoning += f"  [⚠ dropped invalid: {dropped}]"
            notes = (data.get("notes") or "").strip()
            phase = (data.get("phase") or "").strip()
            current_goal = (data.get("current_goal") or "").strip()
            next_goal = (data.get("next_goal") or "").strip()
            return {
                "moves": clean,
                "reasoning": reasoning,
                "ascii_grid": ascii_grid,
                "done": done_flag,
                "notes": notes,
                "phase": phase,
                "current_goal": current_goal,
                "next_goal": next_goal,
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = str(e)
            continue
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
    raise RuntimeError(f"LLM gave unusable response after retries: {last_err}")


def ask_llm_strategy(state):
    """High-level decomposition: ONE call at the start of the mission.
    Asks the LLM to break the mission into checkpointed phases. The server
    will then track phase progression deterministically, freeing the LLM
    from having to remember the full mission across calls."""
    rover = state["rover"]
    target = state["target"]
    targets = state.get("targets", [])
    gs = state["grid_size"]
    mission = state.get("mission", "Reach the target.")
    start_pos = state.get("start_pos", rover)

    marker_lines = [
        f"  {t['label']} at (row {t['pos'][0]}, col {t['pos'][1]})"
        for t in targets
    ]
    marker_block = "\n".join(marker_lines) if marker_lines else "  (none placed)"

    prompt = f"""You are a strategic mission planner for an autonomous rover. Decompose the
operator's mission into an ORDERED LIST OF PHASES. Each phase has a goal in plain text
and an END CONDITION expressed as either a target cell position or a marker label.

The rover will execute phases sequentially. The server detects when a phase's end
condition is met and auto-advances. Your job is ONLY to define the phases.

MISSION TEXT (operator):
  {mission}

WORLD INFO:
  Grid size: {gs}x{gs}
  Rover start position: (row {start_pos[0]}, col {start_pos[1]})
  Markers on the map:
{marker_block}

OUTPUT JSON SCHEMA (no markdown, no commentary):
{{
  "phases": [
    {{"goal": "short description of this phase", "end_when_pos": [row, col]}},
    {{"goal": "next phase", "end_when_pos": [row, col]}}
  ]
}}

RULES:
  • Use exact coordinates [row, col]. If a phase ends at marker A, use A's coordinates.
  • If the mission says "return to origin" or "go back home", use the rover's start position.
  • Order matters — phases run in sequence.
  • Keep phases atomic: one location per phase.
  • If the mission has no real phases (single destination), output ONE phase only.
  • Be literal: respect order ("then", "after that"), exclusions ("ignore B"), and conditions.

Now produce the decomposition for the mission above."""

    model_name = state.get("model", MODEL)
    client = get_client(model_name)

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        max_tokens=600,
        response_format={"type": "json_object"},
        timeout=15,
        messages=[
            {"role": "system", "content": "You are a strategic mission planner. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    elapsed_ms = (time.time() - t0) * 1000
    m = state.get("metrics", {})
    m["llm_calls"] = m.get("llm_calls", 0) + 1
    m["llm_total_ms"] = m.get("llm_total_ms", 0.0) + elapsed_ms
    m["llm_last_ms"] = elapsed_ms

    data = json.loads(resp.choices[0].message.content)
    raw_phases = data.get("phases", [])
    phases = []
    for i, p in enumerate(raw_phases):
        pos = p.get("end_when_pos") or p.get("pos") or p.get("target")
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        phases.append({
            "idx": i,
            "goal": p.get("goal", f"phase {i+1}"),
            "end_when_pos": [int(pos[0]), int(pos[1])],
            "done": False,
        })
    if not phases:
        # Fallback: single phase to the primary target
        phases = [{"idx": 0, "goal": "reach target", "end_when_pos": list(target), "done": False}]
    return phases


def current_phase(state):
    strat = state.get("strategy") or []
    idx = state.get("current_phase_idx", 0)
    if 0 <= idx < len(strat):
        return strat[idx]
    return None


def check_phase_completion(state):
    """If the rover is at the end_when_pos of the current phase, mark it done and advance.
    Returns the label of the completed phase (or None)."""
    phase = current_phase(state)
    if not phase:
        return None
    if list(state["rover"]) == list(phase["end_when_pos"]):
        phase["done"] = True
        state["current_phase_idx"] = state.get("current_phase_idx", 0) + 1
        return phase
    return None


def iterative_plan(state, iterations, tactical_phase=None):
    """Turn a normal model into a 'thinking' one: draft a plan, then refine it
    over N passes. Optionally constrained to a tactical phase."""
    iterations = max(1, min(int(iterations), 6))
    result = None
    history = []
    for i in range(iterations):
        if result is None:
            result = ask_llm(state, tactical_phase=tactical_phase)
        else:
            result = ask_llm(state, tactical_phase=tactical_phase, refine_context={
                "pass": i + 1,
                "total": iterations,
                "prev_plan": result["moves"],
                "prev_reasoning": result["reasoning"],
            })
        history.append({
            "pass": i + 1,
            "moves": list(result["moves"]),
            "reasoning": result["reasoning"],
        })
        if iterations > 1:
            tag = "draft" if i == 0 else f"refine {i+1}/{iterations}"
            state["log"].append(
                f"🔄 Plan {tag}: {len(result['moves'])} moves — {result['reasoning'][:90]}"
            )
    result["refine_history"] = history
    return result


# ─── Decision recorder ────────────────────────────────────────────────────
def record_plan_completion(state):
    """When a plan ends (replan or new plan), log the % used."""
    m = state["metrics"]
    if m["last_plan_total"] > 0:
        ratio = m["last_plan_used"] / m["last_plan_total"]
        m["plan_completion_rates"].append(ratio)


def start_new_plan_tracking(state, plan_len):
    state["metrics"]["last_plan_total"] = plan_len
    state["metrics"]["last_plan_used"] = 0


def record_decision(state, source, plan, reasoning, ascii_grid):
    state["decisions"].append({
        "id": len(state["decisions"]),
        "step": state["steps"],
        "rover": list(state["rover"]),
        "source": source,        # "llm" | "bfs" | "llm_stuck"
        "plan": list(plan),
        "reasoning": reasoning,
        "ascii_grid": ascii_grid,
        "timestamp": time.time(),
    })


def replans_in_window(state):
    threshold = state["steps"] - REPLAN_WINDOW
    return sum(1 for s in state["replan_steps"] if s >= threshold)


# ─── HTTP API ─────────────────────────────────────────────────────────────
def client_state(state):
    m = state.get("metrics", {})
    rates = m.get("plan_completion_rates", [])

    # Stale walls: cells the rover still believes are walls, but the world
    # has actually moved on (hidden obstacle drifted away unseen). God-view debug.
    gs = state["grid_size"]
    grid = state["grid"]
    real_walls = {(r, c) for r in range(gs) for c in range(gs) if grid[r][c] in (FIXED, HIDDEN)}
    known_set = {tuple(w) for w in state["known_walls"]}
    stale_walls = [list(w) for w in (known_set - real_walls)]
    rover = state["rover"]
    target = state["target"]
    start = state.get("start_pos", rover)
    manhattan_remaining = abs(target[0] - rover[0]) + abs(target[1] - rover[1])
    manhattan_initial = abs(target[0] - start[0]) + abs(target[1] - start[1])
    progress = manhattan_initial - manhattan_remaining
    steps = max(1, state["steps"])
    metrics = {
        "elapsed_s": round(time.time() - state.get("started_at", time.time()), 1),
        "manhattan_remaining": manhattan_remaining,
        "manhattan_initial": manhattan_initial,
        "path_efficiency": round(progress / steps, 3) if state["steps"] else 0.0,
        "plan_completion_avg": round(sum(rates) / len(rates), 3) if rates else None,
        "llm_calls": m.get("llm_calls", 0),
        "llm_avg_ms": round(m.get("llm_total_ms", 0) / max(1, m.get("llm_calls", 0)), 0) if m.get("llm_calls") else 0,
        "llm_last_ms": round(m.get("llm_last_ms", 0), 0),
        "bfs_calls": m.get("bfs_calls", 0),
        "stuck_calls": m.get("stuck_calls", 0),
        "crashes": m.get("crashes", 0),
        "aborts": m.get("aborts", 0),
        "boundary_hits": m.get("boundary_hits", 0),
        "rotations": m.get("rotations", 0),
        "forwards": m.get("forwards", 0),
        "backwards": m.get("backwards", 0),
    }
    return {
        "rover": state["rover"],
        "target": state["target"],
        "targets": state.get("targets", []),
        "landmarks": [lm for lm in state.get("landmarks", []) if lm.get("discovered")],
        "mission_notes": state.get("mission_notes", ""),
        "plan_iterations": state.get("plan_iterations", 1),
        "strategy": state.get("strategy", []),
        "current_phase_idx": state.get("current_phase_idx", 0),
        "mission_phase": state.get("mission_phase", ""),
        "current_goal": state.get("current_goal", ""),
        "next_goal": state.get("next_goal", ""),
        "revealed": state["revealed"],
        "known_walls": state["known_walls"],
        "stale_walls": stale_walls,
        "plan": state["plan"],
        "log": state["log"][-60:],
        "decisions": state["decisions"][-30:],
        "done": state["done"],
        "steps": state["steps"],
        "recalculations": state["recalculations"],
        "replans_window": replans_in_window(state),
        "grid_size": state["grid_size"],
        "heading": state["heading"],
        "forward_range": state["forward_range"],
        "back_range": state["back_range"],
        "side_range": state["side_range"],
        "metrics": metrics,
        "model": state.get("model", MODEL),
        "mission": state.get("mission", ""),
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if not PASSWORD:
        return redirect("/")
    error = None
    if request.method == "POST":
        if request.form.get("password") == PASSWORD:
            session["authed"] = True
            return redirect("/")
        error = "Wrong password"
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login" if PASSWORD else "/")


SETTINGS_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Settings — Rover Navigator</title>
<style>
  body { background:#080c18; color:#c8d8c0; font-family:'Courier New',monospace;
         margin:0; padding:30px; min-height:100vh; }
  .box { max-width:520px; margin:0 auto; background:#0c1020;
         border:1px solid #1a3a2a; border-radius:10px; padding:28px; }
  h1 { color:#00ff88; font-size:16px; letter-spacing:3px; margin-bottom:18px;
       text-transform:uppercase; text-shadow:0 0 18px #00ff8855;
       display:flex; justify-content:space-between; align-items:center; }
  h1 a { color:#667; font-size:11px; text-decoration:none; }
  h3 { font-size:10px; letter-spacing:3px; text-transform:uppercase;
       color:#44cc88; margin:18px 0 8px; border-bottom:1px solid #1a2e1a; padding-bottom:6px; }
  label { display:block; font-size:11px; color:#889988; margin:8px 0 4px; }
  .current { font-size:11px; color:#667; margin-bottom:4px; font-style:italic; }
  input { width:100%; background:#070a14; color:#c8d8c0; border:1px solid #1a3a2a;
          padding:9px; border-radius:4px; font-family:inherit; font-size:13px; }
  input:focus { outline:none; border-color:#00ff88; }
  button { width:100%; background:#0c1e14; color:#00ff88; border:1px solid #00ff88;
           padding:10px; border-radius:4px; cursor:pointer; font-family:inherit; font-size:12px;
           letter-spacing:2px; text-transform:uppercase; margin-top:18px; }
  button:hover { background:#00ff88; color:#080c18; }
  .msg { padding:10px; border-radius:4px; margin:10px 0; font-size:12px; display:none; }
  .msg.ok  { display:block; background:#0d2d1a; border:1px solid #00ff88; color:#00ff88; }
  .msg.err { display:block; background:#2a0d0d; border:1px solid #ff4444; color:#ff8866; }
  .note { font-size:10px; color:#556; margin-top:6px; line-height:1.5; }
</style></head><body>
<div class="box">
  <h1>⚙ Settings <a href="/">↩ back to simulator</a></h1>

  <h3>OpenAI</h3>
  <div class="current">current: {{ openai_masked }}</div>
  <label>OpenAI API key (gpt-* models)</label>
  <input id="openai_key" type="password" placeholder="sk-proj-..." autocomplete="off">

  <h3>Google Gemini</h3>
  <div class="current">current: {{ gemini_masked }}</div>
  <label>Gemini API key (gemini-* models)</label>
  <input id="gemini_key" type="password" placeholder="AIza..." autocomplete="off">
  <div class="note">Get yours at <span style="color:#88aacc">aistudio.google.com/apikey</span></div>

  <h3>Access password</h3>
  <div class="current">current: {{ password_status }}</div>
  <label>New password (leave blank to keep current)</label>
  <input id="new_password" type="password" placeholder="new password" autocomplete="off">

  <div id="msg" class="msg"></div>
  <button onclick="save()">▶ SAVE</button>
</div>
<script>
async function save() {
  const body = {
    openai_key: document.getElementById('openai_key').value.trim(),
    gemini_key: document.getElementById('gemini_key').value.trim(),
    new_password: document.getElementById('new_password').value.trim(),
  };
  const msg = document.getElementById('msg');
  msg.className = 'msg'; msg.textContent = '';
  try {
    const r = await fetch('/api/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if (!r.ok || data.error) { msg.className='msg err'; msg.textContent='✗ '+(data.error||r.statusText); return; }
    msg.className='msg ok'; msg.textContent='✓ Saved. ' + (data.note || '');
    ['openai_key','gemini_key','new_password'].forEach(id => document.getElementById(id).value='');
    setTimeout(()=>location.reload(), 1500);
  } catch(e) {
    msg.className='msg err'; msg.textContent='✗ '+e.message;
  }
}
</script>
</body></html>"""


def _mask(value):
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return "***"
    return value[:4] + "…" + value[-4:]


@app.route("/settings", methods=["GET"])
@login_required
def settings_page():
    return render_template_string(
        SETTINGS_HTML,
        openai_masked=_mask(os.environ.get("OPENAI_API_KEY")),
        gemini_masked=_mask(os.environ.get("GEMINI_API_KEY")),
        password_status="set" if PASSWORD else "(none — auth disabled)",
    )


@app.route("/api/settings", methods=["POST"])
@login_required
def api_settings():
    global PASSWORD
    data = request.json or {}
    updates = {}
    notes = []
    if data.get("openai_key"):
        updates["OPENAI_API_KEY"] = data["openai_key"]
        notes.append("OpenAI key updated.")
    if data.get("gemini_key"):
        updates["GEMINI_API_KEY"] = data["gemini_key"]
        notes.append("Gemini key updated.")
    if data.get("new_password"):
        updates["ROVER_PASSWORD"] = data["new_password"]
        PASSWORD = data["new_password"]
        notes.append("Password changed.")
    if not updates:
        return jsonify({"error": "Nothing to update."}), 400
    try:
        update_env_file(updates)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "note": " ".join(notes)})


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
@login_required
def start():
    global game
    d = request.json or {}
    game = init_game(
        forward_range=d.get("forward_range", 3),
        grid_size=d.get("grid_size", 15),
        hidden_count=d.get("hidden_count", 7),
        move_prob=d.get("move_prob", 0.5),
        move_prob_fixed=d.get("move_prob_fixed", 0.0),
        manual_grid=d.get("grid"),
        rover=d.get("rover"),
        target=d.get("target"),
        model=d.get("model"),
        mission=d.get("mission"),
        plan_iterations=d.get("plan_iterations", 1),
        targets=d.get("targets"),
    )
    game["log"].append(
        f"🚀 Mission start: {game['grid_size']}×{game['grid_size']}, sensor fwd={game['forward_range']}/back={BACK_RANGE}/side={SIDE_RANGE}, "
        f"heading={game['heading']}, rover={game['rover']}, target={game['target']}"
    )
    return jsonify(client_state(game))


@app.route("/api/plan", methods=["POST"])
@login_required
def plan():
    global game
    if not game:
        return jsonify({"error": "no game running"}), 400

    n = replans_in_window(game)
    rover = tuple(game["rover"])
    target = tuple(game["target"])

    # --- BFS fallback if LLM keeps failing ---
    if n >= REPLAN_THRESHOLD:
        abs_path = bfs_path(game)
        if abs_path:
            commands = absolute_to_relative(abs_path, game["heading"])
            ascii_grid = build_llm_ascii(game)
            record_plan_completion(game)
            game["plan"] = commands
            game["replan_steps"] = []
            game["metrics"]["bfs_calls"] += 1
            start_new_plan_tracking(game, len(commands))
            reasoning = (
                f"BFS fallback after {n} replans in {REPLAN_WINDOW} steps. "
                f"BFS abs path: {len(abs_path)} steps → {len(commands)} relative commands."
            )
            record_decision(game, "bfs", commands, reasoning, ascii_grid)
            game["log"].append(f"⚙ {reasoning}")
            game["log"].append(f"📋 Context shown to planner (step {game['steps']}):\n{ascii_grid}")
            return jsonify({**client_state(game), "source": "bfs", "reasoning": reasoning})

        # No BFS path either: hand it to the LLM with explicit context
        try:
            hint = (
                f"BFS could find NO path with current memory ({len(game['known_walls'])} known walls). "
                "Options: (a) move toward ? cells to gather info, (b) accept risk and pass through ? cells, "
                "(c) backtrack to a previous position. Choose and justify."
            )
            r = ask_llm(game, stuck_hint=hint)
            record_plan_completion(game)
            game["plan"] = r["moves"]
            game["replan_steps"] = []
            game["metrics"]["stuck_calls"] += 1
            start_new_plan_tracking(game, len(r["moves"]))
            record_decision(game, "llm_stuck", r["moves"], r["reasoning"], r["ascii_grid"])
            game["log"].append(f"✦ STUCK MODE — {r['reasoning']}")
            game["log"].append(f"📋 Context shown to LLM (step {game['steps']}):\n{r['ascii_grid']}")
            return jsonify({**client_state(game), "source": "llm_stuck", "reasoning": r["reasoning"]})
        except Exception as e:
            game["log"].append(f"✗ LLM error: {e}")
            return jsonify({"error": str(e)}), 500

    # --- Generate strategy on first plan call (hierarchical decomposition) ---
    if not game.get("strategy"):
        try:
            strategy = ask_llm_strategy(game)
            game["strategy"] = strategy
            game["current_phase_idx"] = 0
            phase_lines = " → ".join(f"{p['goal']} @{tuple(p['end_when_pos'])}" for p in strategy)
            game["log"].append(f"🗺 Strategy decomposed: {phase_lines}")
        except Exception as e:
            game["log"].append(f"✗ Strategy generation failed: {e} — falling back to single-phase")
            game["strategy"] = [{"idx": 0, "goal": "reach target", "end_when_pos": list(game["target"]), "done": False}]
            game["current_phase_idx"] = 0

    # If all phases completed → mission done
    if game.get("current_phase_idx", 0) >= len(game["strategy"]):
        game["done"] = True
        game["log"].append("🏁 All phases complete — mission done.")
        return jsonify({**client_state(game), "source": "strategy_complete", "reasoning": "All strategic phases finished."})

    phase = current_phase(game)

    # --- Normal LLM path (tactical: only plan for the current phase) ---
    try:
        r = iterative_plan(game, game.get("plan_iterations", 1), tactical_phase=phase)

        # Persist the LLM's structured mission state for next call.
        # Always overwrite (even with empty) so "" means "no longer applicable",
        # not "keep the stale value".
        if "notes" in r:        game["mission_notes"]  = r["notes"]
        if "phase" in r:        game["mission_phase"]  = r["phase"]
        if "current_goal" in r: game["current_goal"]   = r["current_goal"]
        if "next_goal" in r:    game["next_goal"]      = r["next_goal"]

        # If the LLM declared mission done, honor it (no movement plan needed)
        if r.get("done") and not r["moves"]:
            game["done"] = True
            game["plan"] = []
            record_decision(game, "llm_done", [], r["reasoning"], r["ascii_grid"])
            game["log"].append(f"🏁 LLM declared mission complete — {r['reasoning']}")
            return jsonify({**client_state(game), "source": "llm_done", "reasoning": r["reasoning"]})

        # BFS rescue points to the CURRENT PHASE target, not the global one
        phase_goal_pos = tuple(phase["end_when_pos"])
        final_pos, _ = simulate_plan(game["rover"], game["heading"], r["moves"])
        reaches = final_pos == phase_goal_pos

        source = "llm"
        rescue_note = ""
        if not reaches:
            # Use BFS pointing to the current phase target
            saved_target = game["target"]
            game["target"] = list(phase_goal_pos)
            abs_path = bfs_path(game)
            game["target"] = saved_target
            if abs_path:
                rescued = absolute_to_relative(abs_path, game["heading"])
                source = "llm_rescued"
                rescue_note = (
                    f"  ⚠ LLM plan stopped at {final_pos}; BFS extended to phase target {phase_goal_pos} "
                    f"({len(rescued)} moves)."
                )
                game["metrics"]["bfs_calls"] += 1
                r["moves"] = rescued
            else:
                rescue_note = f"  ⚠ plan ends at {final_pos}, BFS also has no path."

        record_plan_completion(game)
        game["plan"] = r["moves"]
        start_new_plan_tracking(game, len(r["moves"]))
        record_decision(game, source, r["moves"], r["reasoning"] + rescue_note, r["ascii_grid"])
        preview = " ".join(r["moves"][:8]) + ("…" if len(r["moves"]) > 8 else "")
        icon = "🧠" if source == "llm" else "🛟"
        game["log"].append(
            f"{icon} {source} from ({rover[0]},{rover[1]}) → ({target[0]},{target[1]}): "
            f"{len(r['moves'])} moves [{preview}] — {r['reasoning']}{rescue_note}"
        )
        game["log"].append(f"📋 Context shown to LLM (step {game['steps']}):\n{r['ascii_grid']}")
        return jsonify({**client_state(game), "source": source, "reasoning": r["reasoning"] + rescue_note})
    except Exception as e:
        game["log"].append(f"✗ LLM error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/step", methods=["POST"])
@login_required
def step():
    global game
    if not game:
        return jsonify({"error": "no game running"}), 400
    if game.get("done"):
        return jsonify({"error": "done"})
    if not game["plan"]:
        return jsonify({"error": "no_plan"})

    # 1. World ticks (obstacles may drift)
    mp_h = game.get("move_prob", 0.5)
    mp_f = game.get("move_prob_fixed", 0.0)
    if mp_h > 0 or mp_f > 0:
        move_obstacles(game, mp_h, mp_f)

    # 2. Rover sensor reads fresh state BEFORE committing the move
    reveal_sensor(game, log_events=True)

    cmd = game["plan"].pop(0)
    r, c = game["rover"]
    gs = game["grid_size"]
    heading = game["heading"]
    new_heading, dr, dc = apply_command(heading, cmd)
    step_n = game["steps"] + 1

    # Pure rotation (L or R) — no movement, just heading change
    if dr == 0 and dc == 0:
        game["heading"] = new_heading
        game["steps"] += 1
        game["metrics"]["rotations"] += 1
        game["metrics"]["last_plan_used"] += 1
        game["log"].append(
            f"↻ step {step_n}: {cmd} (rotate) heading {heading} → {new_heading}, {len(game['plan'])} moves left"
        )
        reveal_sensor(game, log_events=True)
        return jsonify({**client_state(game), "event": "moved", "move": cmd})

    nr, nc = r + dr, c + dc

    # 3. Boundary check
    if not (0 <= nr < gs and 0 <= nc < gs):
        game["plan"] = []
        game["recalculations"] += 1
        game["replan_steps"].append(game["steps"])
        game["metrics"]["boundary_hits"] += 1
        game["log"].append(f"⚠ step {step_n}: {cmd} from ({r},{c}) heading {heading} → out of bounds")
        return jsonify({**client_state(game), "event": "replan"})

    # 4. Smart abort
    if [nr, nc] in game["known_walls"]:
        game["plan"] = []
        game["recalculations"] += 1
        game["replan_steps"].append(game["steps"])
        game["metrics"]["aborts"] += 1
        game["log"].append(
            f"🛑 step {step_n}: {cmd} aborted — sensor now sees wall at ({nr},{nc})"
        )
        return jsonify({**client_state(game), "event": "replan"})

    # 5. True crash
    if game["grid"][nr][nc] in (FIXED, HIDDEN):
        kind = "fixed" if game["grid"][nr][nc] == FIXED else "mobile"
        walls = set(map(tuple, game["known_walls"]))
        walls.add((nr, nc))
        game["known_walls"] = [list(x) for x in walls]
        game["plan"] = []
        game["recalculations"] += 1
        game["replan_steps"].append(game["steps"])
        game["metrics"]["crashes"] += 1
        game["log"].append(
            f"💥 step {step_n}: CRASH at ({nr},{nc}) — {kind} obstacle outside sensor"
        )
        return jsonify({**client_state(game), "event": "replan"})

    # 6. Commit move. F updates heading to direction of motion; B keeps heading.
    game["rover"] = [nr, nc]
    game["heading"] = new_heading
    game["steps"] += 1
    game["metrics"]["last_plan_used"] += 1
    if cmd == "F":
        game["metrics"]["forwards"] += 1
    elif cmd == "B":
        game["metrics"]["backwards"] += 1
    label = "fwd" if cmd == "F" else "back"
    game["log"].append(
        f"→ step {step_n}: {cmd} ({label}) → ({nr},{nc}), heading={new_heading}, {len(game['plan'])} moves left"
    )

    # 7. Scan from new position
    reveal_sensor(game, log_events=True)

    # 8. Track visits to markers
    newly = update_target_visits(game)
    for label in newly:
        game["log"].append(f"🎯 Passed marker {label}")

    # 9. Strategic phase completion check — server auto-advances phases
    completed = check_phase_completion(game)
    if completed:
        game["log"].append(f"✓ Phase {completed['idx']+1} complete: {completed['goal']}")
        # Force a fresh plan for the new phase
        game["plan"] = []
        # Check if mission is done (all phases complete)
        if game.get("current_phase_idx", 0) >= len(game.get("strategy", [])):
            game["done"] = True
            game["log"].append(f"🏁 ALL PHASES COMPLETE — {game['steps']} steps total")
            return jsonify({**client_state(game), "event": "done"})
        # Otherwise: signal frontend to request new plan
        return jsonify({**client_state(game), "event": "phase_complete", "move": cmd})

    return jsonify({**client_state(game), "event": "moved", "move": cmd})


@app.route("/api/export", methods=["GET"])
@login_required
def export_mission():
    if not game:
        return jsonify({"error": "no game"}), 400
    return jsonify({
        "meta": {
            "model": game.get("model"),
            "started_at": game.get("started_at"),
            "elapsed_s": round(time.time() - game.get("started_at", time.time()), 1),
            "done": game.get("done"),
        },
        "config": {
            "grid_size": game["grid_size"],
            "forward_range": game["forward_range"],
            "back_range": game["back_range"],
            "side_range": game["side_range"],
            "move_prob": game["move_prob"],
            "move_prob_fixed": game.get("move_prob_fixed", 0),
        },
        "endpoints": {
            "start_pos": game.get("start_pos"),
            "target": game["target"],
            "final_rover": game["rover"],
            "final_heading": game["heading"],
        },
        "initial_grid": game.get("initial_grid"),
        "totals": {
            "steps": game["steps"],
            "recalculations": game["recalculations"],
            **game["metrics"],
        },
        "memory": {
            "revealed": game["revealed"],
            "known_walls": game["known_walls"],
        },
        "decisions": game["decisions"],
        "log": game["log"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
