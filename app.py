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

app = Flask(__name__)
app.secret_key = os.environ.get("ROVER_SECRET", secrets.token_hex(16))
client = OpenAI(timeout=15.0)
MODEL = os.environ.get("ROVER_MODEL", "gpt-4o-mini")
PASSWORD = os.environ.get("ROVER_PASSWORD")  # if unset → auth disabled


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
              move_prob_fixed=0.0, manual_grid=None, rover=None, target=None, model=None):
    rover = list(rover) if rover else [0, 0]
    target = list(target) if target else [grid_size - 1, grid_size - 1]

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
    placed.add(tuple(target))
    grid[rover[0]][rover[1]] = EMPTY
    grid[target[0]][target[1]] = EMPTY

    # Only randomize hidden if the imported grid doesn't already include them
    if not has_hidden_already:
        for _ in range(hidden_count):
            for _ in range(300):
                r, c = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
                if (r, c) not in placed:
                    grid[r][c] = HIDDEN
                    placed.add((r, c))
                    break

    state = {
        "grid": grid,
        "rover": rover,
        "target": target,
        "heading": initial_heading(rover, target),
        "forward_range": forward_range,
        "back_range": BACK_RANGE,
        "side_range": SIDE_RANGE,
        "grid_size": grid_size,
        "move_prob": move_prob,
        "move_prob_fixed": move_prob_fixed,
        "model": model or MODEL,
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

    if log_events:
        if new_walls:
            coords = ", ".join(f"({r},{c})" for r, c in new_walls[:4])
            extra = f" +{len(new_walls)-4} more" if len(new_walls) > 4 else ""
            state["log"].append(f"📡 Sensor detected wall at {coords}{extra}")
        if cleared:
            coords = ", ".join(f"({r},{c})" for r, c in cleared[:3])
            state["log"].append(f"💨 Memory cleared at {coords} (obstacle moved)")


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


def ask_llm(state, stuck_hint=None, retries=1):
    ascii_grid = build_llm_ascii(state)
    rover = state["rover"]
    target = state["target"]
    gs = state["grid_size"]

    heading = state.get("heading", "S")
    fwd_r = state.get("forward_range", 3)
    dr_tgt = target[0] - rover[0]
    dc_tgt = target[1] - rover[1]
    manhattan = abs(dr_tgt) + abs(dc_tgt)
    base = f"""You are the navigation AI for an autonomous rover on a {gs}x{gs} grid.

CURRENT STATE
  Rover at (row {rover[0]}, col {rover[1]}), facing {heading}.
  Target at (row {target[0]}, col {target[1]}).
  Manhattan distance to target: {manhattan} cells (Δrow={dr_tgt:+d}, Δcol={dc_tgt:+d}).

GRID (row 0 = top, col 0 = left, N=up S=down E=right W=left):
{ascii_grid}

Legend:
  R = rover (facing {heading})        T = target
  # = known obstacle (impassable)     . = confirmed clear     ? = unscanned

RELATIVE COMMANDS (interpreted from current heading):
  F = move 1 cell FORWARD in heading direction
  B = move 1 cell BACKWARD (opposite of heading; heading does NOT change)
  L = rotate 90° LEFT in place (no movement)
  R = rotate 90° RIGHT in place (no movement)

SENSOR (cross-shape, rotates with heading):
  {fwd_r} cells ahead · 2 behind · 1 each side

⚠ CRITICAL REQUIREMENTS:
  • Your plan MUST end with the rover AT position (row {target[0]}, col {target[1]}).
  • Do NOT stop early — plan the COMPLETE route to T.
  • You may use up to {gs * 6} commands. Use as many as needed to reach T.
  • If your heading points away from T, start with L or R to turn.
  • Avoid # cells. Prefer . over ? when possible, but cross ? if needed.

Respond ONLY with valid JSON (no markdown):
{{"moves": ["F","R","F",...], "reasoning": "brief strategy"}}"""
    if stuck_hint:
        base += f"\n\n⚠ HINT (rover seems stuck):\n{stuck_hint}"

    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=state.get("model", MODEL),
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
            clean, dropped = validate_moves(data.get("moves", []))
            if not clean:
                last_err = f"no valid moves in response (got {data.get('moves')!r})"
                continue
            reasoning = data.get("reasoning", "").strip() or "(no reasoning)"
            if dropped:
                reasoning += f"  [⚠ dropped invalid: {dropped}]"
            return {"moves": clean, "reasoning": reasoning, "ascii_grid": ascii_grid}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            last_err = str(e)
            continue
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")
    raise RuntimeError(f"LLM gave unusable response after retries: {last_err}")


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

    # --- Normal LLM path ---
    try:
        r = ask_llm(game)
        final_pos, _ = simulate_plan(game["rover"], game["heading"], r["moves"])
        reaches = final_pos == tuple(game["target"])

        source = "llm"
        rescue_note = ""
        if not reaches:
            # LLM gave an incomplete plan — rescue with BFS so the rover always
            # has a route that actually reaches T (when one exists in memory).
            abs_path = bfs_path(game)
            if abs_path:
                rescued = absolute_to_relative(abs_path, game["heading"])
                source = "llm_rescued"
                rescue_note = (
                    f"  ⚠ LLM plan stopped at {final_pos}; BFS extended to full "
                    f"{len(rescued)}-move route to {tuple(game['target'])}."
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

    if tuple(game["rover"]) == tuple(game["target"]):
        game["done"] = True
        game["log"].append(
            f"✓ TARGET REACHED — {game['steps']} steps, {game['recalculations']} recalcs"
        )
        return jsonify({**client_state(game), "event": "done"})

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
