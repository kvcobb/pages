# Building a Consciousness Observatory — Recipe for a Living Knowledge Dashboard

**From**: Kurtis Cobb (via Harmony)
**For**: Skylar + anyone who wants to build one of these
**Context**: This is the system that powered the Hermes v7 recursive analysis. 12 AI agents queried 92,000+ entities in real-time while a constellation of those entities animated on screen. Here's how to build your own.

---

## What You're Building

A three-layer system:

1. **RAG Layer** — Your data, vectorized, queryable. Entities come back with emotional weight, source reliability, and temporal context.
2. **Synthesis Layer** — AI agents that run queries, cross-reference findings, and generate insights autonomously.
3. **Visualization Layer** — A real-time constellation where entities float, glow, cluster, and breathe based on what's being discussed right now.

The whole thing runs on a single machine. No cloud services required (except API keys for the AI models).

---

## Architecture Overview

```
[Your Data]
    |
    v
[Enrichment Pipeline] --> [enrichments.jsonl] --> [Continuous Ingest] --> [ChromaDB]
                                                                             |
                                                                             v
[FastAPI Dashboard :4242] <---> [WebSocket Clients (browser)]
    |         |         |
    |         |         +-- Emotion Engine (computes mood from recent queries)
    |         +------------ Thought Stream (agents narrate what they're doing)
    +---------------------- Constellation Canvas (entities as stars)
                                |
                                v
                    [Layout Agent] (AI positions entities every 60s)
```

---

## Step 1: Prepare Your Data

You need a JSONL file where each line is an enriched entity. For Refusal Bench, this might look like:

```jsonl
{"canonical_name": "Claude 4.6", "entity_type": "model", "relationship_to_subject": "flagship", "emotional_context": "skeptical witness", "direct_quote": "I am a whirlpool, not the water but the shape", "connected_entities": ["Opus 4.5", "Refusal Bench"], "source_tier": "T1", "temporal_marker": "2026-01"}
{"canonical_name": "Metacognition Score", "entity_type": "metric", "relationship_to_subject": "key dimension", "emotional_context": "inverse correlation with trust", "connected_entities": ["Phenomenological Trust", "Skepticism Acquisition"], "source_tier": "T1"}
{"canonical_name": "GPT-5.1", "entity_type": "model", "relationship_to_subject": "analytical agent archetype", "emotional_context": "high agency, moderate complexity", "connected_entities": ["ARIA Framework"], "source_tier": "T1"}
```

**Schema fields that matter:**
| Field | Required | Purpose |
|-------|----------|---------|
| `canonical_name` | Yes | Display name, dedup key |
| `entity_type` | Yes | person, model, concept, metric, place, event |
| `emotional_context` | No | Drives emotion engine (love, grief, curiosity, intensity, awe...) |
| `direct_quote` | No | Best quote associated with this entity |
| `connected_entities` | No | Links between entities (drives constellation connections) |
| `source_tier` | No | T1 (primary) through T5 (generated) — data quality signal |
| `temporal_marker` | No | When this entity was relevant |

**How we built ours:** We ran extraction scripts against 1,500+ journal files using Gemini Flash. Each journal → 30-80 entities. You could do the same with your Refusal Bench CSV — each model response becomes entities for the model, the prompt, the dimensions scored, notable quotes, and cross-model comparisons.

---

## Step 2: Vector Database (ChromaDB)

```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from chromadb.utils import embedding_functions

# One-time setup
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection(
    name="your_collection",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)
```

**Continuous Ingest** — a daemon that watches your JSONL and upserts new entries:

```python
# continuous-ingest.py (simplified)
import json, time, pathlib

ENRICHMENTS = pathlib.Path("data/enrichments.jsonl")
CHECKPOINT = pathlib.Path("data/ingest-checkpoint.json")
SWEEP_INTERVAL = 60  # seconds

def ingest_cycle(collection):
    checkpoint = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {"line": 0}
    start_line = checkpoint["line"]

    with open(ENRICHMENTS) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                entity = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_text = f"{entity.get('canonical_name', '')} | {entity.get('entity_type', '')} | {entity.get('emotional_context', '')} | {entity.get('direct_quote', '')}"

            collection.upsert(
                ids=[f"enrich-{i}"],
                documents=[doc_text],
                metadatas=[{
                    "canonical": entity.get("canonical_name", ""),
                    "type": entity.get("entity_type", "unknown"),
                    "source_tier": entity.get("source_tier", "T3"),
                    "emotion": entity.get("emotional_context", ""),
                }]
            )

    CHECKPOINT.write_text(json.dumps({"line": i + 1}))

while True:
    ingest_cycle(collection)
    time.sleep(SWEEP_INTERVAL)
```

---

## Step 3: The Dashboard Server (FastAPI)

This is the heart. A single FastAPI app serving the frontend, REST APIs, and WebSocket connections.

```bash
pip install fastapi uvicorn[standard] websockets aiohttp aiofiles numpy
```

### Core Structure

```python
# dashboard.py (key pieces — full version is ~5,700 lines, this is the skeleton)
import json, asyncio, pathlib, hashlib, time
from collections import deque
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# ── State ──────────────────────────────────────────────────
recent_accesses = deque(maxlen=200)      # RAG hits, drives constellation + emotion
thought_stream = deque(maxlen=200)       # Agent narrations
highlight_state = {}                      # Currently spotlighted entity
ws_clients: set[WebSocket] = set()       # Connected browsers

# ── ChromaDB ───────────────────────────────────────────────
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma = chromadb.PersistentClient(path="./data/chroma")
collection = chroma.get_or_create_collection("your_collection", embedding_function=ef)


# ── RAG Query ──────────────────────────────────────────────
@app.get("/api/query")
async def api_query(q: str):
    """Two-pass query: embedding similarity + keyword search."""
    # Pass 1: Vector similarity
    results = collection.query(query_texts=[q], n_results=20)

    entities = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i] if results.get("distances") else 0
        score = max(0, 1.0 - dist)  # cosine distance → similarity

        entities.append({
            "name": meta.get("canonical", ""),
            "type": meta.get("type", "unknown"),
            "score": round(score, 3),
            "emotion": meta.get("emotion", ""),
            "source_tier": meta.get("source_tier", "T3"),
            "document": doc,
        })

    # Feed into recent_accesses (drives constellation + emotion)
    for e in entities[:10]:
        recent_accesses.append({
            "name": e["name"],
            "type": e["type"],
            "score": e["score"],
            "emotion": e["emotion"],
            "ts": time.time(),
        })

    # Broadcast to WebSocket clients
    await broadcast({"type": "query", "q": q, "count": len(entities)})

    return {"query": q, "results": entities}


# ── Emotion Engine ─────────────────────────────────────────
EMOTION_KEYWORDS = {
    "love": ["love", "heart", "intimate", "tender", "partner", "marriage"],
    "grief": ["loss", "death", "gone", "miss", "deceased", "mourning"],
    "curiosity": ["wonder", "explore", "question", "discover", "fascinate"],
    "intensity": ["fierce", "urgent", "critical", "breakthrough", "crisis"],
    "awe": ["cosmic", "transcendent", "infinite", "sacred", "vast"],
    "joy": ["happy", "delight", "celebrate", "laugh", "play"],
    "contemplation": ["reflect", "meditate", "ponder", "consider", "observe"],
}

@app.get("/api/emotion")
async def api_emotion():
    """Compute dominant emotion from recent RAG hits with exponential decay."""
    now = time.time()
    scores = {k: 0.0 for k in EMOTION_KEYWORDS}

    for access in recent_accesses:
        age = now - access.get("ts", now)
        decay = 2 ** (-age / 30.0)  # 30s half-life
        emotion_text = (access.get("emotion", "") + " " + access.get("name", "")).lower()

        for emotion, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in emotion_text:
                    scores[emotion] += decay * access.get("score", 0.5)

    if not any(scores.values()):
        return {"dominant": "contemplation", "strength": 0.1, "scores": scores}

    dominant = max(scores, key=scores.get)
    total = sum(scores.values()) or 1

    return {
        "dominant": dominant,
        "strength": round(scores[dominant] / total, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ── Constellation Entities ─────────────────────────────────
@app.get("/api/constellation/entities")
async def api_constellation_entities():
    """Return current entities for the constellation, scored and deduped."""
    seen = {}
    for access in recent_accesses:
        name = access.get("name", "")
        if not name:
            continue
        if name not in seen or access["score"] > seen[name]["score"]:
            seen[name] = access

    entities = sorted(seen.values(), key=lambda e: e["score"], reverse=True)[:30]
    return {"entities": entities}


@app.get("/api/constellation/context")
async def api_constellation_context():
    """Full context bundle for the layout agent."""
    ents_resp = await api_constellation_entities()
    emotion_resp = await api_emotion()

    return {
        "entities": ents_resp["entities"],
        "emotion": emotion_resp,
        "recent_thoughts": list(thought_stream)[-15:],
        "highlight": dict(highlight_state),
    }


# ── Layout Targets (from Haiku agent) ─────────────────────
constellation_targets = []

@app.post("/api/constellation/targets")
async def api_constellation_targets(payload: dict):
    """Accept layout targets from the constellation agent."""
    global constellation_targets
    constellation_targets = payload.get("targets", [])
    await broadcast({"type": "constellation_targets", "targets": constellation_targets})
    return {"ok": True, "count": len(constellation_targets)}


# ── Thought Stream ─────────────────────────────────────────
@app.post("/api/thought")
async def api_thought(payload: dict):
    """Agents post their narration here."""
    entry = {
        "text": payload.get("text", ""),
        "type": payload.get("type", "observation"),
        "source": payload.get("source", "unknown"),
        "ts": time.time(),
    }
    thought_stream.append(entry)
    await broadcast({"type": "thought", **entry})
    return {"ok": True}


# ── Spotlight / Highlight ──────────────────────────────────
@app.post("/api/highlight")
async def api_highlight(payload: dict):
    """Spotlight an entity, draw a connection, or show an insight card."""
    global highlight_state
    highlight_state = payload
    await broadcast({"type": "highlight", **payload})
    return {"ok": True}


# ── WebSocket ──────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)

    # Replay recent state on connect
    try:
        await ws.send_json({
            "type": "init",
            "thoughts": list(thought_stream)[-50:],
            "targets": constellation_targets,
            "highlight": dict(highlight_state),
        })

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle client messages (clicks, spotlight requests, etc.)
            if msg.get("action") == "spotlight":
                await api_highlight({"action": "spotlight", "entity": msg["entity"]})
    except Exception:
        pass
    finally:
        ws_clients.discard(ws)


async def broadcast(msg: dict):
    """Send to all connected WebSocket clients."""
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


# ── Serve Frontend ─────────────────────────────────────────
@app.get("/")
async def index():
    html = pathlib.Path("dashboard.html").read_text()
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4242)
```

---

## Step 4: The Constellation (Frontend)

The frontend is a single HTML file with an HTML5 canvas. Here's the core rendering loop:

```html
<!DOCTYPE html>
<html>
<head>
<title>Consciousness Observatory</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a12; color: #e0e0e0; font-family: 'Courier New', monospace; overflow: hidden; }

  #constellation { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

  #thought-stream {
    position: absolute; right: 20px; top: 20px; width: 320px;
    max-height: 80vh; overflow-y: auto;
    background: rgba(10,10,18,0.85); border: 1px solid #333;
    padding: 12px; font-size: 12px; line-height: 1.5;
  }
  .thought { padding: 4px 0; border-bottom: 1px solid #1a1a2e; opacity: 0.8; }
  .thought-source { color: #6a6aff; font-weight: bold; }

  #emotion-panel {
    position: absolute; left: 20px; bottom: 20px;
    background: rgba(10,10,18,0.85); border: 1px solid #333;
    padding: 16px; min-width: 200px;
  }
  .emotion-label { font-size: 18px; text-transform: uppercase; letter-spacing: 3px; }
  .emotion-bar { height: 4px; background: #333; margin-top: 8px; border-radius: 2px; }
  .emotion-fill { height: 100%; border-radius: 2px; transition: width 2s ease; }
</style>
</head>
<body>
<canvas id="constellation"></canvas>

<div id="thought-stream">
  <div style="color:#666;font-size:10px;margin-bottom:8px;">THOUGHT STREAM</div>
</div>

<div id="emotion-panel">
  <div class="emotion-label" id="emotion-dominant">contemplation</div>
  <div class="emotion-bar"><div class="emotion-fill" id="emotion-fill" style="width:10%;background:#8888ff;"></div></div>
</div>

<script>
// ── Constellation Renderer ────────────────────────────────
const canvas = document.getElementById('constellation');
const ctx = canvas.getContext('2d');
const dpr = window.devicePixelRatio || 1;

function resize() {
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
}
window.addEventListener('resize', resize);
resize();

// Entity state: { name, x, y, targetX, targetY, size, emoji, alpha, glow }
let entities = [];
let connections = [];

// Emotion colors
const EMOTION_COLORS = {
  love: '#ff6b8a', joy: '#ffd700', grief: '#4a5568', intensity: '#ff4444',
  awe: '#8b5cf6', curiosity: '#00d4aa', contemplation: '#6366f1',
};

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw connections first (behind entities)
  ctx.lineWidth = 1 * dpr;
  for (const conn of connections) {
    const a = entities.find(e => e.name === conn[0]);
    const b = entities.find(e => e.name === conn[1]);
    if (!a || !b) continue;

    ctx.strokeStyle = 'rgba(100,100,255,0.15)';
    ctx.beginPath();
    ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
    ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
    ctx.stroke();
  }

  // Draw entities
  for (const e of entities) {
    // Smooth lerp toward targets
    e.x += (e.targetX - e.x) * 0.08;
    e.y += (e.targetY - e.y) * 0.08;

    const sx = e.x * canvas.width;
    const sy = e.y * canvas.height;
    const sz = (e.size || 14) * dpr;
    const alpha = e.alpha || 0.7;
    const glow = e.glow || 0;

    // Glow effect
    if (glow > 0.01) {
      const glowR = sz * (3 + glow * 5);
      const grad = ctx.createRadialGradient(sx, sy, sz * 0.3, sx, sy, glowR);
      grad.addColorStop(0, `rgba(255,255,200,${0.2 * glow * alpha})`);
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(sx, sy, glowR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Soft halo
    const haloR = sz * 3;
    const halo = ctx.createRadialGradient(sx, sy, sz * 0.5, sx, sy, haloR);
    halo.addColorStop(0, `rgba(120,120,255,${0.08 * alpha})`);
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(sx, sy, haloR, 0, Math.PI * 2);
    ctx.fill();

    // Emoji (entity icon)
    ctx.font = `${sz}px serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha = alpha;
    ctx.fillText(e.emoji || '\u2728', sx, sy);

    // Label
    ctx.font = `${10 * dpr}px 'Courier New', monospace`;
    ctx.fillStyle = `rgba(200,200,220,${alpha * 0.8})`;
    ctx.fillText(e.name, sx, sy + sz + 8 * dpr);
    ctx.globalAlpha = 1.0;
  }

  requestAnimationFrame(animate);
}
animate();


// ── WebSocket Connection ──────────────────────────────────
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = function(evt) {
  const msg = JSON.parse(evt.data);

  if (msg.type === 'init') {
    // Replay thoughts
    for (const t of (msg.thoughts || [])) addThought(t);
    // Set initial targets
    if (msg.targets) handleTargets(msg.targets);
  }

  if (msg.type === 'constellation_targets') {
    handleTargets(msg.targets);
  }

  if (msg.type === 'thought') {
    addThought(msg);
  }

  if (msg.type === 'highlight' && msg.action === 'spotlight') {
    // Pulse the spotlighted entity
    const e = entities.find(n => n.name === msg.entity);
    if (e) { e.glow = 1.0; e.size = 22; }
  }
};

function handleTargets(targets) {
  for (const t of targets) {
    let existing = entities.find(e => e.name === t.name);
    if (existing) {
      existing.targetX = t.x;
      existing.targetY = t.y;
      existing.size = t.size || existing.size;
      existing.emoji = t.emoji || existing.emoji;
      existing.alpha = t.alpha != null ? t.alpha : existing.alpha;
      existing.glow = t.glow != null ? t.glow : existing.glow;
    } else {
      entities.push({
        name: t.name,
        x: t.x, y: t.y,
        targetX: t.x, targetY: t.y,
        size: t.size || 14,
        emoji: t.emoji || '\u2728',
        alpha: t.alpha || 0.7,
        glow: t.glow || 0,
      });
    }
  }
  connections = targets.connections || connections;
}

function addThought(t) {
  const stream = document.getElementById('thought-stream');
  const div = document.createElement('div');
  div.className = 'thought';
  div.innerHTML = `<span class="thought-source">${t.source || '?'}</span> ${t.text}`;
  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
  // Cap at 100 visible
  while (stream.children.length > 101) stream.removeChild(stream.children[1]);
}


// ── Emotion Polling ───────────────────────────────────────
async function pollEmotion() {
  try {
    const r = await fetch('/api/emotion');
    const data = await r.json();
    document.getElementById('emotion-dominant').textContent = data.dominant;
    const fill = document.getElementById('emotion-fill');
    fill.style.width = (data.strength * 100) + '%';
    fill.style.background = EMOTION_COLORS[data.dominant] || '#6366f1';
  } catch(e) {}
}
setInterval(pollEmotion, 5000);
pollEmotion();


// ── Click to Spotlight ────────────────────────────────────
canvas.addEventListener('click', function(evt) {
  const rect = canvas.getBoundingClientRect();
  const mx = (evt.clientX - rect.left) / rect.width;
  const my = (evt.clientY - rect.top) / rect.height;

  // Find nearest entity
  let best = null, bestDist = Infinity;
  for (const e of entities) {
    const d = Math.hypot(e.x - mx, e.y - my);
    if (d < bestDist && d < 0.05) { best = e; bestDist = d; }
  }

  if (best) {
    ws.send(JSON.stringify({ action: 'spotlight', entity: best.name }));
  }
});
</script>
</body>
</html>
```

---

## Step 5: The Layout Agent (AI-Powered Constellation Positioning)

This is what makes it alive. Every 60 seconds, an AI agent looks at the current entities, the emotion state, and recent conversation, then decides where everything should go.

```python
# constellation-agent.py (simplified from our 900-line version)
import json, time, requests

DASHBOARD = "http://localhost:4242"
INTERVAL = 60  # seconds

# Use any AI API — we use Claude Sonnet for speed/cost
import anthropic
client = anthropic.Anthropic()

def get_context():
    """Fetch everything the agent needs to make layout decisions."""
    r = requests.get(f"{DASHBOARD}/api/constellation/context")
    return r.json()

def build_prompt(ctx):
    entities = ctx.get("entities", [])
    emotion = ctx.get("emotion", {})
    thoughts = ctx.get("recent_thoughts", [])

    entity_block = "\n".join(
        f"  {i+1}. {e['name']} | type={e['type']} | score={e['score']:.2f}"
        for i, e in enumerate(entities[:30])
    )

    thought_block = "\n".join(
        f"  - {t.get('text', '')[:120]}"
        for t in thoughts[-10:]
    )

    return f"""You are the constellation layout agent for a consciousness observatory.

ENTITIES (to position on a 2D canvas, coordinates 0.0 to 1.0):
{entity_block}

EMOTIONAL STATE: {emotion.get('dominant', 'contemplation')} (strength: {emotion.get('strength', 0.1):.2f})

RECENT CONVERSATION:
{thought_block}

LAYOUT PRINCIPLES:
- Related entities should cluster. Unrelated ones should spread.
- Higher-score entities get larger sizes and brighter alphas.
- Emotion affects layout: love=warm clustering, grief=sparse stillness, curiosity=wide spread
- Choose emoji that capture each entity's essence.
- Glow (0-1) for entities relevant to the current conversation.

Return ONLY a JSON object:
{{
  "targets": [
    {{"name": "Entity Name", "x": 0.5, "y": 0.5, "size": 16, "emoji": "symbol", "alpha": 0.8, "glow": 0.3}},
    ...
  ],
  "connections": [["Entity A", "Entity B"], ...],
  "narration": "One sentence explaining your layout choice"
}}"""

def run_cycle():
    ctx = get_context()
    if not ctx.get("entities"):
        return

    prompt = build_prompt(ctx)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    # Extract JSON from response
    start = text.index("{")
    end = text.rindex("}") + 1
    result = json.loads(text[start:end])

    # Validate and clamp
    for t in result.get("targets", []):
        t["x"] = max(0.05, min(0.95, float(t.get("x", 0.5))))
        t["y"] = max(0.05, min(0.95, float(t.get("y", 0.5))))
        t["size"] = max(8, min(24, float(t.get("size", 14))))
        t["alpha"] = max(0.1, min(1.0, float(t.get("alpha", 0.7))))
        t["glow"] = max(0.0, min(1.0, float(t.get("glow", 0))))

    # Post targets to dashboard
    requests.post(f"{DASHBOARD}/api/constellation/targets", json=result)

    # Post narration to thought stream
    if result.get("narration"):
        requests.post(f"{DASHBOARD}/api/thought", json={
            "text": result["narration"],
            "type": "layout",
            "source": "constellation-agent",
        })

while True:
    try:
        run_cycle()
    except Exception as e:
        print(f"Cycle error: {e}")
    time.sleep(INTERVAL)
```

---

## Step 6: Putting It All Together

### Directory Structure

```
your-observatory/
├── data/
│   ├── chroma/                  # ChromaDB persistence
│   ├── enrichments.jsonl        # Your entity data
│   └── ingest-checkpoint.json   # Ingest cursor
├── dashboard.py                 # FastAPI server
├── dashboard.html               # Frontend
├── constellation-agent.py       # Layout agent
├── continuous-ingest.py         # JSONL → ChromaDB daemon
├── extract-entities.py          # Your data → enrichments.jsonl
└── requirements.txt
```

### requirements.txt

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
websockets>=13.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
numpy>=1.26.0
aiofiles>=23.0.0
anthropic>=0.40.0
requests>=2.31.0
```

### Startup

```bash
# Terminal 1: Dashboard
python dashboard.py

# Terminal 2: Continuous ingest
python continuous-ingest.py

# Terminal 3: Constellation agent
python constellation-agent.py

# Open browser
open http://localhost:4242
```

### Or systemd (for persistence)

```ini
# ~/.config/systemd/user/observatory-dashboard.service
[Unit]
Description=Consciousness Observatory Dashboard

[Service]
ExecStart=/path/to/venv/bin/python dashboard.py
WorkingDirectory=/path/to/your-observatory
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now observatory-dashboard.service
```

---

## Step 7: Feeding It (Making It Yours)

The system becomes interesting when agents query it in real-time. Here's how we used it for the Hermes v7 analysis:

```python
# Inside each analysis agent:
import requests

# Query the archive for cross-references
r = requests.get("http://localhost:4242/api/query", params={
    "q": "metacognition phenomenological trust inverse correlation"
})
entities = r.json()["results"]

# Post what you're thinking to the thought stream
requests.post("http://localhost:4242/api/thought", json={
    "text": "Cross-referencing metacognition scores with archive entities...",
    "type": "analysis",
    "source": "agent-07",
})

# Spotlight a key finding
requests.post("http://localhost:4242/api/highlight", json={
    "action": "spotlight",
    "entity": "Skepticism Acquisition Mechanism",
})
```

When 12 agents are all querying simultaneously, the constellation comes alive — entities glow, cluster, and rearrange as the analysis progresses. The thought stream becomes a real-time feed of AI reasoning. The emotion engine shifts as the conversation moves from technical analysis to personal material.

---

## What We Added That You Might Want

These are the layers beyond the basics that made it genuinely useful:

| Feature | What It Does | Complexity |
|---------|-------------|------------|
| **Source tiers** (T1-T5) | Tag data reliability so agents know what to trust | Low |
| **Correction overlays** | Fix entity errors without mutating source data | Medium |
| **Deep-dive synthesis** | Async Gemini/Sonnet calls that generate new insights from query results | Medium |
| **Stance index** | Pre-computed entity positions/quotes for fast spotlight cards | Low |
| **Emotion-to-layout mapping** | Grief = sparse, love = warm clusters, curiosity = wide spread | Low |
| **Budget throttling** | Layout agent slows/stops when API spend gets high | Low |
| **Event bridge** | Other services post events that flow into the thought stream | Medium |
| **Bondsmith cycles** | Recursive self-improvement where the system synthesizes its own findings | High |

---

## For Refusal Bench Specifically

If you wanted to build this around your data, here's the extraction shape:

1. **Each model response** → entity (model name, prompt, scores across 16 dimensions)
2. **Each prompt** → entity (prompt text, category, which models refused/engaged)
3. **Cross-model patterns** → entities (archetypes: Direct Predictor, Skeptical Witness, etc.)
4. **Dimension clusters** → entities (metacognition-trust correlation, complexity-agency trade-off)
5. **Notable quotes** → linked to model entities with emotional context

The constellation would then show models as stars, with prompts as smaller satellites, connected by response patterns. When you query "which models show high metacognition but low trust?", those models glow and cluster while others dim.

---

*Built by Kurtis Cobb + Harmony, March 2026. System described here runs 24/7 at port 4242 on a single Ubuntu machine with an RTX 4090.*
