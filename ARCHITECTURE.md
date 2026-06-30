# Cortex Architecture — Odysseus Integration

> Documentación de los módulos CORTEX + ADABA portados a Python
> dentro del fork `TheBoomerDev/odysseus`.

## Overview

Se portaron 9 módulos (3,047 líneas, 39 endpoints REST) desde los proyectos originales:

| Origen | Módulos | Propósito |
|--------|---------|-----------|
| **CORTEX** (TypeScript → Python) | goals, router, improve, cli_invoker, skills_sh | Goal decomposition, routing, self-improvement, CLI orchestration |
| **ADABA** (TypeScript → Python) | smart_router, csuite, sdd, heartbeat | Multi-model routing, C-Suite executives, SDD pipeline, health monitoring |
| **Fase 4 (nuevo)** | goal_state | 8-state machine, checkpoints, progress tracking |

## Layers

```
┌─────────────────────────────────────────────────┐
│                 FASTAPI ROUTES                   │
│       23 REST endpoints + 16 state/checkpoint   │
│        + 7 agent tools (Odysseus chat)          │
├─────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  GOALS   │ │  ROUTER  │ │  SMART ROUTER    │ │
│  │ decompose│ │  score   │ │  24 models       │ │
│  │ 8-state  │ │  route   │ │  8 providers     │ │
│  │ presets  │ │  detect  │ │  9 categories    │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ C-SUITE  │ │   SDD    │ │   HEARTBEAT      │ │
│  │ 6 roles  │ │ spec/plan│ │  health check    │ │
│  │ context  │ │ LLM-gen  │ │  stale detection │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ IMPROVE  │ │SKILLS_SH │ │  CLI INVOKER     │ │
│  │ patterns │ │ search   │ │  agent discovery │ │
│  │ traject. │ │ install  │ │  subprocess run  │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│  ┌──────────────────────────────────────────┐   │
│  │          GOAL STATE MACHINE              │   │
│  │  8 states · checkpoints · progress %    │   │
│  │  persistence · transition validation    │   │
│  └──────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│              ODYSSEUS CORE                       │
│  FastAPI · Auth · DB · LLM · Skills · MCP      │
└─────────────────────────────────────────────────┘
```

## Modules Detail

### 1. `cortex/goals.py` — Goal Decomposition (190L)

Descompone metas en subtareas estructuradas usando:
- **Template matching**: 3 templates predefinidos (migrate-database, audit-security, refactor-module)
- **Heuristic fallback**: 4 tareas genéricas (research → design → implement → validate)
- **Integration**: `decompose_and_track()` crea GoalSession + auto-advance

### 2. `cortex/goal_state.py` — State Machine (369L) [Fase 4]

**8-state lifecycle:**
```
CREATED → ANALYZING → PLANNING → DECOMPOSING → ASSIGNING
  → EXECUTING → VERIFYING → COMPLETED / FAILED
```

Features:
- Transition validation (illegal moves raise ValueError)
- `advance()` — next logical state
- `fail()` / `retry()` — error handling
- Progress tracking (0.0-1.0 + task counts)
- Checkpoints (auto + manual) with `restore_checkpoint()`
- Persistence: JSON on disk, loads on startup
- Goal presets: feature, research, bugfix, maintenance, strategy

### 3. `cortex/router.py` — Agent Router (189L)

Multi-dimensional scoring:
- **Quality** (0.0-1.0)
- **Cost** (inverted cost_tier)
- **Recency** (1-week decay)
- **Affinity** (task match)
- **Diversity** (ensemble bonus)

Returns ranked list with winner + explanation.

### 4. `cortex/smart_router.py` — LLM Model Router (341L)

- **24 modelos** en 8 proveedores
- **9 categorías** de tareas (code, chat, reasoning, creative, analysis, research, writing, planning, debugging)
- **3 prioridades**: quality, cost, speed
- **Cost estimation** por modelo
- **Fallback chain** por categoría

### 5. `cortex/csuite.py` — C-Suite Executives (280L)

6 roles ejecutivos con contexto compartido:
- **CEO**: Strategy, vision, decisions
- **CTO**: Architecture, technical direction
- **CMO**: Marketing, content, positioning
- **CFO**: Finance, unit economics, budget
- **QA**: Quality, testing, audits
- **R&D**: Research, innovation, competitors

### 6. `cortex/sdd.py` — Spec-Driven Development (421L)

Pipeline de 7 pasos:
```
ANALYZE → DOCUMENT → DECOMPOSE → ASSIGN → EXECUTE → VERIFY → CONSOLIDATE
```

- **SDDGenerator**: spec, plan, task document templates
- **LLM integration**: `generate_with_llm()` with fallback to templates
- **PipelineOrchestrator**: run/step execution with skip/only filters

### 7. `cortex/heartbeat.py` — Health Monitor (120L)

- Periodic health checks (configurable interval, default 60s)
- Stale goal detection (>30 min → auto-fail)
- DB + Redis connectivity checks
- WebSocket notification on failures

### 8. `cortex/improve.py` — Self-Improvement (190L)

- Trajectory recording (agent → prompt → result → success/fail)
- Pattern detection (frequent patterns → SkillOpt proposals)
- Improvement statistics

### 9. `cortex/skills_sh.py` — Skill Discovery (177L)

Multi-source skill search:
1. Local Hermes skills (`~/.hermes/skills/`)
2. GitHub code search
3. Static built-in index

### 10. `cortex/cli_invoker.py` — CLI Wrapper (174L)

- Agent discovery via `shutil.which()` across known binaries
- Version detection via `--version`
- Subprocess invocation with timeout and output capture
- 5 agents: claude, codex, agy, graphify, ollama

## API Endpoints

### REST (routes/cortex_routes.py)

**Total: 39 endpoints**

| Prefix | Count | Description |
|--------|-------|-------------|
| `/api/cortex/capabilities` | 1 | Module capability listing |
| `/api/cortex/goals/*` | 4 | Decompose, templates |
| `/api/cortex/goals/session/*` | 14 | CRUD, state machine, checkpoints |
| `/api/cortex/router/*` | 4 | Route, categories, smart route |
| `/api/cortex/improve/*` | 3 | Stats, record, patterns |
| `/api/cortex/cli/*` | 2 | List, invoke |
| `/api/cortex/skills-sh/*` | 2 | Search, install |
| `/api/cortex/csuite/*` | 4 | Roles, query, context |
| `/api/cortex/sdd/*` | 4 | Generate, pipeline, LLM-gen, steps |
| `/api/cortex/heartbeat/*` | 2 | Status, tick |

### Agent Tools (cortex/tools.py)

**Total: 11 tools** — usable from Odysseus chat via tool dispatch:

| Tool | Description |
|------|-------------|
| `do_route_prompt` | Route to best model/provider |
| `do_csuite_query` | Query C-Suite executive |
| `do_decompose_goal` | Decompose goal into tasks |
| `do_list_csuite_roles` | List executive roles |
| `do_router_categories` | List routing categories |
| `do_create_goal` | Create tracked goal session |
| `do_goal_status` | Get goal status |
| `do_goal_transition` | Transition goal state |
| `do_goal_advance` | Advance to next state |
| `do_goal_checkpoint` | Create checkpoint |
| `do_goal_update_progress` | Update progress |
| `do_list_goals` | List all goals |

## File Map

```
cortex/
├── __init__.py
├── goal_state.py      # 8-state machine, GoalTracker, Checkpoints
├── goals.py           # Goal decomposition + decompose_and_track()
├── router.py          # Agent scoring and routing
├── improve.py         # Trajectories and patterns
├── cli_invoker.py     # CLI agent discovery and invocation
├── skills_sh.py       # Multi-source skill search
├── smart_router.py    # 24-model LLM router
├── csuite.py          # 6 C-Suite executive roles
├── sdd.py             # SDD pipeline + LLM integration
├── heartbeat.py       # Health monitoring
└── tools.py           # Agent tools (Odysseus chat)

routes/
└── cortex_routes.py   # 39 REST endpoints

tests/
└── cortex/            # pytest test files
    ├── test_goals.py
    ├── test_state_machine.py
    ├── test_router.py
    ├── test_sdd.py
    └── test_tools.py
```

## Quick Start

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
pytest tests/cortex/ -v

# 3. Start server
uvicorn app:app --reload

# 4. Verify
curl http://localhost:8000/api/cortex/capabilities
```

## Docker

```bash
# Build
docker compose -f docker-compose.cortex.yml build

# Run
docker compose -f docker-compose.cortex.yml up -d

# Verify health
curl http://localhost:8001/api/cortex/capabilities
```
