"""
cortex/integration.py — Puente entre módulos CORTEX y core de Odysseus.

Conecta:
- SmartRouter al flujo de chat (selección automática de modelo)
- Goals al TaskScheduler (goals → ScheduledTask en DB)
- Heartbeat a WebSocket (notificaciones de stale goals)
- SDD al LLM real de Odysseus
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 1. SmartRouter → Chat
# -------------------------------------------------------------------


def route_chat_message(
    message: str,
    current_model: str = "",
    current_endpoint: str = "",
) -> Dict[str, Any]:
    """Analiza un mensaje de chat y recomienda el mejor modelo.

    Usa SmartRouter para detectar la categoría de la tarea y sugerir
    el modelo óptimo. Si el modelo actual ya es adecuado, no sugiere cambio.

    Returns:
        Dict con: category, label, recommended_model, explanation, should_switch
    """
    from cortex.smart_router import SmartRouter, RoutingPriority

    router = SmartRouter()
    route_result = router.route(message, priority=RoutingPriority.QUALITY)

    recommended = route_result["recommended_model"]
    should_switch = False

    if current_model and current_model != recommended:
        # Solo recomienda cambio si la categoría lo justifica
        category = route_result.get("category", "")
        if category in ("code", "reasoning", "creative"):
            should_switch = True

    return {
        "category": route_result.get("category", "chat"),
        "label": route_result.get("label", "General Chat"),
        "recommended_model": recommended,
        "recommended_provider": route_result.get("recommended_provider", "local"),
        "estimated_cost_usd": route_result.get("estimated_cost_usd", 0),
        "fallback_models": route_result.get("fallback_models", []),
        "should_switch": should_switch,
        "current_model": current_model,
        "explanation": (
            f"Task category: {route_result.get('label', 'chat')}. "
            f"Recommended: {recommended} ({route_result.get('recommended_provider', 'local')})."
        ),
    }


def route_model_for_session(
    session_history: List[Dict[str, str]],
    current_model: str = "",
) -> Dict[str, Any]:
    """Analiza el historial de la sesión y recomienda un modelo.

    Útil para selección inicial de modelo al crear una sesión nueva.
    """
    # Usar el último mensaje del usuario para categorizar
    last_user_msg = ""
    for msg in reversed(session_history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    if not last_user_msg:
        return {"model": current_model or "deepseek-chat", "category": "chat"}

    result = route_chat_message(last_user_msg, current_model=current_model)
    return {
        "model": result["recommended_model"],
        "category": result["category"],
        "label": result["label"],
        "provider": result["recommended_provider"],
    }


# -------------------------------------------------------------------
# 2. Goals → ScheduledTask
# -------------------------------------------------------------------


def goal_to_scheduled_tasks(
    goal_id: str,
    goal_text: str,
    db_session,
    owner: str = "admin",
) -> List[Dict[str, Any]]:
    """Convierte un goal descompuesto en ScheduledTasks de Odysseus.

    Cada subtarea del goal se convierte en una ScheduledTask independiente
    con dependencias entre ellas.

    Args:
        goal_id: ID único del goal
        goal_text: Descripción del goal
        db_session: SQLAlchemy session de Odysseus
        owner: Propietario de las tareas

    Returns:
        Lista de dicts con las tareas creadas
    """
    from cortex.goals import decompose
    from core.database import ScheduledTask

    decomp = decompose(goal_text)
    created_tasks = []

    for task in decomp.tasks:
        task_id = f"{goal_id}-{task.id}"
        # Build prompt from task description
        prompt = (
            f"Task: {task.description}\n"
            f"Agent: {task.agent}\n"
            f"Capabilities: {', '.join(task.capabilities)}\n"
            f"Dependencies: {', '.join(task.depends_on) if task.depends_on else 'none'}\n"
            f"\nExecute this task and report the results."
        )

        scheduled_task = ScheduledTask(  # type: ignore[call-arg]
            id=task_id,
            owner=owner,
            name=f"[Goal] {task.description[:60]}",
            prompt=prompt,
            task_type="llm",
            schedule="once",
            scheduled_date=datetime.utcnow(),
            status="active",
            output_target="session",
        )
        db_session.add(scheduled_task)
        created_tasks.append(
            {
                "task_id": task_id,
                "description": task.description,
                "agent": task.agent,
                "depends_on": task.depends_on,
            }
        )

    db_session.commit()
    logger.info(
        "Created %d ScheduledTasks for goal '%s' (%s)",
        len(created_tasks),
        goal_id,
        goal_text[:60],
    )
    return created_tasks


def decompose_and_schedule(
    goal_id: str,
    goal_text: str,
    owner: str = "admin",
) -> Dict[str, Any]:
    """Descompone un goal y programa las tareas en una transacción.

    High-level API: goal → decompose → ScheduledTasks → GoalSession.
    """
    from core.database import SessionLocal
    from cortex.goals import decompose
    from cortex.goal_state import GoalTracker, GoalStatus

    # 1. Decompose
    decomp = decompose(goal_text)

    # 2. Create GoalSession
    tracker = GoalTracker()
    session = tracker.create_goal(goal_id, goal_text)
    tracker.transition(goal_id, GoalStatus.DECOMPOSING)

    # 3. Create ScheduledTasks
    db = SessionLocal()
    try:
        tasks = goal_to_scheduled_tasks(goal_id, goal_text, db, owner=owner)
        tracker.transition(goal_id, GoalStatus.ASSIGNING)
        tracker.update_progress(
            goal_id,
            progress=0.3,
            task_count=len(decomp.tasks),
        )
        db.commit()
    except Exception as e:
        db.rollback()
        tracker.fail(goal_id, str(e))
        raise
    finally:
        db.close()

    return {
        "goal_id": goal_id,
        "status": session.status.value,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "agent": t.agent,
                "depends_on": t.depends_on,
            }
            for t in decomp.tasks
        ],
        "scheduled_tasks": tasks,
    }


# -------------------------------------------------------------------
# 3. SDD → Odysseus LLM
# -------------------------------------------------------------------


def generate_sdd_with_odysseus_llm(
    goal_id: str,
    goal_text: str,
) -> Dict[str, Any]:
    """Genera documentos SDD usando el LLM de Odysseus.

    Utiliza el LLM configurado en Odysseus para generar spec, plan y tasks
    reales (no templates).
    """
    from cortex.sdd import SDDGenerator

    generator = SDDGenerator()
    docs = generator.generate_all_with_llm(goal_id, goal_text)
    path = docs.save()

    return {
        "goal_id": goal_id,
        "documents_path": path,
        "llm_used": generator._llm_available,
        "spec_preview": docs.spec_content[:300] if docs.spec_content else "",
        "plan_preview": docs.plan_content[:300] if docs.plan_content else "",
        "tasks_preview": docs.tasks_content[:300] if docs.tasks_content else "",
    }


# -------------------------------------------------------------------
# 4. Heartbeat → WebSocket
# -------------------------------------------------------------------

# Almacén de conexiones WebSocket para notificaciones de heartbeat
_ws_clients: Dict[str, Any] = {}


def register_ws_client(session_id: str, websocket: Any) -> None:
    """Registra un cliente WebSocket para recibir notificaciones."""
    _ws_clients[session_id] = websocket


def unregister_ws_client(session_id: str) -> None:
    """Elimina un cliente WebSocket."""
    _ws_clients.pop(session_id, None)


async def notify_heartbeat_event(event_type: str, data: Dict[str, Any]) -> None:
    """Envía un evento de heartbeat a todos los clientes WebSocket conectados.

    Args:
        event_type: Tipo de evento (stale_goal, health_change, tick)
        data: Datos del evento
    """

    message = json.dumps(
        {
            "type": "heartbeat",
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    dead_clients = []
    for sid, ws in _ws_clients.items():
        try:
            await ws.send_text(message)
        except Exception:
            dead_clients.append(sid)

    for sid in dead_clients:
        unregister_ws_client(sid)


async def heartbeat_health_check() -> Dict[str, Any]:
    """Ejecuta un tick del heartbeat y notifica si hay goals stale.

    Returns:
        Dict con resultados del health check.
    """
    from cortex.heartbeat import HeartbeatManager

    hb = HeartbeatManager()
    result = await hb.tick()

    if result.stale_goals_found > 0:
        await notify_heartbeat_event(
            "stale_goal",
            {
                "count": result.stale_goals_found,
                "goals_marked_failed": result.goals_marked_failed,
            },
        )

    if not result.db_connected or not result.redis_connected:
        await notify_heartbeat_event(
            "health_change",
            {
                "db_connected": result.db_connected,
                "redis_connected": result.redis_connected,
                "errors": result.errors,
            },
        )

    return {
        "db_connected": result.db_connected,
        "redis_connected": result.redis_connected,
        "stale_goals_found": result.stale_goals_found,
        "goals_marked_failed": result.goals_marked_failed,
        "duration_ms": result.duration_ms,
        "healthy": result.db_connected,
    }
