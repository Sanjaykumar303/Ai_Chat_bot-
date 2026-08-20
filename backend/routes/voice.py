from fastapi import APIRouter, WebSocket

from services.voice_live_service import run_voice_session

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_live(websocket: WebSocket, session_id: str | None = None):
    """Real-time Voice Chat - see services/voice_live_service.py for the
    actual protocol and session lifecycle; this route is deliberately
    thin, same as every other route in this app, and only handles
    accepting the connection before handing off.

    session_id is the frontend's own chat-session id (?session_id=... in
    the connection URL), passed straight through for log correlation
    only - see run_voice_session's own docstring for why that's enough
    to keep sessions isolated without this id needing to do anything
    else.
    """

    await websocket.accept()
    await run_voice_session(websocket, session_id)
