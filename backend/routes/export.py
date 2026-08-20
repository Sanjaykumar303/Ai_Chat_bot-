from fastapi import APIRouter, HTTPException, Response

from services import export_store

router = APIRouter()


@router.get("/export/download/{export_id}")
async def download_export(export_id: str):
    """Serves a file routes/chat.py generated moments earlier (a
    docx/xlsx export - see services/export_store.py) as a plain binary
    download. No rate limiting here, matching DELETE /documents/{id}'s
    own precedent - this is an in-memory byte lookup, not a Gemini/DB
    call, so it carries none of the cost the rate limiter exists to
    bound."""

    entry = export_store.get_export(export_id)

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="This export has expired or doesn't exist. Please ask again.",
        )

    return Response(
        content=entry["bytes"],
        media_type=entry["content_type"],
        headers={"Content-Disposition": f'attachment; filename="{entry["filename"]}"'},
    )
