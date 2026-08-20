import axios from "axios";

// VITE_API_URL points at the deployed backend in production (set it in
// Vercel's project settings). Falls back to the local backend for dev,
// unchanged from before.
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_BASE_URL,
});

export default api;

// Named wrappers for the endpoints ChatBox.jsx calls, so call sites read
// as what they do rather than a raw path + payload shape repeated inline.

// Sends a /chat request and streams the answer back as it's generated,
// via a plain fetch() + ReadableStream reader - not axios, whose
// streaming support in the browser is unreliable (its "stream"
// responseType is effectively a Node-only feature); fetch's
// response.body.getReader() is the standard way to read a response body
// incrementally in a browser. The backend (routes/chat.py) answers with
// ONE of two shapes depending on the request:
//   - A DOCX/XLSX export request answers as a single, ordinary JSON body
//     (content-type: application/json) - never streamed, since there's
//     no prose being generated token-by-token for those (see the
//     backend's chat_service.answer_docx_export's own docstring).
//   - Every other question streams newline-delimited JSON (NDJSON,
//     content-type: application/x-ndjson) - one {"type": "chunk"/"done"/
//     "error", ...} object per line; see routes/chat.py's own
//     _stream_chat_response docstring for the exact wire protocol.
//
// callbacks:
//   onChunk(text) - called once per streamed text chunk, in arrival order.
//   onDone({sources, resolved_question}) - called once, after the last
//     chunk, for a streamed answer.
//   onExport({answer, sources, export, resolved_question}) - called
//     once, INSTEAD OF onChunk/onDone, for the non-streamed export case.
//   onError(message) - called on an HTTP-level failure (bad status,
//     network error) OR a {"type": "error", ...} event streamed
//     mid-answer. The two are different situations - the former means
//     nothing was ever shown, the latter means a partial answer may
//     already be visible - but both are reported through this one
//     callback; the caller (pages/Chat.jsx) is what actually knows
//     whether a partial answer is already on screen and reacts
//     accordingly (see its own sendMessage).
export async function streamChatMessage(payload, { onChunk, onDone, onExport, onError }) {
    let response;

    try {
        response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    } catch {
        onError("Could not reach the server. Please check that the backend is running.");
        return;
    }

    if (!response.ok) {
        // FastAPI sends its error text inside response.data.detail (see
        // Chat.jsx's own pre-streaming error handling for the same
        // pattern) - only a plain string is ours, so it's checked before
        // use rather than assumed.
        let detail = null;
        try {
            const body = await response.json();
            detail = typeof body.detail === "string" ? body.detail : null;
        } catch {
            // Response body wasn't JSON (or was empty) - fall through to
            // the generic message below.
        }
        onError(detail || "Could not reach the server. Please check that the backend is running.");
        return;
    }

    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        // The non-streamed export path - one complete body, no chunks.
        onExport(await response.json());
        return;
    }

    // NDJSON: read the body incrementally, splitting on newlines. A
    // network-level read doesn't necessarily land on a JSON-object
    // boundary, so any incomplete trailing line is buffered and
    // prepended to the next read rather than parsed prematurely.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
            const line = buffer.slice(0, newlineIndex);
            buffer = buffer.slice(newlineIndex + 1);
            if (!line) {
                continue;
            }
            const event = JSON.parse(line);
            if (event.type === "chunk") {
                onChunk(event.text);
            } else if (event.type === "done") {
                onDone(event);
            } else if (event.type === "error") {
                onError(event.detail);
            }
        }
    }
}

export function uploadDocument(formData, onUploadProgress) {
    return api.post("/documents/upload", formData, { onUploadProgress });
}

export function deleteDocument(documentId) {
    return api.delete(`/documents/${documentId}`);
}

export function transcribeAudio(formData) {
    return api.post("/transcribe", formData);
}

// Real-time Voice Chat (components/VoiceChat.jsx) connects here
// directly with the browser's native WebSocket, not axios - a plain
// URL string is all that's needed, derived from the same API_BASE_URL
// every other endpoint here uses (http(s) -> ws(s), same host/port).
// sessionId is passed through purely for the backend's own log
// correlation (see backend's services/voice_live_service.py) - omitted
// entirely when not given, which the backend treats as optional.
export function voiceLiveWebSocketUrl(sessionId) {
    const httpUrl = new URL(API_BASE_URL);
    const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
    const url = new URL(`${wsProtocol}//${httpUrl.host}/ws/voice`);

    if (sessionId) {
        url.searchParams.set("session_id", sessionId);
    }

    return url.toString();
}

// A generated export (DOCX answer / XLSX database export - see
// routes/export.py) is fetched via a plain browser download, not axios -
// same reasoning as voiceLiveWebSocketUrl() above: a raw URL is all a
// <a download> link needs, and a plain GET navigation isn't subject to
// the same CORS preflight an XHR/fetch would be, so this works across
// the deployed frontend/backend origins with no extra configuration.
export function exportDownloadUrl(exportId) {
    return `${API_BASE_URL}/export/download/${exportId}`;
}

// GET /health does a real, uncached connectivity check (see
// backend/main.py's own docstring) rather than reporting cached/assumed
// state - every call here is a fresh, live check, not a rehash of a
// previous one. Returns 503 (not 200) when a component is unhealthy;
// callers need the response body either way, so this doesn't try to
// hide that behind a resolved/rejected split - see Navbar.jsx's
// checkDatabaseStatus for how both cases are read from it.
export function checkHealth() {
    return api.get("/health");
}