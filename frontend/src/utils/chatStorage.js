// localStorage persistence for chat sessions - the Claude/ChatGPT-style
// "Recent Chats" list. Deliberately just localStorage, no backend
// storage (per the task's own requirement): sessions only ever hold
// { id, title, messages, createdAt, updatedAt } - never an attachment
// reference. The backend's own document_store.py is itself in-memory
// and TTL-based (see backend/services/document_store.py), so it
// wouldn't survive a page refresh anyway; persisting a document_id here
// would just produce a UI that claims a file is still attached when the
// backend has already forgotten it. Attachment state instead lives in
// Chat.jsx's own sessionRuntime map, keyed by session id and never
// persisted here - it survives switching away from a chat and back
// (only one <ChatBox> is ever mounted, for the active session, but
// sessionRuntime itself lives above it in Chat.jsx, unaffected by that
// component mounting/unmounting) and is only ever cleared by an
// explicit remove/replace or by the session itself being deleted, never
// merely by switching.

const SESSIONS_KEY = "ai_chat_sessions_v1";
const ACTIVE_SESSION_KEY = "ai_chat_active_session_v1";

const TITLE_MAX_LENGTH = 40;

export function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // Corrupted/foreign localStorage content - start fresh rather than crash.
    return [];
  }
}

export function saveSessions(sessions) {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch {
    // Storage full/unavailable (private browsing, quota exceeded) - chats
    // still work for the current tab, they just won't persist.
  }
}

export function loadActiveSessionId() {
  try {
    return localStorage.getItem(ACTIVE_SESSION_KEY);
  } catch {
    return null;
  }
}

export function saveActiveSessionId(id) {
  try {
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
  } catch {
    // Same as saveSessions - non-fatal.
  }
}

export function createSession() {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New Chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

// Short, single-line title from the first user message - trimmed to
// TITLE_MAX_LENGTH characters (not words, to keep this simple and
// predictable), with an ellipsis if it was actually cut short.
export function deriveTitle(text) {
  const cleaned = text.trim().replace(/\s+/g, " ");

  if (cleaned.length <= TITLE_MAX_LENGTH) {
    return cleaned;
  }

  return cleaned.slice(0, TITLE_MAX_LENGTH).trim() + "…";
}
