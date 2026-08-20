import { useState } from "react";
import { Close, Edit, Loader, Plus, Trash, Waveform } from "../icons";

// Derives one inactive-session sidebar status from that session's own
// runtime entry (see pages/Chat.jsx's sessionRuntime/pendingNotice) -
// null for the active session (it already has the Thinking UI inside
// ChatBox; no separate sidebar status is wanted there) or for a session
// with nothing pending. Pure function of the existing runtime state, not
// a state value of its own - there is nowhere else in this component (or
// anywhere) that tracks a session's status independently of it.
function sessionStatus(runtime, isActive) {
  if (isActive || !runtime) {
    return null;
  }
  if (runtime.loading) {
    return "processing";
  }
  if (runtime.pendingNotice === "new" || runtime.pendingNotice === "failed") {
    return runtime.pendingNotice;
  }
  return null;
}

// Claude/ChatGPT-style chat session sidebar: "+ New Chat" plus a list of
// past sessions (title + rename + delete), most-recently-active first.
// Purely presentational - Chat.jsx owns the actual session list and its
// localStorage persistence (see utils/chatStorage.js), and now also the
// runtime state (sessionRuntime) this reads a background request's
// status from for every session that isn't the active one.
//
// The "Voice Chat" button below New Chat only opens components/
// VoiceChat.jsx (Chat.jsx owns that state and renders it at the page
// level, outside any session) - it has nothing to do with any one
// session, which is exactly why it lives here rather than inside
// ChatBox.jsx/the per-session mic button. voiceChatSupported mirrors the
// same feature-detection ChatBox.jsx's own mic button already does for
// MediaRecorder, just for the getUserMedia/AudioContext/WebSocket trio
// Live Voice itself needs (see pages/Chat.jsx).
function ChatSidebar({
  sessions,
  activeSessionId,
  sessionRuntime = {},
  onNewChat,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  onOpenVoiceChat,
  voiceChatSupported,
  open,
  onClose,
}) {

  // The session currently being renamed (its title becomes an <input>
  // in place of the plain label), or null when nothing's being edited.
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingValue, setEditingValue] = useState("");

  function startRename(event, session) {
    event.stopPropagation(); // don't also trigger onSelectSession on the row
    setEditingSessionId(session.id);
    setEditingValue(session.title);
  }

  function commitRename() {
    if (editingSessionId) {
      onRenameSession(editingSessionId, editingValue);
    }
    setEditingSessionId(null);
  }

  function handleRenameKeyDown(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      commitRename();
    } else if (event.key === "Escape") {
      setEditingSessionId(null);
    }
  }

  function handleDelete(event, sessionId) {
    event.stopPropagation(); // don't also trigger onSelectSession on the row
    if (window.confirm("Delete this chat? This can't be undone.")) {
      onDeleteSession(sessionId);
    }
  }

  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onClose} />}

      <div className={`chat-sidebar ${open ? "open" : ""}`}>

        <div className="chat-sidebar-header">
          <button type="button" className="chat-sidebar-new-button" onClick={onNewChat}>
            <Plus size={18} /> New Chat
          </button>

          <button
            type="button"
            className="chat-sidebar-close"
            onClick={onClose}
            aria-label="Close chat list"
          >
            <Close size={20} />
          </button>
        </div>

        {/* A second launch point for the exact same components/
            VoiceChat.jsx instance ChatBox.jsx's own Waveform button
            opens (see pages/Chat.jsx's voiceChatOpen/onOpenVoiceChat,
            which pins the new/reused chat session it opens Voice
            against) - not itself scoped to one particular session row,
            so it lives here below New Chat rather than in the session
            list below. */}
        <button
          type="button"
          className="chat-sidebar-voice-button"
          onClick={onOpenVoiceChat}
          disabled={!voiceChatSupported}
          aria-label={voiceChatSupported ? "Start real-time voice chat" : "Voice chat is not supported in this browser"}
          title={voiceChatSupported ? "Start real-time voice chat" : "Voice chat is not supported in this browser"}
        >
          <Waveform size={18} /> Voice Chat
        </button>

        <div className="chat-sidebar-label">Recent Chats</div>

        <div className="chat-session-list">

          {sessions.length === 0 && (
            <p className="chat-session-empty">No chats yet.</p>
          )}

          {sessions.map((session) => {
            const status = sessionStatus(sessionRuntime[session.id], session.id === activeSessionId);

            return (
            <div
              key={session.id}
              className={`chat-session-item ${session.id === activeSessionId ? "active" : ""}`}
              onClick={() => onSelectSession(session.id)}
            >
              {editingSessionId === session.id ? (
                <input
                  type="text"
                  className="chat-session-title-input"
                  value={editingValue}
                  autoFocus
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => setEditingValue(event.target.value)}
                  onBlur={commitRename}
                  onKeyDown={handleRenameKeyDown}
                />
              ) : (
                <span className="chat-session-title">{session.title}</span>
              )}

              {status === "processing" && (
                <span className="chat-session-status chat-session-status-processing">
                  <Loader size={12} className="icon-spin" /> Processing...
                </span>
              )}

              {status === "new" && (
                <span className="chat-session-status chat-session-status-new">
                  <span className="chat-session-status-dot" aria-hidden="true" /> New answer
                </span>
              )}

              {status === "failed" && (
                <span className="chat-session-status chat-session-status-failed">
                  <Close size={12} /> Failed
                </span>
              )}

              <button
                type="button"
                className="chat-session-rename"
                onClick={(event) => startRename(event, session)}
                aria-label="Rename chat"
                title="Rename chat"
              >
                <Edit size={16} />
              </button>

              <button
                type="button"
                className="chat-session-delete"
                onClick={(event) => handleDelete(event, session.id)}
                aria-label="Delete chat"
                title="Delete chat"
              >
                <Trash size={16} />
              </button>
            </div>
            );
          })}

        </div>

      </div>
    </>
  );
}

export default ChatSidebar;
