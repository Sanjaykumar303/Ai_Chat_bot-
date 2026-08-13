import { useState } from "react";

// Claude/ChatGPT-style chat session sidebar: "+ New Chat" plus a list of
// past sessions (title + rename + delete), most-recently-active first.
// Purely presentational - Chat.jsx owns the actual session list and its
// localStorage persistence (see utils/chatStorage.js).
function ChatSidebar({ sessions, activeSessionId, onNewChat, onSelectSession, onRenameSession, onDeleteSession, open, onClose }) {

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
            + New Chat
          </button>

          <button
            type="button"
            className="chat-sidebar-close"
            onClick={onClose}
            aria-label="Close chat list"
          >
            ✕
          </button>
        </div>

        <div className="chat-sidebar-label">Recent Chats</div>

        <div className="chat-session-list">

          {sessions.length === 0 && (
            <p className="chat-session-empty">No chats yet.</p>
          )}

          {sessions.map((session) => (
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

              <button
                type="button"
                className="chat-session-rename"
                onClick={(event) => startRename(event, session)}
                aria-label="Rename chat"
                title="Rename chat"
              >
                ✏️
              </button>

              <button
                type="button"
                className="chat-session-delete"
                onClick={(event) => handleDelete(event, session.id)}
                aria-label="Delete chat"
                title="Delete chat"
              >
                🗑
              </button>
            </div>
          ))}

        </div>

      </div>
    </>
  );
}

export default ChatSidebar;
