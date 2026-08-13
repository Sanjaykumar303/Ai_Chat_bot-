// Claude/ChatGPT-style chat session sidebar: "+ New Chat" plus a list of
// past sessions (title + delete), most-recently-active first. Purely
// presentational - Chat.jsx owns the actual session list and its
// localStorage persistence (see utils/chatStorage.js).
function ChatSidebar({ sessions, activeSessionId, onNewChat, onSelectSession, onDeleteSession, open, onClose }) {

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
              <span className="chat-session-title">{session.title}</span>
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
