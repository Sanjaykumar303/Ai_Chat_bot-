import { useEffect, useState } from "react";
import "../styles/Chat.css";
import Navbar from "../components/Navbar";
import ChatSidebar from "../components/ChatSidebar";
import ChatBox from "../components/ChatBox";
import {
  loadSessions,
  saveSessions,
  loadActiveSessionId,
  saveActiveSessionId,
  createSession,
  deriveTitle,
} from "../utils/chatStorage";

function Chat() {

  // { id, title, messages, createdAt, updatedAt }[] - the full Recent
  // Chats list. Always at least one session (a brand-new install starts
  // with a single empty "New Chat").
  const [sessions, setSessions] = useState(() => {
    const stored = loadSessions();
    return stored.length > 0 ? stored : [createSession()];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return loadActiveSessionId() || sessions[0].id;
  });

  // Mobile-only drawer state for the sidebar - CSS keeps the sidebar
  // permanently visible on desktop regardless of this value, so it's a
  // harmless no-op there.
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Persisted on every change - chat history is small (text only), so
  // no debouncing needed.
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    saveActiveSessionId(activeSessionId);
  }, [activeSessionId]);

  // Most recently active first, like ChatGPT/Claude's own history list.
  const sortedSessions = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);

  // If the id restored from localStorage no longer exists (its session
  // was deleted, or storage was hand-edited), fall back to the most
  // recent session instead of rendering a ChatBox for one that's gone.
  const activeSession = sortedSessions.find((session) => session.id === activeSessionId) || sortedSessions[0];

  function handleNewChat() {
    const session = createSession();
    setSessions((previous) => [session, ...previous]);
    setActiveSessionId(session.id);
    setSidebarOpen(false);
  }

  function handleSelectSession(sessionId) {
    setActiveSessionId(sessionId);
    setSidebarOpen(false);
  }

  function handleDeleteSession(sessionId) {
    const remaining = sessions.filter((session) => session.id !== sessionId);

    if (sessionId !== activeSessionId) {
      setSessions(remaining);
      return;
    }

    if (remaining.length > 0) {
      setActiveSessionId(remaining[0].id);
      setSessions(remaining);
    } else {
      const fresh = createSession();
      setActiveSessionId(fresh.id);
      setSessions([fresh]);
    }
  }

  // Passed to ChatBox as onMessagesChange - called whenever that
  // session's message list changes (a question sent, an answer
  // received). Also derives the session's title from its first user
  // message the first time one appears, while it's still "New Chat".
  function handleMessagesChange(sessionId, messages) {
    setSessions((previous) =>
      previous.map((session) => {
        if (session.id !== sessionId) {
          return session;
        }

        const firstUserMessage = messages.find((message) => message.sender === "user");
        const title =
          session.title === "New Chat" && firstUserMessage
            ? deriveTitle(firstUserMessage.text)
            : session.title;

        return { ...session, messages, title, updatedAt: Date.now() };
      })
    );
  }

  return (
    <div className="app-shell">

      <Navbar onToggleSidebar={() => setSidebarOpen((value) => !value)} />

      <div className="chat-layout">

        <ChatSidebar
          sessions={sortedSessions}
          activeSessionId={activeSession.id}
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
          onDeleteSession={handleDeleteSession}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <div className="chat-container">

          <h1>Chat</h1>

          <p>
            Ask a general knowledge question, ask about the connected database,
            or attach a PDF/image using the + button to ask about that too.
          </p>

          {/* key=activeSession.id forces a full remount on chat switch -
              this is what keeps attachment state (owned inside ChatBox,
              never passed as a prop) from leaking between chats or
              carrying over into a new one, with no manual reset code
              needed. Only the message history is handed in/out, since
              that's the only part of a chat this feature persists. */}
          <ChatBox
            key={activeSession.id}
            initialMessages={activeSession.messages}
            onMessagesChange={(messages) => handleMessagesChange(activeSession.id, messages)}
          />

        </div>

      </div>

    </div>
  );
}

export default Chat;
