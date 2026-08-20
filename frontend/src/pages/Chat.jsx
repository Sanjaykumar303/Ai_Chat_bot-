import { useEffect, useRef, useState } from "react";
import "../styles/Chat.css";
import Navbar from "../components/Navbar";
import ChatSidebar from "../components/ChatSidebar";
import ChatBox from "../components/ChatBox";
import VoiceChat from "../components/VoiceChat";
import { streamChatMessage, uploadDocument, deleteDocument, transcribeAudio, exportDownloadUrl } from "../services/api";
import { speak } from "../utils/speech";
import {
  loadSessions,
  saveSessions,
  loadActiveSessionId,
  saveActiveSessionId,
  createSession,
  deriveTitle,
} from "../utils/chatStorage";

// Real-time Voice Chat (components/VoiceChat.jsx) needs a microphone and
// an AudioContext, but never MediaRecorder - it streams raw PCM via
// ScriptProcessorNode instead of recording a clip, so this is its own
// feature-detection check rather than whatever ChatBox.jsx's mic-input
// button uses. Computed once here (not duplicated in both ChatBox.jsx
// and ChatSidebar.jsx) since this is the one place that decides whether
// either of their "Voice Chat" buttons is enabled - see voiceChatOpen
// below for why both buttons need to share this same source of truth.
const VOICE_LIVE_SUPPORTED =
  typeof navigator !== "undefined" &&
  Boolean(navigator.mediaDevices?.getUserMedia) &&
  typeof window !== "undefined" &&
  Boolean(window.AudioContext || window.webkitAudioContext) &&
  Boolean(window.WebSocket);

// Runtime (never persisted) state for a session that isn't part of its
// saved history - a pending request's loading flag, its last error, the
// question still being composed, and its attached document. All of it
// is keyed by session id in Chat.jsx's own sessionRuntime state (see
// below) rather than living inside ChatBox, which is what lets exactly
// ONE <ChatBox> be mounted at a time (fixing the visual/layout sticking
// that came from keeping one hidden instance per visited session alive)
// while a request started in a session the user has since switched away
// from keeps running and still lands its result in the right place.
const DEFAULT_SESSION_RUNTIME = {
  loading: false,
  error: "",
  question: "",
  detectedLanguage: null,
  attachment: null,
  attachmentStatus: "idle", // idle | uploading | processing | error
  attachmentError: "",
  // The backend's resolved_question from this session's last exchange
  // (see routes/chat.py) - not necessarily what the user literally
  // typed, since a follow-up like "yesterday" resolves to something
  // like "What is the total profit yesterday?". Sent back as the next
  // request's previous_question so a chain of follow-ups each build on
  // the fully-expanded form of the one before it.
  lastResolvedQuestion: null,
  // null | "new" | "failed" - an inactive-session sidebar notice (see
  // ChatSidebar.jsx) for a /chat request that finished while the user
  // was on a different session. Deliberately separate from `error`
  // above: `error` is reused for several foreground-only UI failures
  // (mic/copy/share) that can only ever happen while this session is
  // already active, so it can't reliably tell "the answer failed while
  // I was away" apart from "the answer failed and I'm looking at it
  // right now" (no sidebar notice wanted for the latter). Set at the
  // same request-completion points that already flip `loading` back to
  // false (see sendMessage below), and cleared whenever this session
  // becomes active again (see activateSession) - never read or written
  // anywhere but those two places, so there's exactly one place a
  // session's pending notice can change.
  pendingNotice: null,
};

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

  // { [sessionId]: { loading, error, question, detectedLanguage,
  // attachment, attachmentStatus, attachmentError, lastResolvedQuestion } }
  // - see DEFAULT_SESSION_RUNTIME above. A session with no entry here
  // yet (never touched) reads as DEFAULT_SESSION_RUNTIME via
  // getSessionRuntime, so nothing needs to pre-populate this map.
  const [sessionRuntime, setSessionRuntime] = useState({});

  // Mobile-only drawer state for the sidebar - CSS keeps the sidebar
  // permanently visible on desktop regardless of this value, so it's a
  // harmless no-op there.
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Whether components/VoiceChat.jsx (real-time Gemini Live voice) is
  // open - owned here, not inside ChatBox.jsx, specifically so
  // ChatSidebar.jsx's own "Voice Chat" button and ChatBox.jsx's own
  // Waveform button can open the exact same instance/WebSocket
  // connection instead of each mounting their own (which is what would
  // happen if this were still local state duplicated in two places).
  // Rendered once below, outside <ChatBox>, so switching sessions or
  // deleting one never touches it while it's open (the sidebar/ChatBox
  // aren't interactable behind Voice's full-screen takeover anyway).
  const [voiceChatOpen, setVoiceChatOpen] = useState(false);
  // The ONE chat session a given Voice Chat call is pinned to for its
  // whole lifetime - set once in handleOpenVoiceChat (below) at the
  // moment Voice opens, never re-derived afterward and never changed on
  // close. This is what lets a voice conversation's turns land in the
  // same session's persisted history (see handleVoiceTurnSaved) and lets
  // closing Voice simply return to whatever session was already showing,
  // with its history (text + voice) intact.
  const [voiceSessionId, setVoiceSessionId] = useState(null);

  // Mirrors activeSessionId into a ref so a background request's
  // completion handler (sendMessage's success/catch blocks, possibly
  // resolving long after the user has switched to a different session)
  // can read the CURRENT active session, not the one captured in its own
  // closure at the moment it was called - the same "ref mirrors state
  // for a long-lived async callback" pattern components/VoiceChat.jsx
  // already uses for its statusRef/mutedRef.
  const activeSessionIdRef = useRef(activeSessionId);
  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  // Persisted on every change - chat history is small (text only), so
  // no debouncing needed. sessionRuntime is deliberately never
  // persisted - a document_id would outlive the backend's own in-memory
  // TTL store (see utils/chatStorage.js's own comment on this), and a
  // stuck "loading: true" surviving a refresh would just be wrong.
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
  // recent session instead of rendering nothing.
  const activeSession = sortedSessions.find((session) => session.id === activeSessionId) || sortedSessions[0];

  function getSessionRuntime(sessionId) {
    return sessionRuntime[sessionId] || DEFAULT_SESSION_RUNTIME;
  }

  // The text of this session's last AI message, or null if it hasn't
  // answered anything yet - sent as previous_answer on every /chat
  // request (see sendMessage below) purely so a DOCX export request
  // ("convert the summary into a document") has something to convert
  // without the backend needing to store or re-derive it. Read
  // synchronously from `sessions` at send time, same as
  // getSessionRuntime(sessionId) already is above - not subject to the
  // stale-closure concern activeSessionIdRef exists for, since this only
  // ever runs before the request goes out, never in an async completion
  // handler.
  function lastAiAnswer(sessionId) {
    const session = sessions.find((candidate) => candidate.id === sessionId);
    const aiMessages = session ? session.messages.filter((message) => message.sender === "ai") : [];
    return aiMessages.length > 0 ? aiMessages[aiMessages.length - 1].text : null;
  }

  // Merges `patch` (an object, or a function of the current runtime
  // returning one - same shape as React's own functional setState) into
  // one session's runtime entry. Every request/upload/transcription
  // handler below funnels its state changes through this, always
  // addressed by the sessionId it was started for - never "whichever
  // session happens to be active right now" - which is what makes a
  // background request land its result in the right session even after
  // the user has switched to a different one.
  function patchSessionRuntime(sessionId, patch) {
    setSessionRuntime((previous) => {
      const current = previous[sessionId] || DEFAULT_SESSION_RUNTIME;
      const next = typeof patch === "function" ? patch(current) : patch;
      return { ...previous, [sessionId]: { ...current, ...next } };
    });
  }

  // Appends one or more messages to a session's persisted history, and
  // derives its title from the first user message the first time one
  // appears, while it's still "New Chat" - same logic the old
  // onMessagesChange callback used to run in ChatBox, now driven
  // directly from here since Chat.jsx is what actually knows when a
  // message was sent or received.
  function appendMessages(sessionId, newMessages) {
    setSessions((previous) =>
      previous.map((session) => {
        if (session.id !== sessionId) {
          return session;
        }

        const messages = [...session.messages, ...newMessages];
        const firstUserMessage = messages.find((message) => message.sender === "user");
        const title =
          session.title === "New Chat" && firstUserMessage
            ? deriveTitle(firstUserMessage.text)
            : session.title;

        return { ...session, messages, title, updatedAt: Date.now() };
      })
    );
  }

  // Applies `patch` (an object, or a function of the current last
  // message returning one) to the LAST message in a session's history -
  // used while streaming an AI reply (see sendMessage below) to grow its
  // text incrementally chunk by chunk, then once the stream ends, to
  // attach its final sources and clear its `streaming` flag. Safe to
  // assume "the last message" is the right target because every call
  // site only ever calls this between appending that message (as the
  // first streamed chunk arrives) and the stream actually finishing -
  // there's no message id to look up by instead, and the Send button
  // stays disabled (loading=true) for a session's whole in-flight
  // request, so nothing else can append a newer message in between.
  function updateLastMessage(sessionId, patch) {
    setSessions((previous) =>
      previous.map((session) => {
        if (session.id !== sessionId || session.messages.length === 0) {
          return session;
        }
        const messages = [...session.messages];
        const lastIndex = messages.length - 1;
        const current = messages[lastIndex];
        const next = typeof patch === "function" ? patch(current) : patch;
        messages[lastIndex] = { ...current, ...next };
        return { ...session, messages, updatedAt: Date.now() };
      })
    );
  }

  // Makes `sessionId` the active session and clears any inactive-session
  // notice it was showing in the sidebar (see pendingNotice above) -
  // "clear the dot when user returns" means the moment it becomes
  // active again, not just when explicitly picked from the list, so
  // every place activeSessionId changes goes through this one function
  // rather than calling setActiveSessionId directly.
  function activateSession(sessionId) {
    setActiveSessionId(sessionId);
    patchSessionRuntime(sessionId, { pendingNotice: null });
  }

  function handleNewChat() {
    const session = createSession();
    setSessions((previous) => [session, ...previous]);
    activateSession(session.id);
    setSidebarOpen(false);
  }

  // Opens Voice Chat pinned to a session - reuses the currently active
  // one as-is (never creates a new session in this branch, per this
  // feature's own requirement) so a voice conversation started from an
  // existing chat lands in that same chat's history. `activeSession` is
  // always defined today (sessions never becomes empty - see
  // handleDeleteSession), but this stays defensive rather than assuming
  // that invariant: only when there's genuinely no active session is a
  // fresh one created and activated, exactly like handleNewChat, so
  // Voice's own turns have somewhere durable to land and there's
  // something to return to once it closes.
  function handleOpenVoiceChat() {
    if (activeSession) {
      setVoiceSessionId(activeSession.id);
    } else {
      const session = createSession();
      setSessions((previous) => [session, ...previous]);
      activateSession(session.id);
      setVoiceSessionId(session.id);
    }
    setVoiceChatOpen(true);
  }

  // Closing Voice never touches activeSessionId or sessions - whichever
  // session was already showing (the one voiceSessionId was pinned to)
  // stays exactly as it was, history intact, satisfying "closing Voice
  // must return to the same session" by simply never having left it.
  function handleCloseVoiceChat() {
    setVoiceChatOpen(false);
    setVoiceSessionId(null);
  }

  // One settled voice turn (see components/VoiceChat.jsx's
  // "user_turn_saved"/"assistant_turn_saved" handling) appended to its
  // pinned session's own persisted history via the exact same
  // appendMessages used for typed messages - same message shape
  // ({sender, text}), same title-deriving logic, no separate voice-only
  // history to keep in sync. Addressed to the sessionId THIS turn was
  // saved under (closed over at call time from VoiceChat's own props,
  // which never changes for as long as one Voice Chat call stays open),
  // not whatever `activeSessionId` happens to be - matches sendMessage's
  // own "always address the session this was started for" pattern.
  function handleVoiceTurnSaved(sessionId, turn) {
    appendMessages(sessionId, [{ sender: turn.sender, text: turn.text }]);
  }

  // Seeds VoiceChat's own on-screen transcript from this session's
  // ALREADY-persisted history (typed and voice both - see
  // handleVoiceTurnSaved's own comment on why voice turns land in the
  // exact same `messages` list as typed ones, with no separate voice-
  // only history to track) - reopening Voice Chat mid-session shows the
  // conversation so far instead of starting blank, even though the
  // messages themselves were never lost (they were always in `messages`,
  // just not re-displayed in VoiceChat's own panel before this). Maps
  // {sender:"user"|"ai"} (this store's own convention, shared with
  // ChatBox.jsx) to {speaker:"you"|"assistant"} (VoiceChat.jsx's own,
  // older convention - unrelated to this feature, not worth renaming
  // just to unify). No `time` field - real timestamps were never kept
  // per-message in this store, and VoiceChat's own bubble only renders
  // one when present, so omitting it here is a clean, correct "unknown"
  // rather than a fabricated one.
  function voiceTranscriptFor(sessionId) {
    const session = sessions.find((candidate) => candidate.id === sessionId);
    if (!session) {
      return [];
    }
    return session.messages.map((message) => ({
      speaker: message.sender === "user" ? "you" : "assistant",
      text: message.text,
    }));
  }

  function handleSelectSession(sessionId) {
    activateSession(sessionId);
    setSidebarOpen(false);
  }

  function handleRenameSession(sessionId, newTitle) {
    const title = newTitle.trim().replace(/\s+/g, " ") || "New Chat";

    setSessions((previous) =>
      previous.map((session) => (session.id === sessionId ? { ...session, title } : session))
    );
  }

  function handleDeleteSession(sessionId) {
    const remaining = sessions.filter((session) => session.id !== sessionId);

    // Revoke this session's attachment preview URL (if any) before its
    // runtime entry is dropped - the same cleanup ChatBox's own unmount
    // effect used to do implicitly when a deleted session's instance
    // went away; now that attachments live here, deleting the session
    // is the only remaining place that needs to do it.
    const runtime = getSessionRuntime(sessionId);
    if (runtime.attachment?.previewUrl) {
      URL.revokeObjectURL(runtime.attachment.previewUrl);
    }
    setSessionRuntime((previous) => {
      if (!(sessionId in previous)) {
        return previous;
      }
      const next = { ...previous };
      delete next[sessionId];
      return next;
    });

    if (sessionId !== activeSessionId) {
      setSessions(remaining);
      return;
    }

    if (remaining.length > 0) {
      activateSession(remaining[0].id);
      setSessions(remaining);
    } else {
      const fresh = createSession();
      activateSession(fresh.id);
      setSessions([fresh]);
    }
  }

  // Composing-input change - always clears detectedLanguage too, same
  // as the input's onChange always did before this was lifted out of
  // ChatBox (typing anything new means whatever voice-detected language
  // produced the previous text no longer applies).
  function handleQuestionChange(sessionId, text) {
    patchSessionRuntime(sessionId, { question: text, detectedLanguage: null });
  }

  function handleErrorMessage(sessionId, message) {
    patchSessionRuntime(sessionId, { error: message });
  }

  function handleAttachmentValidationError(sessionId, message) {
    patchSessionRuntime(sessionId, { attachmentStatus: "error", attachmentError: message });
  }

  // Sends one chat message for `sessionId` - the actual /chat request
  // and everything around it (appending the user's message, the
  // loading flag, appending/growing the answer or recording an error,
  // optionally speaking it) all address THIS sessionId regardless of
  // what's active by the time the response arrives, which is what lets
  // switching sessions never cancel or misdirect it.
  //
  // The answer now arrives incrementally (see services/api.js's
  // streamChatMessage) instead of as one complete response: the FIRST
  // chunk both ends the "Thinking..." wait (ChatBox.jsx hides its own
  // Loader once the last message is an in-progress AI reply - see its
  // `streaming` flag below) and appends a new AI message; every
  // following chunk grows that same message's text via
  // updateLastMessage rather than appending a new one. A DOCX/XLSX
  // export answers in one shot instead (never streamed - see
  // chat_service.answer_docx_export's own docstring), handled by
  // onExport below exactly like the old single-response flow did.
  //
  // autoSpeak is passed in rather than read from state because it's a
  // ChatBox-local UI toggle (see that component's own docstring) - this
  // captures its value at the moment the user hit Send, exactly the
  // same closure semantics the original single-component version had
  // (toggling Auto Speak off while a request is still in flight doesn't
  // retroactively silence an answer already on its way).
  async function sendMessage(sessionId, trimmedQuestion, autoSpeak, webSearch) {
    const runtime = getSessionRuntime(sessionId);
    const spokenLanguage = runtime.detectedLanguage;

    appendMessages(sessionId, [{ sender: "user", text: trimmedQuestion }]);
    // pendingNotice is cleared here too, not just in activateSession -
    // starting a fresh request supersedes any stale "New answer"/"Failed"
    // notice this session was still showing from a previous exchange the
    // user hasn't gotten back to yet.
    patchSessionRuntime(sessionId, { question: "", detectedLanguage: null, error: "", loading: true, pendingNotice: null });

    // Sidebar status (see ChatSidebar.jsx) only applies to a session the
    // user isn't currently looking at - read via the ref, not the
    // `activeSessionId` state closed over when sendMessage was called,
    // since the user may well have switched sessions by the time this
    // resolves (see activeSessionIdRef's own comment above).
    const wasInactiveOnCompletion = () => sessionId !== activeSessionIdRef.current;

    // Plain closure variables, not state - scoped to this one request,
    // read/written only by the callbacks below (which all run
    // sequentially, never concurrently, for a single streamed response).
    // fullAnswer accumulates the complete text for autoSpeak/onError,
    // since neither has the whole answer in one place otherwise once
    // it's only ever stored incrementally, split across many
    // updateLastMessage calls.
    let streamingStarted = false;
    let fullAnswer = "";

    function handleChunk(text) {
      fullAnswer += text;
      if (!streamingStarted) {
        streamingStarted = true;
        appendMessages(sessionId, [{ sender: "ai", text, sources: [], download: null, streaming: true }]);
      } else {
        updateLastMessage(sessionId, (message) => ({ text: message.text + text }));
      }
    }

    try {
      await streamChatMessage(
        {
          question: trimmedQuestion,
          language: spokenLanguage,
          document_id: runtime.attachment?.documentId ?? null,
          previous_question: runtime.lastResolvedQuestion,
          previous_answer: lastAiAnswer(sessionId),
          // The existing per-chat session id (utils/chatStorage.js) - reused
          // as-is so the backend can persist this conversation's memory
          // (see backend/services/chat_memory.py). No new id is minted here.
          session_id: sessionId,
          // Manual Web Search toggle (ChatBox.jsx) - false/omitted
          // reproduces today's exact DB/PDF/RAG/general behavior
          // unchanged (see routes/chat.py's ChatRequest.web_search).
          web_search: webSearch,
        },
        {
          onChunk: handleChunk,

          onDone: ({ sources = [], resolved_question }) => {
            patchSessionRuntime(sessionId, { lastResolvedQuestion: resolved_question || trimmedQuestion });

            if (streamingStarted) {
              // Attaches the sources (only known once the whole answer
              // is in) and settles the message out of "streaming" state
              // - ChatBox.jsx uses that to show the speaker button and
              // stop treating it as still-growing.
              updateLastMessage(sessionId, { sources, streaming: false });
            } else {
              // Every answer_* generator always yields at least one
              // chunk in practice (even a fixed-text answer is yielded
              // as one), so this is a defensive fallback, not the
              // common case - without it, a genuinely chunkless answer
              // would silently vanish instead of landing as a message.
              appendMessages(sessionId, [{ sender: "ai", text: "", sources, download: null }]);
            }

            if (wasInactiveOnCompletion()) {
              patchSessionRuntime(sessionId, { pendingNotice: "new" });
            }

            if (autoSpeak) {
              speak(fullAnswer, (message) => patchSessionRuntime(sessionId, { error: message }));
            }
          },

          onExport: ({ answer, sources = [], resolved_question, export: exportInfo }) => {
            patchSessionRuntime(sessionId, { lastResolvedQuestion: resolved_question || trimmedQuestion });
            // exportInfo (routes/chat.py's "export" field - only present
            // for a recognized DOCX/XLSX export request, see services/
            // export_intent.py) becomes a downloadable badge on this
            // same AI message rather than a separate message - one
            // <a download> link, no new message shape for every other,
            // non-export answer to account for.
            const download = exportInfo
              ? { url: exportDownloadUrl(exportInfo.id), filename: exportInfo.filename }
              : null;
            appendMessages(sessionId, [{ sender: "ai", text: answer, sources, download }]);

            if (wasInactiveOnCompletion()) {
              patchSessionRuntime(sessionId, { pendingNotice: "new" });
            }

            if (autoSpeak) {
              speak(answer, (message) => patchSessionRuntime(sessionId, { error: message }));
            }
          },

          onError: (detail) => {
            if (streamingStarted) {
              // A partial answer is already visible - settle it out of
              // "streaming" state and surface the failure alongside it,
              // rather than discarding what was already shown (there's
              // no way to "un-show" text the user has already read, and
              // routes/chat.py's own docstring on this same trade-off
              // applies here too: once some of the answer streamed
              // through, this can never retroactively become the plain
              // "nothing happened" error state the pre-streaming version
              // always had).
              updateLastMessage(sessionId, { streaming: false });
              patchSessionRuntime(sessionId, {
                error: `${detail} (the answer above may be incomplete)`,
                pendingNotice: wasInactiveOnCompletion() ? "failed" : null,
              });
            } else {
              patchSessionRuntime(sessionId, {
                error: detail,
                pendingNotice: wasInactiveOnCompletion() ? "failed" : null,
              });
            }
          },
        }
      );
    } finally {
      patchSessionRuntime(sessionId, { loading: false });
    }
  }

  // Shared upload path for a PDF, a picked image, or a camera-captured
  // image blob - all three are just bytes to POST /documents/upload,
  // which already validates/branches PDF vs. image server-side
  // (routes/documents.py). previewUrl is only ever set for images.
  async function uploadAttachment(sessionId, file, filename, previewUrl) {
    const before = getSessionRuntime(sessionId);

    // Replacing an existing attachment: drop the old temporary context
    // first, so it's never left around or mistakenly reused once the
    // new one is ready.
    if (before.attachment?.documentId) {
      try {
        await deleteDocument(before.attachment.documentId);
      } catch {
        // Already gone/expired - fine, proceed with the new upload regardless.
      }
      if (before.attachment.previewUrl) {
        URL.revokeObjectURL(before.attachment.previewUrl);
      }
    }

    patchSessionRuntime(sessionId, { attachment: null, attachmentError: "", attachmentStatus: "uploading" });

    const formData = new FormData();
    formData.append("file", file, filename);

    try {
      // Bytes fully sent to the server means the "uploading" phase is
      // over - what's left (extraction + chunking + indexing, or OCR
      // for a scanned PDF/image) is "processing", which has no
      // progress events of its own.
      const response = await uploadDocument(formData, (progressEvent) => {
        if (progressEvent.loaded >= progressEvent.total) {
          patchSessionRuntime(sessionId, { attachmentStatus: "processing" });
        }
      });
      const { document_id, filename: returnedFilename } = response.data;
      patchSessionRuntime(sessionId, {
        attachment: { documentId: document_id, filename: returnedFilename, previewUrl: previewUrl || null },
        attachmentStatus: "idle",
      });
    } catch (err) {
      const detail = err.response?.data?.detail;
      patchSessionRuntime(sessionId, {
        attachmentStatus: "error",
        attachmentError: typeof detail === "string" ? detail : "Unable to process this file.",
      });
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    }
  }

  async function handleRemoveAttachment(sessionId) {
    const runtime = getSessionRuntime(sessionId);

    if (runtime.attachment?.documentId) {
      try {
        await deleteDocument(runtime.attachment.documentId);
      } catch {
        // Already gone/expired on the backend - clear it from the UI regardless.
      }
    }

    if (runtime.attachment?.previewUrl) {
      URL.revokeObjectURL(runtime.attachment.previewUrl);
    }

    patchSessionRuntime(sessionId, { attachment: null, attachmentStatus: "idle", attachmentError: "" });
  }

  // Uploads a just-recorded clip to POST /transcribe and fills the
  // input with the result - addressed to `sessionId` (captured by
  // ChatBox at the moment recording started, see its own
  // onRecordingComplete), so the transcript lands in the composing box
  // of the session that was actually being recorded for, even if the
  // user has since switched to another one.
  async function transcribeForSession(sessionId, audioBlob) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "question.webm");

    try {
      const response = await transcribeAudio(formData);
      const { transcript, language } = response.data;
      patchSessionRuntime(sessionId, { question: transcript, detectedLanguage: language || null });
    } catch (err) {
      const detail = err.response?.data?.detail;
      patchSessionRuntime(sessionId, {
        error:
          typeof detail === "string"
            ? detail
            : "Voice input failed. Please try again or type your question.",
      });
    }
  }

  const activeRuntime = getSessionRuntime(activeSession.id);

  return (
    <div className="app-shell">

      <Navbar onToggleSidebar={() => setSidebarOpen((value) => !value)} />

      <div className="chat-layout">

        <ChatSidebar
          sessions={sortedSessions}
          activeSessionId={activeSession.id}
          sessionRuntime={sessionRuntime}
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
          onRenameSession={handleRenameSession}
          onDeleteSession={handleDeleteSession}
          onOpenVoiceChat={handleOpenVoiceChat}
          voiceChatSupported={VOICE_LIVE_SUPPORTED}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        <div className="chat-container">

          <h1>Chat</h1>

          <p>
            Ask a general knowledge question, ask about the connected database,
            or attach a PDF/image using the + button to ask about that too.
          </p>

          {/* Exactly ONE <ChatBox>, for the active session only - no
              hidden/display:none siblings. key={activeSession.id} gives
              it a fresh instance per session (resetting its own
              foreground-only local state: mic recording, auto-speak/
              language-hint toggles, the camera modal, copy/share
              feedback - see ChatBox's own docstring for why those don't
              need to survive a switch), which is what actually fixes
              the visual/layout sticking the old
              keep-every-visited-session-mounted approach had. Every
              callback below closes over activeSession.id at THIS
              render, so even a request that resolves after the user has
              switched away (its callback captured from the render where
              this session was still active) keeps writing into the
              right session's runtime entry, never whatever happens to
              be active by the time it completes. */}
          <ChatBox
            key={activeSession.id}
            messages={activeSession.messages}
            loading={activeRuntime.loading}
            error={activeRuntime.error}
            question={activeRuntime.question}
            attachment={activeRuntime.attachment}
            attachmentStatus={activeRuntime.attachmentStatus}
            attachmentError={activeRuntime.attachmentError}
            onQuestionChange={(text) => handleQuestionChange(activeSession.id, text)}
            onSubmit={(trimmedQuestion, autoSpeak, webSearch) => sendMessage(activeSession.id, trimmedQuestion, autoSpeak, webSearch)}
            onErrorMessage={(message) => handleErrorMessage(activeSession.id, message)}
            onUploadPdf={(file) => uploadAttachment(activeSession.id, file, file.name, null)}
            onUploadImage={(file) => uploadAttachment(activeSession.id, file, file.name, URL.createObjectURL(file))}
            onCameraCapture={(blob) =>
              uploadAttachment(activeSession.id, blob, `camera-capture-${Date.now()}.jpg`, URL.createObjectURL(blob))
            }
            onAttachmentValidationError={(message) => handleAttachmentValidationError(activeSession.id, message)}
            onRemoveAttachment={() => handleRemoveAttachment(activeSession.id)}
            onRecordingComplete={(audioBlob) => transcribeForSession(activeSession.id, audioBlob)}
            onOpenVoiceChat={handleOpenVoiceChat}
            voiceChatSupported={VOICE_LIVE_SUPPORTED}
          />

        </div>

      </div>

      {/* Exactly ONE instance for the whole page, opened from either
          ChatSidebar.jsx's or ChatBox.jsx's own "Voice Chat" button (see
          voiceChatOpen above) - rendered here, not inside <ChatBox>, so
          it's structurally impossible for a session switch/remount to
          affect an open voice session, and impossible for both buttons
          to ever open two separate connections. sessionId is pinned at
          open time by handleOpenVoiceChat (reusing the active session,
          or a freshly created one if none was active) and onTurnSaved
          appends each settled turn straight into that same session's
          history - see handleOpenVoiceChat/handleVoiceTurnSaved above
          for why closing Voice therefore always "returns to" that exact
          session with its history (text + voice) intact, with no session
          ever created or switched on close. */}
      {voiceChatOpen && (
        <VoiceChat
          sessionId={voiceSessionId}
          initialTranscript={voiceTranscriptFor(voiceSessionId)}
          onClose={handleCloseVoiceChat}
          onTurnSaved={(turn) => handleVoiceTurnSaved(voiceSessionId, turn)}
        />
      )}

    </div>
  );
}

export default Chat;
