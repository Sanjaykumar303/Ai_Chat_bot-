import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { speak, SPEECH_SYNTHESIS_SUPPORTED } from "../utils/speech";
import Loader from "./Loader";
import AttachmentMenu from "./AttachmentMenu";
import CameraCapture from "./CameraCapture";
import logo from "../assets/logo.jpg";
import { Camera, Chat, Close, Copy, Database, Edit, File, Globe, Image, Loader as LoaderIcon, Mic, Refresh, Send, Share, Speaker, User, Waveform } from "../icons";

// Feature detection happens once, at module load, since these APIs
// don't change while the app is running.
const MIC_RECORDING_SUPPORTED =
  typeof navigator !== "undefined" &&
  Boolean(navigator.mediaDevices?.getUserMedia) &&
  typeof window !== "undefined" &&
  Boolean(window.MediaRecorder);

// navigator.clipboard.writeText requires a secure context (https, or
// localhost in dev) - present in every real deployment of this app, but
// still feature-detected the same way every other browser API here is,
// rather than assumed. Backs both the Copy action and the Share action's
// clipboard fallback (see handleCopyMessage/handleShareMessage below).
const CLIPBOARD_SUPPORTED =
  typeof navigator !== "undefined" && Boolean(navigator.clipboard?.writeText);

// The Web Share API - mainly a mobile/OS-level "send to..." sheet.
// Absent on most desktop browsers, where handleShareMessage falls back
// to CLIPBOARD_SUPPORTED above.
const WEB_SHARE_SUPPORTED =
  typeof navigator !== "undefined" && typeof navigator.share === "function";

// Mirrors backend/services/pdf_service.py's PDF_MAX_BYTES (15 MB) and
// image_service.py's IMAGE_MAX_BYTES (10 MB) - purely a fast client-side
// check so an oversized file doesn't even start uploading; the backend
// enforces its own limits regardless.
const MAX_PDF_BYTES = 15 * 1024 * 1024;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

// Auto-stops a recording that's run this long, so a forgotten-open mic
// doesn't grow into a clip large/long enough for routes/transcribe.py's
// MAX_AUDIO_BYTES (15 MB, "well over a minute of compressed audio") to
// reject outright - stopping cleanly here always produces something
// transcribable instead of risking that late rejection.
const MAX_RECORDING_SECONDS = 90;

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
const IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

const CAMERA_MODAL_TITLE = (
  <>
    <Camera size={18} /> Camera
  </>
);

// Clickable starting points shown only on the empty-chat state (see
// "chat-empty" below) - the app supports several genuinely different
// question types (a connected database, general knowledge, an attached
// document, voice) with no other hint of that beyond one sentence of
// prose above these, so a first-time user has no obvious way to discover
// most of it. Deliberately domain-agnostic wording (no assumed table/
// metric name like "revenue" or "students") - this app is designed to
// work against ANY connected database's schema, not just the one
// currently deployed (see backend/services/db_query_service.py's own
// SQL_PROMPT), so a hardcoded business-specific example would be wrong
// on a differently-shaped database. Clicking one fills the input for the
// user to review/send, the same as re-editing a past question
// (handleEditMessage below) - never auto-submits, so nothing is sent
// (and no Gemini call made) without the user actually choosing to.
const EXAMPLE_PROMPTS = [
  "What data do you have access to?",
  "What is machine learning?",
];

// Sources come in two shapes: {type: "document"|"database", ...} from
// the PDF/hybrid answer paths (routes/chat.py), or plain {filename,
// page} from any future document-retrieval path that doesn't tag a
// type - rendered as simple badges either way.
function SourceBadge({ source }) {
  if (source.type === "document") {
    return (
      <span className="source-badge">
        <File size={14} /> {source.filename}
      </span>
    );
  }
  if (source.type === "database") {
    return (
      <span className="source-badge">
        <Database size={14} /> Database
      </span>
    );
  }
  // Web pages the backend's entity verification actually cited (see
  // services/entity_resolution.py). Linked out rather than plain text,
  // since the whole point of showing these is that the reader can check
  // the entity was resolved to the right organisation.
  if (source.type === "web") {
    return (
      <a className="source-badge" href={source.url} target="_blank" rel="noreferrer noopener">
        <Globe size={14} />
        <span className="source-badge-label">{source.title}</span>
      </a>
    );
  }
  return (
    <span className="source-badge">
      {source.page ? `${source.filename} (p. ${source.page})` : source.filename}
    </span>
  );
}

function isImageFilename(filename) {
  const lower = (filename || "").toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// ChatBox is now purely presentational for exactly one session at a
// time - pages/Chat.jsx renders a single instance for the active
// session (key={session.id}, so switching sessions is a normal
// unmount/mount of one subtree, never several DOM trees alive at once)
// and owns every piece of state that a background request needs to
// keep writing to after the user switches away: messages, loading,
// error, the attachment, and the composing question are all controlled
// props here, not local state - see Chat.jsx's own sendMessage/
// uploadAttachment/etc. for where they're actually mutated. This
// component's own local state is intentionally limited to things that
// only ever make sense in the foreground for whichever session is
// currently on screen: live mic recording, the auto-speak/web-search
// toggles, the camera modal, and the transient copy/share feedback
// pill - none of those have a meaningful "carry on in the background"
// version, so resetting them on every session switch (which remounting
// via key already does for free) is correct, not a gap.
//
// Real-time Voice Chat (components/VoiceChat.jsx) is the one exception
// to "this component's own state" above: it's controlled from here (the
// button below) but OWNED by pages/Chat.jsx, which is what lets
// ChatSidebar.jsx's own "Voice Chat" button open the exact same
// instance/connection rather than a second one - see Chat.jsx's own
// voiceChatOpen state and onOpenVoiceChat/voiceChatSupported props for
// both this component and ChatSidebar.jsx.
function ChatBox({
  messages,
  loading,
  error,
  question,
  attachment,
  attachmentStatus,
  attachmentError,
  onQuestionChange,
  onSubmit,
  onErrorMessage,
  onUploadPdf,
  onUploadImage,
  onCameraCapture,
  onAttachmentValidationError,
  onRemoveAttachment,
  onRecordingComplete,
  onOpenVoiceChat,
  voiceChatSupported,
}) {

  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcribing, setTranscribing] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);

  // Transient "Copied!"/"Shared!"-style feedback for the per-user-message
  // actions row below - { index, label } | null. Keyed by the message's
  // array index (messages have no separate id, matching how the message
  // list is already keyed for rendering below), auto-clears itself after
  // a couple seconds via actionFeedbackTimeoutRef.
  const [actionFeedback, setActionFeedback] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const recordingIntervalRef = useRef(null);
  const actionFeedbackTimeoutRef = useRef(null);
  const questionInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Keeps the latest message (or the typing indicator, while a reply is
  // pending) in view automatically - without this, a long conversation
  // would leave a new reply below the fold, silently requiring the user
  // to notice and scroll down themselves every single time. Also fires
  // on mount (a fresh instance every session switch, see the module
  // docstring), which is exactly what's wanted: land scrolled to this
  // session's latest state, not wherever the previous session happened
  // to be scrolled.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  // Stop any voice activity left running, if the user switches away
  // from this session (this whole component unmounts - see Chat.jsx's
  // key={session.id}) or navigates off the page. Recording/speaking are
  // both inherently foreground-only (one physical microphone, one
  // speechSynthesis singleton) - there's no meaningful "keep going in
  // the background for a session I've switched away from" version of
  // either, so stopping them here is correct, not a loss of state.
  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      clearInterval(recordingIntervalRef.current);
      clearTimeout(actionFeedbackTimeoutRef.current);
      if (SPEECH_SYNTHESIS_SUPPORTED) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // Records a clip from the microphone, uploads it to the backend for
  // Gemini-based transcription (services/transcription.py), then fills
  // the input with the result - the user can still edit it before
  // sending, same as the old live-transcript behavior.
  async function startRecording() {
    if (!MIC_RECORDING_SUPPORTED || loading || transcribing) {
      return;
    }

    onErrorMessage("");

    let stream;
    try {
      // Mono, and the browser's own noise/gain handling - cheap, built-in
      // preprocessing that's still worth doing even though transcription
      // is now Gemini-based rather than a local model. The backend
      // (services/transcription.py's _convert_to_wav) resamples to a
      // fixed rate internally regardless of what's recorded here, so no
      // sample-rate constraint is needed on this end.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      // getUserMedia's error .name distinguishes real denial from a
      // missing/busy device - a blanket "access denied" message is
      // actively misleading for the latter two, which no permission
      // prompt would ever fix.
      if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        onErrorMessage("No microphone was found. Please connect a microphone and try again.");
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        onErrorMessage("The microphone is already in use by another application.");
      } else {
        onErrorMessage("Microphone access was denied. Please allow microphone access and try again.");
      }
      return;
    }

    streamRef.current = stream;
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    // onRecordingComplete (Chat.jsx's transcribeForSession, bound to
    // *this* session via closure at render time) is what actually keeps
    // the transcript landing in the right session even if the user
    // switches away before it resolves - this component may already be
    // unmounted by the time onstop fires (switching sessions stops any
    // live recording, see the cleanup effect above), but the callback
    // reference itself was captured when this instance was still
    // mounted for this session, so it still resolves correctly. If that
    // happens, the setTranscribing(false) below lands on an unmounted
    // instance and is silently dropped by React - harmless, since the
    // mic-button spinner this drives is foreground-only UI, not
    // conversation state.
    mediaRecorder.onstop = async () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      const audioBlob = new Blob(chunks, { type: mediaRecorder.mimeType });
      setTranscribing(true);
      await onRecordingComplete(audioBlob);
      setTranscribing(false);
    };

    mediaRecorderRef.current = mediaRecorder;

    try {
      mediaRecorder.start();
      setRecording(true);
      setRecordingSeconds(0);

      recordingIntervalRef.current = setInterval(() => {
        setRecordingSeconds((seconds) => {
          const next = seconds + 1;
          if (next >= MAX_RECORDING_SECONDS) {
            stopRecording();
          }
          return next;
        });
      }, 1000);
    } catch {
      onErrorMessage("Could not start voice input. Please try again.");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    clearInterval(recordingIntervalRef.current);
  }

  function handleUploadPdf(file) {
    if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
      onAttachmentValidationError("Only PDF files are supported.");
      return;
    }
    if (file.size > MAX_PDF_BYTES) {
      onAttachmentValidationError(`File is too large (max ${MAX_PDF_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    onUploadPdf(file);
  }

  function handleUploadImage(file) {
    const name = file.name.toLowerCase();
    const isImage = IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext)) || IMAGE_MIME_TYPES.includes(file.type);

    if (!isImage) {
      onAttachmentValidationError("Only JPG, PNG, or WEBP images are supported.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      onAttachmentValidationError(`File is too large (max ${MAX_IMAGE_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    onUploadImage(file);
  }

  function handleCameraCapture(blob) {
    setCameraOpen(false);
    onCameraCapture(blob);
  }

  // Shows `label` next to the given user message's actions row for a
  // couple seconds, then clears itself - shared by the Copy action and
  // the Share action's clipboard fallback, which both need the same
  // "it worked" confirmation.
  function showActionFeedback(index, label) {
    // Clears any error left over from an earlier failed copy/share on
    // this or another message - otherwise a stale "Could not copy to
    // clipboard." banner could keep showing under a since-succeeded
    // action, which reads as if this one failed too.
    onErrorMessage("");
    clearTimeout(actionFeedbackTimeoutRef.current);
    setActionFeedback({ index, label });
    actionFeedbackTimeoutRef.current = setTimeout(() => setActionFeedback(null), 1800);
  }

  // Copies a user message's exact text - no markdown rendering, no
  // trimming/reformatting, byte-for-byte what they sent.
  async function handleCopyMessage(text, index) {
    if (!CLIPBOARD_SUPPORTED) {
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      showActionFeedback(index, "Copied!");
    } catch {
      onErrorMessage("Could not copy to clipboard.");
    }
  }

  // Loads a past user message back into the input for editing and
  // resubmission - deliberately does NOT touch `messages` itself (the
  // old message stays exactly where it is in history; resubmitting adds
  // a new message afterward, same as typing any other question), so
  // there's no history to keep consistent here beyond what onSubmit
  // already does for every send.
  function handleEditMessage(text) {
    onQuestionChange(text);
    onErrorMessage("");
    questionInputRef.current?.focus();
  }

  // Re-asks the exact question that produced the AI message at `index`,
  // by resubmitting the immediately preceding user message through the
  // same onSubmit prop handleSubmit itself calls below - not a special
  // "regenerate" endpoint, just asking again. Unlike handleEditMessage
  // above, this sends immediately rather than filling the input for
  // review first: the whole point of a one-click regenerate is not
  // having to manually retype/resend, and the original question text is
  // never in doubt here (unlike an edit, where the user might want to
  // change it) - see project_ux_review_2026-08-20 memory for why this
  // exists (a DB answer was found to occasionally give a different
  // number for the same question; this gives a user a way to double-
  // check one without retyping it).
  function handleRegenerate(index) {
    const precedingQuestion = messages[index - 1];
    if (precedingQuestion?.sender !== "user") {
      return;
    }
    onErrorMessage("");
    onSubmit(precedingQuestion.text, autoSpeak, webSearchEnabled);
  }

  // Web Share API first (the mobile/OS "send to..." sheet) - falls back
  // to copying the same text to the clipboard when it's unavailable
  // (most desktop browsers), reusing the exact same feedback mechanism
  // as handleCopyMessage so a fallback share still gets a clear "it
  // worked" confirmation.
  async function handleShareMessage(text, index) {
    if (WEB_SHARE_SUPPORTED) {
      try {
        await navigator.share({ text });
      } catch (err) {
        // The user closing the native share sheet without picking
        // anything throws AbortError - that's a cancel, not a failure.
        if (err.name !== "AbortError") {
          onErrorMessage("Could not share this message.");
        }
      }
      return;
    }

    if (!CLIPBOARD_SUPPORTED) {
      return;
    }

    try {
      await navigator.clipboard.writeText(text);
      showActionFeedback(index, "Copied to share!");
    } catch {
      onErrorMessage("Could not copy to clipboard.");
    }
  }

  // Runs when the user submits the form. Validates and gathers the
  // input, then delegates the actual send (and everything after it -
  // appending messages, the /chat request, loading/error state) to
  // Chat.jsx's onSubmit, which is what keeps all of that alive if the
  // user switches to another session before the response comes back.
  function handleSubmit(event) {

    // Stops the browser from reloading the page.
    event.preventDefault();

    // Submitting (Enter, or clicking Send) while a recording is still
    // in progress, or its transcript hasn't come back yet, isn't a
    // genuinely empty question - the question box is only empty because
    // the transcription hasn't arrived yet. Stop the recording (if any)
    // and return without touching the error, rather than falsely
    // claiming nothing was typed - onRecordingComplete fills the input
    // in as soon as it resolves, and the user can submit again then.
    if (recording) {
      stopRecording();
      return;
    }

    if (transcribing) {
      return;
    }

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      onErrorMessage("Please type a question.");
      return;
    }

    onSubmit(trimmedQuestion, autoSpeak, webSearchEnabled);
  }

  const attachmentBusy = attachmentStatus === "uploading" || attachmentStatus === "processing";

  return (
    <div className="chat-box">

      <div className="chat-toolbar">
        <label
          className="auto-speak-toggle"
          title={SPEECH_SYNTHESIS_SUPPORTED ? "" : "Voice output is not supported in this browser"}
        >
          <input
            type="checkbox"
            checked={autoSpeak}
            onChange={(event) => setAutoSpeak(event.target.checked)}
            disabled={!SPEECH_SYNTHESIS_SUPPORTED}
          />
          Auto Speak
        </label>
      </div>

      <div className="chat-messages" role="log" aria-live="polite" aria-relevant="additions">


        {messages.length === 0 && !loading && (
          <div className="chat-empty">
            <Chat size={22} className="chat-empty-icon" />
            <p>Ask me anything - general knowledge, your connected database, or attach a PDF/image to ask about that too.</p>
            <div className="chat-example-prompts">
              {EXAMPLE_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="chat-example-prompt"
                  onClick={() => handleEditMessage(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.sender} ${message.streaming ? "streaming" : ""}`}>
            {message.sender === "user" ? (
              <span className="chat-avatar chat-avatar-user">
                <User size={18} />
              </span>
            ) : (
              <img className="chat-avatar chat-avatar-ai" src={logo} alt="" />
            )}

            {/* Wraps the bubble itself and (for a user message) the
                actions row below it, so the actions can sit outside the
                bubble's own padding/background while still stacking
                under it and sharing its left/right alignment - see
                .chat-message-content in styles/Chat.css. */}
            <div className="chat-message-content">
              <div className="chat-message-body">
                <span className="chat-sender">
                  {message.sender === "user" ? "You" : "Assistant"}
                </span>

                <div className="chat-message-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
                </div>

                {message.sender === "ai" && !message.streaming && (
                  <button
                    type="button"
                    className="speaker-button"
                    onClick={() => speak(message.text, onErrorMessage)}
                    disabled={!SPEECH_SYNTHESIS_SUPPORTED}
                    aria-label={SPEECH_SYNTHESIS_SUPPORTED ? "Read this answer aloud" : "Voice output is not supported in this browser"}
                    title={SPEECH_SYNTHESIS_SUPPORTED ? "Read this answer aloud" : "Voice output is not supported in this browser"}
                  >
                    <Speaker size={18} />
                  </button>
                )}

                {message.sources?.length > 0 && (
                  <p className="chat-sources">
                    Sources:{" "}
                    {message.sources.map((source, sourceIndex) => (
                      <SourceBadge key={sourceIndex} source={source} />
                    ))}
                  </p>
                )}

                {/* A generated DOCX/XLSX export (see pages/Chat.jsx's
                    exportInfo handling) - a plain browser download link,
                    same .source-badge look every other small pill in
                    this message body already uses, so this doesn't need
                    any styling of its own. */}
                {message.download && (
                  <p className="chat-sources">
                    <a
                      className="source-badge"
                      href={message.download.url}
                      download={message.download.filename}
                    >
                      <File size={14} /> {message.download.filename}
                    </a>
                  </p>
                )}
              </div>

              {/* ChatGPT-style per-message actions - user messages only,
                  below the bubble rather than inside it. Hidden until
                  hover/focus on desktop (see .chat-message-actions in
                  styles/Chat.css), always visible on touch devices (no
                  hover to reveal them there) via that same rule's
                  @media (hover: none) override. */}
              {message.sender === "user" && (
                <div className="chat-message-actions">
                  <button
                    type="button"
                    className="chat-message-action"
                    onClick={() => handleCopyMessage(message.text, index)}
                    disabled={!CLIPBOARD_SUPPORTED}
                    aria-label={CLIPBOARD_SUPPORTED ? "Copy message" : "Copying is not supported in this browser"}
                    title={CLIPBOARD_SUPPORTED ? "Copy message" : "Copying is not supported in this browser"}
                  >
                    <Copy size={16} />
                  </button>

                  <button
                    type="button"
                    className="chat-message-action"
                    onClick={() => handleEditMessage(message.text)}
                    disabled={recording || transcribing}
                    aria-label="Edit message"
                    title="Edit message"
                  >
                    <Edit size={16} />
                  </button>

                  <button
                    type="button"
                    className="chat-message-action"
                    onClick={() => handleShareMessage(message.text, index)}
                    disabled={!WEB_SHARE_SUPPORTED && !CLIPBOARD_SUPPORTED}
                    aria-label="Share prompt"
                    title="Share prompt"
                  >
                    <Share size={16} />
                  </button>

                  {actionFeedback?.index === index && (
                    <span className="chat-message-action-feedback" role="status">
                      {actionFeedback.label}
                    </span>
                  )}
                </div>
              )}

              {message.sender === "ai" && !message.streaming && (
                <div className="chat-message-actions">
                  <button
                    type="button"
                    className="chat-message-action"
                    onClick={() => handleCopyMessage(message.text, index)}
                    disabled={!CLIPBOARD_SUPPORTED}
                    aria-label={CLIPBOARD_SUPPORTED ? "Copy answer" : "Copying is not supported in this browser"}
                    title={CLIPBOARD_SUPPORTED ? "Copy answer" : "Copying is not supported in this browser"}
                  >
                    <Copy size={16} />
                  </button>

                  <button
                    type="button"
                    className="chat-message-action"
                    onClick={() => handleRegenerate(index)}
                    disabled={loading || recording || transcribing}
                    aria-label="Regenerate this answer"
                    title="Regenerate this answer"
                  >
                    <Refresh size={16} />
                  </button>

                  {actionFeedback?.index === index && (
                    <span className="chat-message-action-feedback" role="status">
                      {actionFeedback.label}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Once the first streamed chunk arrives, an in-progress AI
            message (message.streaming - see pages/Chat.jsx's sendMessage)
            takes over as the visual "still working" indicator - showing
            this too would duplicate it. Only shown while genuinely
            waiting for that first chunk (or the whole thing failed
            before any arrived at all). */}
        {loading && !(messages[messages.length - 1]?.sender === "ai" && messages[messages.length - 1]?.streaming) && (
          <Loader text="Thinking..." />
        )}

        <div ref={messagesEndRef} />

      </div>

      {error && <p className="chat-error">{error}</p>}

      {recording && (
        <p className="chat-recording-status">
          <span className="chat-recording-dot" /> Listening… {formatDuration(recordingSeconds)}
        </p>
      )}

      {webSearchEnabled && (
        <p className="chat-web-search-indicator">
          <Globe size={14} /> Web Search is ON - this message will be answered using fresh web results
        </p>
      )}

      {(attachment || attachmentBusy || attachmentStatus === "error") && (
        <div className="attachment-preview">

          {attachmentBusy && (
            <span className="attachment-preview-status">
              {attachmentStatus === "uploading" ? "Uploading..." : "Processing document..."}
            </span>
          )}

          {attachmentStatus === "error" && (
            <span className="attachment-preview-error">{attachmentError}</span>
          )}

          {attachment && !attachmentBusy && (
            <>
              {attachment.previewUrl ? (
                <img className="attachment-preview-thumb" src={attachment.previewUrl} alt="" />
              ) : (
                <span className="attachment-preview-icon">
                  {isImageFilename(attachment.filename) ? <Image size={18} /> : <File size={18} />}
                </span>
              )}
              <span className="attachment-preview-name">{attachment.filename}</span>
              <button
                type="button"
                className="attachment-preview-remove"
                onClick={onRemoveAttachment}
                aria-label="Remove attachment"
                title="Remove attachment"
              >
                <Close size={16} />
              </button>
            </>
          )}

        </div>
      )}

      <form className="chat-form" onSubmit={handleSubmit}>

        <AttachmentMenu
          disabled={loading || attachmentBusy}
          onUploadPdf={handleUploadPdf}
          onUploadImage={handleUploadImage}
          onOpenCamera={() => setCameraOpen(true)}
        />

        <input
          ref={questionInputRef}
          type="text"
          value={question}
          placeholder="Ask a question"
          onChange={(event) => onQuestionChange(event.target.value)}
          disabled={loading}
        />

        <button
          type="button"
          className={`mic-button ${recording ? "recording" : ""} ${transcribing ? "transcribing" : ""}`}
          onClick={recording ? stopRecording : startRecording}
          disabled={!MIC_RECORDING_SUPPORTED || loading || transcribing}
          aria-label={
            !MIC_RECORDING_SUPPORTED
              ? "Voice input is not supported in this browser"
              : recording
                ? "Stop recording"
                : transcribing
                  ? "Transcribing..."
                  : "Speak your question"
          }
          title={
            !MIC_RECORDING_SUPPORTED
              ? "Voice input is not supported in this browser"
              : recording
                ? "Stop recording"
                : transcribing
                  ? "Transcribing..."
                  : "Speak your question"
          }
        >
          {transcribing ? <LoaderIcon size={18} className="icon-spin" /> : <Mic size={18} />}
        </button>

        <button
          type="button"
          className={`web-search-button ${webSearchEnabled ? "active" : ""}`}
          onClick={() => setWebSearchEnabled((enabled) => !enabled)}
          aria-pressed={webSearchEnabled}
          aria-label={webSearchEnabled ? "Web Search is on - click to turn off" : "Turn on Web Search for this message"}
          title={webSearchEnabled ? "Web Search is on - click to turn off" : "Turn on Web Search for this message"}
        >
          <Globe size={18} />
        </button>

        <button
          type="button"
          className="voice-chat-button"
          onClick={onOpenVoiceChat}
          disabled={!voiceChatSupported}
          aria-label={voiceChatSupported ? "Start real-time voice chat" : "Voice chat is not supported in this browser"}
          title={voiceChatSupported ? "Start real-time voice chat" : "Voice chat is not supported in this browser"}
        >
          <Waveform size={18} />
        </button>

        <button type="submit" disabled={loading || recording || transcribing}>
          {loading ? <LoaderIcon size={16} className="icon-spin" /> : <Send size={16} />}
          {loading ? "Sending..." : "Send"}
        </button>

      </form>

      {cameraOpen && (
        <CameraCapture
          title={CAMERA_MODAL_TITLE}
          onCapture={handleCameraCapture}
          onClose={() => setCameraOpen(false)}
        />
      )}

    </div>
  );
}

export default ChatBox;
