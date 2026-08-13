import { useEffect, useRef, useState } from "react";
import { sendChatMessage, uploadDocument, deleteDocument, transcribeAudio } from "../services/api";
import Loader from "./Loader";
import AttachmentMenu from "./AttachmentMenu";
import CameraCapture from "./CameraCapture";

// Feature detection happens once, at module load, since these APIs
// don't change while the app is running.
const MIC_RECORDING_SUPPORTED =
  typeof navigator !== "undefined" &&
  Boolean(navigator.mediaDevices?.getUserMedia) &&
  typeof window !== "undefined" &&
  Boolean(window.MediaRecorder);

const SPEECH_SYNTHESIS_SUPPORTED =
  typeof window !== "undefined" && "speechSynthesis" in window;

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

const CAMERA_MODAL_TITLES = {
  camera: "📷 Camera",
  live: "🎥 Live Camera",
};

// Voice input is transcribed by a local Whisper model on the backend
// (services/transcription.py). Auto-detection is the default and works
// fine for clearly English speech, but real testing showed it under-uses
// Tamil on short/code-mixed clips (small Whisper models are known to be
// English-biased under auto-detect) - so a manual override is offered,
// left on Auto unless the user knows which language they're about to speak.
const LANGUAGE_LABELS = {
  en: "English",
  ta: "Tamil",
  kn: "Kannada",
  bn: "Bengali",
};

const LANGUAGE_HINT_OPTIONS = [
  { code: "auto", label: "Auto-detect" },
  { code: "en", label: "English" },
  { code: "ta", label: "Tamil" },
  { code: "kn", label: "Kannada" },
  { code: "bn", label: "Bengali" },
];

function languageLabel(code) {
  return LANGUAGE_LABELS[code] || code;
}

// Tamil script (U+0B80-U+0BFF). If any of it appears in the answer, the
// answer is treated as Tamil; otherwise it's read as English. A plain
// script check, not a Gemini call, matching how this app already prefers
// deterministic, cost-free classification over an extra API call.
const TAMIL_SCRIPT = /[஀-௿]/;

function detectLanguage(text) {
  return TAMIL_SCRIPT.test(text) ? "ta-IN" : "en-US";
}

// speechSynthesis.getVoices() can return an empty list until the
// voiceschanged event has fired once, in some browsers. Voices are cached
// here and refreshed whenever that event fires, so pickVoice() works
// correctly regardless of when it's first called.
let cachedVoices = [];

if (SPEECH_SYNTHESIS_SUPPORTED) {
  cachedVoices = window.speechSynthesis.getVoices();
  window.speechSynthesis.onvoiceschanged = () => {
    cachedVoices = window.speechSynthesis.getVoices();
  };
}

// Finds a voice for the given language: an exact match first (e.g.
// "ta-IN"), then any voice for the same language regardless of region
// (e.g. "ta-LK") - the "search for another compatible voice" fallback.
// Returns null if nothing matches, rather than guessing.
function pickVoice(lang) {
  const voices = cachedVoices.length > 0 ? cachedVoices : (SPEECH_SYNTHESIS_SUPPORTED ? window.speechSynthesis.getVoices() : []);
  const prefix = lang.split("-")[0];

  return (
    voices.find((voice) => voice.lang === lang) ||
    voices.find((voice) => voice.lang.toLowerCase().startsWith(prefix)) ||
    null
  );
}

// Sources come in two shapes: {type: "document"|"database", ...} from
// the PDF/hybrid answer paths (routes/chat.py), or plain {filename,
// page} from any future document-retrieval path that doesn't tag a
// type - rendered as simple badges either way.
function SourceBadge({ source }) {
  if (source.type === "document") {
    return <span className="source-badge">📄 {source.filename}</span>;
  }
  if (source.type === "database") {
    return <span className="source-badge">🗄️ Database</span>;
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

// initialMessages seeds this chat session's history (see Chat.jsx,
// which remounts ChatBox with key={sessionId} on every chat switch, so
// this only ever runs once per session rather than needing to sync on
// prop changes). onMessagesChange is called whenever the message list
// changes, so Chat.jsx can persist it - ChatBox itself never touches
// localStorage.
function ChatBox({ initialMessages = [], onMessagesChange }) {

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcribing, setTranscribing] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState(null);
  const [languageHint, setLanguageHint] = useState("auto");

  // The backend's resolved_question from the last exchange in *this*
  // chat (see routes/chat.py) - not necessarily what the user literally
  // typed, since a follow-up like "yesterday" resolves to something like
  // "What is the total profit yesterday?". Sent back as the next
  // request's previous_question so a chain of follow-ups ("today" ->
  // "yesterday" -> "and last week?") each build on the fully-expanded
  // form of the one before it. Deliberately local component state, not
  // persisted (see utils/chatStorage.js) - like attachment state, it
  // resets on every chat switch (ChatBox is remounted, key=sessionId,
  // see Chat.jsx), so a follow-up never resolves against a different
  // chat's question.
  const [lastResolvedQuestion, setLastResolvedQuestion] = useState(null);

  // Attachment (uploaded PDF/image/camera capture) state - lives here
  // now that there's no permanent sidebar to hold it. { documentId,
  // filename, previewUrl } | null. previewUrl is a local object URL for
  // images only (revoked on removal/replacement), so the compact
  // preview can show an actual thumbnail rather than just an icon.
  const [attachment, setAttachment] = useState(null);
  const [attachmentStatus, setAttachmentStatus] = useState("idle"); // idle | uploading | processing | error
  const [attachmentError, setAttachmentError] = useState("");
  const [cameraMode, setCameraMode] = useState(null); // null | "camera" | "live"

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const recordingIntervalRef = useRef(null);
  const attachmentPreviewUrlRef = useRef(null);
  // Captured once, at construction - lets the effect below tell "still
  // the untouched initial array" apart from "a real setMessages update
  // happened", by reference rather than by a mutable "have I run
  // before" flag. A flag would break under React 18 StrictMode's
  // dev-only double-invoke of effects (it flips on the first synthetic
  // run, so the second synthetic run - against the same, still-
  // unchanged messages - would wrongly look like a genuine change); a
  // stable reference snapshot gives the same, correct answer no matter
  // how many times the effect happens to run.
  const initialMessagesRef = useRef(initialMessages);

  // Reports this session's message list up to Chat.jsx for persistence
  // (see utils/chatStorage.js) whenever it actually changes. Skipped
  // when messages is still the original initialMessages reference,
  // since reporting that back unchanged would bump this chat's
  // updatedAt (and its position in the Recent Chats list) merely from
  // switching to it, not from any real new message. Deliberately not
  // depending on onMessagesChange itself, since a new inline function
  // identity from the parent every render shouldn't re-fire this.
  useEffect(() => {
    if (messages === initialMessagesRef.current) {
      return;
    }
    onMessagesChange?.(messages);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]);

  // Stop any voice activity left running, and release any local object
  // URL, if the user navigates away.
  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      clearInterval(recordingIntervalRef.current);
      if (SPEECH_SYNTHESIS_SUPPORTED) {
        window.speechSynthesis.cancel();
      }
      if (attachmentPreviewUrlRef.current) {
        URL.revokeObjectURL(attachmentPreviewUrlRef.current);
      }
    };
  }, []);

  // Reads a piece of text aloud, in whichever language it's actually
  // written in. Always stops whatever is currently playing first, whether
  // that came from a manual click, Auto Speak, or replaying an older
  // message - language/voice are re-detected fresh every call, never
  // cached, so replay always matches that message's real language.
  function speak(text) {
    if (!SPEECH_SYNTHESIS_SUPPORTED || !text) {
      return;
    }

    const lang = detectLanguage(text);
    const voice = pickVoice(lang);

    if (lang === "ta-IN" && !voice) {
      setError(
        "No Tamil voice is installed on this device. Add a Tamil voice in " +
        "your operating system's language/speech settings to hear Tamil answers read aloud."
      );
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    if (voice) {
      utterance.voice = voice;
    }
    utterance.onerror = (event) => {
      // "canceled"/"interrupted" fire on the utterance we just stopped with
      // cancel() above - that's expected, not a failure, so stay quiet.
      if (event.error === "canceled" || event.error === "interrupted") {
        return;
      }
      setError("Could not read the answer aloud.");
    };

    window.speechSynthesis.speak(utterance);
  }

  // Records a clip from the microphone, uploads it to the backend for
  // local Whisper transcription (services/transcription.py), then fills
  // the input with the result - the user can still edit it before
  // sending, same as the old live-transcript behavior.
  async function startRecording() {
    if (!MIC_RECORDING_SUPPORTED || loading || transcribing) {
      return;
    }

    setError("");
    setDetectedLanguage(null);

    let stream;
    try {
      // Mono, and the browser's own noise/gain handling - cheap, built-in
      // preprocessing that measurably helps a small model like Whisper
      // tiny. Whisper resamples internally regardless of input rate, so
      // no sample-rate constraint is needed here.
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
        setError("No microphone was found. Please connect a microphone and try again.");
      } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
        setError("The microphone is already in use by another application.");
      } else {
        setError("Microphone access was denied. Please allow microphone access and try again.");
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

    mediaRecorder.onstop = async () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      const audioBlob = new Blob(chunks, { type: mediaRecorder.mimeType });
      await sendForTranscription(audioBlob);
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
      setError("Could not start voice input. Please try again.");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    clearInterval(recordingIntervalRef.current);
  }

  // Uploads the recorded clip to POST /transcribe. No audio ever reaches
  // Gemini or leaves the machine - only the resulting text does, and only
  // once the user submits it through the existing /chat pipeline.
  async function sendForTranscription(audioBlob) {
    setTranscribing(true);

    const formData = new FormData();
    formData.append("audio", audioBlob, "question.webm");
    formData.append("language_hint", languageHint);

    try {
      const response = await transcribeAudio(formData);
      const { transcript, language } = response.data;
      setQuestion(transcript);
      setDetectedLanguage(language || null);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Voice input failed. Please try again or type your question."
      );
    }

    setTranscribing(false);
  }

  // Shared upload path for a PDF, a picked image, or a camera-captured
  // image blob - all three are just bytes to POST /documents/upload,
  // which already validates/branches PDF vs. image server-side
  // (routes/documents.py). previewUrl is only ever set for images.
  async function uploadAttachment(file, filename, previewUrl) {
    // Replacing an existing attachment: drop the old temporary context
    // first, so it's never left around or mistakenly reused once the
    // new one is ready - same as the removed Sidebar's replace logic.
    if (attachment?.documentId) {
      try {
        await deleteDocument(attachment.documentId);
      } catch {
        // Already gone/expired - fine, proceed with the new upload regardless.
      }
      if (attachmentPreviewUrlRef.current) {
        URL.revokeObjectURL(attachmentPreviewUrlRef.current);
      }
      setAttachment(null);
    }

    setAttachmentError("");
    setAttachmentStatus("uploading");
    attachmentPreviewUrlRef.current = previewUrl || null;

    const formData = new FormData();
    formData.append("file", file, filename);

    try {
      // Bytes fully sent to the server means the "uploading" phase is
      // over - what's left (extraction + chunking + indexing, or OCR
      // for a scanned PDF/image) is "processing", which has no
      // progress events of its own.
      const response = await uploadDocument(formData, (progressEvent) => {
        if (progressEvent.loaded >= progressEvent.total) {
          setAttachmentStatus("processing");
        }
      });
      const { document_id, filename: returnedFilename } = response.data;
      setAttachment({ documentId: document_id, filename: returnedFilename, previewUrl: previewUrl || null });
      setAttachmentStatus("idle");
    } catch (err) {
      const detail = err.response?.data?.detail;
      setAttachmentStatus("error");
      setAttachmentError(typeof detail === "string" ? detail : "Unable to process this file.");
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      attachmentPreviewUrlRef.current = null;
    }
  }

  function handleUploadPdf(file) {
    if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
      setAttachmentStatus("error");
      setAttachmentError("Only PDF files are supported.");
      return;
    }
    if (file.size > MAX_PDF_BYTES) {
      setAttachmentStatus("error");
      setAttachmentError(`File is too large (max ${MAX_PDF_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    uploadAttachment(file, file.name, null);
  }

  function handleUploadImage(file) {
    const name = file.name.toLowerCase();
    const isImage = IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext)) || IMAGE_MIME_TYPES.includes(file.type);

    if (!isImage) {
      setAttachmentStatus("error");
      setAttachmentError("Only JPG, PNG, or WEBP images are supported.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setAttachmentStatus("error");
      setAttachmentError(`File is too large (max ${MAX_IMAGE_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    uploadAttachment(file, file.name, URL.createObjectURL(file));
  }

  function handleCameraCapture(blob) {
    setCameraMode(null);
    const filename = `camera-capture-${Date.now()}.jpg`;
    uploadAttachment(blob, filename, URL.createObjectURL(blob));
  }

  async function handleRemoveAttachment() {
    if (attachment?.documentId) {
      try {
        await deleteDocument(attachment.documentId);
      } catch {
        // Already gone/expired on the backend - clear it from the UI regardless.
      }
    }

    if (attachmentPreviewUrlRef.current) {
      URL.revokeObjectURL(attachmentPreviewUrlRef.current);
      attachmentPreviewUrlRef.current = null;
    }

    setAttachment(null);
    setAttachmentStatus("idle");
    setAttachmentError("");
  }

  // Runs when the user submits the form.
  async function handleSubmit(event) {

    // Stops the browser from reloading the page.
    event.preventDefault();

    if (recording) {
      stopRecording();
    }

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setError("Please type a question.");
      return;
    }

    // Only tag this request with a spoken language if the text still is
    // what transcription produced - once the user edits or types their
    // own question, detectedLanguage has already been cleared (see the
    // input's onChange below), so a stale language never gets attached.
    const spokenLanguage = detectedLanguage;

    // Show the question straight away.
    setMessages((previous) => [
      ...previous,
      { sender: "user", text: trimmedQuestion },
    ]);

    setQuestion("");
    setDetectedLanguage(null);
    setError("");
    setLoading(true);

    try {
      const response = await sendChatMessage({
        question: trimmedQuestion,
        language: spokenLanguage,
        document_id: attachment?.documentId ?? null,
        previous_question: lastResolvedQuestion,
      });
      const { answer, sources = [], resolved_question } = response.data;

      setLastResolvedQuestion(resolved_question || trimmedQuestion);

      setMessages((previous) => [
        ...previous,
        { sender: "ai", text: answer, sources },
      ]);

      if (autoSpeak) {
        speak(answer);
      }

    } catch (err) {
      // FastAPI sends its error text inside response.data.detail.
      // It is only a plain string for our own errors, so we check the type.
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Could not reach the server. Please check that the backend is running."
      );
    }

    setLoading(false);
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

      <div className="chat-messages">

        {messages.length === 0 && !loading && (
          <p className="chat-empty">
            Ask a general knowledge question, or ask about the connected database.
          </p>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.sender}`}>
            <span className="chat-sender">
              {message.sender === "user" ? "You" : "Assistant"}
            </span>
            <p>{message.text}</p>

            {message.sender === "ai" && (
              <button
                type="button"
                className="speaker-button"
                onClick={() => speak(message.text)}
                disabled={!SPEECH_SYNTHESIS_SUPPORTED}
                title={SPEECH_SYNTHESIS_SUPPORTED ? "Read this answer aloud" : "Voice output is not supported in this browser"}
              >
                🔊
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
          </div>
        ))}

        {loading && <Loader text="Thinking..." />}

      </div>

      {error && <p className="chat-error">{error}</p>}

      {recording && (
        <p className="chat-recording-status">
          <span className="chat-recording-dot" /> Listening… {formatDuration(recordingSeconds)}
        </p>
      )}

      {detectedLanguage && (
        <p className="chat-detected-language">Heard: {languageLabel(detectedLanguage)}</p>
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
                  {isImageFilename(attachment.filename) ? "🖼️" : "📄"}
                </span>
              )}
              <span className="attachment-preview-name">{attachment.filename}</span>
              <button
                type="button"
                className="attachment-preview-remove"
                onClick={handleRemoveAttachment}
                aria-label="Remove attachment"
                title="Remove attachment"
              >
                ✕
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
          onOpenCamera={setCameraMode}
        />

        <input
          type="text"
          value={question}
          placeholder="Ask a question"
          onChange={(event) => {
            setQuestion(event.target.value);
            setDetectedLanguage(null);
          }}
          disabled={loading}
        />

        <select
          className="language-select"
          value={languageHint}
          onChange={(event) => setLanguageHint(event.target.value)}
          disabled={!MIC_RECORDING_SUPPORTED || loading || recording || transcribing}
          title="Voice input language (leave on Auto-detect unless it keeps missing your language)"
        >
          {LANGUAGE_HINT_OPTIONS.map((option) => (
            <option key={option.code} value={option.code}>
              {option.label}
            </option>
          ))}
        </select>

        <button
          type="button"
          className={`mic-button ${recording ? "recording" : ""} ${transcribing ? "transcribing" : ""}`}
          onClick={recording ? stopRecording : startRecording}
          disabled={!MIC_RECORDING_SUPPORTED || loading || transcribing}
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
          {transcribing ? "…" : "🎤"}
        </button>

        <button type="submit" disabled={loading}>
          {loading ? "Sending..." : "Send"}
        </button>

      </form>

      {cameraMode && (
        <CameraCapture
          title={CAMERA_MODAL_TITLES[cameraMode]}
          onCapture={handleCameraCapture}
          onClose={() => setCameraMode(null)}
        />
      )}

    </div>
  );
}

export default ChatBox;
