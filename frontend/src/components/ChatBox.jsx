import { useEffect, useRef, useState } from "react";
import api from "../services/api";
import Loader from "./Loader";

// Feature detection happens once, at module load, since these APIs
// don't change while the app is running.
const MIC_RECORDING_SUPPORTED =
  typeof navigator !== "undefined" &&
  Boolean(navigator.mediaDevices?.getUserMedia) &&
  typeof window !== "undefined" &&
  Boolean(window.MediaRecorder);

const SPEECH_SYNTHESIS_SUPPORTED =
  typeof window !== "undefined" && "speechSynthesis" in window;

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

function ChatBox({ documentId = null }) {

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState(null);
  const [languageHint, setLanguageHint] = useState("auto");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  // Stop any voice activity left running if the user navigates away.
  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (SPEECH_SYNTHESIS_SUPPORTED) {
        window.speechSynthesis.cancel();
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
    } catch {
      setError("Microphone access was denied. Please allow microphone access and try again.");
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
    } catch {
      setError("Could not start voice input. Please try again.");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
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
      const response = await api.post("/transcribe", formData);
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
      const response = await api.post("/chat", {
        question: trimmedQuestion,
        language: spokenLanguage,
        document_id: documentId,
      });
      const { answer, sources = [] } = response.data;

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

      {detectedLanguage && (
        <p className="chat-detected-language">Heard: {languageLabel(detectedLanguage)}</p>
      )}

      <form className="chat-form" onSubmit={handleSubmit}>

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

    </div>
  );
}

export default ChatBox;
