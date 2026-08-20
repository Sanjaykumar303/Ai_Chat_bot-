// Text-to-speech (Web Speech API) - extracted out of ChatBox.jsx so it
// can be called from pages/Chat.jsx too. Chat.jsx now owns the actual
// /chat request (see its sendMessage) so autoSpeak can fire the instant
// an answer arrives, even for a session that isn't the one currently
// mounted - this module only ever touches the global
// window.speechSynthesis singleton, never any component's state
// directly, so nothing here needs to know which conversation it was
// called on behalf of.

export const SPEECH_SYNTHESIS_SUPPORTED =
  typeof window !== "undefined" && "speechSynthesis" in window;

// Tamil script (U+0B80-U+0BFF). If any of it appears in the answer, the
// answer is treated as Tamil; otherwise it's read as English. A plain
// script check, not a Gemini call, matching how this app already prefers
// deterministic, cost-free classification over an extra API call.
const TAMIL_SCRIPT = /[஀-௿]/;

export function detectLanguage(text) {
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

// Reads a piece of text aloud, in whichever language it's actually
// written in. Always stops whatever is currently playing first, whether
// that came from a manual click, Auto Speak, or replaying an older
// message - language/voice are re-detected fresh every call, never
// cached, so replay always matches that message's real language.
//
// onError, if given, is called with a user-facing message string on
// failure - the caller decides where that message actually belongs
// (e.g. a specific chat session's own error banner), since this
// function has no notion of "which conversation" itself.
export function speak(text, onError) {
  if (!SPEECH_SYNTHESIS_SUPPORTED || !text) {
    return;
  }

  const lang = detectLanguage(text);
  const voice = pickVoice(lang);

  if (lang === "ta-IN" && !voice) {
    onError?.(
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
    onError?.("Could not read the answer aloud.");
  };

  window.speechSynthesis.speak(utterance);
}
