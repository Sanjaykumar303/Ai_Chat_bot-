import { useEffect, useRef, useState } from "react";
import { voiceLiveWebSocketUrl } from "../services/api";
import { encodePcm16, decodePcm16ToAudioBuffer } from "../utils/pcmAudio";
import { createThinkingGate } from "../utils/voiceThinkingGate";
import { createTurnDetector } from "../utils/turnDetector";
import { Close, Loader as LoaderIcon, Mic, Phone, Speaker, Trash, Waveform } from "../icons";

// Gemini's own documented default output rate - overwritten the instant
// the backend's "ready" event states the real one (see
// backend/services/voice_live_service.py), so this is only ever used
// for the handful of milliseconds before that first message arrives.
const DEFAULT_OUTPUT_SAMPLE_RATE = 24000;

// ScriptProcessorNode's standard real-time-streaming buffer size - big
// enough to keep per-chunk overhead low, small enough to keep latency
// reasonable (roughly 100-250ms of audio per chunk, depending on the
// browser's native sample rate). ScriptProcessorNode is deprecated in
// favor of AudioWorklet, but is still supported everywhere and needs no
// separate worklet module file to load - the simpler, still-fully-
// functional choice for a self-contained feature like this one.
const CAPTURE_BUFFER_SIZE = 4096;

// Client-side end-of-turn detection: the backend's Live model does not
// reliably start responding from silence alone (confirmed live - see
// backend/services/gemini_client.py's LIVE_MODEL_NAME comment), so this
// component has to tell it explicitly when the user has stopped
// talking. RMS level of raw mic samples above this counts as "the user
// is actually speaking" (calibrated well above typical room-noise
// floor, which sits close to 0 for a live mic with noise suppression
// on); SILENCE_DURATION_MS of continuous audio below it, after speech
// was confirmed, is treated as "done talking" and fires exactly one
// {"type":"activity_end"} message (see utils/turnDetector.js, wired up
// via the turnDetector below and driven from the onaudioprocess handler -
// this same detector also now drives {"type":"activity_start"} the
// moment speech is confirmed, including during "speaking" for a genuine
// barge-in - see backend/services/voice_live_service.py's _LIVE_CONFIG
// for why manual signaling replaced Gemini's own server-side detection).
//
// Both loosened from an earlier 0.02/700ms after real reports of speech
// being misheard/misunderstood - a quieter opening word never crossing
// the old, higher RMS threshold meant speech was never confirmed at all
// (end_turn then never firing until the user spoke louder), and a short
// natural pause (a breath, "and... revenue") past the old 700ms window
// was enough to end the turn mid-sentence, sending Gemini a truncated
// question it could only guess the rest of. Still short enough to feel
// responsive once the user really has stopped talking. See
// MIN_SPEECH_DURATION_MS below for the third piece of this: how long
// confirming that speech actually started takes.
const SPEECH_RMS_THRESHOLD = 0.015;
const SILENCE_DURATION_MS = 900;

// A single captured audio block (~85-100ms) crossing SPEECH_RMS_THRESHOLD
// is NOT enough by itself to count as "the user started talking" - see
// utils/turnDetector.js's own module comment for the real, reported bug
// this fixes (background noise/echo bleed right after Gemini finishes
// speaking was enough to trigger end_turn with nothing actually said).
// Requiring this much CONTINUOUS above-threshold audio before speech is
// confirmed filters that out without raising SPEECH_RMS_THRESHOLD itself
// (which would reintroduce the missed-quiet-speech bug that threshold
// was already lowered to fix).
const MIN_SPEECH_DURATION_MS = 250;

// A safety net for a turn that gets stuck in "Thinking..." - armed the
// moment status enters "thinking" (both the initial end_turn and the
// later re-entry after a database question's spoken acknowledgment
// finishes, see playAudioChunk's onended below) and disarmed the moment
// it leaves (new audio starts playing, or turn_complete/interrupted
// ends it). If it DOES fire, something is genuinely stuck (a dropped
// turn_complete, a hung tool call) rather than just slow, so it fails
// the turn outright into the existing error screen (Retry already
// tears down and reconnects cleanly) instead of leaving the user
// staring at "Thinking..." with no way out but closing the whole
// screen.
//
// 35s, not the first-guess 25s: a direct live measurement of a plain
// database question through POST /chat (the exact same SQL-generation +
// DB-query + answer-generation pipeline the voice DB tool call reuses,
// see backend/services/voice_live_service.py's own module docstring)
// came back at ~18s on its own - already close to 25s before adding
// this same round trip's OWN extra overhead from running nested inside
// an already-open Live session (the spoken acknowledgment's own
// generation, the tool-call dispatch, ordinary network jitter). 25s was
// observed live cutting off a real, still-working answer, not catching
// a genuinely stuck one - 35s gives real headroom above the measured
// worst case while still bounding a truly hung turn instead of leaving
// it stuck forever.
const THINKING_TIMEOUT_MS = 35000;

// Real, reported bug this fixes: Listening -> Thinking -> Speaking ->
// Thinking -> Listening, with that second "Thinking" a stray flash
// AFTER Gemini had already finished speaking. Root cause: playAudioChunk's
// onended used to flip straight to "thinking" the instant the local
// playback queue emptied, but that doesn't distinguish a genuine
// mid-turn pause (a database question's acknowledgment finishing before
// the real answer starts, several seconds later - the actual case this
// exists for) from the reply simply being completely over and playback
// catching up with delivery before turn_complete's own separate, tiny
// control message has arrived - a real race, not a hypothetical one.
// createThinkingGate (utils/voiceThinkingGate.js) gives turn_complete/a
// new chunk this long to arrive first before it's treated as a genuine
// gap - generous for a control-message race, negligible against a real
// multi-second wait either way. See that module's own test file for a
// deterministic reproduction of the exact race this closes.
const QUEUE_EMPTY_GRACE_MS = 400;

// How fast the orb's displayed size chases the actual audio-level
// target each animation frame (see the rAF loop below) - low enough to
// look like a smooth, organic pulse rather than jittering with every
// individual sample chunk, high enough to still read as "reacting to
// this specific sound" rather than lagging behind it.
const ORB_SMOOTHING = 0.18;
// How much the level target decays per frame when no new audio has
// arrived since the last one - what makes the orb settle back down
// between words instead of just holding at the last loud moment.
const ORB_DECAY = 0.94;
// Baseline idle pulse size (as a fraction of the orb's max growth) per
// status, so the orb never looks completely static even in silence -
// "connecting" breathes slower/dimmer than an actively listening mic.
const ORB_IDLE_LEVEL = { connecting: 0.12, listening: 0.18, thinking: 0.22, speaking: 0.18, error: 0 };

function microphoneErrorMessage(err) {
  if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
    return "No microphone was found. Please connect a microphone and try again.";
  }
  if (err.name === "NotReadableError" || err.name === "TrackStartError") {
    return "The microphone is already in use by another application.";
  }
  return "Microphone access was denied. Please allow microphone access and try again.";
}

// thinkingReason (null | "database") only ever matters while
// status==="thinking" - a more specific label than generic "Thinking..."
// for the one case this app can actually distinguish today (a database
// tool call, per the backend's "tool_call_started" event - see
// handleControlMessage below). Falls back to the generic label for any
// other/unrecognized reason, so a future second tool this component
// doesn't yet know about still shows something reasonable rather than a
// wrong or missing label.
function statusLabel(status, thinkingReason) {
  if (status === "thinking" && thinkingReason === "database") {
    return "Checking the database...";
  }
  switch (status) {
    case "connecting":
      return "Connecting...";
    case "listening":
      return "Listening...";
    case "thinking":
      return "Thinking...";
    case "speaking":
      return "Speaking...";
    default:
      return "";
  }
}

// The left panel's persistent Listening/Thinking/Speaking list (distinct
// from voice-screen-status above, which is just the current one as
// text) - each row's own icon, rendered active when it matches `status`.
const STATE_ROWS = [
  { key: "listening", label: "Listening", icon: <Mic size={16} /> },
  { key: "thinking", label: "Thinking", icon: <LoaderIcon size={16} /> },
  { key: "speaking", label: "Speaking", icon: <Speaker size={16} /> },
];

// A short local wall-clock label ("10:30 AM") next to each transcript
// bubble - purely cosmetic, computed from the `time` each entry is
// stamped with the moment it's first created (see updateUserTranscript/
// appendAssistantTranscript below). Never sent anywhere - the backend's
// own persisted chat_memory timestamps are a separate thing entirely.
function formatClockTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function footerStatusLabel(status) {
  return status === "connecting" ? "Connecting to voice chat..." : "Voice chat is active...";
}

// The mic toggle's own label had only two states (Muted / Tap to mute) -
// both implying "your voice is being sent right now" whenever not
// muted, which is only true for "listening"/"speaking". It is NOT true
// for "thinking": the onaudioprocess handler below deliberately stops
// forwarding audio the instant a turn ends (see its own comment on why -
// nothing to barge into yet, and stray sound risks being misread as the
// next turn's input) - a real, if easy to miss, difference the label
// used to paper over. Reported as unclear; this makes the mic's actual
// behavior explicit instead of just implied by which row is highlighted
// in the state list above.
function micStatusLabel(muted, status) {
  if (muted) {
    return "Muted";
  }
  if (status === "thinking") {
    return "Not sending audio while thinking";
  }
  return "Tap to mute";
}

// Root-mean-square level of a block of Float32 mic samples, in [0, 1] -
// what actually drives the orb's size while listening (see the
// ORB_LEVEL_GAIN scaling in startMicCapture below). Cheap: one pass over
// a single ~4096-sample block already being read for encoding anyway.
function rmsLevel(float32Samples) {
  let sumOfSquares = 0;
  for (let i = 0; i < float32Samples.length; i++) {
    sumOfSquares += float32Samples[i] * float32Samples[i];
  }
  return Math.sqrt(sumOfSquares / float32Samples.length);
}

// Same idea as rmsLevel above, but reading straight from the raw 16-bit
// PCM bytes Gemini sends - drives the orb while "speaking", so it reacts
// to the actual loudness of Gemini's own voice rather than pulsing on a
// generic timer.
function rmsLevelFromPcm16(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const sampleCount = Math.floor(arrayBuffer.byteLength / 2);
  if (sampleCount === 0) {
    return 0;
  }
  let sumOfSquares = 0;
  for (let i = 0; i < sampleCount; i++) {
    const sample = view.getInt16(i * 2, true) / 0x8000;
    sumOfSquares += sample * sample;
  }
  return Math.sqrt(sumOfSquares / sampleCount);
}

// A raw RMS level maps to a fairly narrow, quiet-sounding range for
// normal speech - this scales it up so the orb's motion actually reads
// as responsive rather than barely moving, then clamps to keep a single
// loud spike from blowing the orb up past a sensible size.
function levelToOrbTarget(level) {
  return Math.min(level * 6, 1);
}

// Dedicated, full-screen live voice mode (Gemini Live API) - a
// ChatGPT-style takeover of the whole viewport while active, entered
// from its own button in the sidebar (see ChatSidebar.jsx), not
// anywhere inside ChatBox.jsx. Genuinely separate from the existing
// record-once/transcribe/fill-the-input voice INPUT feature that still
// lives in ChatBox.jsx untouched - see backend's
// services/voice_live_service.py for the full protocol this speaks.
//
// Deliberately foreground-only: there is exactly one physical
// microphone and speaker, so "keep a Live session running in the
// background" isn't a meaningful thing to build - closing this screen
// (the X, End Voice Chat, or the whole component unmounting) always
// fully closes the Gemini Live connection, never leaves it dangling.
//
// Pinned to ONE chat session for its whole lifetime: `sessionId` is
// computed once by pages/Chat.jsx's handleOpenVoiceChat (reusing the
// currently active session, or creating a fresh one if none was active -
// never re-derived or changed while this component stays mounted) and
// passed straight through to the backend over the WebSocket, and
// `onTurnSaved` is called once per settled turn (see the
// "user_turn_saved"/"assistant_turn_saved" cases in handleControlMessage
// below) so that session's own persisted message history +
// backend chat_memory both get the exact same finalized text the user
// heard/spoke - never an interim/partial one. Both props are mirrored
// into refs (sessionIdRef/onTurnSavedRef below), same pattern as
// statusRef/mutedRef already use, so neither has to be a dependency of
// the connection effect - this component is only ever mounted fresh
// each time Voice Chat opens anyway (see pages/Chat.jsx), so `sessionId`
// never actually changes during one mounted lifetime; the ref is just
// what lets the effect read the always-current `onTurnSaved` callback
// identity without tearing the WebSocket down on every parent re-render.
function VoiceChat({ sessionId, initialTranscript, onClose, onTurnSaved }) {

  const [status, setStatus] = useState("connecting"); // connecting | listening | thinking | speaking | error
  // Only meaningful while status==="thinking" (see statusLabel above) -
  // reset to null everywhere a fresh turn begins or the current one
  // ends (tryStartListening, "interrupted", "turn_complete"), same
  // lifecycle as turnActiveRef below, and set to "database" only by the
  // backend's "tool_call_started" event.
  const [thinkingReason, setThinkingReason] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [muted, setMuted] = useState(false);
  // [{ speaker: "you" | "assistant", text, interim, interrupted?, time? }] -
  // seeded once from initialTranscript (pages/Chat.jsx's
  // voiceTranscriptFor - this session's already-persisted history, typed
  // and voice both) via a lazy initializer, so reopening Voice Chat
  // mid-session shows the conversation so far instead of starting blank.
  // A LAZY initializer specifically (not a bare `useState(initialTranscript)`)
  // because this only needs to run once, on first mount - re-running it
  // on every render would be wasted work, and reading the prop again
  // after mount would be wrong anyway (this component is only ever
  // mounted fresh per Voice Chat session - see the module docstring -
  // so initialTranscript reflects "history as of open time", not
  // something that should keep resyncing against a changing prop).
  // Every NEW turn from here on is still appended locally exactly as
  // before AND separately persisted via onTurnSaved - see that prop's
  // own docstring above.
  const [transcript, setTranscript] = useState(() => initialTranscript || []);
  // Bumped by handleRetry to force the connection effect below to fully
  // tear down and re-run from scratch after a failure - see that
  // effect's dependency array.
  const [attempt, setAttempt] = useState(0);

  const wsRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const silentGainRef = useRef(null);
  const outputSampleRateRef = useRef(DEFAULT_OUTPUT_SAMPLE_RATE);
  // { sources: AudioBufferSourceNode[], nextStartTime: number } - queues
  // consecutive Gemini audio chunks back-to-back for gapless playback
  // (see playAudioChunk) and lets stopPlayback cut all of it off
  // instantly on barge-in (see handleControlMessage's "interrupted"
  // case).
  const playbackQueueRef = useRef({ sources: [], nextStartTime: 0 });
  // True from the moment THIS side decides to end the session (the
  // close button, or unmounting) - lets the WebSocket's own onclose/
  // onerror handlers tell "we did this on purpose" apart from a genuine
  // disconnect, without depending on a stale `status` closure.
  const endingRef = useRef(false);
  const mutedRef = useRef(false);
  const transcriptEndRef = useRef(null);
  // True from the moment end_turn is sent until turn_complete/interrupted
  // actually ends it - lets playAudioChunk's onended (below) tell "this
  // reply paused mid-turn" apart from "this reply is actually over".
  // Load-bearing for a database question specifically: backend/services/
  // voice_live_service.py now has Gemini speak a brief "Let me check
  // that..." acknowledgment BEFORE its (13-20s live) database tool call
  // resolves - that short filler's own audio queue empties long before
  // the real answer starts streaming, and without this flag onended
  // would read that as "done" and drop the screen back to "Listening..."
  // for the rest of the wait, silently inviting the user to talk again
  // over what is still, from Gemini's perspective, one uninterrupted
  // turn.
  const turnActiveRef = useRef(false);
  // The pending setTimeout id for THINKING_TIMEOUT_MS's watchdog (see its
  // own comment above) - null whenever the watchdog isn't currently
  // armed, i.e. whenever status isn't "thinking".
  const thinkingTimeoutRef = useRef(null);

  // Mirrors state into refs the animation/audio-capture loops below
  // read from - both run as long-lived callbacks (a rAF loop, an
  // onaudioprocess handler) set up once and never recreated, so they
  // need a way to see the LATEST value without becoming a dependency
  // that would tear down and restart the whole connection.
  const statusRef = useRef(status);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);
  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  // See this component's own docstring above - sessionId never actually
  // changes during one mounted lifetime, but is still read via a ref
  // (not a connection-effect dependency) for consistency with
  // onTurnSavedRef just below, which DOES need this treatment (a fresh
  // arrow-function identity from pages/Chat.jsx every render would
  // otherwise be a real dependency-array footgun).
  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  const onTurnSavedRef = useRef(onTurnSaved);
  useEffect(() => {
    onTurnSavedRef.current = onTurnSaved;
  }, [onTurnSaved]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [transcript]);

  // The orb's own animation: a rAF loop, independent of React's render
  // cycle, smoothly chasing a "target size" that startMicCapture/
  // playAudioChunk below update on every real audio chunk (see
  // rmsLevel/rmsLevelFromPcm16) - driving this through refs and direct
  // style writes rather than setState avoids a re-render on every one
  // of dozens of chunks per second.
  const orbRef = useRef(null);
  const orbLevelTargetRef = useRef(0);
  const orbLevelCurrentRef = useRef(0);

  useEffect(() => {
    let frameId;

    function tick() {
      const idle = ORB_IDLE_LEVEL[statusRef.current] ?? 0;
      orbLevelTargetRef.current = Math.max(orbLevelTargetRef.current * ORB_DECAY, idle);

      const current = orbLevelCurrentRef.current;
      const target = orbLevelTargetRef.current;
      const next = current + (target - current) * ORB_SMOOTHING;
      orbLevelCurrentRef.current = next;

      if (orbRef.current) {
        orbRef.current.style.transform = `scale(${1 + next * 0.4})`;
        orbRef.current.style.opacity = String(0.75 + next * 0.25);
      }

      frameId = requestAnimationFrame(tick);
    }

    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, []);

  useEffect(() => {
    let cancelled = false;
    // Set once each of these two independent prerequisites is actually
    // ready - see tryStartListening below for why "Listening..." and mic
    // capture can't start until BOTH are true, and start()'s own comment
    // for why they're now kicked off in parallel instead of one after
    // the other.
    let micReady = false;
    let geminiReady = false;

    // Flips to "listening" and starts mic capture the moment BOTH the
    // browser's own mic permission (micReady, set at the end of start()
    // below) AND the backend's Gemini Live handshake (geminiReady, set
    // by handleControlMessage's "ready" case) have resolved - whichever
    // arrives second is what actually triggers this, so the two no
    // longer have to happen in a fixed order.
    function tryStartListening(audioContext) {
      if (cancelled || !micReady || !geminiReady) {
        return;
      }
      setStatus("listening");
      setThinkingReason(null);
      resetTurnDetection();
      // turnActiveRef lives on the component instance, not this
      // effect, so a Retry (attempt bump, same instance, no full
      // remount) would otherwise carry a stale `true` over from
      // whatever turn was in flight when the connection broke.
      turnActiveRef.current = false;
      startMicCapture(audioContext);
    }

    async function start() {
      setStatus("connecting");
      setErrorMessage("");
      // Deliberately does NOT reset transcript - it's seeded once from
      // this session's real history (see the transcript state's own
      // comment above) and grows for as long as this screen stays
      // mounted, including across a Retry (attempt bump re-runs this
      // same start()) - a reconnect is not a new conversation, so it
      // should not look like one. Only the explicit Clear button
      // (handleClearTranscript) empties it.
      endingRef.current = false;

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      audioContextRef.current = audioContext;

      // Opened right away, in parallel with the mic permission prompt
      // below - not waited on afterward. The browser's own permission
      // round trip (which can take a couple of seconds even when
      // already granted, longer if the user has to actively click
      // Allow) and the backend's Gemini Live handshake (services/
      // voice_live_service.py opening its own connection to Gemini) are
      // two genuinely independent round trips with no reason to run one
      // after the other - overlapping them cuts real wall-clock off
      // "Connecting...". tryStartListening above is what makes this
      // safe: mic capture only ever starts once BOTH have actually
      // finished, whichever order they land in.
      const ws = new WebSocket(voiceLiveWebSocketUrl(sessionIdRef.current));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onmessage = (event) => {
        if (cancelled) {
          return;
        }
        if (typeof event.data === "string") {
          handleControlMessage(JSON.parse(event.data), audioContext);
          return;
        }
        // A binary frame is always one chunk of Gemini's spoken reply -
        // see the protocol described in services/voice_live_service.py.
        playAudioChunk(audioContext, event.data);
      };

      ws.onerror = () => {
        if (!cancelled && !endingRef.current) {
          setStatus("error");
          setErrorMessage("Could not connect to voice chat. Please check that the backend is running.");
        }
      };

      ws.onclose = (event) => {
        if (cancelled || endingRef.current) {
          return;
        }
        // A close we didn't initiate ourselves (code 1000 is always our
        // own clean stop(), below) means the connection dropped out
        // from under us - worth telling the user, not just going quiet.
        if (event.code !== 1000) {
          setStatus("error");
          setErrorMessage("Voice chat disconnected.");
        }
        stopMicCapture();
      };

      // Requested in parallel with the WebSocket above, not before it -
      // see that block's own comment. A denial/failure here still ends
      // the whole attempt (there's no voice chat without a microphone),
      // closing the socket that was opened above rather than leaving it
      // dangling.
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch (err) {
        if (!cancelled) {
          setStatus("error");
          setErrorMessage(microphoneErrorMessage(err));
        }
        try {
          ws.close();
        } catch {
          // Already closed/never opened - nothing to do.
        }
        return;
      }

      if (cancelled) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      micStreamRef.current = stream;
      micReady = true;
      tryStartListening(audioContext);
    }

    // Parses one JSON control message from the backend - see
    // services/voice_live_service.py's module docstring for the full
    // event list this switches on.
    function handleControlMessage(payload, audioContext) {
      switch (payload.type) {
        case "ready":
          outputSampleRateRef.current = payload.sampleRate || DEFAULT_OUTPUT_SAMPLE_RATE;
          geminiReady = true;
          tryStartListening(audioContext);
          break;

        case "input_transcript":
          updateUserTranscript(payload.text, payload.final);
          break;

        case "output_transcript":
          appendAssistantTranscript(payload.text);
          break;

        // Purely additive - sent by the backend alongside (never instead
        // of) the existing input_transcript/output_transcript events
        // above, once a turn's exact final text is settled and persisted
        // (see backend/services/voice_live_service.py's
        // _VoiceTurnRecorder). Reported to pages/Chat.jsx so it can
        // append the identical text to this voice conversation's pinned
        // chat session - never the interim/partial text still visible in
        // `transcript` above, which stays purely a live, on-screen-only
        // caption exactly as before.
        case "user_turn_saved":
          onTurnSavedRef.current?.({ sender: "user", text: payload.text });
          break;

        case "assistant_turn_saved":
          onTurnSavedRef.current?.({ sender: "ai", text: payload.text });
          break;

        case "interrupted":
          // Barge-in: the user started talking while Gemini's audio was
          // still playing. Gemini has already abandoned that reply, so
          // anything still queued locally has to stop immediately too -
          // letting it keep playing would mean hearing a reply Gemini
          // itself no longer considers current.
          stopPlayback();
          disarmThinkingTimeout();
          setStatus("listening");
          setThinkingReason(null);
          markLastAssistantInterrupted();
          resetTurnDetection();
          turnActiveRef.current = false;
          break;

        case "turn_complete":
          finalizeAssistantTranscript();
          resetTurnDetection();
          disarmThinkingTimeout();
          // Explicit, not just left to playAudioChunk's onended/
          // thinkingGate below - guarantees the screen returns to
          // "Listening..." the moment Gemini is actually done, whether
          // this arrives before local playback has even caught up (the
          // common case) or, more rarely, a beat after it already has
          // (disarmThinkingTimeout above cancels thinkingGate's pending
          // check either way, so it can't still flip to "Thinking..."
          // for a reply that just finished - see QUEUE_EMPTY_GRACE_MS's
          // own comment).
          turnActiveRef.current = false;
          setStatus("listening");
          setThinkingReason(null);
          break;

        // See statusLabel's own comment above and backend/services/
        // voice_live_service.py's matching protocol entry - arrives
        // shortly before the slow (13-20s live) DB round trip actually
        // starts, so "Checking the database..." can replace generic
        // "Thinking..." for the rest of that wait. Only ever changes
        // the LABEL, never `status` itself - whatever already decided
        // this is "thinking" (the turn detector's onEndTurn, or
        // playAudioChunk's thinkingGate re-entry after a spoken
        // acknowledgment) still owns that.
        case "tool_call_started":
          setThinkingReason(payload.name === "query_business_database" ? "database" : null);
          break;

        case "error":
          disarmThinkingTimeout();
          setStatus("error");
          setErrorMessage(payload.message || "Voice chat failed. Please try again.");
          break;

        default:
          break;
      }
    }

    // Clears end-of-turn detection state - called whenever a turn just
    // ended (turn_complete/interrupted) or a fresh one is starting
    // (mic capture start), so a stale "speech already seen" flag from
    // the previous turn can't fire end_turn instantly on the next one.
    function resetTurnDetection() {
      turnDetector.reset();
    }

    // See THINKING_TIMEOUT_MS's own comment above for why this exists
    // and when it needs re-arming vs. disarming. Fires the same "give
    // up cleanly" sequence a real WebSocket failure already goes
    // through (stop the mic, close the socket, show the error screen)
    // rather than just flipping local state, so a reply that eventually
    // does arrive after the timeout can't silently resurrect the UI out
    // from under an error the user has already seen.
    function armThinkingTimeout() {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = setTimeout(() => {
        if (cancelled) {
          return;
        }
        endingRef.current = true;
        stopMicCapture();
        try {
          wsRef.current?.close(1000, "timed out");
        } catch {
          // Already closed/never opened - nothing to do.
        }
        setStatus("error");
        setErrorMessage("Voice chat is taking too long to respond. Please try again.");
      }, THINKING_TIMEOUT_MS);
    }

    // See QUEUE_EMPTY_GRACE_MS's own comment above - playAudioChunk's
    // onended calls thinkingGate.onQueueEmpty() instead of flipping to
    // "thinking" directly, so a reply that's actually already finished
    // (turn_complete just hasn't arrived yet) never shows a stray
    // "Thinking..." flash.
    const thinkingGate = createThinkingGate({
      graceMs: QUEUE_EMPTY_GRACE_MS,
      onEnterThinking: () => {
        if (cancelled) {
          return;
        }
        setStatus("thinking");
        armThinkingTimeout();
      },
    });

    // Disarms BOTH the real "give up" watchdog above AND any pending
    // thinkingGate check - called from every place a turn's "thinking"
    // phase can end or never really begin (new audio starts playing,
    // turn_complete/interrupted/error, or the whole session tearing
    // down), so a stale timer from either one can't fire later.
    function disarmThinkingTimeout() {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
      thinkingGate.cancel();
    }

    // See MIN_SPEECH_DURATION_MS's own comment above and utils/
    // turnDetector.js for the real, reported bug this fixes - fires
    // end_turn (and enters "Thinking...") only once continuous speech has
    // actually been confirmed and then genuinely paused, never for a bare
    // noise/echo blip followed by silence.
    const turnDetector = createTurnDetector({
      speechRmsThreshold: SPEECH_RMS_THRESHOLD,
      minSpeechDurationMs: MIN_SPEECH_DURATION_MS,
      silenceDurationMs: SILENCE_DURATION_MS,
      // Real local speech was just confirmed (see MIN_SPEECH_DURATION_MS's
      // own comment - this never fires for a mere noise/echo blip), so
      // show SOMETHING in the transcript right away rather than leaving
      // it blank until Gemini's own interim ASR result completes its
      // network round trip. updateUserTranscript's own "replaces the
      // last interim entry" behavior means the very first real
      // input_transcript event (interim or final) for this utterance
      // overwrites this placeholder with the real text automatically -
      // no separate cleanup needed here.
      // Also tells the backend real speech has started (relayed as
      // Gemini Live's own activity_start - see backend/services/
      // voice_live_service.py's _LIVE_CONFIG for why this app now drives
      // that manually instead of leaving it to Gemini's own server-side
      // detector). Firing during "speaking" (not just "listening") is
      // what makes this a genuine, deliberate barge-in signal, not just
      // a transcript nicety - Google's Live API defaults to activity_start
      // interrupting an in-progress reply, so this is what actually
      // stops Gemini talking now, the same role Gemini's own detector
      // used to play, just driven by this app's already-hardened
      // detector instead.
      onSpeechStart: () => {
        updateUserTranscript("...", false);
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "activity_start" }));
        }
      },
      onEndTurn: () => {
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "activity_end" }));
        }
        // Without this, the screen keeps showing "Listening..." for
        // however long Gemini takes to start replying - unnoticeable for
        // a general question (well under a second), but a real, observed
        // problem for a database question (the DB tool-call round trip -
        // see backend/services/voice_live_service.py's module docstring -
        // has been measured at 13-20s live), where it reads as the app
        // having silently frozen right after the user finished talking.
        // playAudioChunk below moves status to "speaking" the moment a
        // real reply actually starts arriving, whether that's near-
        // instant or not.
        setStatus("thinking");
        turnActiveRef.current = true;
        armThinkingTimeout();
      },
    });

    function startMicCapture(audioContext) {
      const stream = micStreamRef.current;
      if (!stream) {
        return;
      }

      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(CAPTURE_BUFFER_SIZE, 1, 1);
      // ScriptProcessorNode only fires onaudioprocess while connected to
      // a destination in some browsers (Safari in particular) - routing
      // through a silent (gain 0) node keeps the graph "live" without
      // the user hearing their own mic played back at them.
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        const isListening = statusRef.current === "listening";

        if (isListening) {
          orbLevelTargetRef.current = Math.max(orbLevelTargetRef.current, levelToOrbTarget(rmsLevel(input)));
        }

        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN || mutedRef.current) {
          return;
        }

        // Streamed while "listening" (the normal case) and "speaking" -
        // kept deliberately during "speaking" so a genuine barge-in has
        // real audio to detect it from (see turnDetector.processSample
        // below - it now runs during "speaking" too, not just
        // "listening", for exactly this reason - see MIN_SPEECH_DURATION_MS's
        // own comment and _LIVE_CONFIG's matching comment in
        // backend/services/voice_live_service.py for why this app's own
        // detector now owns that decision instead of Gemini's server-side
        // one).
        //
        // NOT streamed while "thinking": the user's turn already ended
        // (the activity_end signal below already told Gemini so) and,
        // unlike "speaking", there's no Gemini audio in progress yet to
        // barge into - continuing to capture/send audio here would just
        // let the mic pick up stray sound (or the user thinking out loud)
        // and have it misread as new input for a turn that hasn't even
        // been answered yet, which can measure a real 13-20s for a
        // database question (see gemini_client.py's LIVE_MODEL_NAME
        // comment) - long enough for this to matter in practice.
        if (statusRef.current === "thinking") {
          return;
        }

        ws.send(encodePcm16(input, audioContext.sampleRate));

        // Turn/barge-in detection: reached only when status is
        // "listening" or "speaking" (every other status either hasn't
        // started mic capture yet, or already returned above) - see
        // turnDetector's own declaration above for why this is needed at
        // all and what counts as a completed turn or a confirmed
        // barge-in.
        turnDetector.processSample(rmsLevel(input), performance.now());
      };

      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioContext.destination);

      sourceNodeRef.current = source;
      processorNodeRef.current = processor;
      silentGainRef.current = silentGain;
    }

    function playAudioChunk(audioContext, arrayBuffer) {
      orbLevelTargetRef.current = Math.max(orbLevelTargetRef.current, levelToOrbTarget(rmsLevelFromPcm16(arrayBuffer)));

      const audioBuffer = decodePcm16ToAudioBuffer(audioContext, arrayBuffer, outputSampleRateRef.current);
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);

      const queue = playbackQueueRef.current;
      // Schedules chunks back-to-back rather than each starting
      // immediately - two chunks arriving close together would
      // otherwise overlap/garble instead of playing in order.
      const startTime = Math.max(queue.nextStartTime, audioContext.currentTime);
      source.start(startTime);
      queue.nextStartTime = startTime + audioBuffer.duration;
      queue.sources.push(source);

      // Real audio is actively arriving, so whatever was armed while
      // waiting for it no longer applies - onended below re-arms it if
      // this chunk turns out to be a database question's acknowledgment
      // rather than the real answer (see its own comment).
      disarmThinkingTimeout();
      setStatus("speaking");

      source.onended = () => {
        queue.sources = queue.sources.filter((existing) => existing !== source);
        if (queue.sources.length === 0 && !endingRef.current) {
          // Does NOT decide "thinking" vs "listening" instantly - see
          // QUEUE_EMPTY_GRACE_MS's own comment above for why an instant
          // decision right here was the actual reported bug (a stray
          // Listening -> Thinking flash after a reply that had already
          // finished). turnActiveRef distinguishes a genuine mid-turn
          // pause (a database question's acknowledgment finishing
          // before the real answer starts, seconds later) from the
          // reply simply being over - turn_complete/interrupted's own
          // handlers above already set status to "listening" in the
          // latter case, so thinkingGate correctly does nothing once its
          // grace period confirms that's what happened.
          thinkingGate.onQueueEmpty(() => turnActiveRef.current);
        }
      };
    }

    function stopPlayback() {
      const queue = playbackQueueRef.current;
      queue.sources.forEach((source) => {
        source.onended = null;
        try {
          source.stop();
        } catch {
          // Already finished playing on its own - nothing to stop.
        }
      });
      queue.sources = [];
      queue.nextStartTime = audioContextRef.current ? audioContextRef.current.currentTime : 0;
    }

    function stopMicCapture() {
      processorNodeRef.current?.disconnect();
      processorNodeRef.current = null;
      sourceNodeRef.current?.disconnect();
      sourceNodeRef.current = null;
      silentGainRef.current?.disconnect();
      silentGainRef.current = null;
    }

    // A live caption: each interim update REPLACES the last one (it's
    // the model's evolving best guess for the whole utterance, not a
    // chunk to append), and final=true settles it in place.
    function updateUserTranscript(text, final) {
      setTranscript((previous) => {
        const last = previous[previous.length - 1];
        if (last && last.speaker === "you" && last.interim) {
          const updated = [...previous];
          // Spreads `...last` (not a fresh literal) so `time` - stamped
          // once below, when this entry is first created - survives
          // every interim update instead of being overwritten each time.
          updated[updated.length - 1] = { ...last, text, interim: !final };
          return updated;
        }
        return [...previous, { speaker: "you", text, interim: !final, time: Date.now() }];
      });
    }

    // Gemini's spoken reply streams as successive chunks of its own
    // transcript - each one is appended to the current turn, not a
    // replacement, until turn_complete settles it (finalizeAssistantTranscript).
    function appendAssistantTranscript(textChunk) {
      setTranscript((previous) => {
        const last = previous[previous.length - 1];
        if (last && last.speaker === "assistant" && last.interim) {
          const updated = [...previous];
          updated[updated.length - 1] = { ...last, text: last.text + textChunk };
          return updated;
        }
        return [...previous, { speaker: "assistant", text: textChunk, interim: true, time: Date.now() }];
      });
    }

    function finalizeAssistantTranscript() {
      setTranscript((previous) => {
        // Same "search backward, don't assume the last array entry"
        // reasoning as markLastAssistantInterrupted below - now that
        // turnDetector runs during "speaking" too, a genuine (if
        // narrow) race is possible: the user's own onSpeechStart fires
        // and pushes a "..." placeholder in the moment BEFORE an
        // already-in-flight turn_complete for a reply that was ending
        // anyway is actually processed here.
        const index = previous.findLastIndex((entry) => entry.speaker === "assistant" && entry.interim);
        if (index === -1) {
          return previous;
        }
        const updated = [...previous];
        updated[index] = { ...updated[index], interim: false };
        return updated;
      });
    }

    function markLastAssistantInterrupted() {
      setTranscript((previous) => {
        // Searches backward for the last ASSISTANT entry specifically,
        // not just literally the array's last element - the two used to
        // always be the same thing (the server's "interrupted" event was
        // the very first local sign of a barge-in), but now that
        // turnDetector's onSpeechStart fires this same instant a barge-in
        // is confirmed (see its own comment above), the user's own "..."
        // placeholder can already be the last entry by the time this
        // runs, one event later, over the network round trip to Gemini
        // and back. Searching backward finds the actual interrupted
        // reply regardless of what's been pushed after it since.
        const index = previous.findLastIndex((entry) => entry.speaker === "assistant");
        if (index === -1) {
          return previous;
        }
        const updated = [...previous];
        updated[index] = { ...updated[index], interim: false, interrupted: true };
        return updated;
      });
    }

    start();

    return () => {
      cancelled = true;
      endingRef.current = true;
      disarmThinkingTimeout();

      try {
        wsRef.current?.close(1000, "ended");
      } catch {
        // Already closed/never opened - nothing to do.
      }
      wsRef.current = null;

      processorNodeRef.current?.disconnect();
      sourceNodeRef.current?.disconnect();
      silentGainRef.current?.disconnect();

      playbackQueueRef.current.sources.forEach((source) => {
        source.onended = null;
        try {
          source.stop();
        } catch {
          // Already finished playing on its own - nothing to stop.
        }
      });
      playbackQueueRef.current = { sources: [], nextStartTime: 0 };

      audioContextRef.current?.close().catch(() => {});
      audioContextRef.current = null;

      micStreamRef.current?.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;

      orbLevelTargetRef.current = 0;
    };
    // `attempt` is the only dependency - bumping it (see handleRetry) is
    // what triggers a full teardown/reconnect after a failure. onClose
    // is used only by the JSX outside this effect, never inside it, so
    // it's correctly not a dependency here.
  }, [attempt]);

  function handleRetry() {
    setAttempt((previous) => previous + 1);
  }

  function toggleMuted() {
    setMuted((previous) => !previous);
  }

  // Clears only the on-screen log (setTranscript([]) - the same local
  // state this whole redesign renders from), never anything persisted:
  // every turn already saved to this session's chat memory (the
  // "user_turn_saved"/"assistant_turn_saved" cases above) has already
  // happened by the time a turn appears here, so clearing the visible
  // log can't un-save it - there's still exactly one conversation/session
  // system, this button just stops showing its recent local caption.
  function handleClearTranscript() {
    setTranscript([]);
  }

  const canMute = status === "listening" || status === "thinking" || status === "speaking";

  return (
    <div className="voice-screen">

      {status === "error" ? (
        <div className="voice-screen-center">
          <button type="button" className="voice-screen-close voice-screen-close-floating" onClick={onClose} aria-label="Close voice chat">
            <Close size={18} />
          </button>
          <div className="voice-screen-error">
            <span className="voice-screen-error-icon">
              <Waveform size={22} />
            </span>
            <p className="voice-screen-error-message">{errorMessage}</p>
            <button type="button" className="voice-screen-retry" onClick={handleRetry}>
              Retry
            </button>
          </div>
        </div>
      ) : (
        <div className="voice-layout">

          {/* LEFT: the orb + current state + mic status + call controls -
              everything that's about THIS moment, nothing scrollable. */}
          <div className="voice-panel voice-panel-left">

            <div className="voice-panel-header">
              <Waveform size={18} />
              <span>Live Voice Chat</span>
            </div>

            <div className="voice-panel-left-body">

              <div className="voice-orb-wrapper">
                <div className={`voice-orb voice-orb-${status}`} ref={orbRef} />
                {status === "connecting" || status === "thinking" ? (
                  <LoaderIcon size={28} className="icon-spin voice-orb-spinner" />
                ) : (
                  <span className="voice-orb-ring" aria-hidden="true" />
                )}
              </div>

              <p className={`voice-screen-status voice-screen-status-${status}`}>{statusLabel(status, thinkingReason)}</p>

              {(status === "listening" || status === "speaking") && (
                <div className="voice-live-bars" aria-hidden="true">
                  <span /><span /><span /><span /><span />
                </div>
              )}

              <div className="voice-state-list">
                {STATE_ROWS.map((row) => (
                  <div key={row.key} className={`voice-state-row ${status === row.key ? "active" : ""}`}>
                    <span className="voice-state-row-icon">{row.icon}</span>
                    <span className="voice-state-row-label">{row.label}</span>
                    <span className="voice-state-row-dot" />
                  </div>
                ))}
              </div>

            </div>

            <div className="voice-screen-bottom">
              <button
                type="button"
                className={`voice-mic-toggle ${muted ? "muted" : ""}`}
                onClick={toggleMuted}
                disabled={!canMute}
                aria-label={muted ? "Unmute your microphone" : "Mute your microphone"}
                title={muted ? "Unmute your microphone" : "Mute your microphone"}
              >
                <Mic size={28} />
              </button>
              <p className="voice-mic-tap-label">{micStatusLabel(muted, status)}</p>

              <button type="button" className="voice-screen-end" onClick={onClose}>
                <Phone size={16} /> End Voice Chat
              </button>
            </div>

          </div>

          {/* RIGHT: the full, scrollable conversation - every entry
              VoiceChat has ever pushed into `transcript` this session
              (see updateUserTranscript/appendAssistantTranscript above),
              not just a recent slice - the growing interim entry at the
              end (entry.interim) is what shows the current transcript/
              response as it arrives, live. */}
          <div className="voice-panel voice-panel-right">

            <div className="voice-panel-header voice-panel-header-right">
              <span className="voice-panel-header-title">Conversation</span>
              <div className="voice-panel-header-actions">
                <button
                  type="button"
                  className="voice-clear-button"
                  onClick={handleClearTranscript}
                  disabled={transcript.length === 0}
                >
                  <Trash size={14} /> Clear
                </button>
                <button type="button" className="voice-screen-close" onClick={onClose} aria-label="Close voice chat">
                  <Close size={16} />
                </button>
              </div>
            </div>

            <div className="voice-transcript-panel" role="log" aria-live="polite">
              {transcript.length === 0 && (
                <p className="voice-transcript-empty">Say something to get started...</p>
              )}
              {transcript.map((entry, index) => (
                <div key={index} className={`voice-bubble voice-bubble-${entry.speaker}`}>
                  <span className="voice-bubble-sender">{entry.speaker === "you" ? "You" : "AI"}</span>
                  <div className="voice-bubble-card">
                    <p className="voice-bubble-text">
                      {entry.text}
                      {entry.interim && <span className="voice-bubble-cursor" aria-hidden="true" />}
                    </p>
                    {entry.time && <span className="voice-bubble-time">{formatClockTime(entry.time)}</span>}
                  </div>
                  {entry.interrupted && <span className="voice-bubble-interrupted">Interrupted</span>}
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>

            <div className="voice-transcript-footer">
              <Waveform size={14} />
              <span>{footerStatusLabel(status)}</span>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}

export default VoiceChat;
