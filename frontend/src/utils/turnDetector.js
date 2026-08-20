// Extracted for the same reason utils/voiceThinkingGate.js was: to make
// this state machine deterministically testable without a real
// AudioContext/mic (see turnDetector.test.js) - components/VoiceChat.jsx's
// onaudioprocess handler calls processSample() with every captured audio
// block, so the test exercises the exact same decision logic the
// component runs, not a re-implementation that could drift out of sync.
//
// The bug this exists to fix (real, reported): a SINGLE sample block
// (~85-100ms of audio) crossing speechRmsThreshold was, by itself,
// enough to mark the turn as "the user has started speaking". One stray
// block of background noise, a mic pop, or Gemini's own just-finished
// reply bleeding back through the mic (browser echo cancellation is not
// fully reliable for audio played via raw Web Audio API buffer nodes -
// see components/VoiceChat.jsx's playAudioChunk, and the matching
// server-side note in backend/services/voice_live_service.py's
// _LIVE_CONFIG) was enough to trigger it. Once that happened, ordinary
// silence immediately afterward - the user never having said anything at
// all - was read as "done talking" the moment silenceDurationMs passed,
// firing end_turn and flipping the UI to "Thinking..." for a turn nobody
// actually spoke, and (separately) making a real pause of only a couple
// hundred milliseconds enough to end a turn the user hadn't finished.
//
// Requiring minSpeechDurationMs of CONTINUOUS above-threshold audio
// before speech counts as "started" (any dip below threshold resets the
// run - see the level <= speechRmsThreshold branch below) filters out a
// transient without meaningfully delaying real speech, which sustains
// far longer than this across even a single spoken syllable. This is
// deliberately a duration fix, not an amplitude one - raising
// speechRmsThreshold instead would reintroduce the earlier, already-
// fixed bug where a quieter opening word never crossed the threshold at
// all (see VoiceChat.jsx's own comment on SPEECH_RMS_THRESHOLD).
// onSpeechStart (optional, defaults to a no-op) fires once per confirmed
// speech run, the moment minSpeechDurationMs of continuous audio above
// the threshold is reached (real speech has genuinely started) - not on
// every block, and not on a transient blip that never gets confirmed at
// all (see the module comment above for why that distinction exists).
// components/VoiceChat.jsx uses this to show an immediate "..." caption
// the instant the user starts talking, rather than waiting for Gemini's
// own (network-round-trip-bound) interim transcript to arrive - genuine
// local signal, not a substitute for the real transcript, which still
// replaces this placeholder the moment it arrives (see
// updateUserTranscript's own "replaces the last interim entry" comment).
export function createTurnDetector({
  speechRmsThreshold,
  minSpeechDurationMs,
  silenceDurationMs,
  onEndTurn,
  onSpeechStart = () => {},
}) {
  let hasSpeech = false;
  let speechStartedAt = null;
  let silenceStartedAt = null;

  return {
    // Called once per captured audio block with that block's RMS level
    // and the current timestamp (performance.now() in production, a
    // plain counter in tests - this module never reads the clock itself,
    // so no fake-timer plumbing is needed to test it). Calls onEndTurn()
    // at most once per confirmed speech run, the moment silenceDurationMs
    // of continuous silence follows it.
    processSample(level, now) {
      if (level > speechRmsThreshold) {
        silenceStartedAt = null;
        if (!hasSpeech) {
          if (speechStartedAt === null) {
            speechStartedAt = now;
          } else if (now - speechStartedAt >= minSpeechDurationMs) {
            hasSpeech = true;
            onSpeechStart();
          }
        }
        return;
      }

      // Below threshold: any dip cancels a not-yet-confirmed speech run
      // so only genuinely continuous energy counts toward
      // minSpeechDurationMs - two separate noise blips can't add up.
      speechStartedAt = null;

      if (!hasSpeech) {
        return; // pure silence with no confirmed speech yet - nothing to end
      }

      if (silenceStartedAt === null) {
        silenceStartedAt = now;
      } else if (now - silenceStartedAt >= silenceDurationMs) {
        hasSpeech = false;
        silenceStartedAt = null;
        onEndTurn();
      }
    },

    // Called whenever a turn genuinely ends/restarts for a reason other
    // than this detector's own onEndTurn (turn_complete, interrupted, a
    // fresh "listening" phase starting) - clears all state so a stale
    // "speech already seen" flag from the previous turn can't fire
    // end_turn instantly on the next one.
    reset() {
      hasSpeech = false;
      speechStartedAt = null;
      silenceStartedAt = null;
    },
  };
}
