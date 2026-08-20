// Extracted specifically to make one race condition in
// components/VoiceChat.jsx deterministically testable without mocking
// AudioContext/WebSocket/React (see voiceThinkingGate.test.js) -
// VoiceChat.jsx's playAudioChunk calls straight into this, so the test
// exercises the exact same logic the component runs, not a
// re-implementation that could quietly drift out of sync with it.
//
// The bug this exists to fix: the playback queue going empty does NOT
// necessarily mean the turn is genuinely paused waiting for more audio
// (the one real case that should show "Thinking..." again - e.g. a
// database question's spoken acknowledgment finishing before the real
// answer starts, several seconds later). It can just as easily mean the
// reply is completely over and local playback has simply caught up with
// delivery before the separate, tiny turn_complete control message has
// physically arrived over the WebSocket - real, observed, and reported:
// Listening -> Thinking -> Speaking -> Thinking -> Listening, with that
// second "Thinking" being a stray flash for a reply that had already
// finished, corrected a moment later once turn_complete landed.
//
// Flipping to "Thinking..." the INSTANT the queue empties can't tell
// these two cases apart. Waiting graceMs for turn_complete (which
// resolves the turn - see VoiceChat.jsx's isTurnActive callback) or a
// new audio chunk (which calls cancel()) to arrive first means
// "Thinking..." only fires for a gap that grace period doesn't resolve
// - i.e. a turn that's still genuinely in progress, not this race.
export function createThinkingGate({
  graceMs,
  onEnterThinking,
  setTimeoutFn = setTimeout,
  clearTimeoutFn = clearTimeout,
}) {
  let pendingId = null;

  return {
    // Called when the playback queue just emptied. isTurnActive is a
    // function, not a plain boolean, because it has to be read at the
    // moment the grace period actually elapses - not a stale snapshot
    // from when the queue emptied, which is exactly the value that's
    // still racing against turn_complete at that instant.
    onQueueEmpty(isTurnActive) {
      clearTimeoutFn(pendingId);
      pendingId = setTimeoutFn(() => {
        pendingId = null;
        if (isTurnActive()) {
          onEnterThinking();
        }
      }, graceMs);
    },

    // Cancels a pending check without acting on it - called whenever
    // something has already resolved what happens next: a new audio
    // chunk started playing, or the turn ended (turn_complete/
    // interrupted/error), or the whole session is tearing down.
    cancel() {
      clearTimeoutFn(pendingId);
      pendingId = null;
    },

    // Test/inspection only - whether a check is currently pending.
    isPending() {
      return pendingId !== null;
    },
  };
}
