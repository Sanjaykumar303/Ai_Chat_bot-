// Deterministic regression test for the race fixed in
// components/VoiceChat.jsx: Listening -> Thinking -> Speaking ->
// Thinking -> Listening, with that second "Thinking" being a stray
// flash for a reply that had ALREADY finished (see
// voiceThinkingGate.js's own module comment for the full story).
//
// Uses Node's built-in test runner (node:test/node:assert) rather than
// adding a new devDependency - this project has no frontend test
// framework at all yet, and one pure module doesn't warrant introducing
// one. Run with: node --test src/utils/voiceThinkingGate.test.js
//
// A hand-rolled fake clock (not real setTimeout) makes every scenario
// below instant and exact - no flaky real-time waiting, and "the grace
// period elapses" is a single deterministic clock.advance() call.
import assert from "node:assert/strict";
import { test } from "node:test";

import { createThinkingGate } from "./voiceThinkingGate.js";

function createFakeClock() {
  let nextId = 1;
  let now = 0;
  const scheduled = new Map();

  function setTimeoutFn(callback, delay) {
    const id = nextId++;
    scheduled.set(id, { fireAt: now + delay, callback });
    return id;
  }

  function clearTimeoutFn(id) {
    scheduled.delete(id);
  }

  function advance(ms) {
    now += ms;
    const due = [...scheduled.entries()]
      .filter(([, entry]) => entry.fireAt <= now)
      .sort((a, b) => a[1].fireAt - b[1].fireAt);
    for (const [id, entry] of due) {
      scheduled.delete(id);
      entry.callback();
    }
  }

  return { setTimeoutFn, clearTimeoutFn, advance };
}

test("THE BUG: queue emptying does not flash Thinking if turn_complete resolves the turn within the grace period", () => {
  const clock = createFakeClock();
  let enteredThinking = false;
  let turnActive = true; // playback just caught up; turn_complete hasn't arrived yet

  const gate = createThinkingGate({
    graceMs: 400,
    onEnterThinking: () => {
      enteredThinking = true;
    },
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  // playAudioChunk's onended: the playback queue just went empty.
  gate.onQueueEmpty(() => turnActive);
  assert.equal(gate.isPending(), true, "a check should now be pending");

  // turn_complete arrives 50ms later - the real backend race this test
  // reproduces. VoiceChat.jsx's turn_complete handler does exactly this:
  // resolve the turn, then disarm (cancel) the gate.
  clock.advance(50);
  turnActive = false;
  gate.cancel();

  // The rest of what would have been the grace period elapses.
  clock.advance(1000);

  assert.equal(
    enteredThinking,
    false,
    "status must go directly to Listening (via turn_complete's own setStatus) - Thinking must never fire for this race"
  );
});

test("a turn that is still genuinely active after the grace period DOES enter Thinking (e.g. a database question's ack-then-wait gap)", () => {
  const clock = createFakeClock();
  let enteredThinking = false;
  const turnActive = true; // never resolved within this test - simulates the real multi-second DB wait

  const gate = createThinkingGate({
    graceMs: 400,
    onEnterThinking: () => {
      enteredThinking = true;
    },
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  gate.onQueueEmpty(() => turnActive);
  clock.advance(400);

  assert.equal(enteredThinking, true, "a genuinely still-in-progress turn must still show Thinking again");
});

test("a new audio chunk arriving before the grace period elapses cancels the pending check", () => {
  const clock = createFakeClock();
  let enteredThinking = false;
  const turnActive = true;

  const gate = createThinkingGate({
    graceMs: 400,
    onEnterThinking: () => {
      enteredThinking = true;
    },
    setTimeoutFn: clock.setTimeoutFn,
    clearTimeoutFn: clock.clearTimeoutFn,
  });

  gate.onQueueEmpty(() => turnActive);
  clock.advance(100);
  gate.cancel(); // playAudioChunk: another chunk started playing
  assert.equal(gate.isPending(), false);

  clock.advance(1000);

  assert.equal(enteredThinking, false, "more audio arriving means this was never a real gap");
});
