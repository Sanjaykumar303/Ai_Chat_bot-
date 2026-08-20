// Deterministic regression test for the bug fixed in
// components/VoiceChat.jsx's end-of-turn detection: a single noisy audio
// block (background noise, a mic pop, or Gemini's own reply bleeding
// back through the mic) was enough to mark the turn as "the user started
// speaking", so ordinary silence right afterward - the user never having
// said anything - fired end_turn and flipped the UI to "Thinking..." for
// a turn nobody actually spoke (see turnDetector.js's own module comment
// for the full story).
//
// Uses Node's built-in test runner, same as voiceThinkingGate.test.js -
// this project has no frontend test framework beyond that yet. Run with:
// node --test src/utils/turnDetector.test.js
//
// `now` is just a plain incrementing counter passed straight into
// processSample() - this module never reads a clock itself, so no fake-
// timer plumbing is needed to test it deterministically.
import assert from "node:assert/strict";
import { test } from "node:test";

import { createTurnDetector } from "./turnDetector.js";

const THRESHOLD = 0.015;
const MIN_SPEECH_MS = 250;
const SILENCE_MS = 900;
const LOUD = 0.05; // well above THRESHOLD - "the user is speaking"
const QUIET = 0.001; // well below THRESHOLD - silence

function makeDetector() {
  let endTurnCount = 0;
  let speechStartCount = 0;
  const detector = createTurnDetector({
    speechRmsThreshold: THRESHOLD,
    minSpeechDurationMs: MIN_SPEECH_MS,
    silenceDurationMs: SILENCE_MS,
    onEndTurn: () => {
      endTurnCount += 1;
    },
    onSpeechStart: () => {
      speechStartCount += 1;
    },
  });
  return { detector, getEndTurnCount: () => endTurnCount, getSpeechStartCount: () => speechStartCount };
}

test("THE BUG: a single loud blip followed by silence never fires end_turn", () => {
  const { detector, getEndTurnCount } = makeDetector();

  // One block crosses the threshold (a noise blip / echo bleed) - not
  // sustained long enough to count as real speech starting.
  detector.processSample(LOUD, 0);

  // Plenty of ordinary silence follows - in the old, buggy behavior this
  // alone was enough to fire end_turn since hasSpeech had already latched.
  detector.processSample(QUIET, 100);
  detector.processSample(QUIET, 1200);
  detector.processSample(QUIET, 5000);

  assert.equal(getEndTurnCount(), 0, "a transient blip must never be treated as a completed user turn");
});

test("sustained real speech followed by a full silence gap fires end_turn exactly once", () => {
  const { detector, getEndTurnCount } = makeDetector();

  // Continuous above-threshold audio for >= MIN_SPEECH_MS confirms real
  // speech (a spoken word/syllable sustains far longer than one block).
  detector.processSample(LOUD, 0);
  detector.processSample(LOUD, 90);
  detector.processSample(LOUD, 180);
  detector.processSample(LOUD, 270); // 270ms >= MIN_SPEECH_MS: speech confirmed

  // Silence starts here; nothing happens until SILENCE_MS has elapsed.
  detector.processSample(QUIET, 360);
  assert.equal(getEndTurnCount(), 0, "must not end the turn before the silence gap is actually reached");

  detector.processSample(QUIET, 360 + SILENCE_MS);
  assert.equal(getEndTurnCount(), 1, "a genuinely completed utterance must end the turn");

  // Further silence after the turn already ended must not fire again.
  detector.processSample(QUIET, 360 + SILENCE_MS + 2000);
  assert.equal(getEndTurnCount(), 1, "end_turn must fire at most once per confirmed speech run");
});

test("two short blips separated by a dip never accumulate into confirmed speech", () => {
  const { detector, getEndTurnCount } = makeDetector();

  detector.processSample(LOUD, 0);
  detector.processSample(LOUD, 90); // 90ms - short of MIN_SPEECH_MS
  detector.processSample(QUIET, 180); // dip resets the run
  detector.processSample(LOUD, 270);
  detector.processSample(LOUD, 360); // another short, separate run

  detector.processSample(QUIET, 450);
  detector.processSample(QUIET, 450 + SILENCE_MS);

  assert.equal(getEndTurnCount(), 0, "non-continuous blips must not add up to a confirmed speech start");
});

test("a natural mid-sentence pause shorter than the silence gap does not end the turn early", () => {
  const { detector, getEndTurnCount } = makeDetector();

  detector.processSample(LOUD, 0);
  detector.processSample(LOUD, 300); // speech confirmed by here

  // A brief pause (e.g. a breath) - well under SILENCE_MS.
  detector.processSample(QUIET, 400);
  detector.processSample(QUIET, 700);

  // Speech resumes before the silence gap completed.
  detector.processSample(LOUD, 800);
  assert.equal(getEndTurnCount(), 0, "a short pause mid-sentence must not end the turn");

  // Now the user genuinely stops.
  detector.processSample(QUIET, 900);
  detector.processSample(QUIET, 900 + SILENCE_MS);
  assert.equal(getEndTurnCount(), 1, "the turn should still end once real silence follows");
});

test("pure silence with no speech ever seen never fires end_turn", () => {
  const { detector, getEndTurnCount } = makeDetector();

  detector.processSample(QUIET, 0);
  detector.processSample(QUIET, 5000);
  detector.processSample(QUIET, 20000);

  assert.equal(getEndTurnCount(), 0, "silence alone, with nothing ever spoken, must not end a turn");
});

test("reset() clears confirmed speech so silence in a fresh turn can't instantly end it", () => {
  const { detector, getEndTurnCount } = makeDetector();

  detector.processSample(LOUD, 0);
  detector.processSample(LOUD, 300); // speech confirmed

  detector.reset(); // e.g. turn_complete / interrupted / a fresh "listening" phase

  detector.processSample(QUIET, 400);
  detector.processSample(QUIET, 400 + SILENCE_MS);

  assert.equal(getEndTurnCount(), 0, "a stale confirmed-speech flag from before reset() must not carry over");
});

// --- onSpeechStart: immediate local "the user is talking" signal --------

test("onSpeechStart fires exactly once, the moment speech is actually confirmed", () => {
  const { detector, getSpeechStartCount } = makeDetector();

  detector.processSample(LOUD, 0);
  assert.equal(getSpeechStartCount(), 0, "must not fire before minSpeechDurationMs has elapsed");
  detector.processSample(LOUD, 90);
  assert.equal(getSpeechStartCount(), 0, "still short of minSpeechDurationMs");
  detector.processSample(LOUD, 270); // 270ms >= MIN_SPEECH_MS: confirmed here
  assert.equal(getSpeechStartCount(), 1, "must fire the instant speech is confirmed");

  detector.processSample(LOUD, 360); // still talking - must not fire again mid-utterance
  assert.equal(getSpeechStartCount(), 1);
});

test("onSpeechStart never fires for a transient blip that's never confirmed", () => {
  const { detector, getSpeechStartCount } = makeDetector();

  detector.processSample(LOUD, 0); // one block, then silence - never reaches MIN_SPEECH_MS
  detector.processSample(QUIET, 100);
  detector.processSample(QUIET, 5000);

  assert.equal(getSpeechStartCount(), 0, "a noise blip must never be reported as the user starting to talk");
});

test("onSpeechStart fires again for a genuinely new turn after reset()", () => {
  const { detector, getSpeechStartCount } = makeDetector();

  detector.processSample(LOUD, 0);
  detector.processSample(LOUD, 300); // first turn's speech confirmed
  assert.equal(getSpeechStartCount(), 1);

  detector.reset();

  detector.processSample(LOUD, 1000);
  detector.processSample(LOUD, 1300); // second turn's speech confirmed
  assert.equal(getSpeechStartCount(), 2, "a fresh turn must be able to report its own speech start again");
});
