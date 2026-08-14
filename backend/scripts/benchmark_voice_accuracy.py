"""
Measures voice-transcription accuracy against a set of known-correct
transcripts, using the exact same transcribe_audio() the real
/transcribe endpoint calls (services/transcription.py) - not a
simulated or separate code path, so the number this produces reflects
what a real user actually gets.

Usage:
    cd backend
    python scripts/benchmark_voice_accuracy.py scripts/samples/manifest.json

Manifest format (JSON array, paths resolved relative to the manifest's
own directory):
    [
      {"audio": "sample1.wav", "expected": "What is the profit today?"},
      {"audio": "sample2.wav", "expected": "...", "language_hint": "ta"}
    ]
`language_hint` is optional - omit it to match how most real usage
leaves the frontend's language dropdown on Auto-detect.

Reports Word Error Rate (WER) - the standard speech-recognition
accuracy metric: (substitutions + deletions + insertions) / words in
the expected transcript, via word-level edit distance. Computed against
the *clean* question (TRANSCRIBE_PROMPT's rewritten output - what
actually fills the user's chat input), since that's the thing accuracy
matters for, not the literal raw transcript.

This only measures what you feed it: a handful of clean English clips
tells you little about Tamil-English code-mixing or noisy real-world
audio. Build a manifest that actually covers the cases you care about.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.transcription import transcribe_audio  # noqa: E402


def _normalize(text):
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split()


def word_error_rate(expected, actual):
    """Word-level edit distance (substitutions + deletions + insertions)
    divided by the number of words in the expected transcript - the
    standard WER formula used to report speech-recognition accuracy."""

    ref = _normalize(expected)
    hyp = _normalize(actual)

    if not ref:
        return 0.0 if not hyp else 1.0

    rows, cols = len(ref) + 1, len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dist[i][j] = dist[i - 1][j - 1]
            else:
                dist[i][j] = 1 + min(
                    dist[i - 1][j],      # deletion
                    dist[i][j - 1],      # insertion
                    dist[i - 1][j - 1],  # substitution
                )

    return dist[-1][-1] / len(ref)


async def main(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent

    results = []

    for case in cases:
        audio_path = base_dir / case["audio"]
        expected = case["expected"]
        language_hint = case.get("language_hint")

        print(f"\n{case['audio']}")

        try:
            raw, question, language = await transcribe_audio(str(audio_path), language_hint)
        except Exception as error:
            print(f"  ERROR: {error}")
            results.append({"audio": case["audio"], "wer": 1.0, "failed": True})
            continue

        if not raw:
            print("  No speech detected.")
            results.append({"audio": case["audio"], "wer": 1.0, "failed": True})
            continue

        wer = word_error_rate(expected, question)
        results.append({"audio": case["audio"], "wer": wer, "failed": False})

        print(f"  expected: {expected!r}")
        print(f"  raw:      {raw!r}")
        print(f"  question: {question!r}")
        print(f"  language: {language}")
        print(f"  WER:      {wer:.1%}")

    print(f"\n{'=' * 60}")
    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        failures = sum(1 for r in results if r["failed"])
        print(f"Samples:          {len(results)}")
        print(f"Failed/no speech: {failures}")
        print(f"Average WER:      {avg_wer:.1%}")
        print(f"Approx. accuracy: {(1 - avg_wer):.1%}")
    else:
        print("No samples in manifest.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/benchmark_voice_accuracy.py path/to/manifest.json")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
