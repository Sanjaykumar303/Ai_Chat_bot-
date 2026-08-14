# Voice accuracy benchmark

```bash
cd backend
python scripts/benchmark_voice_accuracy.py scripts/samples/manifest.json
```

Reports Word Error Rate (WER) for each sample, plus an average, by
running the real `transcribe_audio()` the `/transcribe` endpoint uses -
not a simulated path.

## `scripts/samples/` is a demo, not real coverage

The three samples committed here are Windows SAPI text-to-speech,
English only - clean, artificial audio with no accent, background
noise, or code-mixing. They prove the *tool* works, not that the
*transcription pipeline* is accurate for what this app actually needs
to handle well: Tamil-English (and Kannada-English, Bengali-English)
code-mixed speech from real people.

To build a manifest that actually tells you something:

1. Record real spoken questions - yourself or someone else, on a phone
   or laptop mic, in a normal (not silent-studio) environment. Include:
   - Plain English questions.
   - Tamil-English (or Kannada-English, Bengali-English) code-mixed
     questions, spoken the way someone actually would, not read
     stiffly.
   - A few of the business terms this app cares about (profit, revenue,
     expense, voucher, ledger) - these are exactly what
     `services/transcription.py`'s `TRANSCRIBE_PROMPT` and `HOTWORDS`
     were tuned against, so they're worth verifying stayed fixed.
2. Save each as a `.wav`/`.webm`/`.mp3`/anything `av` (PyAV) can decode -
   whatever your recording tool produces is fine, the benchmark converts
   it the same way the real upload path does.
3. Write down the *exact* correct transcript for each - what you
   actually said, not a cleaned-up version.
4. Add each to a manifest (JSON array, same shape as
   `samples/manifest.json`):
   ```json
   [
     {"audio": "my_sample.wav", "expected": "exact correct transcript"},
     {"audio": "tamil_sample.wav", "expected": "...", "language_hint": "ta"}
   ]
   ```
   `language_hint` is optional - omit it to match how most real usage
   leaves the frontend's language dropdown on Auto-detect.
5. Run the script against your manifest.

15-20 samples covering the mix above gives a real, if small, accuracy
signal. A handful of clean English clips (like the demo set) does not.
