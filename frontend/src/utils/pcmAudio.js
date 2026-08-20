// PCM audio encode/decode helpers for real-time Gemini Live streaming
// (see components/VoiceChat.jsx, the only caller) - kept separate from
// that component since none of this is UI, it's just binary audio math,
// easier to reason about (and unit test) on its own.

// Gemini Live's documented input format (see backend's
// services/voice_live_service.py, INPUT_AUDIO_MIME_TYPE) - 16-bit PCM,
// mono, exactly 16 kHz. The browser's own AudioContext rarely captures
// at exactly this rate (it runs at whatever the OS/hardware default is,
// commonly 44100 or 48000) - every captured chunk is resampled down to
// this rate before being sent, rather than trusting
// `new AudioContext({sampleRate: 16000})` to be honored exactly, which
// isn't reliable enough across browsers/OSes for audio a speech model
// depends on actually being at the rate it's told.
export const GEMINI_INPUT_SAMPLE_RATE = 16000;

// Linear-interpolation resample from `inputSampleRate` down to
// `outputSampleRate`. Good enough for speech - this is the standard,
// lightweight technique browser-side Gemini Live integrations use, not
// a broadcast-quality resampler, which would be a lot more code for no
// real benefit to speech recognition accuracy.
function resample(float32Samples, inputSampleRate, outputSampleRate) {
  if (inputSampleRate === outputSampleRate) {
    return float32Samples;
  }

  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.round(float32Samples.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const sourceIndex = i * ratio;
    const indexBefore = Math.floor(sourceIndex);
    const indexAfter = Math.min(indexBefore + 1, float32Samples.length - 1);
    const weight = sourceIndex - indexBefore;
    output[i] = float32Samples[indexBefore] * (1 - weight) + float32Samples[indexAfter] * weight;
  }

  return output;
}

// Converts one captured audio chunk (Float32 samples in [-1, 1], at
// whatever rate the AudioContext actually captured at) into the exact
// bytes Gemini Live expects: 16-bit signed PCM, little-endian, mono, at
// GEMINI_INPUT_SAMPLE_RATE. Returns an ArrayBuffer, ready to send
// straight over the WebSocket as a binary frame.
export function encodePcm16(float32Samples, inputSampleRate) {
  const resampled = resample(float32Samples, inputSampleRate, GEMINI_INPUT_SAMPLE_RATE);
  const buffer = new ArrayBuffer(resampled.length * 2);
  const view = new DataView(buffer);

  for (let i = 0; i < resampled.length; i++) {
    // Clamp to [-1, 1] before scaling - a mic input can peak slightly
    // above 1.0/below -1.0, which would otherwise wrap around instead
    // of clipping once cast to a 16-bit integer.
    const sample = Math.max(-1, Math.min(1, resampled[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(i * 2, int16, true); // true = little-endian
  }

  return buffer;
}

// Converts raw 16-bit PCM bytes (as received from Gemini) into an
// AudioBuffer ready to play. `sampleRate` is tagged onto the buffer
// itself, not the AudioContext - the context resamples automatically
// during playback if the two differ, so this never needs to match
// audioContext.sampleRate.
export function decodePcm16ToAudioBuffer(audioContext, arrayBuffer, sampleRate) {
  const view = new DataView(arrayBuffer);
  const sampleCount = Math.floor(arrayBuffer.byteLength / 2);
  const audioBuffer = audioContext.createBuffer(1, Math.max(sampleCount, 1), sampleRate);
  const channelData = audioBuffer.getChannelData(0);

  for (let i = 0; i < sampleCount; i++) {
    channelData[i] = view.getInt16(i * 2, true) / 0x8000;
  }

  return audioBuffer;
}
