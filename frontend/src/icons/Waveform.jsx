import IconBase from "./IconBase";

// Marks real-time Voice Chat (components/VoiceChat.jsx) specifically -
// deliberately distinct from Mic.jsx, which starts the older
// record-once/transcribe/fill-the-input flow. A row of bars of varying
// height (a classic waveform/equalizer glyph) reads as "live audio" in
// a way a plain microphone icon, already used for that other feature,
// would not.
function Waveform(props) {
  return (
    <IconBase {...props}>
      <line x1="4" y1="10" x2="4" y2="14" />
      <line x1="8" y1="6" x2="8" y2="18" />
      <line x1="12" y1="3" x2="12" y2="21" />
      <line x1="16" y1="6" x2="16" y2="18" />
      <line x1="20" y1="10" x2="20" y2="14" />
    </IconBase>
  );
}

export default Waveform;
