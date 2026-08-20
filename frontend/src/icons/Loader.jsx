import IconBase from "./IconBase";

// A spinner arc, not a filled circle - callers add the `icon-spin` CSS
// class (see styles/Chat.css) to actually rotate it. Kept as a plain,
// non-animating SVG by default so it's also usable as a static "loading"
// glyph wherever spinning wouldn't make sense.
function Loader(props) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="10" opacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" />
    </IconBase>
  );
}

export default Loader;
