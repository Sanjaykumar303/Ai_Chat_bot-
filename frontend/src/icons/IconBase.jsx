// Shared inline-SVG wrapper every icon in this folder renders through -
// keeps sizing/stroke/accessibility behavior consistent in one place
// instead of repeating the same handful of SVG attributes in every
// icon file. Not itself one of the requested icons - internal plumbing.
//
// Sizing: defaults to 20px (within the 18-22px range every icon in
// this app uses) and is overridable per call site via `size`.
//
// Color: no `fill`/`stroke` color is ever hard-coded - `stroke="currentColor"`
// means every icon inherits whatever CSS `color` the surrounding
// button/element already sets, exactly like the emoji/text it replaced.
//
// Accessibility: icons render as `aria-hidden` by default, since every
// current call site sits inside a control that already has its own
// accessible name (a button's `aria-label`/`title`, or visible text
// next to the icon) - an icon announcing itself on top of that would
// be redundant, not helpful. Pass `title="..."` to the rare standalone
// icon that needs its own accessible name instead.
function IconBase({ size = 20, title, children, ...props }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden={title ? undefined : "true"}
      role={title ? "img" : undefined}
      {...props}
    >
      {title ? <title>{title}</title> : null}
      {children}
    </svg>
  );
}

export default IconBase;
