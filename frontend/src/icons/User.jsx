import IconBase from "./IconBase";

// Not in the originally requested icon list, but required to actually
// satisfy "no emoji icons": the user message avatar used a 🧑 emoji
// that nothing else in the requested set covers. See the assistant's
// summary for this call-out.
function User(props) {
  return (
    <IconBase {...props}>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </IconBase>
  );
}

export default User;
