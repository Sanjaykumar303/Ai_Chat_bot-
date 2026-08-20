import IconBase from "./IconBase";

// Not in the originally requested icon list, but required to actually
// satisfy "no emoji icons": the rename-chat button used a ✏️ emoji
// that nothing else in the requested set covers. See the assistant's
// summary for this call-out.
function Edit(props) {
  return (
    <IconBase {...props}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </IconBase>
  );
}

export default Edit;
