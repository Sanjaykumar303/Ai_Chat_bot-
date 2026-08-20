import IconBase from "./IconBase";

// Not in the originally requested icon set - added for the user-message
// "Share prompt" action (see ChatBox.jsx's chat-message-actions), which
// nothing else in this folder covers. Standard three-node "share" glyph,
// same feather-style stroke shape as every other icon here.
function Share(props) {
  return (
    <IconBase {...props}>
      <circle cx="18" cy="5" r="3" />
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
    </IconBase>
  );
}

export default Share;
