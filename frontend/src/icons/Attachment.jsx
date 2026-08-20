import IconBase from "./IconBase";

// Paperclip - the conventional "attach a file" symbol, used for the
// chat input's attachment-menu trigger.
function Attachment(props) {
  return (
    <IconBase {...props}>
      <path d="M21.44 11.05 12.25 20.24a5.5 5.5 0 0 1-7.78-7.78l9.19-9.19a3.5 3.5 0 0 1 4.95 4.95l-9.2 9.19a1.5 1.5 0 0 1-2.12-2.12l8.49-8.48" />
    </IconBase>
  );
}

export default Attachment;
