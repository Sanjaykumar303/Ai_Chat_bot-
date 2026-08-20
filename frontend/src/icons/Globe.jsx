import IconBase from "./IconBase";

// Marks a web source under an answer, alongside the existing File
// (uploaded document) and Database source badges - added when the
// general-knowledge path started verifying named entities against a live
// web search and reporting the pages it used (see backend
// services/entity_resolution.py).
function Globe(props) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </IconBase>
  );
}

export default Globe;
