import logo from "../assets/logo.jpg";

// Rendered as its own "chat-message ai" bubble (see ChatBox.jsx) so a
// pending answer visually continues the conversation instead of
// interrupting it with plain loose text below the messages.
function Loader({ text = "Loading..." }) {
  return (
    <div className="chat-message ai chat-typing">
      <img className="chat-avatar chat-avatar-ai" src={logo} alt="" />
      <div className="chat-message-body">
        <span className="chat-sender">Assistant</span>
        <p className="chat-typing-text">
          <span className="chat-typing-dots" aria-hidden="true">
            <span></span><span></span><span></span>
          </span>
          {text}
        </p>
      </div>
    </div>
  );
}

export default Loader;
