import { useState } from "react";
import "../styles/Chat.css";
import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";

function Chat() {
  // { documentId, filename } | null - lifted here since both Sidebar
  // (upload/remove) and ChatBox (sending document_id with each question)
  // need it. Named pdfDocument, not "document", to avoid shadowing the
  // global DOM `document` object.
  const [pdfDocument, setPdfDocument] = useState(null);

  return (
    <div>

      <Navbar />

      <div className="chat-layout">

        <Sidebar document={pdfDocument} onDocumentChange={setPdfDocument} />

        <div className="chat-container">

          <h1>Chat</h1>

          <p>
            Ask a general knowledge question, ask about the connected database,
            or upload a PDF to ask about that too.
          </p>

          <ChatBox documentId={pdfDocument?.documentId ?? null} />

        </div>

      </div>

    </div>
  );
}

export default Chat;
