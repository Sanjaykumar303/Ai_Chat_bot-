import { useRef, useState } from "react";
import api from "../services/api";

// Mirrors backend/services/pdf_service.py's PDF_MAX_BYTES (15 MB) and
// image_service.py's IMAGE_MAX_BYTES (10 MB) - purely a fast client-side
// check so an oversized file doesn't even start uploading; the backend
// enforces its own limits regardless.
const MAX_PDF_BYTES = 15 * 1024 * 1024;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
const IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

// The prop is deliberately not named "document" - that would shadow the
// global DOM `document` object for the rest of this component.
function Sidebar({ document: activeDocument, onDocumentChange }) {

  const [status, setStatus] = useState("idle"); // idle | uploading | processing | error
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    event.target.value = ""; // lets choosing the same filename again re-trigger onChange

    if (!file) {
      return;
    }

    setError("");

    const name = file.name.toLowerCase();
    const isPdf = name.endsWith(".pdf") || file.type === "application/pdf";
    const isImage = IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext)) || IMAGE_MIME_TYPES.includes(file.type);

    if (!isPdf && !isImage) {
      setStatus("error");
      setError("Only PDF, JPG, PNG, or WEBP files are supported.");
      return;
    }

    const maxBytes = isPdf ? MAX_PDF_BYTES : MAX_IMAGE_BYTES;
    if (file.size > maxBytes) {
      setStatus("error");
      setError(`File is too large (max ${maxBytes / (1024 * 1024)} MB).`);
      return;
    }

    // Replacing an existing document: drop the old temporary context
    // first, so it's never left around or mistakenly reused once the
    // new one is ready.
    if (activeDocument?.documentId) {
      try {
        await api.delete(`/documents/${activeDocument.documentId}`);
      } catch {
        // Already gone/expired - fine, proceed with the new upload regardless.
      }
      onDocumentChange(null);
    }

    setStatus("uploading");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await api.post("/documents/upload", formData, {
        // Bytes fully sent to the server means the "uploading" phase is
        // over - what's left (extraction + chunking + indexing) is
        // "processing", which has no progress events of its own.
        onUploadProgress: (progressEvent) => {
          if (progressEvent.loaded >= progressEvent.total) {
            setStatus("processing");
          }
        },
      });
      const { document_id, filename } = response.data;
      onDocumentChange({ documentId: document_id, filename });
      setStatus("idle");
    } catch (err) {
      const detail = err.response?.data?.detail;
      setStatus("error");
      setError(typeof detail === "string" ? detail : "Unable to process this file.");
    }
  }

  async function handleRemove() {
    if (!activeDocument?.documentId) {
      return;
    }

    try {
      await api.delete(`/documents/${activeDocument.documentId}`);
    } catch {
      // Already gone/expired on the backend - clear it from the UI regardless.
    }

    onDocumentChange(null);
    setStatus("idle");
    setError("");
  }

  return (
    <div className="sidebar">

      <h2 className="sidebar-title">Document</h2>

      {activeDocument ? (
        <div className="document-attached">
          <p className="document-filename">
            {IMAGE_EXTENSIONS.some((ext) => activeDocument.filename.toLowerCase().endsWith(ext)) ? "🖼️" : "📄"}{" "}
            {activeDocument.filename}
          </p>
          <p className="document-ready">✓ Ready for questions</p>
          <button type="button" className="document-remove" onClick={handleRemove}>
            Remove
          </button>
        </div>
      ) : (
        <div className="document-empty">
          <p className="document-none">No document attached.</p>

          {status === "uploading" && <p className="document-status">Uploading...</p>}
          {status === "processing" && <p className="document-status">Processing document...</p>}
          {status === "error" && <p className="document-error">{error}</p>}

          <button
            type="button"
            className="document-upload-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={status === "uploading" || status === "processing"}
          >
            Upload PDF or Image
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf,image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
            onChange={handleFileChange}
            hidden
          />
        </div>
      )}

      {activeDocument && (
        <p className="document-hint">
          Ask about the document, the connected database, or both.
        </p>
      )}

    </div>
  );
}

export default Sidebar;
