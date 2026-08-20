import { useEffect, useRef, useState } from "react";
import { Attachment, Camera, File, Image } from "../icons";

// The "+" attachment button beside the chat input, ChatGPT-style: click
// to reveal PDF/image upload and camera capture options. Purely
// presentational - all the actual upload/capture logic lives in
// ChatBox.jsx, this just forwards clicks to the callbacks it's given and
// owns its own open/closed menu state.
function AttachmentMenu({ disabled, onUploadPdf, onUploadImage, onOpenCamera }) {

  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);
  const pdfInputRef = useRef(null);
  const imageInputRef = useRef(null);

  // Closes the menu on an outside click or Escape - standard dropdown
  // behavior.
  useEffect(() => {
    if (!open) {
      return;
    }

    function handlePointerDown(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setOpen(false);
      }
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  function handlePdfFileChange(event) {
    const file = event.target.files?.[0];
    event.target.value = ""; // lets choosing the same filename again re-trigger onChange
    setOpen(false);
    if (file) {
      onUploadPdf(file);
    }
  }

  function handleImageFileChange(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    setOpen(false);
    if (file) {
      onUploadImage(file);
    }
  }

  return (
    <div className="attachment-menu-wrapper" ref={wrapperRef}>

      <button
        type="button"
        className="attachment-menu-button"
        onClick={() => setOpen((value) => !value)}
        disabled={disabled}
        title="Attach a file"
        aria-label="Attach a file"
        aria-expanded={open}
      >
        <Attachment size={20} />
      </button>

      {open && (
        <div className="attachment-menu">
          <button type="button" className="attachment-menu-item" onClick={() => pdfInputRef.current?.click()}>
            <File size={18} /> Upload PDF
          </button>
          <button type="button" className="attachment-menu-item" onClick={() => imageInputRef.current?.click()}>
            <Image size={18} /> Upload Image
          </button>
          <button
            type="button"
            className="attachment-menu-item"
            onClick={() => {
              setOpen(false);
              onOpenCamera();
            }}
          >
            <Camera size={18} /> Camera
          </button>
        </div>
      )}

      <input
        ref={pdfInputRef}
        type="file"
        accept="application/pdf,.pdf"
        onChange={handlePdfFileChange}
        hidden
      />
      <input
        ref={imageInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
        onChange={handleImageFileChange}
        hidden
      />

    </div>
  );
}

export default AttachmentMenu;
