import { useEffect, useRef, useState } from "react";

// Full-screen-ish modal: requests the camera, shows a live preview, and
// lets the user capture a single still frame. Both the "Camera" and
// "Live Camera" attachment-menu items open this same component - there's
// no meaningful browser-API difference between them (capturing a still
// frame always requires a live preview stream first), just a different
// title, passed in via props.
function CameraCapture({ title, onCapture, onClose }) {

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });

        // The modal may already have been closed (or capture already
        // taken) by the time getUserMedia resolves - don't attach or
        // leave a stream running past that.
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setReady(true);
      } catch {
        if (!cancelled) {
          setError("Could not access the camera. Please allow camera access and try again.");
        }
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  function handleCapture() {
    const video = videoRef.current;

    if (!video || !video.videoWidth) {
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          onCapture(blob);
        }
      },
      "image/jpeg",
      0.9
    );
  }

  return (
    <div className="camera-modal-backdrop" onClick={onClose}>
      <div className="camera-modal" onClick={(event) => event.stopPropagation()}>

        <div className="camera-modal-header">
          <span>{title}</span>
          <button type="button" className="camera-modal-close" onClick={onClose} aria-label="Close camera">
            ✕
          </button>
        </div>

        {error ? (
          <p className="camera-modal-error">{error}</p>
        ) : (
          <video ref={videoRef} className="camera-modal-video" autoPlay playsInline muted />
        )}

        <div className="camera-modal-actions">
          <button type="button" className="camera-modal-capture" onClick={handleCapture} disabled={!ready}>
            📷 Capture
          </button>
        </div>

      </div>
    </div>
  );
}

export default CameraCapture;
