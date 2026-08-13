import axios from "axios";

// VITE_API_URL points at the deployed backend in production (set it in
// Vercel's project settings). Falls back to the local backend for dev,
// unchanged from before.
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
});

export default api;

// Named wrappers for the endpoints ChatBox.jsx calls, so call sites read
// as what they do rather than a raw path + payload shape repeated inline.

export function sendChatMessage(payload) {
    return api.post("/chat", payload);
}

export function uploadDocument(formData, onUploadProgress) {
    return api.post("/documents/upload", formData, { onUploadProgress });
}

export function deleteDocument(documentId) {
    return api.delete(`/documents/${documentId}`);
}

export function transcribeAudio(formData) {
    return api.post("/transcribe", formData);
}