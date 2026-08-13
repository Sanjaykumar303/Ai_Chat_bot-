import axios from "axios";

// VITE_API_URL points at the deployed backend in production (set it in
// Vercel's project settings). Falls back to the local backend for dev,
// unchanged from before.
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
});

export default api;