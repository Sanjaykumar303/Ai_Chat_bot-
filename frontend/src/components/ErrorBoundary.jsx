import { Component } from "react";

// A class component because React has no hook equivalent for this -
// getDerivedStateFromError/componentDidCatch are the only way to catch
// a render error in a subtree instead of letting it unmount the whole
// app to a blank white screen. Wraps <App/> in main.jsx.
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    // Chat history itself lives in localStorage (see utils/chatStorage.js),
    // not component state, so a reload below recovers the conversation
    // too, not just the UI - this is only here so the failure is visible
    // in the console instead of silently swallowed.
    console.error("Unhandled error in the chat UI:", error, info);
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="error-boundary-fallback">
        <h1>Something went wrong</h1>
        <p>
          The chat app hit an unexpected error and couldn't continue. Your chat
          history is saved and won't be lost.
        </p>
        <button type="button" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
