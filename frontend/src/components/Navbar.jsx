import { useEffect, useRef, useState } from "react";

import logo from "../assets/logo.jpg";
import { Database, Menu } from "../icons";
import { checkHealth } from "../services/api";

// checking/connected/error: result of the most recent GET /health call -
// see checkDatabaseStatus below for how each is reached. Checked once
// automatically on mount (see the effect below) so the dot always shows
// real status without waiting for a click - there's no "never checked
// yet" idle state to render any more.
const CHECKING = "checking";
const CONNECTED = "connected";
const ERROR = "error";

// Turns a resolved/rejected GET /health call into the {status, detail}
// dbState shape - pure (no setState), so both the mount effect and the
// click-triggered re-check below can share the exact same
// interpretation of a response/error without duplicating it.
//
// /health responds 503 (not 200) when a component is unhealthy, which
// axios treats as a rejected promise - resolveDbStateFromError reads
// err.response.data the same way resolveDbState reads response.data, so
// an actual "database unreachable" answer from the server is told apart
// from the request never reaching the backend at all (network error /
// backend down, where err.response is undefined).
function resolveDbState(response) {
  const database = response.data.components.database;
  return database === "ok"
    ? { status: CONNECTED, detail: "" }
    : { status: ERROR, detail: database };
}

function resolveDbStateFromError(err) {
  const database = err.response?.data?.components?.database;
  return database
    ? { status: ERROR, detail: database }
    : { status: ERROR, detail: "Could not reach the backend server." };
}

function Navbar({ onToggleSidebar }) {

  const [dbState, setDbState] = useState({ status: CHECKING, detail: "" });
  const [popoverOpen, setPopoverOpen] = useState(false);
  const wrapperRef = useRef(null);

  // Same outside-click/Escape-closes behavior as AttachmentMenu.jsx.
  useEffect(() => {
    if (!popoverOpen) {
      return;
    }

    function handlePointerDown(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setPopoverOpen(false);
      }
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setPopoverOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [popoverOpen]);

  // Every click is a fresh, live check - flips the dot to CHECKING right
  // away (dbState already starts there on mount, see useState above, so
  // this only actually changes anything visually on a RE-check), then
  // pings GET /health and settles at CONNECTED/ERROR via the shared
  // resolveDbState/resolveDbStateFromError helpers above.
  async function checkDatabaseStatus() {
    setDbState({ status: CHECKING, detail: "" });

    try {
      setDbState(resolveDbState(await checkHealth()));
    } catch (err) {
      setDbState(resolveDbStateFromError(err));
    }
  }

  // Runs once on mount so the dot reflects real status immediately,
  // without the user ever having to click - checkDatabaseStatus itself
  // still re-runs on every click too (see toggleDatabaseStatus below),
  // so this is purely "check eagerly first", not a replacement for that.
  // Written as a plain .then()/.catch() with a cancelled guard (same
  // idea VoiceChat.jsx's own connection effect uses its `cancelled`
  // variable for) rather than an async function called from the effect
  // body - react-hooks/set-state-in-effect flags the latter as a
  // synchronous-setState-in-an-effect risk even when the actual setState
  // only happens after the request resolves.
  useEffect(() => {
    let cancelled = false;

    checkHealth()
      .then((response) => {
        if (!cancelled) setDbState(resolveDbState(response));
      })
      .catch((err) => {
        if (!cancelled) setDbState(resolveDbStateFromError(err));
      });

    return () => {
      cancelled = true;
    };
  }, []);

  function toggleDatabaseStatus() {
    const opening = !popoverOpen;
    setPopoverOpen(opening);

    if (opening) {
      checkDatabaseStatus();
    }
  }

  const dbStatusLabel = {
    [CHECKING]: "Checking database connection...",
    [CONNECTED]: "Database is connected.",
    [ERROR]: "Database is not connected.",
  }[dbState.status];

  return (
    <div className="navbar">

      <button
        type="button"
        className="navbar-menu-button"
        onClick={onToggleSidebar}
        aria-label="Toggle chat list"
      >
        <Menu size={22} />
      </button>

      <img className="navbar-logo" src={logo} alt="" />
      <span className="navbar-title">AI Business Assistant</span>

      <div className="navbar-db-status-wrapper" ref={wrapperRef}>
        <button
          type="button"
          className="navbar-db-status-button"
          onClick={toggleDatabaseStatus}
          aria-label={`${dbStatusLabel} Click to check again.`}
          aria-expanded={popoverOpen}
          title={dbStatusLabel}
        >
          <Database size={20} />
          <span className={`navbar-db-status-dot navbar-db-status-dot-${dbState.status}`} />
        </button>

        {popoverOpen && (
          <div className="navbar-db-status-popover" role="status">
            {dbState.status === CHECKING && <p>Checking database connection...</p>}
            {dbState.status === CONNECTED && (
              <p className="navbar-db-status-ok">Database is connected.</p>
            )}
            {dbState.status === ERROR && (
              <>
                <p className="navbar-db-status-fail">Database is not connected.</p>
                {dbState.detail && <p className="navbar-db-status-detail">{dbState.detail}</p>}
              </>
            )}
          </div>
        )}
      </div>

    </div>
  );
}

export default Navbar;
