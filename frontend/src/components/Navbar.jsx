function Navbar({ onToggleSidebar }) {
  return (
    <div className="navbar">

      <button
        type="button"
        className="navbar-menu-button"
        onClick={onToggleSidebar}
        aria-label="Toggle chat list"
      >
        ☰
      </button>

      <span className="navbar-title">AI Document Assistant</span>

    </div>
  );
}

export default Navbar;
