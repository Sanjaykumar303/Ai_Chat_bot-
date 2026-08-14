import logo from "../assets/logo.jpg";

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

      <img className="navbar-logo" src={logo} alt="" />
      <span className="navbar-title">AI Document Assistant</span>

    </div>
  );
}

export default Navbar;
