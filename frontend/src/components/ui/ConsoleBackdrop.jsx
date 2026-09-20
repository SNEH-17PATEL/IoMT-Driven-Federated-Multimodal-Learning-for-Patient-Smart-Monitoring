// Fixed near-black canvas behind the whole app: a faint blueprint grid,
// a slow drifting scan band, and soft accent glows — a command-console
// backdrop instead of a pastel gradient one.
export default function ConsoleBackdrop() {
  return (
    <div className="console-field" aria-hidden="true">
      <div className="console-grid" />
      <div className="console-scan" />
      <div className="console-noise" />
    </div>
  );
}
