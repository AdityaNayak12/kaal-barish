import { colorAt, type Scale } from "./colors";

function valueAt(t: number, scale: Scale): number {
  let f = t;
  if (scale.sqrt) {
    f = t * t;
  }
  return scale.min + f * (scale.max - scale.min);
}

function format(value: number): string {
  const size = Math.abs(value);
  if (size >= 100) {
    return value.toFixed(0);
  }
  if (size >= 1) {
    return value.toFixed(1);
  }
  return value.toPrecision(2);
}

export default function Legend({ scale }: { scale: Scale }) {
  const parts: string[] = [];
  const n = 12;
  for (let k = 0; k <= n; k++) {
    const t = k / n;
    parts.push(`${colorAt(t, scale.stops)} ${(t * 100).toFixed(0)}%`);
  }
  const gradient = `linear-gradient(to right, ${parts.join(", ")})`;

  const labels: string[] = [];
  for (let k = 0; k <= 4; k++) {
    labels.push(format(valueAt(k / 4, scale)));
  }

  return (
    <div className="legend">
      <div className="legend-bar" style={{ background: gradient }} />
      <div className="legend-labels">
        {labels.map((label, k) => (
          <span key={k}>{label}</span>
        ))}
      </div>
      <div className="legend-unit">{scale.unit}</div>
    </div>
  );
}
