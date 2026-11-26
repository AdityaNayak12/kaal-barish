// Colour scales and display units for each channel.
import { N_LAT, N_LON, type Field } from "./grid";

type Rgb = [number, number, number];

export type Scale = {
  min: number;
  max: number;
  stops: Rgb[];
  unit: string;
  // Rain looks better on a square-root scale: many small values, a few very large ones.
  sqrt: boolean;
  // Values below this are drawn transparent (used for light rain).
  hideBelow: number | null;
};

const THERMAL: Rgb[] = [
  [49, 54, 149],
  [69, 117, 180],
  [171, 217, 233],
  [255, 255, 191],
  [253, 174, 97],
  [215, 48, 39],
  [120, 0, 30],
];
const RAIN: Rgb[] = [
  [199, 233, 252],
  [107, 174, 214],
  [33, 113, 181],
  [84, 39, 143],
  [190, 30, 120],
];
const DIVERGING: Rgb[] = [
  [33, 102, 172],
  [103, 169, 207],
  [247, 247, 247],
  [239, 138, 98],
  [178, 24, 43],
];
const PRESSURE: Rgb[] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
const MOISTURE: Rgb[] = [
  [140, 81, 10],
  [216, 179, 101],
  [246, 232, 195],
  [128, 205, 193],
  [1, 102, 94],
  [0, 50, 90],
];

export type ChannelInfo = {
  name: string;
  title: string;
  // Converts a physical value (K, Pa, ...) to the unit shown on screen.
  toDisplay: (value: number) => number;
  scale: Scale;
};

export const CHANNEL_INFO: ChannelInfo[] = [
  {
    name: "t2m",
    title: "Temperature 2 m",
    toDisplay: (v) => v - 273.15,
    scale: { min: -10, max: 42, stops: THERMAL, unit: "°C", sqrt: false, hideBelow: null },
  },
  {
    name: "tp",
    title: "Rain",
    toDisplay: (v) => v,
    scale: { min: 0, max: 80, stops: RAIN, unit: "mm/day", sqrt: true, hideBelow: 0.5 },
  },
  {
    name: "u10",
    title: "Wind east-west (u10)",
    toDisplay: (v) => v,
    scale: { min: -12, max: 12, stops: DIVERGING, unit: "m/s", sqrt: false, hideBelow: null },
  },
  {
    name: "v10",
    title: "Wind north-south (v10)",
    toDisplay: (v) => v,
    scale: { min: -12, max: 12, stops: DIVERGING, unit: "m/s", sqrt: false, hideBelow: null },
  },
  {
    name: "msl",
    title: "Sea level pressure",
    toDisplay: (v) => v / 100,
    scale: { min: 990, max: 1022, stops: PRESSURE, unit: "hPa", sqrt: false, hideBelow: null },
  },
  {
    name: "tcwv",
    title: "Water vapour",
    toDisplay: (v) => v,
    scale: { min: 0, max: 70, stops: MOISTURE, unit: "kg/m²", sqrt: false, hideBelow: null },
  },
];

// A symmetric blue-white-red scale for "shocked minus baseline".
export function differenceScale(maxAbs: number, unit: string): Scale {
  let limit = maxAbs;
  if (limit < 1e-6) {
    limit = 1e-6;
  }
  return { min: -limit, max: limit, stops: DIVERGING, unit: unit, sqrt: false, hideBelow: null };
}

// Position of a value on the scale, from 0 to 1.
export function scalePosition(value: number, scale: Scale): number {
  let t = (value - scale.min) / (scale.max - scale.min);
  if (scale.sqrt) {
    t = Math.sqrt(Math.max(t, 0));
  }
  if (t < 0) {
    t = 0;
  }
  if (t > 1) {
    t = 1;
  }
  return t;
}

export function colorAt(t: number, stops: Rgb[]): string {
  const position = t * (stops.length - 1);
  let k = Math.floor(position);
  if (k >= stops.length - 1) {
    k = stops.length - 2;
  }
  const f = position - k;
  const a = stops[k];
  const b = stops[k + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

// Returns null when the value should not be drawn.
export function colorFor(value: number, scale: Scale): string | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  if (scale.hideBelow !== null && value < scale.hideBelow) {
    return null;
  }
  return colorAt(scalePosition(value, scale), scale.stops);
}

// Converts a physical field to display units.
export function toDisplayField(field: Field, channel: number): Field {
  const convert = CHANNEL_INFO[channel].toDisplay;
  const out: Field = [];
  for (let i = 0; i < N_LAT; i++) {
    const row: number[] = [];
    for (let j = 0; j < N_LON; j++) {
      row.push(convert(field[i][j]));
    }
    out.push(row);
  }
  return out;
}
