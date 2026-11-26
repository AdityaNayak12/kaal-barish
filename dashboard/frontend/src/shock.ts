// A shock is a change added to the last input day inside a box, before the model runs.
// t2m is shifted in kelvin. Water vapour and wind are scaled by a percentage of their current value.
import { cellInBox, N_LAT, N_LON, T2M, TCWV, U10, V10, type Box, type State } from "./grid";

export type Shock = {
  box: Box;
  t2mK: number; // added kelvin
  tcwvPct: number; // percent of the current tcwv
  windPct: number; // percent of the current u10 and v10
};

export type Preset = { name: string; description: string; shock: Shock };

export const PRESETS: Preset[] = [
  {
    name: "Warm Bay of Bengal",
    description: "+3 K temperature and +15% water vapour over the Bay of Bengal",
    shock: { box: { latMin: 8, latMax: 20, lonMin: 82, lonMax: 94 }, t2mK: 3, tcwvPct: 15, windPct: 0 },
  },
  {
    name: "Monsoon stall",
    description: "-40% of the winds over the Arabian Sea",
    shock: { box: { latMin: 8, latMax: 20, lonMin: 65, lonMax: 75 }, t2mK: 0, tcwvPct: 0, windPct: -40 },
  },
];

// Returns the perturbation [6][32][36] in physical units, using the last input day for the % changes.
export function buildPerturbation(lastDay: State, shock: Shock): State {
  const perturbation: State = [];
  for (let c = 0; c < 6; c++) {
    const channel: number[][] = [];
    for (let i = 0; i < N_LAT; i++) {
      const row: number[] = [];
      for (let j = 0; j < N_LON; j++) {
        let delta = 0;
        if (cellInBox(i, j, shock.box)) {
          if (c === T2M) {
            delta = shock.t2mK;
          }
          if (c === TCWV) {
            delta = (shock.tcwvPct / 100) * lastDay[TCWV][i][j];
          }
          if (c === U10 || c === V10) {
            delta = (shock.windPct / 100) * lastDay[c][i][j];
          }
        }
        row.push(delta);
      }
      channel.push(row);
    }
    perturbation.push(channel);
  }
  return perturbation;
}

export function countCellsInBox(box: Box): number {
  let count = 0;
  for (let i = 0; i < N_LAT; i++) {
    for (let j = 0; j < N_LON; j++) {
      if (cellInBox(i, j, box)) {
        count += 1;
      }
    }
  }
  return count;
}

export function describeShock(shock: Shock): string {
  const parts: string[] = [];
  if (shock.t2mK !== 0) {
    parts.push(`${shock.t2mK > 0 ? "+" : ""}${shock.t2mK} K t2m`);
  }
  if (shock.tcwvPct !== 0) {
    parts.push(`${shock.tcwvPct > 0 ? "+" : ""}${shock.tcwvPct}% tcwv`);
  }
  if (shock.windPct !== 0) {
    parts.push(`${shock.windPct > 0 ? "+" : ""}${shock.windPct}% wind`);
  }
  if (parts.length === 0) {
    parts.push("no change (null shock)");
  }
  const b = shock.box;
  return `${parts.join(", ")} in lat ${b.latMin}-${b.latMax}, lon ${b.lonMin}-${b.lonMax}`;
}
