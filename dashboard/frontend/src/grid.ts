// Grid and channel facts from CONTRACT.md.
// 32 x 36 cells of 1 degree. Cell centres: lat 6.5..37.5 (south to north), lon 65.5..100.5 (west to east).

export const N_LAT = 32;
export const N_LON = 36;
export const LAT0 = 6.5;
export const LON0 = 65.5;

export const CHANNELS = ["t2m", "tp", "u10", "v10", "msl", "tcwv"] as const;
export type Channel = (typeof CHANNELS)[number];

export const T2M = 0;
export const TP = 1;
export const U10 = 2;
export const V10 = 3;
export const MSL = 4;
export const TCWV = 5;

// One map field: [lat][lon]
export type Field = number[][];
// One day: [channel][lat][lon]
export type State = number[][][];
// Several days: [day][channel][lat][lon]
export type States = number[][][][];

export type Box = { latMin: number; latMax: number; lonMin: number; lonMax: number };

export function cellLat(i: number): number {
  return LAT0 + i;
}

export function cellLon(j: number): number {
  return LON0 + j;
}

// A cell is inside the box when its centre is inside (same rule as the kit's box_mask).
export function cellInBox(i: number, j: number, box: Box): boolean {
  const lat = cellLat(i);
  const lon = cellLon(j);
  return lat >= box.latMin && lat <= box.latMax && lon >= box.lonMin && lon <= box.lonMax;
}

// Returns the cell [i, j] under a map point, or null outside the grid.
export function cellAt(lat: number, lon: number): [number, number] | null {
  const i = Math.floor(lat - (LAT0 - 0.5));
  const j = Math.floor(lon - (LON0 - 0.5));
  if (i < 0 || i >= N_LAT || j < 0 || j >= N_LON) {
    return null;
  }
  return [i, j];
}

export function emptyField(value: number): Field {
  const field: Field = [];
  for (let i = 0; i < N_LAT; i++) {
    const row: number[] = [];
    for (let j = 0; j < N_LON; j++) {
      row.push(value);
    }
    field.push(row);
  }
  return field;
}
