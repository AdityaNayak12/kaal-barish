// Impact numbers per region: heatwave risk and rain over the 10 forecast days.
// Regions are rough lat/lon boxes over land. A land cell belongs to the FIRST region in the list whose box holds it.
import { cellLat, cellLon, N_LAT, N_LON, T2M, TP, type States } from "./grid";

export type Region = { name: string; latMin: number; latMax: number; lonMin: number; lonMax: number };

export const REGIONS: Region[] = [
  { name: "Himalaya and Kashmir", latMin: 31, latMax: 38, lonMin: 72, lonMax: 81 },
  { name: "North plains (Punjab, Haryana, Delhi, west UP)", latMin: 26, latMax: 31, lonMin: 73, lonMax: 81 },
  { name: "North-west (Rajasthan, Gujarat)", latMin: 20, latMax: 30, lonMin: 68, lonMax: 76 },
  { name: "Gangetic east (east UP, Bihar, West Bengal)", latMin: 22, latMax: 28, lonMin: 81, lonMax: 89 },
  { name: "North-east", latMin: 22, latMax: 29, lonMin: 89, lonMax: 97 },
  { name: "Central (MP, Chhattisgarh, Vidarbha)", latMin: 18, latMax: 26, lonMin: 76, lonMax: 84 },
  { name: "East coast (Odisha, coastal Andhra)", latMin: 15, latMax: 22, lonMin: 84, lonMax: 88 },
  { name: "West coast (Konkan, Goa, Kerala)", latMin: 8, latMax: 20, lonMin: 72, lonMax: 76 },
  { name: "South interior (Telangana, Karnataka, Tamil Nadu)", latMin: 8, latMax: 18, lonMin: 76, lonMax: 84 },
];

export type RegionImpact = {
  name: string;
  cells: number;
  heatCellDays: number; // land cells above the threshold, summed over all days
  maxTempC: number;
  rainMm: number; // 10-day rain total, averaged over the region's cells
};

// regionOf[i][j] = index into REGIONS, or -1 for sea and cells outside every region.
export function assignRegions(landMask: number[][]): number[][] {
  const regionOf: number[][] = [];
  for (let i = 0; i < N_LAT; i++) {
    const row: number[] = [];
    for (let j = 0; j < N_LON; j++) {
      let found = -1;
      if (landMask[i][j] === 1) {
        const lat = cellLat(i);
        const lon = cellLon(j);
        for (let r = 0; r < REGIONS.length; r++) {
          const reg = REGIONS[r];
          if (found === -1 && lat >= reg.latMin && lat <= reg.latMax && lon >= reg.lonMin && lon <= reg.lonMax) {
            found = r;
          }
        }
      }
      row.push(found);
    }
    regionOf.push(row);
  }
  return regionOf;
}

export function computeImpact(states: States, regionOf: number[][], heatThresholdC: number): RegionImpact[] {
  const results: RegionImpact[] = [];
  for (let r = 0; r < REGIONS.length; r++) {
    results.push({ name: REGIONS[r].name, cells: 0, heatCellDays: 0, maxTempC: -999, rainMm: 0 });
  }
  const thresholdK = heatThresholdC + 273.15;

  for (let i = 0; i < N_LAT; i++) {
    for (let j = 0; j < N_LON; j++) {
      const r = regionOf[i][j];
      if (r === -1) {
        continue;
      }
      const result = results[r];
      result.cells += 1;
      for (let d = 0; d < states.length; d++) {
        const t2m = states[d][T2M][i][j];
        if (t2m > thresholdK) {
          result.heatCellDays += 1;
        }
        if (t2m - 273.15 > result.maxTempC) {
          result.maxTempC = t2m - 273.15;
        }
        result.rainMm += states[d][TP][i][j];
      }
    }
  }

  for (const result of results) {
    if (result.cells > 0) {
      result.rainMm = result.rainMm / result.cells;
    }
  }
  return results;
}

// Percent change of rain, shocked vs baseline. null when the baseline is almost dry.
export function rainChangePct(baselineMm: number, shockedMm: number): number | null {
  if (baselineMm < 1) {
    return null;
  }
  return (100 * (shockedMm - baselineMm)) / baselineMm;
}
