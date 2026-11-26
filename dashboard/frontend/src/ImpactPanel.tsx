// Ranked lists per region: heatwave risk (baseline vs shocked) and rain deficit of the shocked run vs the baseline.
import { useState } from "react";
import type { States } from "./grid";
import { computeImpact, rainChangePct, type RegionImpact } from "./impact";

type Props = {
  baseline: States;
  shocked: States | null;
  regionOf: number[][];
};

type RainRow = { name: string; baselineMm: number; shockedMm: number; changePct: number | null };

function sortByHeat(rows: RegionImpact[]): RegionImpact[] {
  const copy = rows.slice();
  copy.sort((a, b) => {
    if (b.heatCellDays !== a.heatCellDays) {
      return b.heatCellDays - a.heatCellDays;
    }
    return b.maxTempC - a.maxTempC;
  });
  return copy;
}

// Biggest deficit (most negative change) first. Regions without a percentage go last.
function sortByDeficit(rows: RainRow[]): RainRow[] {
  const copy = rows.slice();
  copy.sort((a, b) => {
    if (a.changePct === null && b.changePct === null) return 0;
    if (a.changePct === null) return 1;
    if (b.changePct === null) return -1;
    return a.changePct - b.changePct;
  });
  return copy;
}

function rainClass(changePct: number | null): string {
  if (changePct === null || Math.abs(changePct) < 0.05) {
    return "";
  }
  if (changePct < 0) {
    return "worse";
  }
  return "better";
}

function HeatList({ title, rows, compare }: { title: string; rows: RegionImpact[]; compare: RegionImpact[] | null }) {
  let largest = 1;
  for (const row of rows) {
    if (row.heatCellDays > largest) largest = row.heatCellDays;
  }
  return (
    <div className="ranked">
      <h3>{title}</h3>
      <ol>
        {rows.map((row) => {
          let change = null;
          if (compare) {
            for (const other of compare) {
              if (other.name === row.name) {
                change = row.heatCellDays - other.heatCellDays;
              }
            }
          }
          return (
            <li key={row.name}>
              <div className="ranked-name">{row.name}</div>
              <div className="ranked-bar-row">
                <div className="ranked-bar heat" style={{ width: `${(100 * row.heatCellDays) / largest}%` }} />
                <span className="ranked-value">
                  {row.heatCellDays} cell-days, max {row.maxTempC.toFixed(1)}°C
                  {change !== null && change !== 0 && (
                    <b className={change > 0 ? "worse" : "better"}>
                      {" "}
                      ({change > 0 ? "+" : ""}
                      {change})
                    </b>
                  )}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function ImpactPanel({ baseline, shocked, regionOf }: Props) {
  const [thresholdC, setThresholdC] = useState<number>(34);

  const baseImpact = computeImpact(baseline, regionOf, thresholdC);
  let shockImpact: RegionImpact[] | null = null;
  if (shocked) {
    shockImpact = computeImpact(shocked, regionOf, thresholdC);
  }

  const rainRows: RainRow[] = [];
  for (let r = 0; r < baseImpact.length; r++) {
    let shockedMm = baseImpact[r].rainMm;
    if (shockImpact) {
      shockedMm = shockImpact[r].rainMm;
    }
    rainRows.push({
      name: baseImpact[r].name,
      baselineMm: baseImpact[r].rainMm,
      shockedMm: shockedMm,
      changePct: rainChangePct(baseImpact[r].rainMm, shockedMm),
    });
  }

  // Without a shock there is no deficit yet: rank by baseline rain instead.
  let sortedRain = sortByDeficit(rainRows);
  if (!shockImpact) {
    sortedRain = rainRows.slice();
    sortedRain.sort((a, b) => b.baselineMm - a.baselineMm);
  }

  return (
    <section className="panel impact">
      <h2>Impact by region (10 forecast days, land cells only)</h2>
      <label className="threshold">
        Heatwave threshold: daily mean 2 m temperature above <b>{thresholdC}°C</b>
        <input type="range" min={28} max={40} step={1} value={thresholdC} onChange={(e) => setThresholdC(Number(e.target.value))} />
      </label>

      <div className="impact-grid">
        <HeatList title="Heatwave risk: baseline" rows={sortByHeat(baseImpact)} compare={null} />
        {shockImpact ? (
          <HeatList title="Heatwave risk: shocked (change vs baseline)" rows={sortByHeat(shockImpact)} compare={baseImpact} />
        ) : (
          <div className="ranked placeholder">Run a shocked forecast to rank the shocked heatwave risk.</div>
        )}

        <div className="ranked">
          <h3>Rain deficit: shocked vs baseline</h3>
          {!shockImpact && <div className="hint">Baseline 10-day rain shown. Run a shock to see the deficit.</div>}
          <table>
            <thead>
              <tr>
                <th>Region</th>
                <th>Baseline mm</th>
                <th>Shocked mm</th>
                <th>Change</th>
              </tr>
            </thead>
            <tbody>
              {sortedRain.map((row) => (
                <tr key={row.name}>
                  <td>{row.name}</td>
                  <td>{row.baselineMm.toFixed(1)}</td>
                  <td>{shockImpact ? row.shockedMm.toFixed(1) : "-"}</td>
                  <td className={rainClass(row.changePct)}>
                    {!shockImpact || row.changePct === null
                      ? "-"
                      : `${row.changePct > 0 ? "+" : ""}${row.changePct.toFixed(1)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="hint">
        Heat: number of (cell, day) pairs above the threshold. Rain: 10-day total, averaged over the region's land
        cells. Regions are rough lat/lon boxes.
      </div>
    </section>
  );
}
