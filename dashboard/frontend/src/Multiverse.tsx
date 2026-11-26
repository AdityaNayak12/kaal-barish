// Baseline and shocked forecasts side by side, plus the difference map (shocked minus baseline).
// The difference uses one colour range for all 10 days, so you can watch the shock spread day by day.
import { useState } from "react";
import { CHANNEL_INFO, differenceScale, toDisplayField } from "./colors";
import GridMap, { type MapView } from "./GridMap";
import { N_LAT, N_LON, U10, V10, type Box, type Field, type States } from "./grid";
import Legend from "./Legend";

// A cell counts as "reached by the shock" when it changes by more than this (display units).
const CHANGE_THRESHOLD = [0.1, 0.5, 0.2, 0.2, 0.2, 0.5];

type Props = {
  baseline: States;
  shocked: States;
  channel: number;
  day: number;
  box: Box;
  showWind: boolean;
  onDayClick: (day: number) => void;
};

function differenceField(baseline: States, shocked: States, day: number, channel: number): Field {
  const before = toDisplayField(baseline[day - 1][channel], channel);
  const after = toDisplayField(shocked[day - 1][channel], channel);
  const diff: Field = [];
  for (let i = 0; i < N_LAT; i++) {
    const row: number[] = [];
    for (let j = 0; j < N_LON; j++) {
      row.push(after[i][j] - before[i][j]);
    }
    diff.push(row);
  }
  return diff;
}

function largestChange(field: Field): number {
  let largest = 0;
  for (let i = 0; i < N_LAT; i++) {
    for (let j = 0; j < N_LON; j++) {
      const size = Math.abs(field[i][j]);
      if (size > largest) {
        largest = size;
      }
    }
  }
  return largest;
}

function cellsChanged(field: Field, threshold: number): number {
  let count = 0;
  for (let i = 0; i < N_LAT; i++) {
    for (let j = 0; j < N_LON; j++) {
      if (Math.abs(field[i][j]) > threshold) {
        count += 1;
      }
    }
  }
  return count;
}

export default function Multiverse(props: Props) {
  const { baseline, shocked, channel, day, box, showWind, onDayClick } = props;
  const [view, setView] = useState<MapView | null>(null);
  const info = CHANNEL_INFO[channel];

  // Difference for every day: used for the fixed colour range and the spread counts.
  const diffs: Field[] = [];
  let maxAbs = 0;
  for (let d = 1; d <= baseline.length; d++) {
    const diff = differenceField(baseline, shocked, d, channel);
    diffs.push(diff);
    const largest = largestChange(diff);
    if (largest > maxAbs) {
      maxAbs = largest;
    }
  }
  const diffScale = differenceScale(maxAbs, info.scale.unit);

  const spread: number[] = [];
  for (let d = 0; d < diffs.length; d++) {
    spread.push(cellsChanged(diffs[d], CHANGE_THRESHOLD[channel]));
  }
  let maxSpread = 1;
  for (const count of spread) {
    if (count > maxSpread) {
      maxSpread = count;
    }
  }

  const baseDay = baseline[day - 1];
  const shockDay = shocked[day - 1];
  let baseWind = null;
  let shockWind = null;
  if (showWind) {
    baseWind = { u: baseDay[U10], v: baseDay[V10] };
    shockWind = { u: shockDay[U10], v: shockDay[V10] };
  }

  return (
    <section className="panel multiverse">
      <h2>Multiverse: baseline vs shocked, day {day}</h2>
      <div className="multiverse-maps">
        <div>
          <GridMap
            title={`Baseline: ${info.title}`}
            field={toDisplayField(baseDay[channel], channel)}
            scale={info.scale}
            wind={baseWind}
            box={box}
            view={view}
            onViewChange={setView}
            height={340}
          />
        </div>
        <div>
          <GridMap
            title={`Shocked: ${info.title}`}
            field={toDisplayField(shockDay[channel], channel)}
            scale={info.scale}
            wind={shockWind}
            box={box}
            view={view}
            onViewChange={setView}
            height={340}
          />
        </div>
        <div>
          <GridMap
            title={`Difference (shocked minus baseline)`}
            field={diffs[day - 1]}
            scale={diffScale}
            box={box}
            view={view}
            onViewChange={setView}
            height={340}
          />
        </div>
      </div>
      <div className="multiverse-legends">
        <Legend scale={info.scale} />
        <Legend scale={diffScale} />
      </div>

      <h3>
        How far the shock reaches: cells changed by more than {CHANGE_THRESHOLD[channel]} {info.scale.unit}
      </h3>
      <div className="spread-bars">
        {spread.map((count, k) => (
          <div
            key={k}
            className={k + 1 === day ? "spread-bar current" : "spread-bar"}
            onClick={() => onDayClick(k + 1)}
            title={`Day ${k + 1}: ${count} cells`}
          >
            <div className="spread-fill" style={{ height: `${(100 * count) / maxSpread}%` }} />
            <div className="spread-label">
              d{k + 1}
              <br />
              {count}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
