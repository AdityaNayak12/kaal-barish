import type { Box } from "./grid";
import { countCellsInBox, PRESETS, type Shock } from "./shock";

type Props = {
  shock: Shock;
  onChange: (shock: Shock) => void;
  drawing: boolean;
  onToggleDraw: () => void;
  onRun: () => void;
  onClear: () => void;
  running: boolean;
  canRun: boolean;
  shockMs: number | null;
};

type NumberInputProps = {
  label: string;
  value: number;
  step: number;
  onChange: (value: number) => void;
};

function NumberInput({ label, value, step, onChange }: NumberInputProps) {
  return (
    <label className="number-input">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => {
          const parsed = Number(e.target.value);
          if (Number.isFinite(parsed)) {
            onChange(parsed);
          }
        }}
      />
    </label>
  );
}

export default function ShockBuilder(props: Props) {
  const { shock, onChange, drawing, onToggleDraw, onRun, onClear, running, canRun, shockMs } = props;

  function setBox(key: keyof Box, value: number) {
    const box = { ...shock.box };
    box[key] = value;
    onChange({ ...shock, box: box });
  }

  const cells = countCellsInBox(shock.box);

  return (
    <div className="panel shock-builder">
      <h2>Shock builder</h2>
      <div className="presets">
        {PRESETS.map((preset) => (
          <button key={preset.name} title={preset.description} onClick={() => onChange(preset.shock)}>
            {preset.name}
          </button>
        ))}
      </div>

      <h3>Box</h3>
      <button className={drawing ? "active" : ""} onClick={onToggleDraw}>
        {drawing ? "Drag on the map to draw..." : "Draw box on map"}
      </button>
      <div className="box-inputs">
        <NumberInput label="lat from" value={shock.box.latMin} step={1} onChange={(v) => setBox("latMin", v)} />
        <NumberInput label="lat to" value={shock.box.latMax} step={1} onChange={(v) => setBox("latMax", v)} />
        <NumberInput label="lon from" value={shock.box.lonMin} step={1} onChange={(v) => setBox("lonMin", v)} />
        <NumberInput label="lon to" value={shock.box.lonMax} step={1} onChange={(v) => setBox("lonMax", v)} />
      </div>
      <div className="hint">{cells} grid cells inside the box</div>

      <h3>Change (added to the last input day)</h3>
      <div className="delta-inputs">
        <NumberInput label="Temperature t2m (K)" value={shock.t2mK} step={0.5} onChange={(v) => onChange({ ...shock, t2mK: v })} />
        <NumberInput label="Water vapour tcwv (%)" value={shock.tcwvPct} step={5} onChange={(v) => onChange({ ...shock, tcwvPct: v })} />
        <NumberInput label="Wind u10 and v10 (%)" value={shock.windPct} step={5} onChange={(v) => onChange({ ...shock, windPct: v })} />
      </div>

      <div className="run-row">
        <button className="primary" disabled={!canRun || running || cells === 0} onClick={onRun}>
          {running ? "Running..." : "Run shocked forecast"}
        </button>
        <button onClick={onClear}>Clear</button>
      </div>
      {shockMs !== null && <div className="hint">Shocked forecast took {shockMs.toFixed(0)} ms on the model server</div>}
    </div>
  );
}
