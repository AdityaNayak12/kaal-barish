import { useEffect, useRef, useState } from "react";
import { fetchHealth, fetchSample, fetchSamples, simulate, type Sample, type SampleInfo } from "./api";
import { CHANNEL_INFO, toDisplayField } from "./colors";
import GridMap, { type Wind } from "./GridMap";
import { CHANNELS, U10, V10, type States } from "./grid";
import Legend from "./Legend";

const STEPS = 10;
const PLAY_MS = 900;

type Source = "forecast" | "truth";

export default function App() {
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [sampleId, setSampleId] = useState<string>("");
  const [sample, setSample] = useState<Sample | null>(null);
  const [modelName, setModelName] = useState<string>("");
  const [error, setError] = useState<string>("");

  const [baseline, setBaseline] = useState<States | null>(null);
  const [baselineMs, setBaselineMs] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const [channel, setChannel] = useState<number>(1);
  const [day, setDay] = useState<number>(1);
  const [playing, setPlaying] = useState<boolean>(false);
  const [source, setSource] = useState<Source>("forecast");
  const [showWind, setShowWind] = useState<boolean>(true);

  // Counts sample loads so a slow old reply cannot overwrite a newer one.
  const loadCounter = useRef(0);

  // Start-up: model status and the list of samples.
  useEffect(() => {
    fetchHealth()
      .then((health) => setModelName(health.model_name))
      .catch(() => setModelName("model server not reachable"));
    fetchSamples()
      .then((list) => {
        setSamples(list);
        if (list.length > 0) {
          setSampleId(list[0].id);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  // New sample: load it, then run the baseline forecast live on the model server.
  useEffect(() => {
    if (sampleId === "") {
      return;
    }
    loadCounter.current += 1;
    const myLoad = loadCounter.current;
    setLoading(true);
    setBaseline(null);
    setBaselineMs(null);
    setError("");

    async function load() {
      const loaded = await fetchSample(sampleId);
      if (myLoad !== loadCounter.current) return;
      setSample(loaded);
      const result = await simulate(loaded, STEPS, null);
      if (myLoad !== loadCounter.current) return;
      setBaseline(result.rollout);
      setBaselineMs(result.inference_ms);
    }
    load()
      .catch((e) => setError(String(e)))
      .finally(() => {
        if (myLoad === loadCounter.current) setLoading(false);
      });
  }, [sampleId]);

  // Play button: step through days 1..10 and loop.
  useEffect(() => {
    if (!playing) {
      return;
    }
    const timer = setInterval(() => {
      setDay((d) => (d >= STEPS ? 1 : d + 1));
    }, PLAY_MS);
    return () => clearInterval(timer);
  }, [playing]);

  let mainStates: States | null = null;
  if (source === "truth" && sample) {
    mainStates = sample.truth;
  }
  if (source === "forecast") {
    mainStates = baseline;
  }

  const info = CHANNEL_INFO[channel];
  let mainField = null;
  let mainWind: Wind | null = null;
  if (mainStates) {
    const dayState = mainStates[day - 1];
    mainField = toDisplayField(dayState[channel], channel);
    if (showWind) {
      mainWind = { u: dayState[U10], v: dayState[V10] };
    }
  }

  let dayText = `Day ${day}`;
  if (sample) {
    dayText = `Day ${day} (${sample.truth_days[day - 1]})`;
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Kaal Barish: Multiverse of Monsoon</h1>
          <div className="subtitle">
            Model: <b>{modelName || "..."}</b>
            {baselineMs !== null && <span> · baseline forecast {baselineMs.toFixed(0)} ms</span>}
            {loading && <span> · running forecast...</span>}
          </div>
        </div>
        <label className="sample-picker">
          Sample window (2021 test year)
          <select value={sampleId} onChange={(e) => setSampleId(e.target.value)}>
            {samples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="controls">
        <div className="button-group">
          {CHANNELS.map((name, k) => (
            <button key={name} className={k === channel ? "active" : ""} onClick={() => setChannel(k)}>
              {CHANNEL_INFO[k].title}
            </button>
          ))}
        </div>
        <div className="button-group">
          <button className={source === "forecast" ? "active" : ""} onClick={() => setSource("forecast")}>
            Forecast
          </button>
          <button className={source === "truth" ? "active" : ""} onClick={() => setSource("truth")}>
            Truth
          </button>
        </div>
        <label className="check">
          <input type="checkbox" checked={showWind} onChange={(e) => setShowWind(e.target.checked)} />
          Wind arrows
        </label>
        <div className="day-control">
          <button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
          <input type="range" min={1} max={STEPS} value={day} onChange={(e) => setDay(Number(e.target.value))} />
          <span className="day-text">{dayText}</span>
        </div>
      </section>

      <section className="main-row">
        <div className="main-map">
          <GridMap
            title={`${source === "forecast" ? "Forecast" : "Truth"}: ${info.title}, ${dayText}`}
            field={mainField}
            scale={info.scale}
            wind={mainWind}
            height={520}
          />
          <Legend scale={info.scale} />
        </div>
      </section>
    </div>
  );
}
