// One Leaflet map of India with the 32 x 36 grid drawn as coloured cells.
// Optional extras: wind arrows, a shock box, and drawing a new box with the mouse.
import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { colorFor, type Scale } from "./colors";
import { cellAt, cellLat, cellLon, N_LAT, N_LON, type Box, type Field } from "./grid";

const GRID_BOUNDS: L.LatLngBoundsExpression = [
  [6, 65],
  [38, 101],
];
const ARROW_STEP = 2; // draw an arrow on every 2nd cell
const ARROW_DEGREES_PER_MS = 0.17; // arrow length per m/s of wind
const ARROW_MAX_DEGREES = 1.8;

export type Wind = { u: Field; v: Field };
export type MapView = { lat: number; lon: number; zoom: number };

type Props = {
  title: string;
  field: Field | null; // display units
  scale: Scale;
  wind?: Wind | null;
  box?: Box | null;
  drawing?: boolean;
  onBoxDrawn?: (box: Box) => void;
  view?: MapView | null;
  onViewChange?: (view: MapView) => void;
  height?: number;
};

function boxBounds(box: Box): L.LatLngBoundsExpression {
  return [
    [box.latMin, box.lonMin],
    [box.latMax, box.lonMax],
  ];
}

// Three short lines that make an arrow from the tail to the tip.
function arrowLines(lat: number, lon: number, u: number, v: number): L.LatLngExpression[][] {
  const speed = Math.sqrt(u * u + v * v);
  let length = speed * ARROW_DEGREES_PER_MS;
  if (length > ARROW_MAX_DEGREES) {
    length = ARROW_MAX_DEGREES;
  }
  if (speed < 0.01) {
    return [];
  }
  // Direction in degrees of lat/lon. A degree of longitude is shorter than a degree of latitude.
  const lonStretch = 1 / Math.cos((lat * Math.PI) / 180);
  const dLat = (v / speed) * length;
  const dLon = (u / speed) * length * lonStretch;
  const tail: L.LatLngTuple = [lat - dLat / 2, lon - dLon / 2];
  const tip: L.LatLngTuple = [lat + dLat / 2, lon + dLon / 2];
  // Arrow head: two short lines going back from the tip at +-30 degrees.
  const head = 0.35 * length;
  const angle = Math.atan2(dLat, dLon / lonStretch);
  const left: L.LatLngTuple = [
    tip[0] - head * Math.sin(angle - Math.PI / 6),
    tip[1] - head * Math.cos(angle - Math.PI / 6) * lonStretch,
  ];
  const right: L.LatLngTuple = [
    tip[0] - head * Math.sin(angle + Math.PI / 6),
    tip[1] - head * Math.cos(angle + Math.PI / 6) * lonStretch,
  ];
  return [
    [tail, tip],
    [left, tip, right],
  ];
}

export default function GridMap(props: Props) {
  const { title, field, scale, wind, box, drawing, onBoxDrawn, view, onViewChange } = props;
  const height = props.height || 460;

  const divRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const cellsRef = useRef<L.Rectangle[][]>([]);
  const arrowsRef = useRef<L.Polyline[]>([]);
  const boxRef = useRef<L.Rectangle | null>(null);
  const fieldRef = useRef<Field | null>(null);
  const unitRef = useRef<string>("");
  const callbacksRef = useRef({ onBoxDrawn, onViewChange });
  const [hover, setHover] = useState<string>("");

  fieldRef.current = field;
  unitRef.current = scale.unit;
  callbacksRef.current = { onBoxDrawn, onViewChange };

  // Create the map, the grid cells and the arrows once.
  useEffect(() => {
    if (divRef.current === null) {
      return;
    }
    const renderer = L.canvas({ padding: 0.2 });
    const map = L.map(divRef.current, { zoomSnap: 0.25, renderer: renderer, attributionControl: true });
    map.fitBounds(GRID_BOUNDS);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "© OpenStreetMap contributors",
      maxZoom: 10,
    }).addTo(map);

    const cells: L.Rectangle[][] = [];
    for (let i = 0; i < N_LAT; i++) {
      const row: L.Rectangle[] = [];
      for (let j = 0; j < N_LON; j++) {
        const lat = cellLat(i);
        const lon = cellLon(j);
        const rect = L.rectangle(
          [
            [lat - 0.5, lon - 0.5],
            [lat + 0.5, lon + 0.5],
          ],
          { stroke: false, fillOpacity: 0, interactive: false, renderer: renderer },
        );
        rect.addTo(map);
        row.push(rect);
      }
      cells.push(row);
    }
    cellsRef.current = cells;

    const arrows: L.Polyline[] = [];
    for (let i = 0; i < N_LAT; i += ARROW_STEP) {
      for (let j = 0; j < N_LON; j += ARROW_STEP) {
        const line = L.polyline([], { color: "#222", weight: 1.2, opacity: 0.8, interactive: false, renderer: renderer });
        line.addTo(map);
        arrows.push(line);
      }
    }
    arrowsRef.current = arrows;

    map.on("mousemove", (event: L.LeafletMouseEvent) => {
      const cell = cellAt(event.latlng.lat, event.latlng.lng);
      const current = fieldRef.current;
      if (cell === null || current === null) {
        setHover("");
        return;
      }
      const [i, j] = cell;
      setHover(`${cellLat(i)}°N ${cellLon(j)}°E: ${current[i][j].toFixed(2)} ${unitRef.current}`);
    });
    map.on("mouseout", () => setHover(""));
    map.on("moveend", () => {
      const callback = callbacksRef.current.onViewChange;
      if (callback) {
        const center = map.getCenter();
        callback({ lat: center.lat, lon: center.lng, zoom: map.getZoom() });
      }
    });

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Recolour the cells when the field or the scale changes.
  useEffect(() => {
    const cells = cellsRef.current;
    for (let i = 0; i < N_LAT; i++) {
      for (let j = 0; j < N_LON; j++) {
        let color: string | null = null;
        if (field !== null) {
          color = colorFor(field[i][j], scale);
        }
        if (color === null) {
          cells[i][j].setStyle({ fillOpacity: 0 });
        } else {
          cells[i][j].setStyle({ fillColor: color, fillOpacity: 0.72 });
        }
      }
    }
  }, [field, scale]);

  // Move the wind arrows.
  useEffect(() => {
    const arrows = arrowsRef.current;
    let k = 0;
    for (let i = 0; i < N_LAT; i += ARROW_STEP) {
      for (let j = 0; j < N_LON; j += ARROW_STEP) {
        if (wind) {
          arrows[k].setLatLngs(arrowLines(cellLat(i), cellLon(j), wind.u[i][j], wind.v[i][j]));
        } else {
          arrows[k].setLatLngs([]);
        }
        k += 1;
      }
    }
  }, [wind]);

  // Show the shock box.
  useEffect(() => {
    const map = mapRef.current;
    if (map === null) {
      return;
    }
    if (boxRef.current !== null) {
      boxRef.current.remove();
      boxRef.current = null;
    }
    if (box) {
      boxRef.current = L.rectangle(boxBounds(box), {
        color: "#000",
        weight: 2,
        dashArray: "6 4",
        fill: false,
        interactive: false,
      }).addTo(map);
    }
  }, [box]);

  // Follow the view of another map (keeps the multiverse maps in step).
  useEffect(() => {
    const map = mapRef.current;
    if (map === null || !view) {
      return;
    }
    const center = map.getCenter();
    const same =
      Math.abs(center.lat - view.lat) < 1e-4 && Math.abs(center.lng - view.lon) < 1e-4 && map.getZoom() === view.zoom;
    if (!same) {
      map.setView([view.lat, view.lon], view.zoom, { animate: false });
    }
  }, [view]);

  // Draw a box: press, drag, release. The box snaps to whole degrees (grid cell edges).
  useEffect(() => {
    const map = mapRef.current;
    if (map === null || !drawing) {
      return;
    }
    map.dragging.disable();
    map.getContainer().style.cursor = "crosshair";
    let start: L.LatLng | null = null;
    let preview: L.Rectangle | null = null;

    function snapBox(a: L.LatLng, b: L.LatLng): Box {
      let latMin = Math.floor(Math.min(a.lat, b.lat));
      let latMax = Math.ceil(Math.max(a.lat, b.lat));
      let lonMin = Math.floor(Math.min(a.lng, b.lng));
      let lonMax = Math.ceil(Math.max(a.lng, b.lng));
      if (latMin < 6) latMin = 6;
      if (latMax > 38) latMax = 38;
      if (lonMin < 65) lonMin = 65;
      if (lonMax > 101) lonMax = 101;
      return { latMin, latMax, lonMin, lonMax };
    }

    function onDown(event: L.LeafletMouseEvent) {
      start = event.latlng;
    }
    function onMove(event: L.LeafletMouseEvent) {
      if (start === null || map === null) {
        return;
      }
      const snapped = snapBox(start, event.latlng);
      if (preview === null) {
        preview = L.rectangle(boxBounds(snapped), { color: "#d00", weight: 2, fillOpacity: 0.1, interactive: false });
        preview.addTo(map);
      } else {
        preview.setBounds(L.latLngBounds(boxBounds(snapped) as L.LatLngBoundsLiteral));
      }
    }
    function onUp(event: L.LeafletMouseEvent) {
      if (start === null) {
        return;
      }
      const snapped = snapBox(start, event.latlng);
      start = null;
      if (preview !== null) {
        preview.remove();
        preview = null;
      }
      const callback = callbacksRef.current.onBoxDrawn;
      if (callback && snapped.latMax > snapped.latMin && snapped.lonMax > snapped.lonMin) {
        callback(snapped);
      }
    }

    map.on("mousedown", onDown);
    map.on("mousemove", onMove);
    map.on("mouseup", onUp);
    return () => {
      map.off("mousedown", onDown);
      map.off("mousemove", onMove);
      map.off("mouseup", onUp);
      if (preview !== null) {
        preview.remove();
      }
      map.dragging.enable();
      map.getContainer().style.cursor = "";
    };
  }, [drawing]);

  return (
    <div className="map-card">
      <div className="map-title">{title}</div>
      <div ref={divRef} className="map" style={{ height: height }} />
      <div className="map-hover">{hover || " "}</div>
    </div>
  );
}
