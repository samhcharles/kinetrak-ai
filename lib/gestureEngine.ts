// Gesture engine — all gesture state machines and detectors.
// Uses both image-space (normalized [0,1]) and world-space (metric) landmarks.

export type GestureType =
    | 'idle'
    | 'point'       // index extended → cursor
    | 'pinch'       // thumb+index close → drag
    | 'grab'        // all fingers closed → panel grab / draw clear
    | 'draw'        // index only extended → ink trail
    | 'palm_open'   // all 4 fingers open → freeze / zoom signal
    | 'wrist_roll'  // rotating wrist → dial scrub
    | 'swipe'       // fast lateral → dismiss panel
    | 'poke';       // z-thrust with N fingers → spawn N panels

export interface NormalizedLandmark { x: number; y: number; z: number; visibility?: number; presence?: number; }
export interface Landmark { x: number; y: number; z: number; }

export interface HandGestureState {
    gesture: GestureType;
    pinchDist: number;
    cursorPos: [number, number];    // [0,1] normalized, already mirrored (1-x)
    fingerCount: number;
    wristRotation: number;          // cumulative delta radians
    lateralVelocity: number;        // normalized x/s
    pokeCharge: number;             // 0-1, shows thrust buildup
    confidence: number;
    framesSinceChange: number;
}

export interface PokeEvent {
    fingerCount: number;
    handedness: string;
    normPos: [number, number];      // [0,1] screen-mirrored position of hand center
}

export interface TwoHandState {
    mode: 'none' | 'resize' | 'zoom_skeleton' | 'calibrate_reset';
    midpoint: [number, number];     // [0,1] normalized
    spanDist: number;
    deltaSpan: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const FINGER_TIP = [8, 12, 16, 20];
const FINGER_MCP = [5, 9, 13, 17];
const FINGER_PIP = [6, 10, 14, 18];

function dist2D(a: NormalizedLandmark, b: NormalizedLandmark): number {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function dist3D(a: Landmark, b: Landmark): number {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function dot3(ax: number, ay: number, az: number, bx: number, by: number, bz: number): number {
    return ax * bx + ay * by + az * bz;
}

function countExtendedFingers(img: NormalizedLandmark[]): number {
    let n = 0;
    for (let i = 0; i < 4; i++) {
        // Extended: tip not significantly below PIP
        if (img[FINGER_TIP[i]].y < img[FINGER_PIP[i]].y + 0.03) n++;
    }
    return n;
}

function computePalmArea(img: NormalizedLandmark[]): number {
    // Shoelace area of wrist + 3 MCPs as depth proxy
    const pts = [img[0], img[5], img[9], img[17]];
    let area = 0;
    for (let i = 0; i < 4; i++) {
        const j = (i + 1) % 4;
        area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    return Math.abs(area) / 2;
}

function palmNormal(world: Landmark[]): [number, number, number] {
    // Normal to palm plane from three points: wrist(0), index_mcp(5), pinky_mcp(17)
    const ax = world[5].x - world[0].x, ay = world[5].y - world[0].y, az = world[5].z - world[0].z;
    const bx = world[17].x - world[0].x, by = world[17].y - world[0].y, bz = world[17].z - world[0].z;
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const m = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    return [nx / m, ny / m, nz / m];
}

// ── GestureStateMachine (per hand) ───────────────────────────────────────────

export class GestureStateMachine {
    private current: GestureType = 'idle';
    private candidate: GestureType = 'idle';
    private candidateFrames = 0;
    private exitFrames = 0;
    private framesSinceChange = 0;

    // Hysteresis counts
    private static ON_FRAMES: Partial<Record<GestureType, number>> = {
        pinch: 3, grab: 3, draw: 3, palm_open: 3, swipe: 2, poke: 1,
    };
    private static OFF_FRAMES: Partial<Record<GestureType, number>> = {
        pinch: 5, grab: 5, draw: 5, palm_open: 4, swipe: 3,
    };

    // History for wrist roll and swipe
    private wristHistory: Array<{ n: [number, number, number]; t: number }> = [];
    private lateralHistory: Array<{ x: number; t: number }> = [];
    private lastLateralX = 0;
    private cumulativeRotation = 0;
    private lastNormal: [number, number, number] = [0, 0, 1];

    update(img: NormalizedLandmark[], world: Landmark[], dt: number, nowMs: number): HandGestureState {
        const pinchDist = dist3D(world[4], world[8]);

        // --- Candidate detection ---
        let candidate: GestureType = 'idle';

        // Pinch
        if (pinchDist < 0.07) candidate = 'pinch';

        // Grab (3+ fingers curled below MCP)
        if (candidate === 'idle') {
            let curled = 0;
            for (let i = 0; i < 4; i++) {
                if (img[FINGER_TIP[i]].y > img[FINGER_MCP[i]].y + 0.02) curled++;
            }
            if (curled >= 3) candidate = 'grab';
        }

        // Draw: index extended, middle/ring/pinky closed
        if (candidate === 'idle') {
            const indexUp = img[8].y < img[6].y - 0.03;
            const midClosed = img[12].y > img[10].y + 0.02;
            const ringClosed = img[16].y > img[14].y + 0.02;
            const pinkyClosed = img[20].y > img[18].y + 0.02;
            if (indexUp && midClosed && ringClosed && pinkyClosed) candidate = 'draw';
        }

        // Point: index extended, not draw (middle not strictly closed)
        if (candidate === 'idle') {
            const indexUp = img[8].y < img[6].y - 0.02;
            if (indexUp) candidate = 'point';
        }

        // Palm open: all 4 fingers extended
        if (candidate === 'idle') {
            let extended = 0;
            for (let i = 0; i < 4; i++) {
                if (img[FINGER_TIP[i]].y < img[FINGER_PIP[i]].y + 0.03) extended++;
            }
            if (extended === 4) candidate = 'palm_open';
        }

        // Swipe: fast lateral wrist motion
        const wristX = img[0].x;
        this.lateralHistory.push({ x: wristX, t: nowMs });
        this.lateralHistory = this.lateralHistory.filter(h => nowMs - h.t < 200);
        let lateralVelocity = 0;
        if (this.lateralHistory.length >= 2) {
            const first = this.lateralHistory[0], last = this.lateralHistory[this.lateralHistory.length - 1];
            const dT = (last.t - first.t) / 1000 || dt;
            lateralVelocity = (last.x - first.x) / dT;
        }
        if (candidate === 'idle' && Math.abs(lateralVelocity) > 0.8) candidate = 'swipe';

        // Wrist roll: track palm normal over time
        const normal = palmNormal(world);
        const dot = dot3(normal[0], normal[1], normal[2], this.lastNormal[0], this.lastNormal[1], this.lastNormal[2]);
        const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
        this.cumulativeRotation += angle;
        this.lastNormal = normal;
        this.wristHistory.push({ n: normal, t: nowMs });
        this.wristHistory = this.wristHistory.filter(h => nowMs - h.t < 500);

        // --- Hysteresis state machine ---
        const onRequired = GestureStateMachine.ON_FRAMES[candidate] ?? 3;
        const offRequired = GestureStateMachine.OFF_FRAMES[this.current] ?? 4;

        if (candidate !== this.current) {
            if (candidate === this.candidate) {
                this.candidateFrames++;
            } else {
                this.candidate = candidate;
                this.candidateFrames = 1;
            }

            if (this.candidateFrames >= onRequired) {
                this.current = candidate;
                this.candidateFrames = 0;
                this.exitFrames = 0;
                this.framesSinceChange = 0;
                this.cumulativeRotation = 0;
            }
        } else {
            this.candidateFrames = 0;
            this.framesSinceChange++;
        }

        // Pinch exit hysteresis
        if (this.current === 'pinch' && pinchDist > 0.10) {
            this.exitFrames++;
            if (this.exitFrames >= (GestureStateMachine.OFF_FRAMES.pinch ?? 5)) {
                this.current = 'idle';
                this.exitFrames = 0;
            }
        } else {
            this.exitFrames = 0;
        }

        // Cursor: index tip, mirrored x
        const cursorPos: [number, number] = [1 - img[8].x, img[8].y];

        const confidence = Math.min(
            img[0].visibility ?? 1,
            img[8].visibility ?? 1,
            img[4].visibility ?? 1
        );

        return {
            gesture: this.current,
            pinchDist,
            cursorPos,
            fingerCount: countExtendedFingers(img),
            wristRotation: this.cumulativeRotation,
            lateralVelocity,
            pokeCharge: 0, // set by ZThrustDetector
            confidence,
            framesSinceChange: this.framesSinceChange,
        };
    }
}

// ── ZThrustDetector — panel spawn via z-thrust poke ──────────────────────────

export class ZThrustDetector {
    private sizeHistory: Array<{ size: number; t: number }> = [];
    private lastPokeT = -Infinity;
    private readonly COOLDOWN_MS = 1200;
    private readonly GROWTH_THRESH = 0.28;   // 28% palm area increase triggers poke
    private readonly WINDOW_MS = 250;

    update(
        img: NormalizedLandmark[],
        handedness: string,
        nowMs: number
    ): { poke: PokeEvent | null; charge: number } {
        const size = computePalmArea(img);
        this.sizeHistory.push({ size, t: nowMs });
        this.sizeHistory = this.sizeHistory.filter(h => nowMs - h.t < this.WINDOW_MS);

        if (this.sizeHistory.length < 3 || nowMs - this.lastPokeT < this.COOLDOWN_MS) {
            return { poke: null, charge: 0 };
        }

        const oldest = this.sizeHistory[0].size;
        const newest = size;
        const growth = oldest > 0 ? (newest - oldest) / oldest : 0;

        // Stability check: was size stable in first half before spike
        const midIdx = Math.floor(this.sizeHistory.length / 2);
        const midSize = this.sizeHistory[midIdx]?.size ?? oldest;
        const wasStable = oldest > 0 ? Math.abs((midSize - oldest) / oldest) < 0.1 : false;

        const charge = Math.max(0, Math.min(1, growth / this.GROWTH_THRESH));

        if (wasStable && growth >= this.GROWTH_THRESH) {
            this.lastPokeT = nowMs;
            this.sizeHistory = [];

            const fingerCount = Math.max(1, countExtendedFingers(img));
            // Hand center: average of all 21 landmark positions (mirrored x)
            const cx = img.reduce((s, l) => s + l.x, 0) / img.length;
            const cy = img.reduce((s, l) => s + l.y, 0) / img.length;
            const normPos: [number, number] = [1 - cx, cy];

            return {
                poke: { fingerCount, handedness, normPos },
                charge: 1,
            };
        }

        return { poke: null, charge };
    }

    cooldownRemaining(nowMs: number): number {
        return Math.max(0, this.COOLDOWN_MS - (nowMs - this.lastPokeT));
    }
}

// ── TwoHandDetector ───────────────────────────────────────────────────────────

export class TwoHandDetector {
    private resizeStart: number | null = null;
    private baseSpan = 0;
    private calibrateStart: number | null = null;
    private baseSpanZoom = 0;

    update(
        left: HandGestureState | null,
        right: HandGestureState | null
    ): TwoHandState {
        const both = left && right;

        if (!both) {
            this.resizeStart = null;
            this.calibrateStart = null;
            return { mode: 'none', midpoint: [0.5, 0.5], spanDist: 0, deltaSpan: 0 };
        }

        const lx = left.cursorPos[0], ly = left.cursorPos[1];
        const rx = right.cursorPos[0], ry = right.cursorPos[1];
        const midpoint: [number, number] = [(lx + rx) / 2, (ly + ry) / 2];
        const spanDist = Math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2);

        // Both palm_open → zoom skeleton or calibrate reset
        if (left.gesture === 'palm_open' && right.gesture === 'palm_open') {
            if (this.calibrateStart === null) {
                this.calibrateStart = performance.now();
                this.baseSpanZoom = spanDist;
            }
            const deltaSpan = spanDist - this.baseSpanZoom;
            // Hold for 1.5s with increasing span → calibrate reset
            const held = performance.now() - this.calibrateStart;
            if (held > 1500 && deltaSpan > 0.15) {
                this.calibrateStart = null;
                return { mode: 'calibrate_reset', midpoint, spanDist, deltaSpan };
            }
            return { mode: 'zoom_skeleton', midpoint, spanDist, deltaSpan };
        }
        this.calibrateStart = null;

        // Both pinching → resize panel
        if (left.gesture === 'pinch' && right.gesture === 'pinch') {
            if (this.resizeStart === null) {
                this.resizeStart = performance.now();
                this.baseSpan = spanDist;
            }
            return { mode: 'resize', midpoint, spanDist, deltaSpan: spanDist - this.baseSpan };
        }

        this.resizeStart = null;
        return { mode: 'none', midpoint, spanDist, deltaSpan: 0 };
    }
}
