// Panel system — floating data panels driven by hand gestures.
// Panels store positions in canvas pixel coords (before mirror transform).
// Cursor positions from HandGestureState.cursorPos are [0,1] normalized, mirrored.

import type { HandGestureState, TwoHandState } from './gestureEngine';

export type PanelContent = 'telemetry' | 'gesture' | 'kinematics' | 'signals' | 'draw';

export interface Panel {
    id: string;
    pos: { x: number; y: number };
    size: { w: number; h: number };
    content: PanelContent;
    spawnAnim: number;      // 0 → 1 (ramps up over SPAWN_FRAMES frames)
    dismissVelX: number;    // px/frame dismiss animation
    dismissing: boolean;
}

export interface LiveData {
    fps: number;
    poseConf: number;
    leftHandConf: number;
    rightHandConf: number;
    leftGesture: HandGestureState | null;
    rightGesture: HandGestureState | null;
    twoHand: TwoHandState;
    telemetry: any;         // RIG_SOLVED from server
    landmarkCounts: { pose: number; leftHand: number; rightHand: number; face: number };
    skeletonScale: number;  // zoom factor driven by zoom_skeleton
    drawInkCount: number;
}

// Content type color coding
const CONTENT_COLOR: Record<PanelContent, string> = {
    telemetry: 'rgba(0,200,100,0.55)',
    gesture:   'rgba(0,180,255,0.55)',
    kinematics:'rgba(255,180,0,0.55)',
    signals:   'rgba(180,0,255,0.55)',
    draw:      'rgba(255,90,90,0.55)',
};

const CONTENT_ORDER: PanelContent[] = ['telemetry', 'gesture', 'kinematics', 'signals', 'draw'];
const PANEL_W = 240, PANEL_H = 200;
const TITLE_H = 22;
const SPAWN_FRAMES = 14;
const FONT_MONO = '8px "GeistMono", "Geist Mono", monospace';
const FONT_TITLE = '9px "GeistMono", "Geist Mono", monospace';

let _nextContentIdx = 0;
let _panelSeq = 0;

export class PanelManager {
    panels: Panel[] = [];
    private dragging: { panelId: string; offX: number; offY: number } | null = null;
    private resizingId: string | null = null;
    private resizeBaseSize = { w: 0, h: 0 };
    private resizeBaseSpan = 0;

    spawnPanels(count: number, normX: number, normY: number, cw: number, ch: number): void {
        const clampedCount = Math.max(1, Math.min(5, count));
        const totalW = clampedCount * (PANEL_W + 12) - 12;
        let startX = normX * cw - totalW / 2;
        const y = normY * ch - PANEL_H / 2;

        for (let i = 0; i < clampedCount; i++) {
            const content = CONTENT_ORDER[_nextContentIdx % CONTENT_ORDER.length];
            _nextContentIdx++;
            const x = Math.max(4, Math.min(cw - PANEL_W - 4, startX + i * (PANEL_W + 12)));
            this.panels.push({
                id: `p${_panelSeq++}`,
                pos: { x, y: Math.max(4, Math.min(ch - PANEL_H - 4, y)) },
                size: { w: PANEL_W, h: PANEL_H },
                content,
                spawnAnim: 0,
                dismissVelX: 0,
                dismissing: false,
            });
        }
    }

    // Returns screen-pixel cursor position from normalized [0,1] coords
    toPx(normPos: [number, number], cw: number, ch: number): [number, number] {
        return [normPos[0] * cw, normPos[1] * ch];
    }

    hitTestTitle(px: number, py: number): Panel | null {
        for (let i = this.panels.length - 1; i >= 0; i--) {
            const p = this.panels[i];
            if (!p.dismissing &&
                px >= p.pos.x && px <= p.pos.x + p.size.w &&
                py >= p.pos.y && py <= p.pos.y + TITLE_H) {
                return p;
            }
        }
        return null;
    }

    hitTestBody(px: number, py: number): Panel | null {
        for (let i = this.panels.length - 1; i >= 0; i--) {
            const p = this.panels[i];
            if (!p.dismissing &&
                px >= p.pos.x && px <= p.pos.x + p.size.w &&
                py >= p.pos.y && py <= p.pos.y + p.size.h) {
                return p;
            }
        }
        return null;
    }

    nearest(px: number, py: number): Panel | null {
        let best: Panel | null = null, bestD = Infinity;
        for (const p of this.panels) {
            if (p.dismissing) continue;
            const cx = p.pos.x + p.size.w / 2, cy = p.pos.y + p.size.h / 2;
            const d = (cx - px) ** 2 + (cy - py) ** 2;
            if (d < bestD) { bestD = d; best = p; }
        }
        return best;
    }

    startDrag(panelId: string, px: number, py: number): void {
        const p = this.panels.find(p => p.id === panelId);
        if (!p) return;
        this.dragging = { panelId, offX: px - p.pos.x, offY: py - p.pos.y };
        // Bring to front
        const idx = this.panels.indexOf(p);
        this.panels.splice(idx, 1);
        this.panels.push(p);
    }

    updateDrag(px: number, py: number, cw: number, ch: number): void {
        if (!this.dragging) return;
        const p = this.panels.find(p => p.id === this.dragging!.panelId);
        if (!p) return;
        p.pos.x = Math.max(0, Math.min(cw - p.size.w, px - this.dragging.offX));
        p.pos.y = Math.max(0, Math.min(ch - p.size.h, py - this.dragging.offY));
    }

    endDrag(): void {
        this.dragging = null;
    }

    startResize(panelId: string, spanDist: number): void {
        const p = this.panels.find(p => p.id === panelId);
        if (!p) return;
        this.resizingId = panelId;
        this.resizeBaseSize = { w: p.size.w, h: p.size.h };
        this.resizeBaseSpan = spanDist;
    }

    updateResize(spanDist: number): void {
        if (!this.resizingId || this.resizeBaseSpan === 0) return;
        const p = this.panels.find(p => p.id === this.resizingId);
        if (!p) return;
        const ratio = spanDist / this.resizeBaseSpan;
        p.size.w = Math.max(120, Math.min(600, this.resizeBaseSize.w * ratio));
        p.size.h = Math.max(80, Math.min(400, this.resizeBaseSize.h * ratio));
    }

    endResize(): void {
        this.resizingId = null;
    }

    dismiss(panelId: string, directionX: number): void {
        const p = this.panels.find(p => p.id === panelId);
        if (!p || p.dismissing) return;
        p.dismissing = true;
        p.dismissVelX = directionX > 0 ? 40 : -40;
    }

    tick(cw: number): void {
        for (const p of this.panels) {
            if (p.spawnAnim < 1) p.spawnAnim = Math.min(1, p.spawnAnim + 1 / SPAWN_FRAMES);
            if (p.dismissing) p.pos.x += p.dismissVelX;
        }
        this.panels = this.panels.filter(p => !p.dismissing || (p.pos.x > -p.size.w - 10 && p.pos.x < cw + p.size.w + 10));
    }

    get isDragging(): boolean { return this.dragging !== null; }
    get draggingPanelId(): string | null { return this.dragging?.panelId ?? null; }
    get resizingPanelId(): string | null { return this.resizingId; }

    render(ctx: CanvasRenderingContext2D, data: LiveData): void {
        for (const panel of this.panels) {
            ctx.save();

            // Spawn scale animation from panel center
            const scale = this.easeOut(panel.spawnAnim);
            const cx = panel.pos.x + panel.size.w / 2;
            const cy = panel.pos.y + panel.size.h / 2;
            ctx.translate(cx, cy);
            ctx.scale(scale, scale);
            ctx.translate(-cx, -cy);

            const { pos: { x, y }, size: { w, h } } = panel;
            const color = CONTENT_COLOR[panel.content];

            // Background
            ctx.fillStyle = 'rgba(0,0,0,0.80)';
            ctx.fillRect(x, y, w, h);

            // Border
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

            // Title bar
            ctx.fillStyle = color.replace('0.55)', '0.25)');
            ctx.fillRect(x, y, w, TITLE_H);

            ctx.fillStyle = color.replace('rgba(', 'rgb(').replace(/,[\d.]+\)/, ')');
            ctx.font = FONT_TITLE;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(panel.content.toUpperCase(), x + 6, y + TITLE_H / 2);

            // Content
            ctx.fillStyle = 'rgba(255,255,255,0.75)';
            ctx.font = FONT_MONO;
            this.renderContent(ctx, panel, data, x + 6, y + TITLE_H + 6, w - 12, h - TITLE_H - 10);

            ctx.restore();
        }
    }

    private easeOut(t: number): number {
        return 1 - (1 - t) ** 3;
    }

    private renderContent(
        ctx: CanvasRenderingContext2D,
        panel: Panel,
        d: LiveData,
        x: number, y: number, w: number, _h: number
    ): void {
        const line = (txt: string, row: number) => {
            ctx.fillText(txt, x, y + row * 11, w);
        };
        const bar = (label: string, val: number, row: number, colorStr: string) => {
            const bx = x, by = y + row * 11 + 3;
            const bw = (w - 2) * Math.max(0, Math.min(1, val));
            ctx.fillStyle = 'rgba(255,255,255,0.12)';
            ctx.fillRect(bx, by - 4, w - 2, 7);
            ctx.fillStyle = colorStr;
            ctx.fillRect(bx, by - 4, bw, 7);
            ctx.fillStyle = 'rgba(255,255,255,0.75)';
            ctx.fillText(`${label}: ${(val * 100).toFixed(0)}%`, bx + 2, by - 3, w - 4);
        };

        ctx.fillStyle = 'rgba(255,255,255,0.75)';

        switch (panel.content) {
            case 'telemetry': {
                const t = d.telemetry;
                line(`FPS: ${d.fps}`, 0);
                line(`POSE_CONF: ${(d.poseConf * 100).toFixed(0)}%`, 1);
                line(`L_HAND_CONF: ${(d.leftHandConf * 100).toFixed(0)}%`, 2);
                line(`R_HAND_CONF: ${(d.rightHandConf * 100).toFixed(0)}%`, 3);
                line(`POSE_LMS: ${d.landmarkCounts.pose}`, 4);
                line(`LH_LMS: ${d.landmarkCounts.leftHand}`, 5);
                line(`RH_LMS: ${d.landmarkCounts.rightHand}`, 6);
                line(`FACE_LMS: ${d.landmarkCounts.face}`, 7);
                if (t?.pose) {
                    line(`HEAD_P: ${t.pose.pitch}  Y: ${t.pose.yaw}  R: ${t.pose.roll}`, 8);
                }
                if (t?.gaze) {
                    line(`GAZE: L${t.gaze.l}  R${t.gaze.r}`, 9);
                }
                break;
            }

            case 'gesture': {
                const lg = d.leftGesture, rg = d.rightGesture;
                line(`L: ${lg?.gesture ?? 'none'}`, 0);
                line(`  FINGERS: ${lg?.fingerCount ?? 0}`, 1);
                line(`  PINCH_D: ${lg?.pinchDist.toFixed(3) ?? '---'}`, 2);
                line(`  POKE_CHG: ${((lg?.pokeCharge ?? 0) * 100).toFixed(0)}%`, 3);
                line(`  CURSOR: ${lg?.cursorPos[0].toFixed(2)},${lg?.cursorPos[1].toFixed(2)}`, 4);
                line(`R: ${rg?.gesture ?? 'none'}`, 5);
                line(`  FINGERS: ${rg?.fingerCount ?? 0}`, 6);
                line(`  PINCH_D: ${rg?.pinchDist.toFixed(3) ?? '---'}`, 7);
                line(`  POKE_CHG: ${((rg?.pokeCharge ?? 0) * 100).toFixed(0)}%`, 8);
                line(`TWO_HAND: ${d.twoHand.mode}`, 9);
                if (d.twoHand.mode !== 'none') {
                    line(`  SPAN_DELTA: ${d.twoHand.deltaSpan.toFixed(3)}`, 10);
                }
                break;
            }

            case 'kinematics': {
                const t = d.telemetry;
                line(`HEAD_POSE:`, 0);
                line(`  PITCH: ${t?.pose?.pitch ?? '--'}`, 1);
                line(`  YAW:   ${t?.pose?.yaw ?? '--'}`, 2);
                line(`  ROLL:  ${t?.pose?.roll ?? '--'}`, 3);
                line(`MUSCLE:`, 4);
                line(`  ZYGO: ${t?.face?.zygomatic ?? '--'}`, 5);
                line(`  CORR: ${t?.face?.corrugator ?? '--'}`, 6);
                line(`  ORBI: ${t?.face?.orbicularis ?? '--'}`, 7);
                const lk = t?.hands?.left, rk = t?.hands?.right;
                if (lk) {
                    line(`L_WRIST_ANG: ${lk.wristAngle}°`, 8);
                    line(`L_TIP_VEL: ${lk.tipVelocities?.map((v: number) => v.toFixed(2)).join(' ') ?? '--'}`, 9);
                }
                if (rk) {
                    line(`R_WRIST_ANG: ${rk.wristAngle}°`, 10);
                }
                break;
            }

            case 'signals': {
                const t = d.telemetry;
                const lk = t?.hands?.left;
                const rk = t?.hands?.right;
                const barH = 11;
                ctx.textBaseline = 'middle';

                // L/R tip velocity bars
                const tips = ['IDX', 'MID', 'RNG', 'PNK'];
                for (let i = 0; i < 4; i++) {
                    const lv = (lk?.tipVelocities?.[i] ?? 0) / 0.05; // normalize to 5cm/s
                    const rv = (rk?.tipVelocities?.[i] ?? 0) / 0.05;
                    bar(`L.${tips[i]}`, lv, i * 2, 'rgba(0,180,255,0.7)');
                    bar(`R.${tips[i]}`, rv, i * 2 + 1, 'rgba(255,100,0,0.7)');
                }
                // Head pose bars
                if (t?.pose) {
                    const pn = Math.min(1, Math.abs(parseFloat(t.pose.pitch)) / 30);
                    const yn = Math.min(1, Math.abs(parseFloat(t.pose.yaw)) / 30);
                    bar('PITCH', pn, 8, 'rgba(0,255,120,0.7)');
                    bar('YAW', yn, 9, 'rgba(255,220,0,0.7)');
                }
                break;
            }

            case 'draw': {
                line('DRAW MODE ACTIVE', 0);
                line('INDEX ONLY = INK', 1);
                line('FIST = CLEAR', 2);
                line(`INK POINTS: ${d.drawInkCount}`, 4);
                line(`SCALE: ${d.skeletonScale.toFixed(2)}x`, 5);
                line('', 6);
                line('TWO HAND OPEN = ZOOM', 7);
                line('SWIPE = DISMISS PANEL', 8);
                line('POKE N FINGERS = SPAWN', 9);
                break;
            }
        }
    }
}
