// Tracking renderer — all canvas draw calls.
// All rendering is in MediaPipe landmark space (un-mirrored x [0,1]),
// then the calling context flips the canvas. Cursor positions come in already mirrored.

import type { HandGestureState, TwoHandState, NormalizedLandmark } from './gestureEngine';

// Skeleton connection topology (same as BodyTrackEngine V14.0)
const POSE_C = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const HAND_C = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const FACE_C = {
    lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 61],
    lEye: [33, 160, 158, 133, 153, 144, 33],
    rEye: [263, 387, 385, 362, 380, 373, 263],
    brows: [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    oval: [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
};

// Landmark confidence → color
function landmarkColor(conf: number): string {
    if (conf >= 0.85) return 'rgb(0,255,120)';
    if (conf >= 0.5)  return 'rgb(255,220,0)';
    return 'rgb(255,60,0)';
}

function confOf(lm: NormalizedLandmark): number {
    return lm.visibility ?? lm.presence ?? 1;
}

export class TrackingRenderer {
    private frameCount = 0;

    tick(): void { this.frameCount++; }

    // ── Skeleton ────────────────────────────────────────────────────────────

    renderSkeleton(
        ctx: CanvasRenderingContext2D,
        cw: number, ch: number,
        pose: NormalizedLandmark[] | null,
        leftHand: NormalizedLandmark[] | null,
        rightHand: NormalizedLandmark[] | null,
        face: NormalizedLandmark[] | null,
        skeletonScale = 1
    ): void {
        ctx.clearRect(0, 0, cw, ch);

        // Scale around center (for zoom_skeleton gesture)
        const ox = cw / 2, oy = ch / 2;
        ctx.save();
        ctx.translate(ox, oy);
        ctx.scale(skeletonScale, skeletonScale);
        ctx.translate(-ox, -oy);

        const px = (lm: NormalizedLandmark) => lm.x * cw;
        const py = (lm: NormalizedLandmark) => lm.y * ch;

        const bone = (
            lms: NormalizedLandmark[],
            a: number, b: number,
            baseAlpha: number,
            width: number
        ) => {
            const la = lms[a], lb = lms[b];
            if (!la || !lb) return;
            const confA = confOf(la), confB = confOf(lb);
            if (confA < 0.25 && confB < 0.25) return;
            const weakerConf = Math.min(confA, confB);
            const col = landmarkColor(weakerConf);
            const alpha = baseAlpha * Math.max(0.3, weakerConf);
            ctx.beginPath();
            ctx.strokeStyle = col.replace('rgb(', 'rgba(').replace(')', `, ${alpha})`);
            ctx.lineWidth = width;
            ctx.moveTo(px(la), py(la));
            ctx.lineTo(px(lb), py(lb));
            ctx.stroke();
        };

        // Pose
        if (pose) {
            POSE_C.forEach(([a, b]) => bone(pose, a, b, 0.75, 2.0));
            // Dot each pose landmark
            for (const lm of pose) {
                const c = confOf(lm);
                if (c < 0.25) continue;
                ctx.beginPath();
                ctx.fillStyle = landmarkColor(c).replace('rgb(', 'rgba(').replace(')', `, ${Math.max(0.3, c)}`);
                ctx.arc(px(lm), py(lm), 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Hands
        [leftHand, rightHand].forEach(hand => {
            if (!hand) return;
            HAND_C.forEach(([a, b]) => bone(hand, a, b, 0.85, 1.5));
            // Highlight fingertips
            [4, 8, 12, 16, 20].forEach(i => {
                const lm = hand[i];
                if (!lm || confOf(lm) < 0.3) return;
                ctx.beginPath();
                ctx.fillStyle = landmarkColor(confOf(lm));
                ctx.arc(px(lm), py(lm), 3, 0, Math.PI * 2);
                ctx.fill();
            });
        });

        // Face
        if (face) {
            ctx.lineWidth = 0.5;
            Object.values(FACE_C).forEach(indices => {
                // Confidence of the group (rough: use first landmark)
                const conf = confOf(face[indices[0]] ?? { x:0,y:0,z:0 });
                const alpha = Math.max(0.08, conf * 0.25);
                ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
                ctx.beginPath();
                indices.forEach((idx, i) => {
                    const lm = face[idx];
                    if (!lm) return;
                    if (i === 0) ctx.moveTo(px(lm), py(lm));
                    else ctx.lineTo(px(lm), py(lm));
                });
                ctx.stroke();
            });
        }

        ctx.restore();
    }

    // ── Cursors & gesture rings ─────────────────────────────────────────────

    renderHandOverlay(
        ctx: CanvasRenderingContext2D,
        cw: number, ch: number,
        leftHand: NormalizedLandmark[] | null,
        rightHand: NormalizedLandmark[] | null,
        leftState: HandGestureState | null,
        rightState: HandGestureState | null,
        leftCharge: number,
        rightCharge: number,
        twoHand: TwoHandState
    ): void {
        ctx.clearRect(0, 0, cw, ch);

        // Two-hand connector line
        if (twoHand.mode !== 'none' && leftState && rightState) {
            const lx = leftState.cursorPos[0] * cw, ly = leftState.cursorPos[1] * ch;
            const rx = rightState.cursorPos[0] * cw, ry = rightState.cursorPos[1] * ch;
            ctx.beginPath();
            ctx.strokeStyle = twoHand.mode === 'resize' ? 'rgba(255,220,0,0.6)' : 'rgba(0,255,180,0.6)';
            ctx.lineWidth = 1;
            ctx.moveTo(lx, ly);
            ctx.lineTo(rx, ry);
            ctx.stroke();
            // Midpoint label
            ctx.fillStyle = 'rgba(255,255,255,0.5)';
            ctx.font = '7px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(
                twoHand.mode === 'resize'
                    ? `RZ ${twoHand.deltaSpan > 0 ? '+' : ''}${twoHand.deltaSpan.toFixed(3)}`
                    : twoHand.mode.toUpperCase(),
                (lx + rx) / 2, (ly + ry) / 2 - 6
            );
        }

        // Per-hand rendering
        const hands: Array<[NormalizedLandmark[] | null, HandGestureState | null, number]> = [
            [leftHand, leftState, leftCharge],
            [rightHand, rightState, rightCharge],
        ];

        for (const [hand, state, charge] of hands) {
            if (!state) continue;
            const cx = state.cursorPos[0] * cw;
            const cy = state.cursorPos[1] * ch;

            // Poke charge arc (gold)
            if (charge > 0.02) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(255,200,0,${charge * 0.8})`;
                ctx.lineWidth = 2;
                ctx.arc(cx, cy, 16, -Math.PI / 2, -Math.PI / 2 + charge * Math.PI * 2);
                ctx.stroke();
            }

            // Gesture-colored cursor
            const [r, g, b] = this.gestureRGB(state.gesture);
            const pulse = Math.sin(this.frameCount * 0.15) * 2 + 6;

            // Outer ring
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${r},${g},${b},0.4)`;
            ctx.lineWidth = 1;
            ctx.arc(cx, cy, pulse, 0, Math.PI * 2);
            ctx.stroke();

            // Inner dot
            ctx.beginPath();
            ctx.fillStyle = `rgba(${r},${g},${b},0.9)`;
            ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
            ctx.fill();

            // Gesture label
            ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
            ctx.font = '7px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(state.gesture.toUpperCase(), cx, cy - 14);

            // Pinch line: thumb tip to index tip.
            // UI canvas has no CSS flip, so x must be mirrored manually: (1 - lm.x) * cw
            if (state.gesture === 'pinch' && hand) {
                const thumb = hand[4], index = hand[8];
                if (thumb && index) {
                    const tx = (1 - thumb.x) * cw, ty = thumb.y * ch;
                    const ix = (1 - index.x) * cw, iy = index.y * ch;
                    ctx.beginPath();
                    ctx.strokeStyle = 'rgba(0,220,255,0.7)';
                    ctx.lineWidth = 1;
                    ctx.moveTo(tx, ty);
                    ctx.lineTo(ix, iy);
                    ctx.stroke();
                    ctx.fillStyle = 'rgba(0,220,255,0.8)';
                    ctx.fillText(
                        `${(state.pinchDist * 100).toFixed(1)}cm`,
                        (tx + ix) / 2, (ty + iy) / 2 - 8
                    );
                }
            }

            // Swipe indicator
            if (state.gesture === 'swipe') {
                const arrow = state.lateralVelocity > 0 ? '→' : '←';
                ctx.fillStyle = 'rgba(255,100,100,0.9)';
                ctx.font = '14px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(arrow, cx, cy - 20);
            }
        }
    }

    // Draw stroke on the draw canvas (called incrementally — no clear)
    renderDrawPoint(
        ctx: CanvasRenderingContext2D,
        cw: number, ch: number,
        points: Array<{ x: number; y: number }>,
        velocity: number
    ): void {
        if (points.length < 2) return;
        const last = points[points.length - 1];
        const prev = points[points.length - 2];
        const lightness = Math.min(80, 40 + velocity * 40);
        ctx.beginPath();
        ctx.strokeStyle = `hsl(270, 100%, ${lightness}%)`;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (points.length >= 3) {
            const pp = points[points.length - 3];
            const cpx = (prev.x + last.x) / 2 * cw;
            const cpy = (prev.y + last.y) / 2 * ch;
            ctx.moveTo(pp.x * cw, pp.y * ch);
            ctx.quadraticCurveTo(prev.x * cw, prev.y * ch, cpx, cpy);
        } else {
            ctx.moveTo(prev.x * cw, prev.y * ch);
            ctx.lineTo(last.x * cw, last.y * ch);
        }
        ctx.stroke();
    }

    renderPokeFlash(ctx: CanvasRenderingContext2D, cw: number, ch: number, normX: number, normY: number, frame: number): void {
        const decay = 1 - frame / 10;
        if (decay <= 0) return;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(255,255,255,${decay * 0.8})`;
        ctx.lineWidth = 2;
        ctx.arc(normX * cw, normY * ch, 10 + frame * 8, 0, Math.PI * 2);
        ctx.stroke();
    }

    private gestureRGB(g: string): [number, number, number] {
        switch (g) {
            case 'pinch':     return [0, 220, 255];
            case 'grab':      return [255, 140, 0];
            case 'draw':      return [180, 0, 255];
            case 'swipe':     return [255, 80, 80];
            case 'palm_open': return [0, 255, 180];
            case 'poke':      return [255, 200, 0];
            case 'point':
            default:          return [200, 200, 200];
        }
    }
}
