import React, { useEffect, useRef, useState } from 'react';
import { HandLandmarker, PoseLandmarker, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { FilterBank } from '../lib/oneEuroFilter';
import { GestureStateMachine, ZThrustDetector, TwoHandDetector } from '../lib/gestureEngine';
import type { HandGestureState, TwoHandState } from '../lib/gestureEngine';
import { PanelManager } from '../lib/panelSystem';
import type { LiveData } from '../lib/panelSystem';
import { TrackingRenderer } from '../lib/trackingRenderer';

// ── MediaPipe model CDN paths ─────────────────────────────────────────────────
// WASM version must match the installed @mediapipe/tasks-vision npm package version.
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const HAND_M = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_M = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const FACE_M = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// ── Data packer → server-compatible Float32Array ──────────────────────────────
function packFrame(
    pose: NormalizedLandmark[] | null,
    leftHand: NormalizedLandmark[] | null,
    rightHand: NormalizedLandmark[] | null,
    face: NormalizedLandmark[] | null
): Float32Array {
    const buf = new Float32Array(553 * 4);
    const pack = (lms: NormalizedLandmark[] | null, off: number) => {
        if (!lms) return;
        lms.forEach((lm, i) => {
            const o = (off + i) * 4;
            buf[o] = lm.x; buf[o+1] = lm.y; buf[o+2] = lm.z;
            buf[o+3] = lm.visibility ?? 0;
        });
    };
    pack(pose, 0);     // landmarks 0-32:  pose (33)
    pack(leftHand, 33);// landmarks 33-53: left hand (21)
    pack(rightHand, 54);// landmarks 54-74: right hand (21)
    pack(face, 75);    // landmarks 75-552: face (478)
    return buf;
}

// ── Velocity helper ───────────────────────────────────────────────────────────
function lmVelocity(
    prev: NormalizedLandmark | null,
    curr: NormalizedLandmark | null,
    dt: number
): number {
    if (!prev || !curr || dt <= 0) return 0;
    const dx = curr.x - prev.x, dy = curr.y - prev.y;
    return Math.sqrt(dx*dx + dy*dy) / dt;
}

// ── Component ─────────────────────────────────────────────────────────────────
const KinetrackEngine: React.FC = () => {
    const [status, setStatus] = useState<'loading' | 'tracking' | 'error'>('loading');
    const [statusMsg, setStatusMsg] = useState('INIT');

    // DOM refs
    const videoRef = useRef<HTMLVideoElement>(null);
    const skRef = useRef<HTMLCanvasElement>(null);    // skeleton
    const drRef = useRef<HTMLCanvasElement>(null);    // draw
    const uiRef = useRef<HTMLCanvasElement>(null);    // panels + cursors

    // Engine refs (all initialized async, never trigger re-render)
    const handRef = useRef<HandLandmarker | null>(null);
    const poseRef = useRef<PoseLandmarker | null>(null);
    const faceRef = useRef<FaceLandmarker | null>(null);
    const wsRef   = useRef<WebSocket | null>(null);

    const lBankRef = useRef<FilterBank | null>(null);  // left hand
    const rBankRef = useRef<FilterBank | null>(null);  // right hand
    const pBankRef = useRef<FilterBank | null>(null);  // pose
    const fBankRef = useRef<FilterBank | null>(null);  // face

    const lGestRef = useRef(new GestureStateMachine());
    const rGestRef = useRef(new GestureStateMachine());
    const lPokeRef = useRef(new ZThrustDetector());
    const rPokeRef = useRef(new ZThrustDetector());
    const twoRef   = useRef(new TwoHandDetector());
    const panelRef = useRef<PanelManager | null>(null);
    const rendRef  = useRef<TrackingRenderer | null>(null);

    const rafRef       = useRef(0);
    const lastTRef     = useRef(performance.now());
    const fpsRef       = useRef(0);
    const fcRef        = useRef(0);
    const fpsLastRef   = useRef(performance.now());
    const teleRef      = useRef<any>({});
    const drawPtsRef   = useRef<Array<{ x: number; y: number }>>([]);
    const prevIndexRef = useRef<{ lx: NormalizedLandmark | null; rx: NormalizedLandmark | null }>({ lx: null, rx: null });
    const skScaleRef   = useRef(1);
    const pokeFlashRef = useRef<{ x: number; y: number; frame: number } | null>(null);

    // Gesture transition tracking for drag/dismiss
    const prevLGRef = useRef<HandGestureState | null>(null);
    const prevRGRef = useRef<HandGestureState | null>(null);
    const lChargeRef = useRef(0);
    const rChargeRef = useRef(0);
    const resizingRef = useRef(false);

    useEffect(() => {
        let running = true;

        (async () => {
            try {
                // ── Camera ────────────────────────────────────────────────
                setStatusMsg('CAM');
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
                });
                if (!videoRef.current || !running) return;
                videoRef.current.srcObject = stream;
                await videoRef.current.play();

                // ── MediaPipe Tasks API ────────────────────────────────────
                setStatusMsg('MP_WASM');
                const vision = await FilesetResolver.forVisionTasks(WASM);

                setStatusMsg('HAND_MODEL');
                handRef.current = await HandLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: HAND_M, },
                    runningMode: 'VIDEO',
                    numHands: 2,
                    minHandDetectionConfidence: 0.4,
                    minHandPresenceConfidence: 0.4,
                    minTrackingConfidence: 0.4,
                });

                setStatusMsg('POSE_MODEL');
                poseRef.current = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: POSE_M, },
                    runningMode: 'VIDEO',
                    numPoses: 1,
                    minPoseDetectionConfidence: 0.4,
                    minPosePresenceConfidence: 0.4,
                    minTrackingConfidence: 0.4,
                });

                setStatusMsg('FACE_MODEL');
                faceRef.current = await FaceLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: FACE_M, },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                    minFaceDetectionConfidence: 0.4,
                    minFacePresenceConfidence: 0.4,
                    minTrackingConfidence: 0.4,
                });

                // ── Filter banks ────────────────────────────────────────────
                lBankRef.current = new FilterBank(21, 1.0, 0.012);
                rBankRef.current = new FilterBank(21, 1.0, 0.012);
                pBankRef.current = new FilterBank(33, 1.0, 0.007);
                fBankRef.current = new FilterBank(478, 2.0, 0.005);

                // ── UI systems ──────────────────────────────────────────────
                panelRef.current = new PanelManager();
                rendRef.current  = new TrackingRenderer();

                // ── WebSocket ───────────────────────────────────────────────
                const ws = new WebSocket('ws://localhost:8080');
                ws.onmessage = (e) => {
                    try { const m = JSON.parse(e.data); if (m.type === 'RIG_SOLVED') teleRef.current = m.data; } catch {}
                };
                ws.onerror = () => {}; // silently degrade if server not running
                wsRef.current = ws;

                if (!running) return;
                setStatus('tracking');
                setStatusMsg('');

                // ── Animation loop ──────────────────────────────────────────
                const loop = () => {
                    if (!running) return;
                    rafRef.current = requestAnimationFrame(loop);

                    const video = videoRef.current;
                    const skCtx = skRef.current?.getContext('2d');
                    const drCtx = drRef.current?.getContext('2d');
                    const uiCtx = uiRef.current?.getContext('2d');
                    if (!video || video.readyState < 2 || !skCtx || !drCtx || !uiCtx) return;

                    const cw = video.videoWidth, ch = video.videoHeight;
                    if (!cw || !ch) return;

                    // Sync canvas dimensions once
                    if (skRef.current!.width !== cw) {
                        [skRef, drRef, uiRef].forEach(r => {
                            if (r.current) { r.current.width = cw; r.current.height = ch; }
                        });
                    }

                    const now = performance.now();
                    const dt  = Math.max(0.001, Math.min((now - lastTRef.current) / 1000, 0.1));
                    lastTRef.current = now;

                    // FPS counter
                    fcRef.current++;
                    if (now - fpsLastRef.current >= 1000) {
                        fpsRef.current = Math.round(fcRef.current * 1000 / (now - fpsLastRef.current));
                        fcRef.current = 0;
                        fpsLastRef.current = now;
                    }

                    // ── Detect ──────────────────────────────────────────────
                    const hr = handRef.current?.detectForVideo(video, now);
                    const pr = poseRef.current?.detectForVideo(video, now);
                    const fr = faceRef.current?.detectForVideo(video, now);

                    // Parse hand results → left/right
                    let leftRaw:  NormalizedLandmark[] | null = null;
                    let rightRaw: NormalizedLandmark[] | null = null;
                    let leftWorld  = null, rightWorld = null;
                    let lHandConf = 0, rHandConf = 0;

                    if (hr) {
                        hr.handednesses.forEach((h, i) => {
                            const isLeft = h[0].categoryName === 'Left';
                            const score  = h[0].score ?? 1;
                            if (isLeft) {
                                leftRaw   = hr.landmarks[i] as NormalizedLandmark[];
                                leftWorld = (hr.worldLandmarks as any)[i];
                                lHandConf = score;
                            } else {
                                rightRaw   = hr.landmarks[i] as NormalizedLandmark[];
                                rightWorld = (hr.worldLandmarks as any)[i];
                                rHandConf = score;
                            }
                        });
                    }

                    const poseRaw = pr?.landmarks?.[0] ?? null;
                    const faceRaw = (fr?.faceLandmarks?.[0] ?? null) as NormalizedLandmark[] | null;
                    const poseConf = poseRaw ? (poseRaw[0].visibility ?? 0.5) : 0;

                    // ── Smooth ──────────────────────────────────────────────
                    const leftHand  = leftRaw  && lBankRef.current ? lBankRef.current.smooth(leftRaw,  dt) : null;
                    const rightHand = rightRaw  && rBankRef.current ? rBankRef.current.smooth(rightRaw, dt) : null;
                    const pose      = poseRaw  && pBankRef.current ? pBankRef.current.smooth(poseRaw,  dt) : null;
                    const face      = faceRaw  && fBankRef.current ? fBankRef.current.smooth(faceRaw,  dt) : null;

                    // ── Gesture state machines ──────────────────────────────
                    const lg: HandGestureState | null = leftHand && leftWorld
                        ? lGestRef.current.update(leftHand, leftWorld, dt, now)
                        : null;
                    const rg: HandGestureState | null = rightHand && rightWorld
                        ? rGestRef.current.update(rightHand, rightWorld, dt, now)
                        : null;

                    const two: TwoHandState = twoRef.current.update(lg, rg);

                    // ── Z-thrust poke detection ─────────────────────────────
                    const panels = panelRef.current!;
                    if (leftHand) {
                        const { poke, charge } = lPokeRef.current.update(leftHand, 'Left', now);
                        lChargeRef.current = charge;
                        if (lg) (lg as any).pokeCharge = charge;
                        if (poke) {
                            panels.spawnPanels(poke.fingerCount, poke.normPos[0], poke.normPos[1], cw, ch);
                            pokeFlashRef.current = { x: poke.normPos[0], y: poke.normPos[1], frame: 0 };
                        }
                    } else { lChargeRef.current = 0; }

                    if (rightHand) {
                        const { poke, charge } = rPokeRef.current.update(rightHand, 'Right', now);
                        rChargeRef.current = charge;
                        if (rg) (rg as any).pokeCharge = charge;
                        if (poke) {
                            panels.spawnPanels(poke.fingerCount, poke.normPos[0], poke.normPos[1], cw, ch);
                            pokeFlashRef.current = { x: poke.normPos[0], y: poke.normPos[1], frame: 0 };
                        }
                    } else { rChargeRef.current = 0; }

                    // ── Skeleton zoom (palm_open two-hand) ──────────────────
                    if (two.mode === 'zoom_skeleton') {
                        skScaleRef.current = Math.max(0.5, Math.min(3, 1 + two.deltaSpan * 4));
                    }
                    if (two.mode === 'calibrate_reset') {
                        wsRef.current?.send(JSON.stringify({ type: 'RESET' }));
                        skScaleRef.current = 1;
                    }

                    // ── Panel interaction ───────────────────────────────────
                    // Drag (per hand, pinch over title)
                    const handleDrag = (g: HandGestureState | null, prev: HandGestureState | null) => {
                        if (!g) { if (panels.isDragging) panels.endDrag(); return; }
                        const px = g.cursorPos[0] * cw, py = g.cursorPos[1] * ch;
                        if (g.gesture === 'pinch') {
                            if (prev?.gesture !== 'pinch') {
                                const hit = panels.hitTestTitle(px, py);
                                if (hit) panels.startDrag(hit.id, px, py);
                            } else {
                                panels.updateDrag(px, py, cw, ch);
                            }
                        } else {
                            if (prev?.gesture === 'pinch') panels.endDrag();
                        }
                        // Swipe dismiss
                        if (g.gesture === 'swipe' && prev?.gesture !== 'swipe') {
                            const hit = panels.hitTestBody(px, py);
                            if (hit) panels.dismiss(hit.id, g.lateralVelocity);
                        }
                    };
                    handleDrag(lg, prevLGRef.current);
                    handleDrag(rg, prevRGRef.current);

                    // Resize (two-hand pinch)
                    if (two.mode === 'resize') {
                        if (!resizingRef.current) {
                            const nearest = panels.nearest(two.midpoint[0] * cw, two.midpoint[1] * ch);
                            if (nearest) { panels.startResize(nearest.id, two.spanDist); resizingRef.current = true; }
                        } else {
                            panels.updateResize(two.spanDist);
                        }
                    } else {
                        if (resizingRef.current) { panels.endResize(); resizingRef.current = false; }
                    }

                    // ── Draw canvas ─────────────────────────────────────────
                    const activeDrawGesture = lg?.gesture === 'draw' ? lg : rg?.gesture === 'draw' ? rg : null;
                    const activeGrabGesture = lg?.gesture === 'grab' || rg?.gesture === 'grab';

                    if (activeGrabGesture && (prevLGRef.current?.gesture === 'draw' || prevRGRef.current?.gesture === 'draw' || drawPtsRef.current.length > 0)) {
                        drCtx.clearRect(0, 0, cw, ch);
                        drawPtsRef.current = [];
                    } else if (activeDrawGesture) {
                        const hand = lg?.gesture === 'draw' ? leftHand : rightHand;
                        if (hand) {
                            const tip = hand[8]; // index tip
                            const prev = drawPtsRef.current[drawPtsRef.current.length - 1];
                            const vel = prev ? Math.sqrt((tip.x - prev.x)**2 + (tip.y - prev.y)**2) / dt : 0;
                            drawPtsRef.current.push({ x: tip.x, y: tip.y });
                            if (drawPtsRef.current.length > 1) {
                                // Draw canvas also has CSS scaleX(-1), draw in original coords
                                rendRef.current!.renderDrawPoint(drCtx, cw, ch, drawPtsRef.current, vel);
                            }
                        }
                    } else {
                        // Gesture changed from draw → reset path (start new stroke next time)
                        if (prevLGRef.current?.gesture === 'draw' || prevRGRef.current?.gesture === 'draw') {
                            drawPtsRef.current = [];
                        }
                    }

                    // ── Panel tick ──────────────────────────────────────────
                    panels.tick(cw);

                    // ── Server send ─────────────────────────────────────────
                    if (wsRef.current?.readyState === 1) {
                        const frame = packFrame(pose, leftHand, rightHand, face);
                        wsRef.current.send(frame.buffer);
                    }

                    // ── Live data for panels ────────────────────────────────
                    const liveData: LiveData = {
                        fps: fpsRef.current,
                        poseConf,
                        leftHandConf: lHandConf,
                        rightHandConf: rHandConf,
                        leftGesture: lg,
                        rightGesture: rg,
                        twoHand: two,
                        telemetry: teleRef.current,
                        landmarkCounts: {
                            pose: pose?.length ?? 0,
                            leftHand: leftHand?.length ?? 0,
                            rightHand: rightHand?.length ?? 0,
                            face: face?.length ?? 0,
                        },
                        skeletonScale: skScaleRef.current,
                        drawInkCount: drawPtsRef.current.length,
                    };

                    // ── Render skeleton (z1) ────────────────────────────────
                    // CSS scaleX(-1) on the canvas element handles the mirror.
                    // Draw in original landmark coords — CSS flip aligns it with the video.
                    rendRef.current!.renderSkeleton(skCtx, cw, ch, pose, leftHand, rightHand, face, skScaleRef.current);

                    // ── Render UI (z3): panels + cursors ────────────────────
                    uiCtx.clearRect(0, 0, cw, ch);

                    // Panels are in screen coords (cursor is already mirrored)
                    panels.render(uiCtx, liveData);

                    // Hand cursors drawn in screen coords (no flip needed — cursors already mirrored)
                    rendRef.current!.renderHandOverlay(
                        uiCtx, cw, ch,
                        leftHand, rightHand, lg, rg,
                        lChargeRef.current, rChargeRef.current, two
                    );

                    // Poke flash
                    if (pokeFlashRef.current) {
                        const f = pokeFlashRef.current;
                        rendRef.current!.renderPokeFlash(uiCtx, cw, ch, f.x, f.y, f.frame);
                        f.frame++;
                        if (f.frame >= 12) pokeFlashRef.current = null;
                    }

                    rendRef.current!.tick();

                    // Store prev gesture for next frame
                    prevLGRef.current = lg;
                    prevRGRef.current = rg;
                };

                rafRef.current = requestAnimationFrame(loop);

            } catch (err: any) {
                if (running) {
                    setStatus('error');
                    // DOM Events don't have .message — unwrap to something readable
                    const msg = err?.message
                        ?? err?.type
                        ?? (err instanceof Event ? `Event(${err.type ?? 'unknown'})` : null)
                        ?? String(err);
                    setStatusMsg(msg);
                    console.error('[KineTrak] Init error:', err);
                }
            }
        })();

        return () => {
            running = false;
            cancelAnimationFrame(rafRef.current);
            wsRef.current?.close();
            handRef.current?.close?.();
            poseRef.current?.close?.();
            faceRef.current?.close?.();
            const v = videoRef.current;
            if (v?.srcObject) {
                (v.srcObject as MediaStream).getTracks().forEach(t => t.stop());
            }
        };
    }, []);

    return (
        <div className="w-full h-screen bg-black overflow-hidden select-none cursor-none relative">
            {/* Camera feed — CSS-mirrored */}
            <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover"
                style={{
                    transform: 'scaleX(-1)',
                    opacity: 0.25,
                    filter: 'brightness(1.1) contrast(1.2) saturate(1.1) grayscale(30%)',
                }}
                playsInline
                muted
            />

            {/* Skeleton canvas — CSS-mirrored (gets mirror transform in draw calls too for overlay match) */}
            <canvas
                ref={skRef}
                className="absolute inset-0 w-full h-full z-10"
                style={{ transform: 'scaleX(-1)' }}
            />

            {/* Draw canvas — CSS-mirrored */}
            <canvas
                ref={drRef}
                className="absolute inset-0 w-full h-full z-20"
                style={{ transform: 'scaleX(-1)' }}
            />

            {/* UI canvas — screen coords, NO CSS mirror (panels + cursors) */}
            <canvas
                ref={uiRef}
                className="absolute inset-0 w-full h-full z-30"
            />

            {/* Status overlay (loading / error only) */}
            {status !== 'tracking' && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 text-white font-mono text-xs text-center pointer-events-none">
                    <div className="opacity-40">{status === 'loading' ? `LOADING: ${statusMsg}` : `ERR: ${statusMsg}`}</div>
                </div>
            )}

            {/* Minimal HUD when tracking */}
            {status === 'tracking' && (
                <div className="absolute top-3 left-3 z-50 font-mono text-[6px] text-white/25 pointer-events-none leading-tight">
                    KINETRAK V15 | POKE=SPAWN | PINCH=DRAG | SWIPE=DISMISS | DRAW=INK | FIST=CLEAR | 2×OPEN=ZOOM
                </div>
            )}
        </div>
    );
};

export default KinetrackEngine;
