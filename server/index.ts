import { WebSocketServer, WebSocket } from 'ws';
import dgram from 'dgram';
import { PoseRingBuffer } from './buffer';
import { vec3, Vec3, mag, sub, add } from './kinematics';
import {
    OneEuroFilter, solveHeadPose, solveMuscleTension, solveGaze,
    solveHandTopology, solveHandKinematics, calculateKineticEnergy, constrainBone, solveBodyCollisions,
    angleBetween
} from './solver';

/**
 * V12.0 PALANTIR-GRADE INFRASTRUCTURE:
 * - UDP Relay: Multiplexing binary stream to external engines.
 * - Multi-Volume Collision: Rigid body physics for limb protection.
 * - Data Persistence: Zero-copy temporal stacking.
 */

const BONE_RIG = [
    { n: 'lUA', a: 11, b: 13 }, { n: 'lLA', a: 13, b: 15 },
    { n: 'rUA', a: 12, b: 14 }, { n: 'rLA', a: 14, b: 16 },
    { n: 'lTh', a: 23, b: 25 }, { n: 'lCl', a: 25, b: 27 },
    { n: 'rTh', a: 24, b: 26 }, { n: 'rCl', a: 26, b: 28 }
];

const wss = new WebSocketServer({ port: 8080 });
const udp = dgram.createSocket('udp4');
let relayTarget: { ip: string, port: number } | null = null;

console.log('[KINE-V12] Palantir-Grade Substrate Active');

interface Client {
    buffer: PoseRingBuffer;
    filters: OneEuroFilter[];
    boneLen: Record<string, number>;
    lastT: number;
    scratch: Float32Array;
}

const clients = new Map<WebSocket, Client>();

wss.on('connection', (ws) => {
    clients.set(ws, {
        buffer: new PoseRingBuffer(30, 553 * 4),
        filters: Array.from({length: 553 * 3}, () => new OneEuroFilter(1.2, 0.01)),
        boneLen: {},
        lastT: performance.now(),
        scratch: new Float32Array(553 * 4)
    });

    ws.on('message', (data: Buffer) => {
        try {
            const state = clients.get(ws);
            if (!state) return;

            // Handle Control Commands (Relay Config, etc.)
            if (data.length < 100 && data.toString().startsWith('{')) {
                const cmd = JSON.parse(data.toString());
                if (cmd.type === 'SET_RELAY') {
                    relayTarget = cmd.target;
                    console.log(`[KINE-SYS] UDP Relay targeted to: ${relayTarget?.ip}:${relayTarget?.port}`);
                }
                if (cmd.type === 'RESET') state.boneLen = {};
                return;
            }

            const raw = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
            if (raw.length < 553 * 4) return;

            const now = performance.now(), dt = (now - state.lastT) / 1000;
            state.lastT = now;

            const out = state.scratch;
            const last = state.buffer.getFrame(0);

            // 1. HARDENED PIPELINE
            for (let i = 0; i < 553; i++) {
                const off = i * 4;
                if (raw[off+3] < 0.4 && last.length > 0) {
                    const v = state.buffer.getAverageVelocity(i, 2);
                    out[off] = last[off] + v[0]; out[off+1] = last[off+1] + v[1]; out[off+2] = last[off+2] + v[2]; out[off+3] = last[off+3] * 0.9;
                    continue;
                }
                out[off] = state.filters[i*3].update(raw[off], dt);
                out[off+1] = state.filters[i*3+1].update(raw[off+1], dt);
                out[off+2] = state.filters[i*3+2].update(raw[off+2], dt);
                out[off+3] = raw[off+3];
            }
            state.buffer.push(out);

            // 2. SOLVE & PHYSICS
            const solved = solveV12(state, out);

            // 3. MULTIPLEX BROADCAST
            const payload = JSON.stringify({ type: 'RIG_SOLVED', data: solved });
            ws.send(payload);

            // UDP External Relay
            if (relayTarget) {
                const udpBuf = Buffer.from(payload);
                udp.send(udpBuf, relayTarget.port, relayTarget.ip);
            }

        } catch (e: any) {
            ws.send(JSON.stringify({ type: 'ERROR', msg: e.message }));
        }
    });
    ws.on('close', () => clients.delete(ws));
});

function solveV12(state: Client, frame: Float32Array) {
    const getV = (idx: number): Vec3 => { const off = idx * 4; return vec3(frame[off], frame[off+1], frame[off+2]); };
    const faceOff = 75;

    // Auto-Calibrate
    BONE_RIG.forEach(p => {
        if (frame[p.a*4+3] > 0.8 && frame[p.b*4+3] > 0.8 && !state.boneLen[p.n]) {
            state.boneLen[p.n] = mag(sub(getV(p.b), getV(p.a)));
        }
    });

    // 1. Define Physics Volumes (Dynamic)
    const lSh = getV(11), rSh = getV(12), lHp = getV(23), rHp = getV(24), nose = getV(0);
    const torso = { 
        c: vec3((lSh[0]+rSh[0]+lHp[0]+rHp[0])/4, (lSh[1]+rSh[1]+lHp[1]+rHp[1])/4, (lSh[2]+rSh[2]+lHp[2]+rHp[2])/4),
        r: mag(sub(lSh, rSh)) * 0.55
    };
    const head = { c: nose, r: mag(sub(lSh, rSh)) * 0.25 };

    const getVel = (absIdx: number) => state.buffer.getVelocity(absIdx);

    const res: any = {
        pose:  solveHeadPose(frame, faceOff),
        face:  solveMuscleTension(frame, faceOff),
        gaze:  solveGaze(frame, faceOff),
        hands: {
            left:  frame[33*4+3] > 0.2 ? solveHandKinematics(frame, 33, getVel) : null,
            right: frame[54*4+3] > 0.2 ? solveHandKinematics(frame, 54, getVel) : null,
        },
        constrained: {},
        physics: { torso, head },
    };

    // 2. Rigid Constraints + Collision Clamping
    BONE_RIG.forEach(p => {
        const len = state.boneLen[p.n];
        if (len) {
            let pos = constrainBone(getV(p.a), getV(p.b), len);
            // Protect volumes from hands/wrists (15, 16)
            if (p.b === 15 || p.b === 16) pos = solveBodyCollisions(pos, torso, head);
            res.constrained[p.b] = [pos[0], pos[1], pos[2]];
        }
    });

    return res;
}
