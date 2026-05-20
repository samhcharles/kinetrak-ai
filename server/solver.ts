import { Vec3, vec3, sub, add, mag, normalize, dot, cross, quatFromVectors, angleBetween } from './kinematics';

/**
 * ARCHITECTURAL INTEGRITY:
 * Solvers now take flat Float32Array frames and landmark indices to minimize allocation.
 * Logic: O(1) memory access, zero-copy pointer arithmetic.
 */

const getV = (frame: Float32Array, idx: number): Vec3 => {
    const off = idx * 4;
    return vec3(frame[off], frame[off+1], frame[off+2]);
};

export class OneEuroFilter {
    private lastVal: number = 0;
    private lastDx: number = 0;
    private lastT: number = 0;
    constructor(private minCutoff = 1.0, private beta = 0.007, private dCutoff = 1.0) {}
    private alpha(cutoff: number, dt: number): number {
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }
    update(val: number, dt: number): number {
        if (this.lastT === 0) { this.lastVal = val; this.lastT = 1; return val; }
        const dx = (val - this.lastVal) / dt;
        const edx = (0.2 * dx) + (0.8 * this.lastDx); // Fast smoothing for d
        this.lastDx = edx;
        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        const a = this.alpha(cutoff, dt);
        const res = a * val + (1 - a) * this.lastVal;
        this.lastVal = res;
        return res;
    }
}

/**
 * Rigid Bone Constraint
 */
export function constrainBone(root: Vec3, limb: Vec3, targetLen: number, out = vec3()): Vec3 {
    const dir = sub(limb, root);
    const currentLen = mag(dir);
    if (currentLen === 0) return add(root, vec3(0, -0.001, 0), out);
    const factor = targetLen / currentLen;
    out[0] = root[0] + dir[0] * factor;
    out[1] = root[1] + dir[1] * factor;
    out[2] = root[2] + dir[2] * factor;
    return out;
}

/**
 * 3D Head Pose Solver (Flat-Buffer Edition)
 */
export function solveHeadPose(frame: Float32Array, offset: number) {
    const getL = (idx: number) => getV(frame, offset + idx);
    const nose = getL(1), chin = getL(152), forehead = getL(10);
    const lEye = getL(33), rEye = getL(263);

    const eyeMid = add(lEye, rEye);
    eyeMid[0] *= 0.5; eyeMid[1] *= 0.5; eyeMid[2] *= 0.5;

    const faceH = mag(sub(forehead, chin));
    const pitch = ((nose[1] - eyeMid[1]) / (faceH || 1)) * 100;
    const yaw = (nose[0] - eyeMid[0]) * 100;
    const roll = Math.atan2(rEye[1] - lEye[1], rEye[0] - lEye[0]) * (180 / Math.PI);

    return { pitch: pitch.toFixed(1), yaw: yaw.toFixed(1), roll: roll.toFixed(1) };
}

/**
 * Facial Muscle Tension Solver
 */
export function solveMuscleTension(frame: Float32Array, offset: number) {
    const getD = (i: number, j: number) => mag(sub(getV(frame, offset+i), getV(frame, offset+j)));
    return {
        zygomatic: (getD(61, 291) * 6).toFixed(2), // Normalized lip stretch
        corrugator: (getD(107, 336) * 12).toFixed(2), // Brow compression
        orbicularis: (getD(13, 14) * 10).toFixed(2)  // Lip height
    };
}

/**
 * Iris Gaze Solver
 */
export function solveGaze(frame: Float32Array, offset: number) {
    const getL = (idx: number) => getV(frame, offset + idx);
    const solve = (iris: number, inE: number, outE: number) => {
        const i = getL(iris), inner = getL(inE), outer = getL(outE);
        const w = mag(sub(inner, outer)) || 1;
        const midX = (inner[0] + outer[0]) * 0.5;
        return ((i[0] - midX) / w).toFixed(3);
    };
    return { l: solve(468, 133, 33), r: solve(473, 362, 263) };
}

function solveSphereCollision(p: Vec3, center: Vec3, radius: number, out = vec3()): Vec3 {
    const d = sub(p, center);
    const dist = mag(d);
    if (dist < radius && dist > 0) {
        const scale = radius / dist;
        out[0] = center[0] + d[0] * scale;
        out[1] = center[1] + d[1] * scale;
        out[2] = center[2] + d[2] * scale;
        return out;
    }
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
    return out;
}

/**
 * Multi-Volume Body Collision
 * Protects Torso and Head volumes from limb penetration.
 */
export function solveBodyCollisions(p: Vec3, torso: { c: Vec3, r: number }, head: { c: Vec3, r: number }, out = vec3()): Vec3 {
    let current = solveSphereCollision(p, head.c, head.r, out);
    return solveSphereCollision(current, torso.c, torso.r, out);
}

/**
 * Hand Topology Solver
 */
export function solveHandTopology(frame: Float32Array, offset: number) {
    const getD = (i: number, j: number) => mag(sub(getV(frame, offset+i), getV(frame, offset+j)));
    return {
        pinch: getD(4, 8).toFixed(4),
        spread: getD(4, 20).toFixed(4)
    };
}

/**
 * Biomechanical Energy
 */
export function calculateKineticEnergy(velocities: Float32Array[]): number {
    let energy = 0;
    const masses: Record<number, number> = { 11:0.2, 12:0.2, 13:0.1, 14:0.1, 15:0.05, 16:0.05 };
    for (const [idx, m] of Object.entries(masses)) {
        const v = velocities[parseInt(idx)];
        if (v) energy += 0.5 * m * (v[0]**2 + v[1]**2 + v[2]**2);
    }
    return energy;
}

// ── Per-hand kinematics (finger joint angles + tip speeds) ────────────────────

export function solveHandKinematics(frame: Float32Array, handOffset: number, getVelocity: (absIdx: number) => Float32Array) {
    const getL = (idx: number) => getV(frame, handOffset + idx);

    // 4 fingers: [MCP, PIP, DIP, tip] index sets
    const fingers = [
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ] as const;

    const fingerAngles: number[] = [];
    for (const [mcp, pip, dip, tip] of fingers) {
        // MCP: wrist(0) → MCP → PIP
        const mcpA = angleBetween(sub(getL(mcp), getL(0)), sub(getL(pip), getL(mcp)));
        // PIP: MCP → PIP → DIP
        const pipA = angleBetween(sub(getL(pip), getL(mcp)), sub(getL(dip), getL(pip)));
        // DIP: PIP → DIP → tip
        const dipA = angleBetween(sub(getL(dip), getL(pip)), sub(getL(tip), getL(dip)));
        fingerAngles.push(+mcpA.toFixed(1), +pipA.toFixed(1), +dipA.toFixed(1));
    }

    // Wrist flexion: angle between palm width vector and middle-finger direction
    const wristAngle = angleBetween(sub(getL(9), getL(0)), sub(getL(5), getL(17)));

    // Tip velocities (index=8, middle=12, ring=16, pinky=20)
    const tipVelocities = [8, 12, 16, 20].map(i => {
        const v = getVelocity(handOffset + i);
        return +Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2).toFixed(5);
    });

    return { fingerAngles, wristAngle: +wristAngle.toFixed(1), tipVelocities };
}

export { vec3, sub, add, mag, normalize, dot, cross, quatFromVectors, angleBetween };
