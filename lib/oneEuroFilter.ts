// One Euro Filter — adaptive low-pass filter that minimizes lag while suppressing jitter.
// Reference: Casiez et al. 2012. Same algorithm as server/solver.ts but client-side.

export class OneEuroFilter {
    private lastVal = 0;
    private lastDx = 0;
    private initialized = false;

    constructor(
        private minCutoff = 1.0,
        private beta = 0.007,
        private dCutoff = 1.0
    ) {}

    private alpha(cutoff: number, dt: number): number {
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }

    update(val: number, dt: number): number {
        if (!this.initialized) {
            this.lastVal = val;
            this.initialized = true;
            return val;
        }
        const dx = (val - this.lastVal) / dt;
        const edx = 0.2 * dx + 0.8 * this.lastDx;
        this.lastDx = edx;
        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        const a = this.alpha(cutoff, dt);
        const res = a * val + (1 - a) * this.lastVal;
        this.lastVal = res;
        return res;
    }

    reset(): void {
        this.initialized = false;
        this.lastDx = 0;
    }
}

// FilterBank wraps N×3 filters for a set of 3D landmarks.
export class FilterBank {
    private fx: OneEuroFilter[];
    private fy: OneEuroFilter[];
    private fz: OneEuroFilter[];

    constructor(count: number, minCutoff = 1.0, beta = 0.007) {
        const mk = () => Array.from({ length: count }, () => new OneEuroFilter(minCutoff, beta));
        this.fx = mk();
        this.fy = mk();
        this.fz = mk();
    }

    smooth<T extends { x: number; y: number; z: number }>(landmarks: T[], dt: number): T[] {
        return landmarks.map((lm, i) => ({
            ...lm,
            x: this.fx[i].update(lm.x, dt),
            y: this.fy[i].update(lm.y, dt),
            z: this.fz[i].update(lm.z, dt),
        }));
    }

    reset(): void {
        this.fx.forEach(f => f.reset());
        this.fy.forEach(f => f.reset());
        this.fz.forEach(f => f.reset());
    }
}
