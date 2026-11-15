/**
 * Mediterranean Sea Climate Observatory — Advanced Trend Detection Module (js/adv.js)
 * Team Solari-X · NASA Space Apps Challenge ("Be An Earth System Trend Detective")[cite: 1]
 *
 * Provides autocorrelation-robust regression, memory-corrected non-parametric tests,
 * change-point / hinge detection, quadratic acceleration tests, and sliding-window
 * slope stability analysis.
 *
 * Dependencies (loaded prior to adv.js):
 *   - js/stats.js (window.Stats)[cite: 1]
 */
const Adv = (() => {
    "use strict";

    const S = typeof Stats !== "undefined" ? Stats : require("./stats.js");

    /* --------------------------------------------------------------------------
     * 1. Normal Distribution & Linear Algebra Helpers
     * -------------------------------------------------------------------------- */

    /**
     * Complementary error function (Numerical Recipes Chebyshev approximation).
     * @param {number} x
     * @returns {number}
     */
    function erfc(x) {
        const z = Math.abs(x);
        const t = 1 / (1 + 0.5 * z);
        const r =
            t *
            Math.exp(
                -z * z -
                1.26551223 +
                t *
                (1.00002368 +
                    t *
                    (0.37409196 +
                        t *
                        (0.09678418 +
                            t *
                            (-0.18628806 +
                                t *
                                (0.27886807 +
                                    t *
                                    (-1.13520398 +
                                        t *
                                        (1.48851587 +
                                            t * (-0.82215223 + t * 0.17087277))))))))
            );
        return x >= 0 ? r : 2 - r;
    }

    /**
     * Standard normal cumulative distribution function Φ(z).
     * @param {number} z
     * @returns {number}
     */
    const normCdf = z => 1 - 0.5 * erfc(z / Math.SQRT2);

    /**
     * Two-sided p-value from a standard normal z-score.
     * @param {number} z
     * @returns {number}
     */
    const normP2 = z => 2 * (1 - normCdf(Math.abs(z)));

    /**
     * Solves a 3×3 linear system A * c = B using Gaussian elimination with partial pivoting.
     * @param {number[][]} A - 3×3 coefficient matrix
     * @param {number[]} B - 3-element right-hand side vector
     * @returns {number[]} Solution vector [c0, c1, c2]
     */
    function solve3(A, B) {
        const M = A.map((row, i) => row.concat(B[i]));

        for (let i = 0; i < 3; i++) {
            let p = i;
            for (let r = i + 1; r < 3; r++) {
                if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
            }
            [M[i], M[p]] = [M[p], M[i]];

            for (let r = i + 1; r < 3; r++) {
                const f = M[r][i] / M[i][i];
                for (let c = i; c < 4; c++) {
                    M[r][c] -= f * M[i][c];
                }
            }
        }

        const s = [0, 0, 0];
        for (let i = 2; i >= 0; i--) {
            let v = M[i][3];
            for (let c = i + 1; c < 3; c++) {
                v -= M[i][c] * s[c];
            }
            s[i] = v / M[i][i];
        }
        return s;
    }

    /**
     * Computes the analytical inverse of a 3×3 matrix via cofactors.
     * @param {number[][]} m - 3×3 matrix
     * @returns {number[][]} Inverse 3×3 matrix
     */
    function inv3(m) {
        const [a, b, c] = m[0];
        const [d, e, f] = m[1];
        const [g, h, i] = m[2];

        const A = e * i - f * h;
        const B = -(d * i - f * g);
        const C = d * h - e * g;
        const det = a * A + b * B + c * C;

        return [
            [A, -(b * i - c * h), b * f - c * e].map(v => v / det),
            [B, a * i - c * g, -(a * f - c * d)].map(v => v / det),
            [C, -(a * h - b * g), a * e - b * d].map(v => v / det)
        ];
    }

    /**
     * Fits a 3-parameter linear model y = b0 + b1*(x - x0) + b2*h via normal equations.
     * @param {number[]} x - Year array
     * @param {number[]} h - Hinge basis array max(0, x - bk)
     * @param {number[]} y - Observed values
     * @returns {{ sse: number, b0: number, b1: number, b2: number, x0: number }}
     */
    function ols2(x, h, y) {
        const X = x.map((xi, i) => [1, xi - x[0], h[i]]);
        const A = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ];
        const B = [0, 0, 0];

        X.forEach((r, i) => {
            for (let a = 0; a < 3; a++) {
                B[a] += r[a] * y[i];
                for (let b = 0; b < 3; b++) {
                    A[a][b] += r[a] * r[b];
                }
            }
        });

        const c = solve3(A, B);
        let sse = 0;
        X.forEach((r, i) => {
            const f = c[0] * r[0] + c[1] * r[1] + c[2] * r[2];
            sse += (y[i] - f) ** 2;
        });

        return { sse, b0: c[0], b1: c[1], b2: c[2], x0: x[0] };
    }

    /* --------------------------------------------------------------------------
     * 2. Autocorrelation-Robust & Memory-Corrected Trend Tests
     * -------------------------------------------------------------------------- */

    /**
     * Newey–West (Bartlett kernel) Heteroskedasticity and Autocorrelation Consistent
     * (HAC) standard error, 95% confidence interval, and two-sided p-value for the OLS slope.
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @param {number} lags - Maximum Bartlett lag truncation parameter
     * @returns {{ se: number, z: number, p: number, lo: number, hi: number, lags: number }}
     */
    function hac(x, y, lags) {
        const o = S.ols(x, y);
        const n = x.length;
        const mx = o.mx;

        const u = x.map((xi, i) => (xi - mx) * (y[i] - (o.icpt + o.slope * xi)));
        let s = u.reduce((a, v) => a + v * v, 0);

        for (let l = 1; l <= lags; l++) {
            let g = 0;
            for (let i = l; i < n; i++) {
                g += u[i] * u[i - l];
            }
            s += 2 * (1 - l / (lags + 1)) * g;
        }

        const se = Math.sqrt(s) / o.sxx;
        const z = o.slope / se;
        const p = normP2(z);
        const c = 1.959964;

        return {
            se,
            z,
            p,
            lo: o.slope - c * se,
            hi: o.slope + c * se,
            lags
        };
    }

    /**
     * Hamed & Rao (1998) variance-corrected Mann–Kendall trend test.
     * Adjusts Var(S) using statistically significant lag autocorrelations of detrended ranks.
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @returns {{ z: number, p: number, factor: number }}
     */
    function mkHR(x, y) {
        const n = y.length;
        const o = S.ols(x, y);
        const mk = S.mannKendall(y);

        const res = y.map((v, i) => v - (o.icpt + o.slope * x[i]));
        const idx = res.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        const rk = new Array(n);
        idx.forEach((p, r) => {
            rk[p[1]] = r + 1;
        });

        const m = rk.reduce((a, v) => a + v, 0) / n;
        const den = rk.reduce((a, v) => a + (v - m) ** 2, 0);

        let corr = 1;
        const maxl = Math.min(n - 3, Math.floor(n / 3));
        const bound = 1.959964 / Math.sqrt(n);

        for (let l = 1; l <= maxl; l++) {
            let num = 0;
            for (let i = l; i < n; i++) {
                num += (rk[i] - m) * (rk[i - l] - m);
            }
            const r = num / den;
            if (Math.abs(r) > bound) {
                corr +=
                    (2 / (n * (n - 1) * (n - 2))) *
                    (n - l) *
                    (n - l - 1) *
                    (n - l - 2) *
                    r;
            }
        }

        corr = Math.max(corr, 0.1);
        const varS = ((n * (n - 1) * (2 * n + 5)) / 18) * corr;
        const S0 = mk.S;
        const z = S0 === 0 ? 0 : (S0 - Math.sign(S0)) / Math.sqrt(varS);

        return { z, p: normP2(z), factor: corr };
    }

    /* --------------------------------------------------------------------------
     * 3. Regime Shift, Hinge & Acceleration Diagnostics
     * -------------------------------------------------------------------------- */

    /**
     * Non-parametric Pettitt (1979) change-point test for an abrupt shift in level.
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @returns {{ year: number, idx: number, p: number, K: number, meanBefore: number, meanAfter: number }}
     */
    function pettitt(x, y) {
        const n = y.length;
        let best = 0;
        let kt = 0;

        for (let t = 0; t < n - 1; t++) {
            let U = 0;
            for (let i = 0; i <= t; i++) {
                for (let j = t + 1; j < n; j++) {
                    U += Math.sign(y[j] - y[i]);
                }
            }
            if (Math.abs(U) > best) {
                best = Math.abs(U);
                kt = t;
            }
        }

        const p = Math.min(1, 2 * Math.exp((-6 * best * best) / (n ** 3 + n ** 2)));
        const a = y.slice(0, kt + 1);
        const b = y.slice(kt + 1);

        return {
            year: x[kt],
            idx: kt,
            p,
            K: best,
            meanBefore: S.mean(a),
            meanAfter: S.mean(b)
        };
    }

    /**
     * Continuous two-segment (hinge) linear regression selecting the optimal
     * breakpoint by minimum SSE and evaluating improvement via an F(2, n - 4) test.
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @param {number} [minSeg=4] - Minimum number of points per segment
     * @returns {Object|null}
     */
    function hinge(x, y, minSeg = 4) {
        const n = y.length;
        let best = null;
        const lin = S.ols(x, y);
        const sse1 = lin.s * lin.s * (n - 2);

        for (let k = minSeg; k <= n - minSeg - 1; k++) {
            const bk = x[k];
            const h = x.map(v => Math.max(0, v - bk));
            const f = ols2(x, h, y);
            if (!best || f.sse < best.sse) {
                best = { k, bk, ...f };
            }
        }

        if (!best) return null;

        const df2 = n - 4;
        const F = ((sse1 - best.sse) / 2) / (best.sse / df2);
        // F(2, df2) p-value via incomplete beta: I_{df2/(df2+2F)}(df2/2, 1) = (df2/(df2+2F))^(df2/2)
        const p = Math.pow(df2 / (df2 + 2 * F), df2 / 2);

        return {
            ...best,
            F,
            p: Math.min(1, p),
            slope1: best.b1,
            slope2: best.b1 + best.b2
        };
    }

    /**
     * Quadratic curvature test for trend acceleration or deceleration:
     * y = a + b*t + c*(t^2 - mean(t^2)), where t = x - mean(x).
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @returns {{ c2: number, se: number, t: number, df: number, p: number, per2decade: number }}
     */
    function accel(x, y) {
        const n = y.length;
        const mx = S.mean(x);
        const t = x.map(v => v - mx);
        const q = t.map(v => v * v);
        const qm = S.mean(q);
        const qc = q.map(v => v - qm);

        const X = t.map((v, i) => [1, v, qc[i]]);
        const AA = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ];
        const B = [0, 0, 0];

        X.forEach((r, i) => {
            for (let a = 0; a < 3; a++) {
                B[a] += r[a] * y[i];
                for (let b = 0; b < 3; b++) {
                    AA[a][b] += r[a] * r[b];
                }
            }
        });

        const c = solve3(AA, B);
        let sse = 0;
        X.forEach((r, i) => {
            sse += (y[i] - (c[0] * r[0] + c[1] * r[1] + c[2] * r[2])) ** 2;
        });

        const df = n - 3;
        const s2 = sse / df;
        const inv = inv3(AA);
        const se = Math.sqrt(s2 * inv[2][2]);
        const tt = c[2] / se;
        const p = S.tP(tt, df);

        return {
            c2: c[2],
            se,
            t: tt,
            df,
            p,
            per2decade: 2 * c[2] * 100 // Rate of slope change per decade ((units/yr)/decade * 10)
        };
    }

    /* --------------------------------------------------------------------------
     * 4. Sliding-Window Slope Stability
     * -------------------------------------------------------------------------- */

    /**
     * Computes rolling-window OLS slopes and 95% confidence intervals across sub-periods.
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @param {number} w - Window width in years
     * @returns {Array<{ x: number, slope: number, lo: number, hi: number, p: number, x0: number, x1: number }>}
     */
    function rolling(x, y, w) {
        const out = [];
        for (let i = 0; i + w <= x.length; i++) {
            const o = S.ols(x.slice(i, i + w), y.slice(i, i + w));
            out.push({
                x: (x[i] + x[i + w - 1]) / 2,
                slope: o.slope,
                lo: o.ciLo,
                hi: o.ciHi,
                p: o.p,
                x0: x[i],
                x1: x[i + w - 1]
            });
        }
        return out;
    }

    return { hac, mkHR, pettitt, hinge, accel, rolling };
})();

if (typeof window !== "undefined") window.Adv = Adv;
if (typeof module !== "undefined") module.exports = Adv;
