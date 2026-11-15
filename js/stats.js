/**
 * Mediterranean Sea Climate Observatory — Core Statistical Engine (js/stats.js)
 * Team Solari-X · NASA Space Apps Challenge ("Be An Earth System Trend Detective")
 *
 * Implements ordinary least squares (OLS) trend estimation, exact Student's t
 * inference, lag-1 residual autocorrelation, Mann–Kendall rank trend tests,
 * and Theil–Sen non-parametric slope estimation with 95% confidence bounds.
 */
const Stats = (() => {
    "use strict";

    /* --------------------------------------------------------------------------
     * 1. Descriptive & Special Mathematical Functions
     * -------------------------------------------------------------------------- */

    /**
     * Computes the arithmetic mean of a numeric array.
     * @param {number[]} arr
     * @returns {number}
     */
    function mean(arr) {
        if (!arr || !arr.length) return NaN;
        let sum = 0;
        for (let i = 0; i < arr.length; i++) {
            sum += arr[i];
        }
        return sum / arr.length;
    }

    /**
     * Computes the median of a numeric array without mutating the input.
     * @param {number[]} arr
     * @returns {number}
     */
    function median(arr) {
        if (!arr || !arr.length) return NaN;
        const sorted = arr.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 !== 0
            ? sorted[mid]
            : 0.5 * (sorted[mid - 1] + sorted[mid]);
    }

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
     * Natural logarithm of the Gamma function via Lanczos approximation.
     * @param {number} z
     * @returns {number}
     */
    function logGamma(z) {
        const c = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.001208650973866179,
            -0.000005395239384953
        ];
        let x = z;
        let y = z;
        let tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        let ser = 1.000000000190015;
        for (let j = 0; j < 6; j++) {
            y += 1;
            ser += c[j] / y;
        }
        return -tmp + Math.log((2.5066282746310005 * ser) / x);
    }

    /**
     * Continued fraction evaluation for the regularized incomplete beta function.
     * @param {number} a
     * @param {number} b
     * @param {number} x
     * @returns {number}
     */
    function betaCf(a, b, x) {
        const MAXIT = 200;
        const EPS = 3e-14;
        const FPMIN = 1e-300;

        const qab = a + b;
        const qap = a + 1;
        const qam = a - 1;
        let c = 1;
        let d = 1 - (qab * x) / qap;
        if (Math.abs(d) < FPMIN) d = FPMIN;
        d = 1 / d;
        let h = d;

        for (let m = 1; m <= MAXIT; m++) {
            const m2 = 2 * m;
            let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < FPMIN) d = FPMIN;
            c = 1 + aa / c;
            if (Math.abs(c) < FPMIN) c = FPMIN;
            d = 1 / d;
            h *= d * c;

            aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.abs(d) < FPMIN) d = FPMIN;
            c = 1 + aa / c;
            if (Math.abs(c) < FPMIN) c = FPMIN;
            d = 1 / d;
            const del = d * c;
            h *= del;
            if (Math.abs(del - 1) < EPS) break;
        }
        return h;
    }

    /**
     * Regularized incomplete beta function I_x(a, b).
     * @param {number} a
     * @param {number} b
     * @param {number} x
     * @returns {number}
     */
    function betaInc(a, b, x) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        const bt = Math.exp(
            logGamma(a + b) -
            logGamma(a) -
            logGamma(b) +
            a * Math.log(x) +
            b * Math.log(1 - x)
        );
        if (x < (a + 1) / (a + b + 2)) {
            return (bt * betaCf(a, b, x)) / a;
        }
        return 1 - (bt * betaCf(b, a, 1 - x)) / b;
    }

    /**
     * Two-sided p-value for a Student's t-statistic with `df` degrees of freedom.
     * @param {number} t - t-statistic
     * @param {number} df - Degrees of freedom
     * @returns {number}
     */
    function tP(t, df) {
        if (!Number.isFinite(t) || df <= 0) return NaN;
        if (t === 0) return 1;
        const x = df / (df + t * t);
        return Math.min(1, Math.max(0, betaInc(0.5 * df, 0.5, x)));
    }

    /**
     * Two-sided critical t-value t_{1 - alpha/2, df} via bisection.
     * @param {number} df - Degrees of freedom
     * @param {number} [alpha=0.05] - Two-sided significance level
     * @returns {number}
     */
    function tCrit(df, alpha = 0.05) {
        if (df <= 0) return NaN;
        let lo = 0;
        let hi = 20;
        while (tP(hi, df) > alpha) {
            hi *= 2;
        }
        for (let i = 0; i < 64; i++) {
            const mid = 0.5 * (lo + hi);
            if (tP(mid, df) > alpha) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

    /* --------------------------------------------------------------------------
     * 2. Ordinary Least Squares (OLS) Linear Trend Estimation
     * -------------------------------------------------------------------------- */

    /**
     * Fits y = icpt + slope * x by ordinary least squares and computes standard errors,
     * 95% confidence bounds, two-sided t-test p-value, R², and lag-1 residual autocorrelation.
     *
     * @param {number[]} x - Independent variable (years)
     * @param {number[]} y - Dependent variable (observations)
     * @returns {Object}
     */
    function ols(x, y) {
        const n = x.length;
        const mx = mean(x);
        const my = mean(y);

        let sxx = 0;
        let sxy = 0;
        let syy = 0;
        for (let i = 0; i < n; i++) {
            const dx = x[i] - mx;
            const dy = y[i] - my;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }

        const slope = sxx > 0 ? sxy / sxx : 0;
        const icpt = my - slope * mx;

        const res = new Array(n);
        let sse = 0;
        for (let i = 0; i < n; i++) {
            const r = y[i] - (icpt + slope * x[i]);
            res[i] = r;
            sse += r * r;
        }

        const df = Math.max(1, n - 2);
        const s = Math.sqrt(sse / df);
        const se = sxx > 0 ? s / Math.sqrt(sxx) : NaN;
        const t = se > 0 ? slope / se : 0;
        const p = tP(t, df);
        const tc = tCrit(df, 0.05);
        const ciLo = slope - tc * se;
        const ciHi = slope + tc * se;
        const r2 = syy > 0 ? Math.max(0, Math.min(1, 1 - sse / syy)) : 0;

        let numLag1 = 0;
        for (let i = 1; i < n; i++) {
            numLag1 += res[i] * res[i - 1];
        }
        const lag1 = sse > 0 ? numLag1 / sse : 0;

        return {
            n,
            mx,
            my,
            sxx,
            slope,
            icpt,
            s,
            se,
            t,
            df,
            tc,
            ciLo,
            ciHi,
            p,
            r2,
            lag1
        };
    }

    /* --------------------------------------------------------------------------
     * 3. Non-Parametric Trend Tests (Mann–Kendall & Theil–Sen)
     * -------------------------------------------------------------------------- */

    /**
     * Computes the tie-corrected variance of the Mann–Kendall S statistic.
     * @param {number[]} y
     * @returns {number}
     */
    function mkVariance(y) {
        const n = y.length;
        let v = n * (n - 1) * (2 * n + 5);

        const sorted = y.slice().sort((a, b) => a - b);
        let i = 0;
        while (i < n) {
            let j = i + 1;
            while (j < n && sorted[j] === sorted[i]) j++;
            const tieLen = j - i;
            if (tieLen > 1) {
                v -= tieLen * (tieLen - 1) * (2 * tieLen + 5);
            }
            i = j;
        }
        return v / 18;
    }

    /**
     * Performs the non-parametric Mann–Kendall monotonic trend test with tie adjustment
     * and continuity correction. Accepts either `(y)` or `(x, y)`.
     *
     * @param {number[]} xOrY - Either y-series (when 1 arg) or x-series (when 2 args)
     * @param {number[]} [yOpt] - Optional y-series when x-series is supplied first
     * @returns {{ S: number, varS: number, tau: number, z: number, p: number }}
     */
    function mannKendall(xOrY, yOpt) {
        let y;
        if (Array.isArray(yOpt)) {
            const paired = xOrY
                .map((xi, i) => [xi, yOpt[i]])
                .sort((a, b) => a[0] - b[0]);
            y = paired.map(pair => pair[1]);
        } else {
            y = xOrY;
        }

        const n = y.length;
        let S = 0;
        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const diff = y[j] - y[i];
                if (diff > 0) S += 1;
                else if (diff < 0) S -= 1;
            }
        }

        const varS = mkVariance(y);
        const z =
            S === 0 || varS <= 0 ? 0 : (S - Math.sign(S)) / Math.sqrt(varS);
        const p = erfc(Math.abs(z) / Math.SQRT2);
        const denom = 0.5 * n * (n - 1);
        const tau = denom > 0 ? S / denom : 0;

        return { S, varS, tau, z, p };
    }

    /**
     * Computes the Theil–Sen median pairwise slope estimator, corresponding median
     * intercept, and non-parametric 95% confidence interval (Sen 1968 / Gilbert 1987).
     *
     * @param {number[]} x - Independent variable (years)
     * @param {number[]} y - Dependent variable (observations)
     * @returns {{ slope: number, icpt: number, ciLo: number, ciHi: number }}
     */
    function theilSen(x, y) {
        const n = x.length;
        const slopes = [];

        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const dx = x[j] - x[i];
                if (dx !== 0) {
                    slopes.push((y[j] - y[i]) / dx);
                }
            }
        }

        slopes.sort((a, b) => a - b);
        const N = slopes.length;
        const slope = median(slopes);
        const icpt = median(y.map((yi, i) => yi - slope * x[i]));

        const varS = mkVariance(y);
        const cAlpha = 1.959964 * Math.sqrt(Math.max(0, varS));
        const m1 = (N - cAlpha) / 2;
        const m2 = (N + cAlpha) / 2;

        const loIdx = Math.max(0, Math.min(N - 1, Math.floor(m1)));
        const hiIdx = Math.max(0, Math.min(N - 1, Math.ceil(m2)));

        return {
            slope,
            icpt,
            ciLo: slopes[loIdx],
            ciHi: slopes[hiIdx]
        };
    }

    /**
     * Runs the full baseline statistical suite (OLS, Mann–Kendall, Theil–Sen) on (x, y).
     *
     * @param {number[]} x - Years
     * @param {number[]} y - Observations
     * @returns {{ ols: Object, mk: Object, sen: Object }}
     */
    function analyse(x, y) {
        return {
            ols: ols(x, y),
            mk: mannKendall(x, y),
            sen: theilSen(x, y)
        };
    }

    return { mean, median, tP, tCrit, ols, mannKendall, theilSen, analyse };
})();

if (typeof window !== "undefined") window.Stats = Stats;
if (typeof module !== "undefined") module.exports = Stats;