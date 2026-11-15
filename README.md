# Mediterranean Sea Climate Observatory

**Team Solari-X · NASA Space Apps Challenge (*"Be An Earth System Trend Detective"*)**

An interactive, client-side Earth system trend observatory for the Mediterranean Sea and three coastal cities (**Alexandria**, **Athens**, and **Barcelona**). Built to help researchers, students, and the public separate genuine multi-decadal climate signals from natural year-to-year variability using autocorrelation-robust statistics, non-parametric rank tests, and false-discovery controls.

---

## Observatory Views

1. **Overview** — Headline trend KPIs with sparklines, key synthesis findings, and a normalized Z-score comparison plotting sea surface temperature, sea-level anomaly, chlorophyll-a, and coastal precipitation on a shared scale.
2. **Data Detective** — Interactive trend investigation workspace featuring region/city selectors, nine Earth system variables, guided detective cases, interactive start/end year sliders, a *"Watch the trend appear"* progressive animation, optional 95% confidence bands, Theil–Sen regression lines, structural shift/hinge overlays, live four-test evidence diagnostics, sliding-window slope stability plots, and one-click CSV export.
3. **Rain Compare** — Multi-city precipitation analysis (NASA GPM IMERG, 2001–2024) comparing Alexandria, Athens, and Barcelona via annual time series, a normalized percentage-per-decade forest plot with 95% Newey–West HAC confidence intervals, pairwise spatial correlation, Benjamini–Hochberg FDR *q*-values, and climatological extremes.
4. **Variables** — Complete metadata catalog covering six basin-wide marine records and three single-grid-cell coastal precipitation records, linked directly to the Data Detective.
5. **Seasonality** — Monthly climatologies illustrating the winter-mixing / summer-stratification phytoplankton cycle (Copernicus Ocean Color chlorophyll-a, 1998–2025) alongside seasonal rainfall volume and peak half-hourly convective rain rates (GPM IMERG, 2001–2024).
6. **Region Map** — Interactive spatial footprint reference comparing the Copernicus Mediterranean sea mask, the KNMI Climate Explorer bounding box, and the 0.1° coastal land grid cells for Alexandria, Athens, and Barcelona.
7. **Statistics** — Reproducible benchmark table reporting full-period OLS slopes, 95% HAC confidence intervals, HAC *p*-values, Theil–Sen slopes, Mann–Kendall $\tau$, and Benjamini–Hochberg *q*-values across all nine datasets.
8. **Guardrails** — Methodological guidance on short-window noise, serial autocorrelation, spatial scale differences between basin averages and 0.1° grid cells, and distinguishing statistical trends from causal attribution.

---

## Project Structure

```text
.
├── .nojekyll              # Disables Jekyll processing for direct GitHub Pages deployment
├── index.html             # Single-page application shell and accessible tab views
├── index.html             # Presentation PDF
├── assets/
│   └── Solari-X_V2.png    # Team Solari-X emblem and favicon
├── css/
│   └── styles.css         # Responsive layout, light/dark theme tokens, and component styles
└── js/
    ├── data.js            # Curated multi-decadal datasets, station metadata, and benchmark stats
    ├── stats.js           # Core statistical engine (OLS, exact Student's t, Mann–Kendall, Theil–Sen)
    ├── adv.js             # Newey–West HAC variance, Benjamini–Hochberg FDR, regime/change-point tools
    └── app.js             # UI state management, Plotly visualizations, SVG map, and interactivity
```

---

## Statistical Methodology

All live window re-computations run directly in the browser with zero backend dependencies:

* **Ordinary Least Squares (OLS) & Lag-1 Autocorrelation (`js/stats.js`)** — Computes linear trend slopes, intercepts, $R^2$, residual standard error, exact two-sided Student's *t* inference (via Lanczos $\ln\Gamma$ and regularized incomplete beta continued fractions), and lag-1 residual autocorrelation ($r_1$).
* **Newey–West HAC Standard Errors (`js/adv.js`)** — Accounts for multi-year climate persistence using Bartlett-weighted residual autocovariances so serial correlation does not artificially deflate *p*-values.
* **Mann–Kendall & Theil–Sen (`js/stats.js`)** — Evaluates monotonic rank order with tie-adjusted variance and continuity correction alongside distribution-free median pairwise slopes and non-parametric 95% confidence bounds.
* **Benjamini–Hochberg FDR (`js/adv.js`)** — Controls the expected false discovery rate across multi-variable and multi-city hypothesis families.

---

## Data Sources

* **NOAA OI SST v2** — Mediterranean Sea surface temperature
* **Met Office Hadley Centre (HadISST1 & EN4)** — Historical SST, upper-ocean heat content, and salinity
* **Copernicus Marine Service** — Mediterranean Sea level anomaly and Ocean Color chlorophyll-a (1998–2025)
* **KNMI Climate Explorer** — Basin-averaged marine time series extraction
* **NASA GPM IMERG** — Coastal precipitation totals and half-hourly rain rates for Alexandria, Athens, and Barcelona (2001–2024)
