"""
Zero-Coupon Spot Curve: Nelson-Siegel-Svensson Fitted to Bond Prices
=====================================================================

Author: Quang Le
Date: 5 November 2026
Version: 3.0 (NSS fitted directly to dirty prices)

Output
------
A zero-coupon spot curve r(t) in Nelson-Siegel-Svensson (NSS) form: annual
compounding, t in years = days / 365.25. A payment due at t is worth
(1 + r(t)) ** (-t) today. Same NSS function and parameters as earlier versions.

Method
------
1. Cash flows. Each bond's remaining coupons are generated back from maturity at
   its coupon frequency (a month-end maturity keeps coupons on month ends). If the
   file has a 'First Settlement Date' later than the previous regular coupon date,
   the next coupon is a short first coupon: the regular coupon times the accrual
   fraction (ACT/ACT: actual days / actual days in the regular period;
   30E/360 and ISMA 30/360: 30/360 days / days in the regular period).
2. Schedule check (rule-based exclusion). The GRY of each bond's Dirty Price on its
   modelled cash flows is compared with the quoted 'Annual Yield'. GRYs here use the
   quoted basis: annual compounding, time counted in coupon periods (ISMA).
   A difference above 0.5 bp means the modelled cash flows are not those behind the
   quoted yield; the bond is excluded.
3. Fit. NSS parameters minimise
       sum_i  w_i * [ (P_model_i - P_dirty_i) / (P_dirty_i * D_i) ] ** 2
   w_i = index weight, D_i = modified duration at the quoted yield, so each term is
   approximately the bond's squared GRY error. P_model_i discounts every cash flow
   of bond i on the NSS spot curve. Multi-start L-BFGS-B, objective in bp^2.
4. Outliers, one pass. Bonds whose NSS-curve GRY differs from the quoted GRY by
   more than the threshold are excluded and the curve is refitted once.

Settlement: T+0, i.e. prices are for settlement on the report date (iBoxx convention).

Interface
---------
Class name, constructor arguments and method names are those of the bootstrap
versions, so the Streamlit app runs this script unchanged:
  load_and_prepare_data()  - load, parse, filter
  run_robust_bootstrap()   - steps 1-2 (name kept for the app; no bootstrap is run)
  fit_nss_curve_robust()   - steps 3-4; returns the NSS parameter dictionary
  plot_results(), export_results(), create_comprehensive_report()

Replaces v041126 (bootstrap to dirty price, then NSS fitted to bootstrapped spots).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize, brentq
from dateutil.relativedelta import relativedelta
import warnings
import sys
import os
import argparse
warnings.filterwarnings('ignore')

# --- Project folders ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from /scripts
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available - plots will be skipped")


class Logger:
    """Logger class to capture console output and save to file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class RobustBootstrapSpotRates:
    """NSS zero-coupon spot curve fitted to dirty prices (class name kept for the app)."""

    SCHEDULE_CHECK_BPS = 0.5          # max |GRY(dirty, modelled flows) - quoted GRY|
    NSS_BOUNDS = [(0.001, 0.20), (-0.20, 0.20), (-0.20, 0.20),
                  (-0.20, 0.20), (0.1, 30.0), (0.1, 30.0)]   # as in v041126
    RANDOM_STARTS = 20
    RANDOM_SEED = 12345               # fixed: the same file gives the same curve

    def __init__(self, csv_file_path, report_date='2025-07-31', outlier_threshold_bps=50.0,
                 min_mty_years=0.01, max_mty_years=100.0,
                 min_clean_price=10.0, max_clean_price=200.0,
                 min_coupon=0.0, max_coupon=50.0,
                 allowed_freq=(1, 2, 4, 12),
                 min_ytm=0.0, max_ytm=50.0,
                 exclude_isins=None):
        """
        Parameters
        ----------
        csv_file_path : str
            CSV with bond data (ISIN, Index Weight, Maturity Date, Coupon in %,
            Coupon Frequency, Dirty Price, Annual Yield; optional First Settlement
            Date, Accrued Interest, Day Count, Report Date).
        report_date : str
            'YYYY-MM-DD'. Prices are taken as settling on this date (T+0).
        outlier_threshold_bps : float
            Bonds whose NSS-curve GRY differs from the quoted GRY by more than this
            are excluded and the curve is refitted once.
        Remaining arguments: filters applied before fitting, as in earlier versions.
        """
        self.csv_file_path = csv_file_path
        self.report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
        self.outlier_threshold_bps = float(outlier_threshold_bps)

        self.min_mty_years = float(min_mty_years)
        self.max_mty_years = float(max_mty_years)
        self.min_clean_price = float(min_clean_price)
        self.max_clean_price = float(max_clean_price)
        self.min_coupon = float(min_coupon)
        self.max_coupon = float(max_coupon)
        self.allowed_freq = tuple(allowed_freq) if allowed_freq is not None else (1, 2, 4, 12)
        self.min_ytm = float(min_ytm)
        self.max_ytm = float(max_ytm)
        self.exclude_isins = set(exclude_isins or [])

        self.bonds_df = None
        self.results_df = None
        self.nss_parameters = None
        self.cash_flows = []          # per bond: dict(dates, amounts, times, prev_regular, first_fraction)
        self.file_report_dates = []

    # ------------------------------------------------------------------ data

    @staticmethod
    def _parse_dates(series, name):
        """Parse a date column as YYYY-MM-DD or DD/MM/YYYY, whichever reads more rows.
        Each format is applied strictly, so day and month are never swapped."""
        s = series.astype(str).str.strip()
        best = None
        for fmt in ('%Y-%m-%d', '%d/%m/%Y'):
            parsed = pd.to_datetime(s, format=fmt, errors='coerce')
            if best is None or parsed.notna().sum() > best.notna().sum():
                best = parsed
        if best.notna().sum() == 0:
            raise ValueError(f"Column '{name}': no dates in YYYY-MM-DD or DD/MM/YYYY format")
        return best

    def load_and_prepare_data(self):
        """Load the CSV, parse dates, apply filters, sort by maturity."""
        print("Loading bond data...")
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded {len(df)} bonds")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

        required = ['ISIN', 'Maturity Date', 'Index Weight', 'Coupon in %',
                    'Coupon Frequency', 'Dirty Price', 'Annual Yield']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: Missing required columns: {missing}")
            return False
        if 'Accrued Interest' not in df.columns:
            df['Accrued Interest'] = 0.0
        if 'Day Count' not in df.columns:
            df['Day Count'] = 'ACT/ACT'

        numeric = ['Index Weight', 'Coupon in %', 'Coupon Frequency',
                   'Dirty Price', 'Accrued Interest', 'Annual Yield']
        for c in numeric:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        bad = df[numeric].isna().any(axis=1)
        if bad.any():
            print(f"Warning: dropping {bad.sum()} bonds with invalid numeric data")
            df = df[~bad]

        try:
            df['Maturity Date'] = self._parse_dates(df['Maturity Date'], 'Maturity Date')
            bad = df['Maturity Date'].isna()
            if bad.any():
                print(f"Warning: dropping {bad.sum()} bonds with unreadable maturity dates")
                df = df[~bad]
            if 'First Settlement Date' in df.columns:
                try:
                    df['First Settlement Date'] = self._parse_dates(df['First Settlement Date'], 'First Settlement Date')
                except ValueError:
                    df['First Settlement Date'] = pd.NaT
                n_missing = df['First Settlement Date'].isna().sum()
                if n_missing:
                    print(f"Warning: {n_missing} bonds without a readable First Settlement Date "
                          f"(their next coupon is taken as regular)")
            else:
                df['First Settlement Date'] = pd.NaT
                print("Note: no 'First Settlement Date' column; every next coupon is taken as regular")
            if 'Report Date' in df.columns and df['Report Date'].notna().any():
                rd = self._parse_dates(df['Report Date'], 'Report Date').dropna().dt.date.unique()
                self.file_report_dates = sorted(rd)
                if len(rd) and any(d != self.report_date for d in rd):
                    print(f"Warning: report date {self.report_date} differs from the file's "
                          f"Report Date {', '.join(str(d) for d in rd)}")
        except Exception as e:
            print(f"Error processing dates: {e}")
            return False

        df['Clean_Price'] = df['Dirty Price'] - df['Accrued Interest']
        rpt = pd.Timestamp(self.report_date)
        df['Days_to_Maturity'] = (df['Maturity Date'] - rpt).dt.days
        df['Time_to_Maturity_Years'] = df['Days_to_Maturity'] / 365.25

        initial = len(df)
        if self.exclude_isins:
            before = len(df)
            df = df[~df['ISIN'].isin(self.exclude_isins)].copy()
            if before - len(df):
                print(f"Excluded by ISIN list: {before - len(df)}")

        keep = (
            (df['Time_to_Maturity_Years'] > self.min_mty_years) &
            (df['Time_to_Maturity_Years'] < self.max_mty_years) &
            (df['Clean_Price'] > self.min_clean_price) &
            (df['Clean_Price'] < self.max_clean_price) &
            (df['Coupon in %'] >= self.min_coupon) &
            (df['Coupon in %'] <= self.max_coupon) &
            (df['Coupon Frequency'].isin(self.allowed_freq)) &
            (df['Annual Yield'] > self.min_ytm) &
            (df['Annual Yield'] < self.max_ytm)
        )
        df = df[keep].sort_values('Time_to_Maturity_Years').reset_index(drop=True)
        if initial != len(df):
            print(f"Filtered out {initial - len(df)} bonds")
        if len(df) == 0:
            print("Error: No valid bonds remaining after filtering")
            return False

        self.bonds_df = df
        print(f"Processing {len(df)} bonds; maturity {df['Time_to_Maturity_Years'].min():.4f} "
              f"to {df['Time_to_Maturity_Years'].max():.4f} years")
        return True

    # ------------------------------------------------------------- cash flows

    @staticmethod
    def _days_30e_360(d1, d2):
        """Day count between d1 and d2 under 30E/360."""
        return (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month)
                + (min(d2.day, 30) - min(d1.day, 30)))

    def _first_coupon_fraction(self, accrual_start, first_coupon, prev_regular, day_count, frequency):
        """Fraction of a regular coupon accrued from accrual_start to first_coupon."""
        dc = str(day_count).upper().replace('-', ' ')
        if '30' in dc and '360' in dc:
            return self._days_30e_360(accrual_start, first_coupon) / (360.0 / frequency)
        return (first_coupon - accrual_start).days / (first_coupon - prev_regular).days

    def _build_cash_flows(self, bond_row):
        """
        Remaining payments of a bond, per 100 nominal.
        Coupon dates are maturity minus whole periods, each counted from maturity
        (a month-end maturity keeps month-end dates). prev_regular is the regular
        coupon date on or before the report date. If the bond's First Settlement Date
        is after prev_regular, the next coupon is prorated (short first coupon).
        """
        maturity = bond_row['Maturity Date'].date()
        f = int(bond_row['Coupon Frequency'])
        if f not in (1, 2, 4, 12):
            raise ValueError(f"Unsupported frequency {f} for {bond_row['ISIN']}")
        months = 12 // f
        month_end = maturity == maturity + relativedelta(day=31)
        regular_coupon = bond_row['Coupon in %'] / f

        dates, n = [], 0
        while True:
            d = maturity - relativedelta(months=months * n)
            if month_end:
                d = d + relativedelta(day=31)
            if d <= self.report_date:
                prev_regular = d
                break
            dates.append(d)
            n += 1
        dates.reverse()
        if not dates:
            raise ValueError(f"No future payments for {bond_row['ISIN']}")

        amounts = [regular_coupon] * len(dates)
        first_fraction = 1.0
        settle = bond_row.get('First Settlement Date', pd.NaT)
        if pd.notna(settle) and settle.date() > prev_regular:
            first_fraction = self._first_coupon_fraction(settle.date(), dates[0], prev_regular,
                                                         bond_row.get('Day Count', 'ACT/ACT'), f)
            amounts[0] = regular_coupon * first_fraction
        amounts[-1] += 100.0
        times = [(d - self.report_date).days / 365.25 for d in dates]
        return {'dates': dates, 'amounts': amounts, 'times': times,
                'prev_regular': prev_regular, 'first_fraction': first_fraction, 'frequency': f}

    def calculate_exact_payment_schedule(self, bond_row):
        """[(payment date, amount per 100, time in years)] - kept for compatibility."""
        cf = self._build_cash_flows(bond_row)
        return list(zip(cf['dates'], cf['amounts'], cf['times']))

    # ------------------------------------------------------------------ yields

    def _gry_times(self, cf):
        """Time to each payment in years on the quoted basis: (i + w) / f, with
        w = (first payment - report date) / (first payment - previous regular date)."""
        first = cf['dates'][0]
        w = (first - self.report_date).days / (first - cf['prev_regular']).days
        f = cf['frequency']
        return np.array([(i + w) / f for i in range(len(cf['dates']))])

    def gross_redemption_yield(self, price, cf):
        """GRY (decimal) on the quoted 'Annual Yield' basis: annual compounding,
        ISMA period fractions. NaN if no root in [-50%, 100%]."""
        tau = self._gry_times(cf)
        amounts = np.array(cf['amounts'])
        try:
            return brentq(lambda y: np.sum(amounts * (1.0 + y) ** (-tau)) - price,
                          -0.5, 1.0, xtol=1e-14, maxiter=500)
        except ValueError:
            return float('nan')

    def _modified_duration(self, gry, cf):
        """Modified duration at yield gry on the quoted basis."""
        tau = self._gry_times(cf)
        amounts = np.array(cf['amounts'])
        pv = amounts * (1.0 + gry) ** (-tau)
        return float(np.sum(tau * pv) / ((1.0 + gry) * np.sum(pv)))

    # ------------------------------------------------------------- preparation

    def run_robust_bootstrap(self):
        """
        Preparation (name kept for the app; no bootstrap is run): build cash flows,
        compute the schedule check and the modified duration of every bond, and mark
        bonds that fail the schedule check.
        """
        if self.bonds_df is None:
            print("Error: No bond data loaded")
            return False

        print("\nPreparing cash flows and schedule check...")
        self.cash_flows, rows = [], []
        for _, b in self.bonds_df.iterrows():
            cf = self._build_cash_flows(b)
            self.cash_flows.append(cf)
            quoted = b['Annual Yield'] / 100
            market_gry = self.gross_redemption_yield(b['Dirty Price'], cf)
            check = (market_gry - quoted) * 10000
            failed = not (abs(check) <= self.SCHEDULE_CHECK_BPS)      # NaN fails too
            settle = b['First Settlement Date']
            rows.append({
                'ISIN': b['ISIN'],
                'Maturity_Date': b['Maturity Date'].strftime('%Y-%m-%d'),
                'First_Settlement_Date': settle.strftime('%Y-%m-%d') if pd.notna(settle) else '',
                'Time_to_Maturity_Years': b['Time_to_Maturity_Years'],
                'Index_Weight_Percent': b['Index Weight'],
                'Coupon_Rate_Percent': b['Coupon in %'],
                'Coupon_Frequency': int(b['Coupon Frequency']),
                'Day_Count': b['Day Count'],
                'Next_Coupon_Date': cf['dates'][0].strftime('%Y-%m-%d'),
                'Next_Coupon_Amount': cf['amounts'][0] - (100.0 if len(cf['dates']) == 1 else 0.0),
                'First_Coupon_Fraction': cf['first_fraction'],
                'Dirty_Price': b['Dirty Price'],
                'Quoted_GRY_Percent': b['Annual Yield'],
                'Schedule_Check_bps': check,
                'Modified_Duration': self._modified_duration(quoted, cf),
                'Status': 'excluded_schedule_check' if failed else 'fitted',
            })
        self.results_df = pd.DataFrame(rows)
        self.results_df['Bootstrap_Method'] = self.results_df['Status']   # read by the app

        # Flattened cash flows for fast pricing
        self._cf_bond = np.concatenate([[i] * len(cf['times']) for i, cf in enumerate(self.cash_flows)])
        self._cf_time = np.concatenate([cf['times'] for cf in self.cash_flows])
        self._cf_amount = np.concatenate([cf['amounts'] for cf in self.cash_flows])

        n_short = int((self.results_df['First_Coupon_Fraction'] < 1).sum())
        failed = self.results_df[self.results_df['Status'] == 'excluded_schedule_check']
        print(f"  Bonds: {len(self.results_df)}; short first coupons modelled: {n_short}")
        print(f"  Schedule check failures (> {self.SCHEDULE_CHECK_BPS} bps): {len(failed)}")
        for _, r in failed.iterrows():
            print(f"    {r['ISIN']}: {r['Schedule_Check_bps']:+.3f} bps")
        return True

    # --------------------------------------------------------------- NSS fit

    def nelson_siegel_svensson(self, t, beta0, beta1, beta2, beta3, tau1, tau2):
        """Nelson-Siegel-Svensson spot rate (unchanged from earlier versions)"""
        t = np.maximum(t, 1e-6)

        term1 = beta0
        term2 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
        term3 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
        term4 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))

        return term1 + term2 + term3 + term4

    def _model_prices(self, params, idx):
        """Prices of bonds idx (positions in results_df) discounted on the NSS curve."""
        sel = np.isin(self._cf_bond, idx)
        t, a = self._cf_time[sel], self._cf_amount[sel]
        local = np.searchsorted(idx, self._cf_bond[sel])
        df = np.where(t > 0, (1.0 + self.nelson_siegel_svensson(t, *params)) ** (-t), 1.0)
        return np.bincount(local, weights=a * df, minlength=len(idx))

    def _fit_prices(self, idx):
        """Fit NSS to the dirty prices of bonds idx. Returns (params, info)."""
        idx = np.sort(np.asarray(idx))
        r = self.results_df
        P = r['Dirty_Price'].values[idx]
        D = r['Modified_Duration'].values[idx]
        w = r['Index_Weight_Percent'].values[idx]
        sel = np.isin(self._cf_bond, idx)
        t, a = self._cf_time[sel], self._cf_amount[sel]
        local = np.searchsorted(idx, self._cf_bond[sel])
        pos = t > 0

        def objective(q):
            rate = self.nelson_siegel_svensson(t, *q)
            if np.any(1.0 + rate[pos] <= 0):
                return 1e12
            df = np.where(pos, (1.0 + rate) ** (-t), 1.0)
            model = np.bincount(local, weights=a * df, minlength=len(idx))
            e = (model - P) / (P * D) * 10000.0                 # ~ GRY error in bp
            return float(np.sum(w * e * e))

        y = r['Quoted_GRY_Percent'].values[idx] / 100
        T = r['Time_to_Maturity_Years'].values[idx]
        level, short = np.average(y, weights=w), y[np.argmin(T)]
        starts = [[level, short - level, 0.0, 0.0, t1, t2]
                  for t1 in (0.5, 1.0, 2.0, 5.0) for t2 in (5.0, 10.0, 20.0, 30.0) if t2 > t1]
        rng = np.random.default_rng(self.RANDOM_SEED)
        for _ in range(self.RANDOM_STARTS):
            starts.append([rng.uniform(0.02, 0.10), rng.uniform(-0.04, 0.02), rng.uniform(-0.06, 0.06),
                           rng.uniform(-0.06, 0.06), rng.uniform(0.3, 10.0), rng.uniform(2.0, 30.0)])

        best = None
        for x0 in starts:
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=self.NSS_BOUNDS,
                               options={'maxiter': 5000})
            except Exception:
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            return None, None
        at_bound = [name for name, v, (lo, hi) in
                    zip(['beta0', 'beta1', 'beta2', 'beta3', 'tau1', 'tau2'], best.x, self.NSS_BOUNDS)
                    if min(abs(v - lo), abs(v - hi)) < 1e-4 * (hi - lo)]
        info = {'objective_bp2': float(best.fun), 'converged': bool(best.success),
                'message': str(best.message), 'starts': len(starts), 'params_at_bound': at_bound,
                'weighted_rms_price_bps': float(np.sqrt(best.fun / w.sum()))}
        return best.x, info

    def _nss_gry_errors(self, params):
        """For every bond: (model price, NSS-curve GRY in %, NSS GRY - quoted GRY in bp)."""
        n = len(self.results_df)
        prices = self._model_prices(params, np.arange(n))
        gry = np.array([self.gross_redemption_yield(p, cf) for p, cf in zip(prices, self.cash_flows)])
        err = (gry - self.results_df['Quoted_GRY_Percent'].values / 100) * 10000
        return prices, gry * 100, err

    @staticmethod
    def _weighted_stats(err_bps, quoted_pct, w):
        ok = ~np.isnan(err_bps)
        e, y, w = err_bps[ok] / 10000, quoted_pct[ok] / 100, w[ok]
        rms = float(np.sqrt(np.average((e * 10000) ** 2, weights=w)))
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        return rms, float(1 - np.sum(w * e ** 2) / ss_tot)

    def fit_nss_curve_robust(self):
        """
        Fit NSS to the dirty prices of bonds that passed the schedule check; exclude
        bonds whose NSS-curve GRY differs from the quoted GRY by more than the
        threshold; refit once. Returns the parameter dictionary (keys as before).
        """
        if self.results_df is None:
            print("Error: run_robust_bootstrap() (preparation) has not been run")
            return None
        r = self.results_df
        thr = self.outlier_threshold_bps
        fit_idx = np.where(r['Status'] == 'fitted')[0]
        if len(fit_idx) < 6:
            print("Error: Too few bonds for NSS fitting")
            return None

        print(f"\nFit 1: NSS to the dirty prices of {len(fit_idx)} bonds...")
        params1, info1 = self._fit_prices(fit_idx)
        if params1 is None:
            print("NSS optimisation failed")
            return None
        _, _, err1 = self._nss_gry_errors(params1)
        r['Fit1_GRY_minus_Quoted_bps'] = np.where(r['Status'] == 'fitted', err1, np.nan)
        outliers = [i for i in fit_idx if abs(err1[i]) > thr]
        print(f"  Weighted RMS error {info1['weighted_rms_price_bps']:.2f} bps; "
              f"outliers (|NSS GRY - quoted GRY| > {thr} bps): {len(outliers)}")
        for i in outliers:
            print(f"    {r['ISIN'].iloc[i]}: {err1[i]:+.2f} bps")

        if outliers:
            r.loc[outliers, 'Status'] = 'excluded_outlier'
            final_idx = np.where(r['Status'] == 'fitted')[0]
            if len(final_idx) < 6:
                print("Error: Too few bonds left after outlier exclusion")
                return None
            print(f"Fit 2: refit to {len(final_idx)} bonds...")
            params, info = self._fit_prices(final_idx)
            if params is None:
                print("NSS optimisation failed")
                return None
        else:
            final_idx, params, info = fit_idx, params1, info1
        r['Bootstrap_Method'] = r['Status']

        prices, gry, err = self._nss_gry_errors(params)
        r['Model_Price'] = prices
        r['Price_Error'] = prices - r['Dirty_Price'].values
        r['NSS_Spot_At_Maturity_Percent'] = self.nelson_siegel_svensson(
            r['Time_to_Maturity_Years'].values, *params) * 100
        r['NSS_GRY_Percent'] = gry
        r['NSS_GRY_minus_Quoted_bps'] = err

        w = r['Index_Weight_Percent'].values
        q = r['Quoted_GRY_Percent'].values
        fitted = (r['Status'] == 'fitted').values
        rms_fit, r2_fit = self._weighted_stats(err[fitted], q[fitted], w[fitted])
        rms_all, r2_all = self._weighted_stats(err, q, w)

        # Forward rates on a quarterly grid out to the longest fitted maturity
        T_max = float(r['Time_to_Maturity_Years'].values[fitted].max())
        grid = np.arange(0.25, T_max + 0.25, 0.25)
        lnD = -grid * np.log1p(self.nelson_siegel_svensson(grid, *params))
        fwd = np.expm1(-np.diff(lnD) / np.diff(grid))

        beta0, beta1, beta2, beta3, tau1, tau2 = params
        self.nss_parameters = {
            'beta0': beta0, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3,
            'tau1': tau1, 'tau2': tau2,
            # R^2 on the GRY basis: NSS-curve GRY vs quoted GRY, index-weighted
            'weighted_r_squared_clean': r2_fit,          # bonds in the final fit
            'weighted_r_squared_all': r2_all,            # all bonds, excluded ones included
            'weighted_rms_gry_bps_fit': rms_fit,
            'weighted_rms_gry_bps_all': rms_all,
            'outlier_count': len(outliers),
            'outlier_isins': [r['ISIN'].iloc[i] for i in outliers],
            'schedule_excluded_isins': list(r.loc[r['Status'] == 'excluded_schedule_check', 'ISIN']),
            'outlier_mask': (r['Status'] != 'fitted').values,
            'clean_count': int(fitted.sum()),
            'outlier_threshold_bps': thr,
            'fit_basis': 'dirty prices, duration-scaled (GRY-equivalent) errors',
            'r_squared_basis': 'GRY',
            'optimiser': info,
            'first_fit': info1,
            'forward_min': float(fwd.min()), 'forward_max': float(fwd.max()),
            'forward_negative_count': int((fwd < 0).sum()),
            'longest_fitted_maturity': T_max,
        }

        print("NSS fit completed:")
        print(f"  Bonds in fit: {fitted.sum()}; excluded: schedule check "
              f"{len(self.nss_parameters['schedule_excluded_isins'])}, outliers {len(outliers)}")
        print(f"  GRY R² (in fit / all): {r2_fit:.6f} / {r2_all:.6f}; "
              f"weighted RMS GRY error {rms_fit:.2f} / {rms_all:.2f} bps")
        print(f"  Forward rates to {T_max:.1f}y: {fwd.min()*100:.3f}% to {fwd.max()*100:.3f}%")
        if info['params_at_bound']:
            print(f"  Note: parameters on a bound: {', '.join(info['params_at_bound'])}")
        return self.nss_parameters

    # ------------------------------------------------------------------ output

    def export_results(self, output_filename='robust_bootstrap_results.csv'):
        """Per-bond results, every bond with its Status."""
        if self.results_df is None:
            return False
        try:
            self.results_df.to_csv(output_filename, index=False, float_format='%.8g')
            print(f"Results exported to: {output_filename}")
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

    def create_comprehensive_report(self, filename='bootstrap_comprehensive_report.txt'):
        """Text report: data, exclusions, fit statistics, spot curve, per-bond results."""
        if self.results_df is None:
            print("Error: No results available for report.")
            return
        r = self.results_df
        p = self.nss_parameters
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("ZERO-COUPON SPOT CURVE: NSS FITTED TO BOND PRICES\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Report Date: {self.report_date} (T+0 settlement)\n")
                if self.file_report_dates and any(d != self.report_date for d in self.file_report_dates):
                    f.write(f"WARNING: file Report Date is {', '.join(str(d) for d in self.file_report_dates)}\n")
                f.write(f"Data file: {os.path.basename(self.csv_file_path)}\n")
                f.write(f"Outlier threshold: {self.outlier_threshold_bps} bps (|NSS-curve GRY - quoted GRY|)\n")
                f.write("Spot rates: annual compounding, t = days / 365.25. "
                        "GRYs: annual compounding, ISMA period fractions.\n\n")

                f.write("DATA AND EXCLUSIONS\n" + "-" * 19 + "\n")
                f.write(f"Bonds after filters: {len(r)}\n")
                f.write(f"Short first coupons modelled from First Settlement Date: "
                        f"{int((r['First_Coupon_Fraction'] < 1).sum())}\n")
                for _, x in r[r['First_Coupon_Fraction'] < 1].iterrows():
                    f.write(f"    {x['ISIN']}: settled {x['First_Settlement_Date']}, next coupon "
                            f"{x['Next_Coupon_Date']} = {x['Next_Coupon_Amount']:.4f} "
                            f"({x['First_Coupon_Fraction']:.4f} of a regular coupon)\n")
                sc = r[r['Status'] == 'excluded_schedule_check']
                f.write(f"Excluded by schedule check (> {self.SCHEDULE_CHECK_BPS} bps): {len(sc)}\n")
                for _, x in sc.iterrows():
                    f.write(f"    {x['ISIN']}: {x['Schedule_Check_bps']:+.3f} bps\n")
                if p:
                    f.write(f"Excluded as outliers in fit 1: {p['outlier_count']}\n")
                    for isin in p['outlier_isins']:
                        v = r.loc[r['ISIN'] == isin, 'Fit1_GRY_minus_Quoted_bps'].iloc[0]
                        f.write(f"    {isin}: {v:+.2f} bps\n")
                    f.write(f"Bonds in final fit: {p['clean_count']}\n\n")

                    f.write("NSS FIT\n" + "-" * 7 + "\n")
                    f.write(f"Fitted to: {p['fit_basis']}, index-weighted\n")
                    f.write(f"GRY R^2: {p['weighted_r_squared_clean']:.6f} (bonds in fit), "
                            f"{p['weighted_r_squared_all']:.6f} (all bonds)\n")
                    f.write(f"Weighted RMS GRY error: {p['weighted_rms_gry_bps_fit']:.2f} bps (bonds in fit), "
                            f"{p['weighted_rms_gry_bps_all']:.2f} bps (all bonds)\n")
                    o = p['optimiser']
                    f.write(f"Optimiser: {o['starts']} starts, converged={o['converged']}, "
                            f"parameters on a bound: {', '.join(o['params_at_bound']) or 'none'}\n")
                    f.write(f"Forward rates to {p['longest_fitted_maturity']:.1f}y (quarterly): "
                            f"{p['forward_min']*100:.3f}% to {p['forward_max']*100:.3f}%, "
                            f"negative: {p['forward_negative_count']}\n\n")
                    f.write("NSS Parameters:\n")
                    f.write(f"  beta0: {p['beta0']*100:.4f}%\n  beta1: {p['beta1']*100:.4f}%\n")
                    f.write(f"  beta2: {p['beta2']*100:.4f}%\n  beta3: {p['beta3']*100:.4f}%\n")
                    f.write(f"  tau1: {p['tau1']:.4f} years\n  tau2: {p['tau2']:.4f} years\n")
                    f.write(f"  tau2/tau1: {p['tau2']/p['tau1']:.3f} (near 1: the beta2 and beta3 terms have "
                            f"nearly the same shape and offset each other)\n\n")

                    f.write("ZERO-COUPON SPOT CURVE\n" + "-" * 22 + "\n")
                    f.write("Maturity (years) | Spot rate (%) | Discount factor\n")
                    params = [p[k] for k in ['beta0', 'beta1', 'beta2', 'beta3', 'tau1', 'tau2']]
                    for t in [0.5, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]:
                        s = float(self.nelson_siegel_svensson(t, *params))
                        note = '  (beyond longest fitted maturity)' if t > p['longest_fitted_maturity'] else ''
                        f.write(f"{t:16.1f} | {s*100:13.4f} | {(1+s)**(-t):15.6f}{note}\n")
                    f.write("\n")

                f.write("PER-BOND RESULTS\n" + "-" * 16 + "\n")
                f.write(f"{'ISIN':<14}{'Mty(y)':>8}{'Wt%':>7}{'Cpn%':>7}{'Dirty':>10}{'Model':>10}"
                        f"{'QuotedGRY':>10}{'NSS GRY':>9}{'Err bp':>8}  Status\n")
                for _, x in r.iterrows():
                    mp = x.get('Model_Price', np.nan)
                    ng = x.get('NSS_GRY_Percent', np.nan)
                    er = x.get('NSS_GRY_minus_Quoted_bps', np.nan)
                    f.write(f"{x['ISIN']:<14}{x['Time_to_Maturity_Years']:8.3f}{x['Index_Weight_Percent']:7.3f}"
                            f"{x['Coupon_Rate_Percent']:7.3f}{x['Dirty_Price']:10.4f}{mp:10.4f}"
                            f"{x['Quoted_GRY_Percent']:10.4f}{ng:9.4f}{er:8.2f}  {x['Status']}\n")
            print(f"Report saved to: {filename}")
        except Exception as e:
            print(f"Error creating report: {e}")

    def plot_results(self, save_plot=True, filename='robust_bootstrap_analysis.png'):
        """Four panels: spot and forward curve; GRYs; GRY errors; bond status."""
        if self.results_df is None or not MATPLOTLIB_AVAILABLE or not self.nss_parameters:
            print("Cannot create plots - no fit results or matplotlib unavailable")
            return
        r = self.results_df
        p = self.nss_parameters
        params = [p[k] for k in ['beta0', 'beta1', 'beta2', 'beta3', 'tau1', 'tau2']]
        T = r['Time_to_Maturity_Years'].values
        w = r['Index_Weight_Percent'].values
        status = r['Status'].values
        fit = status == 'fitted'
        outl = status == 'excluded_outlier'
        sched = status == 'excluded_schedule_check'
        thr = self.outlier_threshold_bps

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Zero-Coupon Spot Curve: NSS Fitted to Bond Prices', fontsize=16, fontweight='bold')

        # 1: spot and forward curves
        grid = np.linspace(0.25, max(T.max(), 1.0), 400)
        spot = self.nelson_siegel_svensson(grid, *params)
        lnD = -grid * np.log1p(spot)
        fwd = np.expm1(-np.diff(lnD) / np.diff(grid))
        ax1.plot(grid, spot * 100, 'red', linewidth=2, label='NSS spot curve')
        ax1.plot(grid[1:], fwd * 100, 'red', linestyle='--', alpha=0.6, label='Implied forward rate')
        ax1.plot(T[fit], np.full(fit.sum(), np.nanmin(spot * 100) - 0.2), '|', color='grey',
                 markersize=12, label='Maturities of bonds in fit')
        ax1.axvline(p['longest_fitted_maturity'], color='grey', linestyle=':', alpha=0.7)
        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Rate (%, annual compounding)')
        ax1.set_title('Zero-Coupon Spot Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2: GRYs, quoted vs NSS curve
        ax2.scatter(T[fit], r['Quoted_GRY_Percent'].values[fit], s=w[fit]*20, c='orange', alpha=0.7,
                    label='Quoted GRY')
        ax2.scatter(T[fit], r['NSS_GRY_Percent'].values[fit], s=15, c='blue', alpha=0.8,
                    label='NSS-curve GRY')
        if (~fit).any():
            ax2.scatter(T[~fit], r['Quoted_GRY_Percent'].values[~fit], marker='x', c='red', s=40,
                        label='Quoted GRY, excluded bond')
        ax2.set_xlabel('Maturity (Years)')
        ax2.set_ylabel('GRY (%, annual compounding)')
        ax2.set_title(f"GRYs (R²={p['weighted_r_squared_clean']:.3f}, "
                      f"RMS={p['weighted_rms_gry_bps_fit']:.1f} bps, bonds in fit)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3: GRY errors
        err = r['NSS_GRY_minus_Quoted_bps'].values
        inside = fit & (np.abs(err) <= thr)
        beyond = fit & (np.abs(err) > thr)
        ax3.scatter(T[inside], err[inside], c='green', s=w[inside]*15, alpha=0.7, label='In fit')
        if beyond.any():
            ax3.scatter(T[beyond], err[beyond], c='orange', s=w[beyond]*15, alpha=0.8,
                        label='In fit, beyond threshold after refit')
        if outl.any():
            ax3.scatter(T[outl], err[outl], c='red', marker='x', s=40, label='Excluded as outlier')
        if sched.any():
            ax3.scatter(T[sched], err[sched], c='grey', marker='x', s=40, label='Excluded by schedule check')
        ax3.axhline(0, color='black', alpha=0.5)
        ax3.axhline(thr, color='red', linestyle='--', alpha=0.7, label=f'±{thr} bps threshold')
        ax3.axhline(-thr, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Maturity (Years)')
        ax3.set_ylabel('NSS-curve GRY - quoted GRY (bps)')
        ax3.set_title('Pricing Errors in GRY Terms')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4: bond status
        counts = r['Status'].value_counts()
        wedges, _ = ax4.pie(counts.values, startangle=90)
        ax4.legend(wedges, [f"{k}: {v} bond{'s' if v != 1 else ''} ({v / counts.sum():.1%})" for k, v in counts.items()],
                   loc='lower center', bbox_to_anchor=(0.5, -0.12))
        ax4.set_title('Bond Status')

        plt.tight_layout()
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {filename}")
        plt.show()


def main():
    """Command-line run: fit, then write chart, CSV and report to Outputs."""
    parser = argparse.ArgumentParser(description="Fit an NSS zero-coupon spot curve to bond prices.")
    parser.add_argument("--csv", required=True, help="CSV file name under ./Data/")
    parser.add_argument("--report_date", required=True, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--outlier_bps", type=float, default=50.0)
    parser.add_argument("--min_mty", type=float, default=0.01)
    parser.add_argument("--max_mty", type=float, default=100.0)
    parser.add_argument("--min_px", type=float, default=10.0)
    parser.add_argument("--max_px", type=float, default=200.0)
    parser.add_argument("--min_coupon", type=float, default=0.0)
    parser.add_argument("--max_coupon", type=float, default=50.0)
    parser.add_argument("--min_ytm", type=float, default=0.0)
    parser.add_argument("--max_ytm", type=float, default=50.0)
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated ISINs to exclude")
    args = parser.parse_args()

    csv_path    = os.path.join(DATA_DIR, args.csv)
    output_csv  = os.path.join(OUTPUT_DIR, "robust_bootstrap_results.csv")
    report_path = os.path.join(OUTPUT_DIR, "bootstrap_comprehensive_report.txt")
    plot_path   = os.path.join(OUTPUT_DIR, "robust_bootstrap_analysis.png")
    exclude_isins = [s.strip() for s in args.exclude.split(',') if s.strip()]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(OUTPUT_DIR, f'nss_price_fit_log_{timestamp}.txt')
    logger = Logger(log_filename)
    sys.stdout = logger
    try:
        calc = RobustBootstrapSpotRates(
            csv_file_path=csv_path, report_date=args.report_date,
            outlier_threshold_bps=args.outlier_bps,
            min_mty_years=args.min_mty, max_mty_years=args.max_mty,
            min_clean_price=args.min_px, max_clean_price=args.max_px,
            min_coupon=args.min_coupon, max_coupon=args.max_coupon,
            min_ytm=args.min_ytm, max_ytm=args.max_ytm,
            exclude_isins=exclude_isins)
        if not calc.load_and_prepare_data():
            print("Failed to load data")
            return
        if not calc.run_robust_bootstrap():
            print("Preparation failed")
            return
        if not calc.fit_nss_curve_robust():
            print("NSS fit failed")
            return
        try:
            calc.plot_results(save_plot=True, filename=plot_path)
        except Exception as e:
            print(f"Visualization failed: {e}")
        calc.export_results(output_csv)
        calc.create_comprehensive_report(report_path)
        print(f"\nGenerated: {output_csv}, {report_path}, {plot_path}, {log_filename}")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
