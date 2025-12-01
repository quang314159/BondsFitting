"""
Self-contained Robust Bootstrap Zero-Coupon Spot Rates Calculator
UI Light (CLI) – Version: 1.1.25
- Runtime limits for filters
- ISIN exclusion list
- NSS curve fitting
- CSV/report/plot outputs to ./Outputs
"""

import os, sys, math, argparse, warnings
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy.optimize import brentq, fsolve, minimize
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


# --------------------------- Logger ------------------------------------------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


# ----------------- Robust Bootstrap Spot Rates Engine ------------------------
class RobustBootstrapSpotRates:
    def __init__(self, csv_file_path, report_date='2025-07-31', outlier_threshold_bps=50.0,
                 min_mty_years=0.01, max_mty_years=100.0,
                 min_clean_price=10.0, max_clean_price=200.0,
                 min_coupon=0.0, max_coupon=50.0,
                 allowed_freq=(1, 2, 4, 12),
                 min_ytm=0.0, max_ytm=50.0,
                 exclude_isins=None):
        self.csv_file_path = csv_file_path
        self.report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
        self.outlier_threshold_bps = float(outlier_threshold_bps)

        # Runtime filter knobs
        self.min_mty_years = float(min_mty_years)
        self.max_mty_years = float(max_mty_years)
        self.min_clean_price = float(min_clean_price)
        self.max_clean_price = float(max_clean_price)
        self.min_coupon = float(min_coupon)
        self.max_coupon = float(max_coupon)
        self.allowed_freq = tuple(allowed_freq)
        self.min_ytm = float(min_ytm)
        self.max_ytm = float(max_ytm)
        self.exclude_isins = set(exclude_isins or [])

        # Data & results
        self.bonds_df = None
        self.spot_rates = {}          # maturity_years -> spot rate
        self.results_df = None
        self.nss_parameters = None
        self.calculation_log = []
        self.precision_tolerance = 1e-10

    # --------------------------- Data Prep -----------------------------------
    def load_and_prepare_data(self):
        print("Loading bond data for robust bootstrap analysis...")
        try:
            self.bonds_df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded {len(self.bonds_df)} bonds")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

        required = ['ISIN','Maturity Date','Index Weight','Coupon in %',
                    'Coupon Frequency','Dirty Price','Accrued Interest','Annual Yield']
        miss = [c for c in required if c not in self.bonds_df.columns]
        if miss:
            print(f"Error: Missing required columns: {miss}")
            return False

        # Numerics
        num_cols = ['Index Weight','Coupon in %','Coupon Frequency',
                    'Dirty Price','Accrued Interest','Annual Yield']
        for c in num_cols:
            self.bonds_df[c] = pd.to_numeric(self.bonds_df[c], errors='coerce')
        bad_num = self.bonds_df[num_cols].isna().any(axis=1)
        if bad_num.any():
            print(f"Warning: {bad_num.sum()} bonds with invalid numeric data removed")
            self.bonds_df = self.bonds_df[~bad_num].copy()

        # Dates (robust: let pandas infer)
        try:
            self.bonds_df['Maturity Date'] = pd.to_datetime(self.bonds_df['Maturity Date'], errors='coerce')
            bad_dt = self.bonds_df['Maturity Date'].isna()
            if bad_dt.any():
                print(f"Warning: {bad_dt.sum()} invalid maturity dates removed")
                self.bonds_df = self.bonds_df[~bad_dt].copy()
        except Exception as e:
            print(f"Error processing dates: {e}")
            return False

        # Derived
        self.bonds_df['Clean_Price'] = self.bonds_df['Dirty Price'] - self.bonds_df['Accrued Interest']
        rep = pd.to_datetime(self.report_date)
        self.bonds_df['Days_to_Maturity'] = (self.bonds_df['Maturity Date'] - rep).dt.days
        self.bonds_df['Time_to_Maturity_Years'] = self.bonds_df['Days_to_Maturity'] / 365.25

        # Exclude ISINs
        if self.exclude_isins:
            before = len(self.bonds_df)
            self.bonds_df = self.bonds_df[~self.bonds_df['ISIN'].isin(self.exclude_isins)].copy()
            print(f"Excluded {before - len(self.bonds_df)} bonds by ISIN list")

        # Runtime filter
        valid = (
            (self.bonds_df['Time_to_Maturity_Years'] > self.min_mty_years) &
            (self.bonds_df['Time_to_Maturity_Years'] < self.max_mty_years) &
            (self.bonds_df['Clean_Price'] > self.min_clean_price) &
            (self.bonds_df['Clean_Price'] < self.max_clean_price) &
            (self.bonds_df['Coupon in %'] >= self.min_coupon) &
            (self.bonds_df['Coupon in %'] <= self.max_coupon) &
            (self.bonds_df['Coupon Frequency'].isin(self.allowed_freq)) &
            (self.bonds_df['Annual Yield'] > self.min_ytm) &
            (self.bonds_df['Annual Yield'] < self.max_ytm)
        )
        init = len(self.bonds_df)
        self.bonds_df = self.bonds_df[valid].copy()
        removed = init - len(self.bonds_df)
        if removed:
            print(f"Filtered out {removed} bonds due to data-quality limits")

        if self.bonds_df.empty:
            print("Error: No valid bonds remain after filtering.")
            return False

        self.bonds_df.sort_values('Time_to_Maturity_Years', inplace=True)
        print(f"Processing {len(self.bonds_df)} valid bonds ("
              f"{self.bonds_df['Time_to_Maturity_Years'].min():.2f}y - "
              f"{self.bonds_df['Time_to_Maturity_Years'].max():.2f}y)")
        return True

    # ---------------------- Helpers for Pricing -------------------------------
    def calculate_exact_payment_schedule(self, bond_row):
        maturity_date = bond_row['Maturity Date'].date()
        coupon_rate = bond_row['Coupon in %'] / 100.0
        frequency = int(bond_row['Coupon Frequency'])
        face = 100.0
        period_coupon = (coupon_rate * face) / frequency

        payment_dates = []
        current = maturity_date
        while current > self.report_date:
            payment_dates.append(current)
            if frequency == 1:
                try:
                    current = current.replace(year=current.year - 1)
                except ValueError:
                    current = current - relativedelta(years=1)
            elif frequency == 2:
                current = current - relativedelta(months=6)
            elif frequency == 4:
                current = current - relativedelta(months=3)
            elif frequency == 12:
                current = current - relativedelta(months=1)
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")

        future = [d for d in reversed(payment_dates) if d > self.report_date]

        schedule = []
        for i, dt in enumerate(future):
            days = (dt - self.report_date).days
            t = days / 365.25
            amt = period_coupon if i < len(future) - 1 else period_coupon + face
            schedule.append((dt, amt, t))

        if not schedule:
            raise ValueError(f"No future payments for {bond_row['ISIN']}")
        return schedule

    def robust_interpolate_spot_rate(self, t_target):
        if not self.spot_rates:
            return 0.05
        items = sorted(self.spot_rates.items())
        T = [x[0] for x in items]
        R = [x[1] for x in items]
        if len(T) == 1:
            return R[0]
        if t_target <= T[0]: return R[0]
        if t_target >= T[-1]: return R[-1]
        for i in range(len(T)-1):
            if T[i] <= t_target <= T[i+1]:
                w = (t_target - T[i])/(T[i+1]-T[i])
                return R[i] + w*(R[i+1]-R[i])
        return R[-1]

    def calculate_precise_bond_pv(self, final_spot_rate, bond_row, schedule):
        total = 0.0
        for i, (_, amt, t) in enumerate(schedule):
            r = self.robust_interpolate_spot_rate(t) if i < len(schedule)-1 else final_spot_rate
            df = 1.0 if t <= 0 else (1.0 + r) ** (-t)
            total += amt * df
        return total

    def solve_bootstrap_spot_rate(self, bond_row):
        clean = bond_row['Clean_Price']
        ytm_guess = bond_row['Annual Yield'] / 100.0

        try:
            schedule = self.calculate_exact_payment_schedule(bond_row)
        except Exception as e:
            return ytm_guess, {'method':'error_fallback','error':str(e),'isin':bond_row['ISIN']}

        # single payment => direct
        if len(schedule) == 1:
            _, amt, t = schedule[0]
            r = 0.0 if t <= 0 else (amt/clean)**(1.0/t) - 1.0
            return r, {'method':'direct_calculation','isin':bond_row['ISIN'],'payment_schedule':schedule,'target_price':clean}

        def f(r):
            return self.calculate_precise_bond_pv(r, bond_row, schedule) - clean

        methods = []

        # Brent (robust)
        try:
            lb = max(-0.05, ytm_guess - 0.10)
            ub = min(0.50, ytm_guess + 0.10)
            fl, fu = f(lb), f(ub)
            tries = 0
            while fl*fu > 0 and tries < 10:
                if abs(fl) < abs(fu): lb -= 0.02
                else: ub += 0.02
                fl, fu = f(lb), f(ub)
                tries += 1
            if fl*fu <= 0:
                sol = brentq(f, lb, ub, xtol=self.precision_tolerance, maxiter=500)
                err = abs(f(sol))
                methods.append(('brent', sol, err))
        except Exception as e:
            methods.append(('brent', None, f'Failed: {e}'))

        # fsolve with multiple starts
        for start in [ytm_guess, ytm_guess*0.8, ytm_guess*1.2, 0.02,0.03,0.04,0.05,0.06,0.07,0.08]:
            try:
                sol, infodict, ier, _ = fsolve(f, [start], xtol=self.precision_tolerance, maxfev=1000, full_output=True)
                if ier == 1:
                    r = sol[0]
                    err = abs(f(r))
                    methods.append(('fsolve', r, err))
            except Exception:
                continue

        valid = [(m, r, e) for (m, r, e) in methods if isinstance(e, float) and e < 0.1]
        if valid:
            best = min(valid, key=lambda x: x[2])
            method, rate, err = best
            return rate, {'method':f'bootstrap_{method}','convergence_error':err,'isin':bond_row['ISIN'],
                          'payment_schedule':schedule,'target_price':clean}

        return ytm_guess, {'method':'ytm_fallback','isin':bond_row['ISIN'],'payment_schedule':schedule,'target_price':clean}

    # --------------------------- Main run -------------------------------------
    def run_robust_bootstrap(self):
        if self.bonds_df is None:
            print("Error: No bond data loaded")
            return False

        results = []
        self.calculation_log = []
        ver_errors = []

        for idx, bond in self.bonds_df.iterrows():
            print(f"\nProcessing {idx+1}/{len(self.bonds_df)}: {bond['ISIN']} "
                  f"(T={bond['Time_to_Maturity_Years']:.2f}y, Clean={bond['Clean_Price']:.4f})")
            r, details = self.solve_bootstrap_spot_rate(bond)
            self.spot_rates[bond['Time_to_Maturity_Years']] = r
            print(f"  Method: {details['method']}; Spot={r*100:.4f}%")

            if 'payment_schedule' in details:
                pv = self.calculate_precise_bond_pv(r, bond, details['payment_schedule'])
                err = abs(pv - bond['Clean_Price'])
                print(f"  Verification PV={pv:.6f} vs Target={bond['Clean_Price']:.6f} -> Err={err:.8f}")
                ver_errors.append(err)
            else:
                err = 0.0
                ver_errors.append(0.0)

            ytm = bond['Annual Yield']/100.0
            diff_bps = (r - ytm)*10000.0

            results.append({
                'Bond_Number': len(results)+1,
                'ISIN': bond['ISIN'],
                'Maturity_Date': bond['Maturity Date'].strftime('%Y-%m-%d'),
                'Time_to_Maturity_Years': f"{bond['Time_to_Maturity_Years']:.6f}",
                'Index_Weight_Percent': f"{bond['Index Weight']:.3f}",
                'Coupon_Rate_Percent': f"{bond['Coupon in %']:.3f}",
                'Coupon_Frequency': int(bond['Coupon Frequency']),
                'Clean_Price': f"{bond['Clean_Price']:.6f}",
                'Bond_YTM_Percent': f"{ytm*100:.6f}",
                'Bootstrap_Spot_Rate_Percent': f"{r*100:.6f}",
                'Spot_vs_YTM_Difference_bps': f"{diff_bps:.2f}",
                'Bootstrap_Method': details['method'],
                'Verification_Error': f"{err:.8f}",
                'Convergence_Error': f"{details.get('convergence_error', 0):.2e}"
            })
            self.calculation_log.append(details)

        self.results_df = pd.DataFrame(results)

        print("\n=== BOOTSTRAP COMPLETION SUMMARY ===")
        print(f"Total bonds processed: {len(results)}")
        print(f"Mean verification error: {np.mean(ver_errors):.6f}")
        print(f"Median verification error: {np.median(ver_errors):.6f}")
        print(f"Max verification error: {np.max(ver_errors):.6f}")
        print(f"Bonds with err > 0.01: {sum(e>0.01 for e in ver_errors)}")
        print(f"Bonds with err > 0.001: {sum(e>0.001 for e in ver_errors)}")

        # Method distribution
        mc = self.results_df['Bootstrap_Method'].value_counts()
        print("\nCalculation method distribution:")
        for m, c in mc.items():
            print(f"  {m}: {c} bonds ({c/len(self.results_df)*100:.1f}%)")
        return True

    # ------------------------- NSS curve fit ----------------------------------
    @staticmethod
    def _nss(t, b0, b1, b2, b3, tau1, tau2):
        t = np.maximum(np.asarray(t), 1e-6)
        term1 = b0
        term2 = b1 * (1 - np.exp(-t/tau1)) / (t/tau1)
        term3 = b2 * ((1 - np.exp(-t/tau1)) / (t/tau1) - np.exp(-t/tau1))
        term4 = b3 * ((1 - np.exp(-t/tau2)) / (t/tau2) - np.exp(-t/tau2))
        return term1 + term2 + term3 + term4

    def fit_nss_curve_robust(self):
        if self.results_df is None or self.results_df.empty:
            print("Error: No bootstrap results available")
            return None

        print("\nFitting NSS curve to bootstrap spot rates...")
        T_all = self.results_df['Time_to_Maturity_Years'].astype(float).values
        S_all = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values / 100.0
        Y_all = self.results_df['Bond_YTM_Percent'].astype(float).values / 100.0
        W_all = self.results_df['Index_Weight_Percent'].astype(float).values
        diff_bps = (S_all - Y_all) * 10000.0

        out_mask = np.abs(diff_bps) > self.outlier_threshold_bps
        print(f"Outlier analysis (>|{self.outlier_threshold_bps}| bps): {out_mask.sum()} excluded of {len(T_all)}")

        mask = ~out_mask
        T, S, W = T_all[mask], S_all[mask], W_all[mask]
        if len(T) < 6:
            print("Error: Too few clean bonds for NSS fitting")
            return None

        def objective(p):
            b0,b1,b2,b3,t1,t2 = p
            fit = self._nss(T, b0,b1,b2,b3,t1,t2)
            r = fit - S
            return np.sum((r**2) * W)

        mean_rate = np.average(S, weights=W)
        r_range = np.max(S) - np.min(S)
        starts = [
            [mean_rate, 0.0, 0.0, 0.0, 2.0, 5.0],
            [mean_rate, -r_range/4, r_range/4, -r_range/8, 1.5, 8.0],
            [mean_rate, r_range/4, -r_range/4, r_range/8, 3.0, 10.0],
            [0.04, -0.01, -0.01, 0.01, 2.0, 5.0],
            [0.06, 0.01, -0.02, 0.01, 1.8, 7.0]
        ]
        bounds = [(0.001,0.20),(-0.20,0.20),(-0.20,0.20),(-0.20,0.20),(0.1,30.0),(0.1,30.0)]

        best = None
        best_r2 = -1
        for i, s in enumerate(starts, 1):
            try:
                res = minimize(objective, s, method='L-BFGS-B', bounds=bounds, options={'maxiter':2000,'ftol':1e-12})
                if res.success:
                    b0,b1,b2,b3,t1,t2 = res.x
                    fit = self._nss(T, b0,b1,b2,b3,t1,t2)
                    resid = fit - S
                    ss_res = np.sum((resid**2)*W)
                    w_mean = np.average(S, weights=W)
                    ss_tot = np.sum(((S - w_mean)**2)*W)
                    r2 = 1 - ss_res/ss_tot
                    print(f"  Run {i}: R^2={r2:.6f}" + ("  (best)" if r2 > best_r2 else ""))
                    if r2 > best_r2:
                        best, best_r2 = res, r2
            except Exception as e:
                print(f"  Run {i} failed: {e}")

        if best is None:
            print("NSS optimization failed")
            return None

        b0,b1,b2,b3,t1,t2 = best.x
        fit_all = self._nss(T_all, b0,b1,b2,b3,t1,t2)
        w_mean_all = np.average(S_all, weights=W_all)
        r2_all = 1 - np.sum(((fit_all - S_all)**2)*W_all) / np.sum(((S_all - w_mean_all)**2)*W_all)

        self.nss_parameters = {
            'beta0': b0, 'beta1': b1, 'beta2': b2, 'beta3': b3,
            'tau1': t1, 'tau2': t2,
            'weighted_r_squared_clean': best_r2,
            'weighted_r_squared_all': r2_all,
            'outlier_threshold_bps': self.outlier_threshold_bps,
            'outlier_count': int(out_mask.sum()),
            'clean_count': int(len(T))
        }
        print(f"NSS fitting complete. Clean R^2={best_r2:.6f}, All R^2={r2_all:.6f}")
        return self.nss_parameters

    # ----------------------- Outputs ------------------------------------------
    def export_results(self, output_filename):
        if self.results_df is None:
            return False
        try:
            self.results_df.to_csv(output_filename, index=False)
            print(f"Results exported to: {output_filename}")
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

    def create_comprehensive_report(self, filename):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("COMPREHENSIVE BOOTSTRAP ANALYSIS REPORT\n")
                f.write("="*45 + "\n")
                f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"Report Date: {self.report_date}\n")
                f.write(f"Outlier Threshold: {self.outlier_threshold_bps} bps\n\n")

                f.write("DATA SUMMARY\n")
                f.write("-"*12 + "\n")
                f.write(f"Total bonds processed: {len(self.results_df)}\n")
                mats = self.results_df['Time_to_Maturity_Years'].astype(float).values
                spots = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values
                verr = self.results_df['Verification_Error'].astype(float).values
                f.write(f"Maturity range: {mats.min():.4f} to {mats.max():.4f} years\n")
                f.write(f"Spot rate range: {spots.min():.3f}% to {spots.max():.3f}%\n")
                f.write(f"Mean verification error: {np.mean(verr):.6f}\n")
                f.write(f"Max verification error: {np.max(verr):.6f}\n\n")

                f.write("CALCULATION METHODS\n")
                f.write("-"*18 + "\n")
                vc = self.results_df['Bootstrap_Method'].value_counts()
                for m, c in vc.items():
                    f.write(f"{m}: {c} bonds ({c/len(self.results_df)*100:.1f}%)\n")
                f.write("\n")

                if self.nss_parameters:
                    p = self.nss_parameters
                    f.write("NSS CURVE FITTING\n")
                    f.write("-"*16 + "\n")
                    f.write(f"Clean R²: {p['weighted_r_squared_clean']:.6f}\n")
                    f.write(f"All R²: {p['weighted_r_squared_all']:.6f}\n")
                    f.write(f"Outliers excluded: {p['outlier_count']}\n")
                    f.write(f"Clean bonds used: {p['clean_count']}\n")
                    f.write("Parameters:\n")
                    f.write(f"  β0={p['beta0']*100:.3f}%  β1={p['beta1']*100:.3f}%  β2={p['beta2']*100:.3f}%  β3={p['beta3']*100:.3f}%\n")
                    f.write(f"  τ1={p['tau1']:.2f}y  τ2={p['tau2']:.2f}y\n")
            print(f"Comprehensive report saved to: {filename}")
        except Exception as e:
            print(f"Error creating report: {e}")

    def plot_results(self, save_plot=True, filename='robust_bootstrap_analysis.png'):
        if self.results_df is None or not MATPLOTLIB_AVAILABLE:
            print("Cannot create plots - no results or matplotlib unavailable")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Robust Bootstrap Analysis Results', fontsize=16, fontweight='bold')

        mats = self.results_df['Time_to_Maturity_Years'].astype(float).values
        spots = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values
        ytms  = self.results_df['Bond_YTM_Percent'].astype(float).values
        wts   = self.results_df['Index_Weight_Percent'].astype(float).values
        diffs = self.results_df['Spot_vs_YTM_Difference_bps'].astype(float).values
        verr  = self.results_df['Verification_Error'].astype(float).values

        # Plot 1: Spot vs YTM (+ NSS)
        ax1.scatter(mats, spots, alpha=0.7, s=np.clip(wts, 1, None)*20, label='Bootstrap Spot (%)')
        ax1.scatter(mats, ytms, alpha=0.5, s=20, label='Bond YTM (%)')
        if self.nss_parameters is not None:
            P = self.nss_parameters
            grid = np.linspace(mats.min(), mats.max(), 200)
            curve = self._nss(grid, P['beta0'], P['beta1'], P['beta2'], P['beta3'], P['tau1'], P['tau2'])*100
            ax1.plot(grid, curve, linewidth=2, label=f"NSS Fit (R²={P['weighted_r_squared_clean']:.3f})")
        ax1.set_xlabel("Maturity (Years)"); ax1.set_ylabel("Rate (%)"); ax1.set_title("Spot vs YTM")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # Plot 2: Verification errors (in currency units)
        colors = ['green' if e < 0.001 else 'orange' if e < 0.01 else 'red' for e in verr]
        ax2.scatter(mats, np.array(verr)*100, c=colors, alpha=0.7, s=np.clip(wts,1,None)*15)
        ax2.axhline(0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 cent')
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1 cent')
        ax2.set_xlabel("Maturity (Years)"); ax2.set_ylabel("Verification Error (cents)")
        ax2.set_title("Bootstrap Verification Errors"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: Spot - YTM (bps)
        colors_diff = ['green' if abs(d) < self.outlier_threshold_bps else 'red' for d in diffs]
        ax3.scatter(mats, diffs, c=colors_diff, alpha=0.7, s=np.clip(wts,1,None)*15)
        ax3.axhline(0, color='black', linestyle='-')
        ax3.axhline(self.outlier_threshold_bps, color='red', linestyle='--', alpha=0.7,
                    label=f'±{self.outlier_threshold_bps} bps')
        ax3.axhline(-self.outlier_threshold_bps, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel("Maturity (Years)"); ax3.set_ylabel("Spot - YTM (bps)")
        ax3.set_title("Bootstrap vs YTM Differences"); ax3.legend(); ax3.grid(True, alpha=0.3)

        # Plot 4: Method distribution
        vc = self.results_df['Bootstrap_Method'].value_counts()
        ax4.pie(vc.values, labels=vc.index, autopct='%1.1f%%'); ax4.set_title("Method Distribution")

        plt.tight_layout()
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {filename}")
        plt.show()


# ------------------------------ CLI / Main -----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run robust zero-coupon bootstrap (self-contained).")
    parser.add_argument("--csv", default="IBoxx_with_weights_310725.csv", help="CSV file name in ./Data/")
    parser.add_argument("--report_date", default="2025-07-31", help="YYYY-MM-DD")
    parser.add_argument("--outlier_bps", type=float, default=50.0)

    parser.add_argument("--min_mty", type=float, default=0.01)
    parser.add_argument("--max_mty", type=float, default=100.0)
    parser.add_argument("--min_px", type=float, default=10.0)
    parser.add_argument("--max_px", type=float, default=200.0)
    parser.add_argument("--min_coupon", type=float, default=0.0)
    parser.add_argument("--max_coupon", type=float, default=50.0)
    parser.add_argument("--min_ytm", type=float, default=0.0)
    parser.add_argument("--max_ytm", type=float, default=50.0)
    parser.add_argument("--freq", type=str, default="1,2,4,12", help="Allowed coupon frequencies, e.g. 1,2,4,12")
    parser.add_argument("--exclude", type=str, default="", help="Comma/space separated ISINs to exclude")

    args = parser.parse_args()

    # Project folders
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "Data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, args.csv)
    allowed_freq = tuple(int(x.strip()) for x in args.freq.replace(",", " ").split() if x.strip())
    exclude_isins = {x.strip() for x in args.exclude.replace(",", " ").split() if x.strip()}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"robust_bootstrap_log_{timestamp}.txt")
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print("ROBUST BOOTSTRAP ZERO COUPON SPOT RATES CALCULATOR")
        print("="*60)
        print(f"Input file: {csv_path}")
        print(f"Excluded ISINs: {exclude_isins if exclude_isins else 'None'}")
        print(f"Outlier threshold: {args.outlier_bps} bps\n")

        calc = RobustBootstrapSpotRates(
            csv_file_path=csv_path,
            report_date=args.report_date,
            outlier_threshold_bps=args.outlier_bps,
            min_mty_years=args.min_mty,
            max_mty_years=args.max_mty,
            min_clean_price=args.min_px,
            max_clean_price=args.max_px,
            min_coupon=args.min_coupon,
            max_coupon=args.max_coupon,
            allowed_freq=allowed_freq if allowed_freq else (1,2,4,12),
            min_ytm=args.min_ytm,
            max_ytm=args.max_ytm,
            exclude_isins=exclude_isins
        )

        if not calc.load_and_prepare_data():
            print("Failed to load/prepare data.")
            return

        if not calc.run_robust_bootstrap():
            print("Bootstrap calculation failed.")
            return

        calc.fit_nss_curve_robust()
        calc.export_results(os.path.join(OUTPUT_DIR, "robust_bootstrap_results.csv"))
        calc.create_comprehensive_report(os.path.join(OUTPUT_DIR, "bootstrap_comprehensive_report.txt"))
        if MATPLOTLIB_AVAILABLE:
            calc.plot_results(save_plot=True, filename=os.path.join(OUTPUT_DIR, "robust_bootstrap_analysis.png"))

        print("\nAll outputs saved in:", OUTPUT_DIR)

    except Exception as e:
        import traceback
        print("Error:", e)
        traceback.print_exc()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        print(f"\nAnalysis complete. Log saved to {log_file}")


if __name__ == "__main__":
    main()
