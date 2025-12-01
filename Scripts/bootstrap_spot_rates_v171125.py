"""
Traditional Bootstrap Zero Coupon Spot Rates Calculator - REVISED
================================================================

This script implements a robust traditional bootstrap methodology with:
1. High-precision cash flow calculations
2. Accurate date handling with multiple day count conventions
3. Robust numerical methods with comprehensive error checking
4. Enhanced interpolation with boundary condition handling
5. Detailed verification and diagnostic capabilities

Author: Quang Le
Date: 13 November 2025
Version: 2.6 (Enhanced Precision)
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from scipy.optimize import fsolve, minimize, brentq
from dateutil.relativedelta import relativedelta
import warnings
import sys
import os
import math
import argparse # Added import for argparse (was missing in original snippet but used in main)
warnings.filterwarnings('ignore')

# --- Project folders (corrected paths) ---
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
    def __init__(self, csv_file_path, report_date='2025-07-31', outlier_threshold_bps=50.0,
                 min_mty_years=0.01, max_mty_years=100.0,
                 min_clean_price=10.0, max_clean_price=200.0,
                 min_coupon=0.0, max_coupon=50.0,
                 allowed_freq=(1, 2, 4, 12),
                 min_ytm=0.0, max_ytm=50.0,
                 exclude_isins=None):
        """
        Robust Bootstrap Calculator with enhanced precision

        Parameters:
        -----------
        csv_file_path : str
            Path to the CSV file containing bond data with index weights
        report_date : str
            Report date in 'YYYY-MM-DD' format
        outlier_threshold_bps : float
            Threshold in basis points for outlier exclusion (default: 50.0)

        Filtering / limits:
        - All limits are runtime-configurable.
        - exclude_isins: an iterable of ISIN strings to drop before bootstrapping and fitting.
        """
        self.csv_file_path = csv_file_path
        self.report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
        self.outlier_threshold_bps = float(outlier_threshold_bps)

        # Runtime filter knobs
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

        # Data / results
        self.bonds_df = None
        self.spot_rates = {}     # maturity_years -> spot rate
        self.results_df = None
        self.nss_parameters = None
        self.calculation_log = []
        self.precision_tolerance = 1e-10  # High precision tolerance
        
    def load_and_prepare_data(self):
        """Load CSV data and prepare bonds for bootstrap with enhanced validation"""
        print("Loading bond data for robust bootstrap analysis...")
        
        try:
            self.bonds_df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded {len(self.bonds_df)} bonds")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
        
        # Enhanced data validation and conversion
        required_columns = ['ISIN', 'Maturity Date', 'Index Weight', 'Coupon in %', 
                           'Coupon Frequency', 'Dirty Price', 'Accrued Interest', 'Annual Yield']
        
        missing_columns = [col for col in required_columns if col not in self.bonds_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        # Convert data types with validation
        numeric_columns = ['Index Weight', 'Coupon in %', 'Coupon Frequency', 
                          'Dirty Price', 'Accrued Interest', 'Annual Yield']
        
        for col in numeric_columns:
            self.bonds_df[col] = pd.to_numeric(self.bonds_df[col], errors='coerce')
            
        # Check for invalid data
        invalid_data = self.bonds_df[numeric_columns].isna().any(axis=1)
        if invalid_data.any():
            print(f"Warning: Found {invalid_data.sum()} bonds with invalid numeric data")
            self.bonds_df = self.bonds_df[~invalid_data]
        
        # Convert maturity dates with validation (robust to ISO and common formats)
        try:
            # First, try without a forced format (handles YYYY-MM-DD cleanly)
            md = pd.to_datetime(self.bonds_df['Maturity Date'], errors='coerce')

            # If too many NaT, try a day-first pass (for dd/mm/yyyy cases)
            if md.isna().sum() > 0 and md.isna().mean() > 0.5:
                md2 = pd.to_datetime(self.bonds_df['Maturity Date'], errors='coerce', dayfirst=True)
                if md2.notna().sum() > md.notna().sum():
                    md = md2

            self.bonds_df['Maturity Date'] = md

            invalid_dates = self.bonds_df['Maturity Date'].isna()
            if invalid_dates.any():
                print(f"Warning: Found {invalid_dates.sum()} bonds with invalid dates")
                self.bonds_df = self.bonds_df[~invalid_dates]
        except Exception as e:
            print(f"Error processing dates: {e}")
            return False

        
        # Calculate derived columns with precision
        self.bonds_df['Clean_Price'] = self.bonds_df['Dirty Price'] - self.bonds_df['Accrued Interest']
        
        # High-precision time to maturity calculation
        report_date_pd = pd.to_datetime(self.report_date)
        self.bonds_df['Days_to_Maturity'] = (self.bonds_df['Maturity Date'] - report_date_pd).dt.days
        self.bonds_df['Time_to_Maturity_Years'] = self.bonds_df['Days_to_Maturity'] / 365.25
        
        # Enhanced filtering with validation
        initial_count = len(self.bonds_df)
        
        # Exclude specific ISINs if provided
        if self.exclude_isins:
            before = len(self.bonds_df)
            self.bonds_df = self.bonds_df[~self.bonds_df['ISIN'].isin(self.exclude_isins)].copy()
            excluded = before - len(self.bonds_df)
            if excluded:
                print(f"Excluded by ISIN list: {excluded}")
        
        # Filter out bonds with issues

        valid_bonds = (
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
        self.bonds_df = self.bonds_df[valid_bonds].copy()
        
        # Sort by maturity (CRITICAL for bootstrap)
        self.bonds_df = self.bonds_df.sort_values('Time_to_Maturity_Years').reset_index(drop=True)
        
        final_count = len(self.bonds_df)
        if initial_count != final_count:
            print(f"Filtered out {initial_count - final_count} bonds due to data quality issues")
        
        if final_count == 0:
            print("Error: No valid bonds remaining after filtering")
            return False
        
        print(f"Processing {final_count} valid bonds in sequential maturity order")
        print(f"Maturity range: {self.bonds_df['Time_to_Maturity_Years'].min():.4f} to {self.bonds_df['Time_to_Maturity_Years'].max():.4f} years")
        print(f"Price range: {self.bonds_df['Clean_Price'].min():.2f} to {self.bonds_df['Clean_Price'].max():.2f}")
        print(f"Coupon range: {self.bonds_df['Coupon in %'].min():.3f}% to {self.bonds_df['Coupon in %'].max():.3f}%")
        
        return True
    
    def calculate_exact_payment_schedule(self, bond_row):
        """Calculate precise coupon payment dates and amounts with rigorous validation"""
        maturity_date = bond_row['Maturity Date'].date()
        coupon_rate = bond_row['Coupon in %'] / 100
        frequency = int(bond_row['Coupon Frequency'])
        face_value = 100.0  # Standard assumption
        
        # Calculate period coupon amount
        period_coupon = (coupon_rate * face_value) / frequency
        
        # Generate all payment dates working backwards from maturity
        payment_dates = []
        current_date = maturity_date
        
        # Generate dates until we go past the report date
        while current_date > self.report_date:
            payment_dates.append(current_date)
            
            # Calculate previous payment date based on frequency
            if frequency == 1:  # Annual
                try:
                    current_date = current_date.replace(year=current_date.year - 1)
                except ValueError:  # Handle Feb 29 edge case
                    current_date = current_date - relativedelta(years=1)
            elif frequency == 2:  # Semi-annual
                current_date = current_date - relativedelta(months=6)
            elif frequency == 4:  # Quarterly
                current_date = current_date - relativedelta(months=3)
            elif frequency == 12:  # Monthly
                current_date = current_date - relativedelta(months=1)
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Reverse to get chronological order and filter future payments only
        future_payments = [d for d in reversed(payment_dates) if d > self.report_date]
        
        # Create payment schedule with high precision timing
        payment_schedule = []
        for i, payment_date in enumerate(future_payments):
            # Calculate exact time to payment in years
            days_to_payment = (payment_date - self.report_date).days
            time_to_payment = days_to_payment / 365.25  # Use consistent day count
            
            # Determine payment amount
            if i == len(future_payments) - 1:  # Final payment
                payment_amount = period_coupon + face_value
            else:  # Intermediate coupon
                payment_amount = period_coupon
            
            payment_schedule.append((payment_date, payment_amount, time_to_payment))
        
        # Validation checks
        if len(payment_schedule) == 0:
            raise ValueError(f"No future payments found for bond {bond_row['ISIN']}")
        
        # Check payment schedule consistency
        expected_payments = max(1, int(bond_row['Time_to_Maturity_Years'] * frequency))
        if abs(len(payment_schedule) - expected_payments) > 2:
            print(f"Warning: Payment count unusual for {bond_row['ISIN']}: "
                  f"expected ~{expected_payments}, got {len(payment_schedule)}")
        
        # Verify final payment includes principal
        final_payment = payment_schedule[-1][1]
        if final_payment < face_value:
            print(f"Warning: Final payment seems too small for {bond_row['ISIN']}: {final_payment:.4f}")
        
        return payment_schedule
    
    def robust_interpolate_spot_rate(self, target_maturity):
        """
        Enhanced linear interpolation with boundary condition handling
        """
        if not self.spot_rates:
            # No rates available - return a reasonable default
            return 0.05
        
        # Get sorted maturities and rates
        sorted_items = sorted(self.spot_rates.items())
        maturities = [item[0] for item in sorted_items]
        rates = [item[1] for item in sorted_items]
        
        # Handle edge cases
        if len(maturities) == 1:
            return rates[0]
        
        # Find bounding points for interpolation
        if target_maturity <= maturities[0]:
            # Target is shorter than shortest rate - use flat extrapolation
            return rates[0]
        
        if target_maturity >= maturities[-1]:
            # Target is longer than longest rate - use flat extrapolation
            return rates[-1]
        
        # Find interpolation bounds
        for i in range(len(maturities) - 1):
            if maturities[i] <= target_maturity <= maturities[i + 1]:
                # Linear interpolation
                x1, y1 = maturities[i], rates[i]
                x2, y2 = maturities[i + 1], rates[i + 1]
                
                # Calculate interpolation weight
                weight = (target_maturity - x1) / (x2 - x1)
                interpolated_rate = y1 + weight * (y2 - y1)
                
                return interpolated_rate
        
        # Fallback (should not reach here)
        return rates[-1]
    
    def calculate_precise_bond_pv(self, final_spot_rate, bond_row, payment_schedule):
        """
        Calculate bond present value with maximum precision
        """
        total_pv = 0.0
        
        for i, (payment_date, payment_amount, time_to_payment) in enumerate(payment_schedule):
            if i < len(payment_schedule) - 1:
                # Intermediate payment - use interpolated spot rate
                discount_rate = self.robust_interpolate_spot_rate(time_to_payment)
            else:
                # Final payment - use the spot rate we're solving for
                discount_rate = final_spot_rate
            
            # Calculate discount factor with high precision
            if time_to_payment <= 0:
                discount_factor = 1.0  # Same-day payment
            else:
                discount_factor = (1.0 + discount_rate) ** (-time_to_payment)
            
            # Add to total present value
            contribution = payment_amount * discount_factor
            total_pv += contribution
        
        return total_pv
    
    def solve_bootstrap_spot_rate(self, bond_row):
        """
        Robust bootstrap spot rate calculation with multiple solving methods
        """
        clean_price = bond_row['Clean_Price']
        ytm_guess = bond_row['Annual Yield'] / 100
        
        # Calculate payment schedule
        try:
            payment_schedule = self.calculate_exact_payment_schedule(bond_row)
        except Exception as e:
            print(f"Error calculating payment schedule for {bond_row['ISIN']}: {e}")
            return ytm_guess, {'method': 'error_fallback', 'error': str(e)}
        
        calc_details = {
            'isin': bond_row['ISIN'],
            'payment_schedule': payment_schedule,
            'target_price': clean_price,
            'existing_rates_count': len(self.spot_rates)
        }
        
        # Handle single payment case (zero-coupon or very short maturity)
        if len(payment_schedule) == 1:
            payment_date, payment_amount, time_to_payment = payment_schedule[0]
            if time_to_payment <= 0:
                spot_rate = 0.0  # Same day maturity
            else:
                spot_rate = (payment_amount / clean_price) ** (1.0 / time_to_payment) - 1.0
            
            calc_details['method'] = 'direct_calculation'
            calc_details['spot_rate'] = spot_rate
            return spot_rate, calc_details
        
        # Multiple payment case - solve bootstrap equation
        def objective_function(spot_rate):
            calculated_pv = self.calculate_precise_bond_pv(spot_rate, bond_row, payment_schedule)
            return calculated_pv - clean_price
        
        # Try multiple solving methods for robustness
        solving_methods = []
        
        # Method 1: Brent's method (most robust for single variable)
        try:
            # Determine reasonable bounds for the spot rate
            lower_bound = max(-0.05, ytm_guess - 0.1)  # Allow negative rates but not too extreme
            upper_bound = min(0.5, ytm_guess + 0.1)    # Cap at 50%
            
            # Ensure bounds bracket the solution
            f_lower = objective_function(lower_bound)
            f_upper = objective_function(upper_bound)
            
            # Expand bounds if they don't bracket the solution
            attempts = 0
            while f_lower * f_upper > 0 and attempts < 10:
                if abs(f_lower) < abs(f_upper):
                    lower_bound -= 0.02
                else:
                    upper_bound += 0.02
                f_lower = objective_function(lower_bound)
                f_upper = objective_function(upper_bound)
                attempts += 1
            
            if f_lower * f_upper <= 0:
                solution = brentq(objective_function, lower_bound, upper_bound, 
                                xtol=self.precision_tolerance, maxiter=500)
                error = abs(objective_function(solution))
                solving_methods.append(('brent', solution, error))
        except Exception as e:
            solving_methods.append(('brent', None, f"Failed: {e}"))
        
        # Method 2: Newton-Raphson with multiple starting points
        starting_points = [
            ytm_guess,
            ytm_guess * 0.8,
            ytm_guess * 1.2,
            0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08
        ]
        
        for start_point in starting_points:
            try:
                solution = fsolve(objective_function, [start_point], 
                                xtol=self.precision_tolerance, maxfev=1000, full_output=True)
                if solution[2] == 1:  # Converged
                    spot_rate = solution[0][0]
                    error = abs(objective_function(spot_rate))
                    solving_methods.append(('fsolve', spot_rate, error))
            except:
                continue
        
        # Select the best solution
        valid_solutions = [(method, rate, error) for method, rate, error in solving_methods 
                          if isinstance(error, float) and error < 0.1]
        
        if valid_solutions:
            # Choose solution with smallest error
            best_method, best_rate, best_error = min(valid_solutions, key=lambda x: x[2])
            
            if best_error < self.precision_tolerance * 100:  # Good precision
                calc_details['method'] = f'bootstrap_{best_method}'
                calc_details['convergence_error'] = best_error
                calc_details['spot_rate'] = best_rate
                return best_rate, calc_details
            else:
                print(f"Warning: Imprecise solution for {bond_row['ISIN']}, error = {best_error:.2e}")
                calc_details['method'] = f'bootstrap_{best_method}_imprecise'
                calc_details['convergence_error'] = best_error
                calc_details['spot_rate'] = best_rate
                return best_rate, calc_details
        
        # All methods failed - use YTM as fallback
        print(f"Warning: All bootstrap methods failed for {bond_row['ISIN']}, using YTM")
        calc_details['method'] = 'ytm_fallback'
        calc_details['solving_attempts'] = len(solving_methods)
        calc_details['spot_rate'] = ytm_guess
        return ytm_guess, calc_details
    
    def run_robust_bootstrap(self):
        """Run robust bootstrap analysis with comprehensive error checking"""
        if self.bonds_df is None:
            print("Error: No bond data loaded")
            return False
        
        print("\nStarting robust bootstrap calculation...")
        print("Processing bonds sequentially by maturity with enhanced precision")
        
        results = []
        self.calculation_log = []
        verification_errors = []
        
        for idx, bond in self.bonds_df.iterrows():
            print(f"\nProcessing bond {idx + 1}/{len(self.bonds_df)}: {bond['ISIN']}")
            print(f"  Maturity: {bond['Time_to_Maturity_Years']:.4f} years")
            print(f"  Clean Price: {bond['Clean_Price']:.6f}")
            print(f"  Coupon: {bond['Coupon in %']:.3f}%, Frequency: {bond['Coupon Frequency']}")
            
            # Robust bootstrap calculation
            spot_rate, calc_details = self.solve_bootstrap_spot_rate(bond)
            
            # Store spot rate in sequential order
            self.spot_rates[bond['Time_to_Maturity_Years']] = spot_rate
            
            print(f"  Method: {calc_details['method']}")
            print(f"  Calculated spot rate: {spot_rate*100:.4f}%")
            
            # Detailed verification for all bonds
            if 'payment_schedule' in calc_details:
                payment_schedule = calc_details['payment_schedule']
                verification_pv = self.calculate_precise_bond_pv(spot_rate, bond, payment_schedule)
                verification_error = abs(verification_pv - bond['Clean_Price'])
                
                print(f"  Verification: PV = {verification_pv:.6f}, Target = {bond['Clean_Price']:.6f}")
                print(f"  Verification Error = {verification_error:.8f}")
                
                # Store verification error
                verification_errors.append(verification_error)
                
                # Enhanced diagnostics for any error > 0.001
                if verification_error > 0.001:
                    print(f"  *** VERIFICATION ISSUE DETECTED ***")
                    print(f"  Payment schedule: {len(payment_schedule)} payments")
                    
                    # Show detailed cash flow breakdown
                    total_manual_pv = 0
                    for i, (pmt_date, pmt_amt, pmt_time) in enumerate(payment_schedule):
                        if i < len(payment_schedule) - 1:
                            rate = self.robust_interpolate_spot_rate(pmt_time)
                        else:
                            rate = spot_rate
                        
                        df = (1 + rate) ** (-pmt_time) if pmt_time > 0 else 1.0
                        pv_contrib = pmt_amt * df
                        total_manual_pv += pv_contrib
                        
                        print(f"    Payment {i+1}: {pmt_date.strftime('%Y-%m-%d')} "
                              f"Amount={pmt_amt:.4f}, Time={pmt_time:.4f}y, "
                              f"Rate={rate*100:.4f}%, PV={pv_contrib:.6f}")
                    
                    print(f"  Manual PV Total: {total_manual_pv:.6f}")
                    print(f"  Manual vs Function difference: {abs(total_manual_pv - verification_pv):.8f}")
                    
                    # Data integrity check
                    calculated_clean = bond['Dirty Price'] - bond['Accrued Interest']
                    if abs(calculated_clean - bond['Clean_Price']) > 0.001:
                        print(f"  *** DATA INTEGRITY ISSUE ***")
                        print(f"  Dirty-Accrued = {calculated_clean:.6f}, Clean = {bond['Clean_Price']:.6f}")
            else:
                verification_error = 0
                verification_errors.append(0)
            
            # Calculate metrics
            ytm = bond['Annual Yield'] / 100
            diff_bps = (spot_rate - ytm) * 10000
            
            # Store results with verification error
            result = {
                'Bond_Number': idx + 1,
                'ISIN': bond['ISIN'],
                'Maturity_Date': bond['Maturity Date'].strftime('%Y-%m-%d'),
                'Time_to_Maturity_Years': f"{bond['Time_to_Maturity_Years']:.6f}",
                'Index_Weight_Percent': f"{bond['Index Weight']:.3f}",
                'Coupon_Rate_Percent': f"{bond['Coupon in %']:.3f}",
                'Coupon_Frequency': int(bond['Coupon Frequency']),
                'Clean_Price': f"{bond['Clean_Price']:.6f}",
                'Bond_YTM_Percent': f"{ytm*100:.6f}",
                'Bootstrap_Spot_Rate_Percent': f"{spot_rate*100:.6f}",
                'Spot_vs_YTM_Difference_bps': f"{diff_bps:.2f}",
                'Bootstrap_Method': calc_details['method'],
                'Verification_Error': f"{verification_error:.8f}",
                'Convergence_Error': f"{calc_details.get('convergence_error', 0):.2e}"
            }
            results.append(result)
            self.calculation_log.append(calc_details)
        
        self.results_df = pd.DataFrame(results)
        
        # Summary statistics
        print(f"\n=== BOOTSTRAP COMPLETION SUMMARY ===")
        print(f"Total bonds processed: {len(results)}")
        print(f"Verification error statistics:")
        print(f"  Mean error: {np.mean(verification_errors):.6f}")
        print(f"  Median error: {np.median(verification_errors):.6f}")
        print(f"  Max error: {np.max(verification_errors):.6f}")
        print(f"  Bonds with error > 0.01: {sum(1 for e in verification_errors if e > 0.01)}")
        print(f"  Bonds with error > 0.001: {sum(1 for e in verification_errors if e > 0.001)}")
        
        # Method distribution
        method_counts = {}
        for calc_detail in self.calculation_log:
            method = calc_detail['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"\nCalculation method distribution:")
        for method, count in method_counts.items():
            percentage = count / len(self.calculation_log) * 100
            print(f"  {method}: {count} bonds ({percentage:.1f}%)")
        
        return True
    
    def nelson_siegel_svensson(self, t, beta0, beta1, beta2, beta3, tau1, tau2):
        """Nelson-Siegel-Svensson yield curve model"""
        t = np.maximum(t, 1e-6)
        
        term1 = beta0
        term2 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
        term3 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
        term4 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))
        
        return term1 + term2 + term3 + term4
    
    def fit_nss_curve_robust(self):
        """Fit NSS curve with enhanced robustness and outlier handling"""
        if self.results_df is None:
            print("Error: No bootstrap results available")
            return None
        
        print("\nFitting NSS curve to bootstrap spot rates...")
        
        # Extract data
        maturities_all = self.results_df['Time_to_Maturity_Years'].astype(float).values
        spot_rates_all = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values / 100
        ytm_rates_all = self.results_df['Bond_YTM_Percent'].astype(float).values / 100
        weights_all = self.results_df['Index_Weight_Percent'].astype(float).values
        differences_bps = (spot_rates_all - ytm_rates_all) * 10000
        
        # Outlier identification using configurable threshold
        outlier_mask = np.abs(differences_bps) > self.outlier_threshold_bps
        outlier_count = np.sum(outlier_mask)
        
        print(f"Outlier analysis (threshold: {self.outlier_threshold_bps} bps):")
        print(f"  Total bonds: {len(maturities_all)}")
        print(f"  Outliers identified: {outlier_count} ({outlier_count/len(maturities_all)*100:.1f}%)")
        
        # Use clean data for fitting
        clean_mask = ~outlier_mask
        maturities = maturities_all[clean_mask]
        spot_rates = spot_rates_all[clean_mask]
        weights = weights_all[clean_mask]
        
        if len(maturities) < 6:
            print("Error: Too few clean bonds for NSS fitting")
            return None
        
        # NSS optimization with multiple starting points
        def weighted_objective(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            fitted_rates = self.nelson_siegel_svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
            residuals = fitted_rates - spot_rates
            return np.sum((residuals ** 2) * weights)
        
        # Generate starting points based on data
        mean_rate = np.average(spot_rates, weights=weights)
        rate_range = np.max(spot_rates) - np.min(spot_rates)
        
        starting_points = [
            [mean_rate, 0.0, 0.0, 0.0, 2.0, 5.0],
            [mean_rate, -rate_range/4, rate_range/4, -rate_range/8, 1.5, 8.0],
            [mean_rate, rate_range/4, -rate_range/4, rate_range/8, 3.0, 10.0],
            [0.04, -0.01, -0.01, 0.01, 2.0, 5.0],
            [0.06, 0.01, -0.02, 0.01, 1.8, 7.0]
        ]
        
        bounds = [
            (0.001, 0.20), (-0.20, 0.20), (-0.20, 0.20), 
            (-0.20, 0.20), (0.1, 30.0), (0.1, 30.0)
        ]
        
        best_result = None
        best_r_squared = -1
        
        for i, start_point in enumerate(starting_points):
            try:
                result = minimize(weighted_objective, start_point, 
                                method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 2000, 'ftol': 1e-12})
                
                if result.success:
                    beta0, beta1, beta2, beta3, tau1, tau2 = result.x
                    fitted_rates = self.nelson_siegel_svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
                    
                    # Calculate R-squared
                    residuals = fitted_rates - spot_rates
                    weighted_ss_res = np.sum((residuals ** 2) * weights)
                    weighted_mean = np.average(spot_rates, weights=weights)
                    weighted_ss_tot = np.sum(((spot_rates - weighted_mean) ** 2) * weights)
                    r_squared = 1 - (weighted_ss_res / weighted_ss_tot)
                    
                    if r_squared > best_r_squared:
                        best_result = result
                        best_r_squared = r_squared
                        print(f"  Run {i+1}: R² = {r_squared:.6f} (best)")
                    else:
                        print(f"  Run {i+1}: R² = {r_squared:.6f}")
                        
            except Exception as e:
                print(f"  Run {i+1}: Failed - {e}")
        
        if best_result is None:
            print("NSS optimization failed")
            return None
        
        # Calculate final statistics
        beta0, beta1, beta2, beta3, tau1, tau2 = best_result.x
        fitted_rates_clean = self.nelson_siegel_svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2)
        fitted_rates_all = self.nelson_siegel_svensson(maturities_all, beta0, beta1, beta2, beta3, tau1, tau2)
        
        # Store parameters
        self.nss_parameters = {
            'beta0': beta0, 'beta1': beta1, 'beta2': beta2, 'beta3': beta3,
            'tau1': tau1, 'tau2': tau2,
            'weighted_r_squared_clean': best_r_squared,
            'weighted_r_squared_all': 1 - np.sum(((fitted_rates_all - spot_rates_all) ** 2) * weights_all) / np.sum(((spot_rates_all - np.average(spot_rates_all, weights=weights_all)) ** 2) * weights_all),
            'outlier_mask': outlier_mask,
            'outlier_count': outlier_count,
            'clean_count': len(maturities),
            'outlier_threshold_bps': self.outlier_threshold_bps
        }
        
        print(f"NSS fitting completed:")
        print(f"  Clean data R²: {best_r_squared:.6f}")
        print(f"  All data R²: {self.nss_parameters['weighted_r_squared_all']:.6f}")
        
        return self.nss_parameters
    
    def export_results(self, output_filename='robust_bootstrap_results.csv'):
        """Export results to CSV"""
        if self.results_df is None:
            return False
        
        try:
            self.results_df.to_csv(output_filename, index=False)
            print(f"Results exported to: {output_filename}")
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

    def create_comprehensive_report(self, filename='bootstrap_comprehensive_report.txt'):
        """Create detailed analysis report, including full spot rate curve listing
           and detailed payment schedules for all bonds.
        """
        if self.results_df is None:
            print("Error: No results available for comprehensive report.")
            return

        try:
            # Force UTF-8 to avoid Windows default encoding issues
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("COMPREHENSIVE BOOTSTRAP ANALYSIS REPORT\n")
                f.write("=" * 45 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Report Date: {self.report_date}\n")
                f.write(f"Outlier Threshold: {self.outlier_threshold_bps} bps\n\n")
                
                # Data summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 12 + "\n")
                f.write(f"Total bonds processed: {len(self.results_df)}\n")
                
                maturities = self.results_df['Time_to_Maturity_Years'].astype(float).values
                spot_rates = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values
                verification_errors = self.results_df['Verification_Error'].astype(float).values
                
                f.write(f"Maturity range: {maturities.min():.4f} to {maturities.max():.4f} years\n")
                f.write(f"Spot rate range: {spot_rates.min():.3f}% to {spot_rates.max():.3f}%\n")
                f.write(f"Mean verification error: {np.mean(verification_errors):.6f}\n")
                f.write(f"Max verification error: {np.max(verification_errors):.6f}\n\n")
                
                # --- FULL SPOT RATE CURVE LISTING ---
                f.write("FULL BOOTSTRAP SPOT RATE CURVE\n")
                f.write("-" * 32 + "\n")
                f.write("Maturity (Years) | Spot Rate (%)\n")
                f.write("-" * 32 + "\n")

                sorted_rates = sorted(self.spot_rates.items())
                for maturity, rate in sorted_rates:
                    f.write(f"{maturity:.6f}         | {rate * 100:.6f}\n")
                f.write("-" * 32 + "\n\n")
                
                # Method analysis
                method_counts = self.results_df['Bootstrap_Method'].value_counts()
                f.write("CALCULATION METHODS\n")
                f.write("-" * 18 + "\n")
                for method, count in method_counts.items():
                    pct = count / len(self.results_df) * 100
                    f.write(f"{method}: {count} bonds ({pct:.1f}%)\n")
                f.write("\n")
                
                # Error analysis
                large_errors = self.results_df[verification_errors > 0.01]
                if len(large_errors) > 0:
                    f.write("BONDS WITH LARGE VERIFICATION ERRORS (>0.01)\n")
                    f.write("-" * 42 + "\n")
                    for idx, row in large_errors.iterrows():
                        f.write(f"{row['ISIN']}: Error = {row['Verification_Error']}\n")
                        f.write(f"  Method: {row['Bootstrap_Method']}\n")
                        f.write(f"  Spot Rate: {row['Bootstrap_Spot_Rate_Percent']}%\n")
                        f.write(f"  Clean Price: {row['Clean_Price']}\n\n")
                
                # NSS analysis
                if self.nss_parameters:
                    f.write("NSS CURVE FITTING\n")
                    f.write("-" * 16 + "\n")
                    params = self.nss_parameters
                    f.write(f"Clean data R^2: {params['weighted_r_squared_clean']:.6f}\n")
                    f.write(f"All data R^2: {params['weighted_r_squared_all']:.6f}\n")
                    f.write(f"Outliers excluded: {params['outlier_count']}\n")
                    f.write(f"Clean bonds used: {params['clean_count']}\n\n")
                    
                    f.write("NSS Parameters:\n")
                    # Use ASCII labels so they are safe across encodings
                    f.write(f"  beta0: {params['beta0']*100:.3f}%\n")
                    f.write(f"  beta1: {params['beta1']*100:.3f}%\n")
                    f.write(f"  beta2: {params['beta2']*100:.3f}%\n")
                    f.write(f"  beta3: {params['beta3']*100:.3f}%\n")
                    f.write(f"  tau1: {params['tau1']:.2f} years\n")
                    f.write(f"  tau2: {params['tau2']:.2f} years\n\n")

                # ==========================================================
                # DETAILED PAYMENT SCHEDULES FOR ALL BONDS
                # ==========================================================
                f.write("DETAILED PAYMENT SCHEDULES (ALL BONDS)\n")
                f.write("=" * 44 + "\n")
                f.write("Bonds are listed in order of increasing time to maturity.\n\n")

                for idx, row in self.results_df.iterrows():
                    if idx >= len(self.calculation_log) or idx >= len(self.bonds_df):
                        continue

                    calc_details = self.calculation_log[idx]
                    payment_schedule = calc_details.get('payment_schedule', None)
                    if not payment_schedule:
                        continue  # zero-coupon or error fallback

                    bond_row = self.bonds_df.iloc[idx]
                    spot_rate = float(row['Bootstrap_Spot_Rate_Percent']) / 100.0
                    verification_error = float(row['Verification_Error'])
                    clean_price = float(row['Clean_Price'])
                    maturity_years = float(row['Time_to_Maturity_Years'])
                    method = row['Bootstrap_Method']

                    verification_pv = self.calculate_precise_bond_pv(
                        spot_rate,
                        bond_row,
                        payment_schedule
                    )

                    f.write(f"Bond {idx+1}: ISIN {row['ISIN']}\n")
                    f.write(f"  Maturity Date: {row['Maturity_Date']}\n")
                    f.write(f"  Time to Maturity: {maturity_years:.4f} years\n")
                    f.write(f"  Clean Price: {clean_price:.6f}\n")
                    f.write(
                        f"  Coupon: {bond_row['Coupon in %']:.3f}%, "
                        f"Frequency: {int(bond_row['Coupon Frequency'])}\n"
                    )
                    f.write(f"  Bootstrap Method: {method}\n")
                    f.write(f"  Final Spot Rate (last payment): {spot_rate*100:.4f}%\n")
                    f.write(
                        f"  Verification: PV = {verification_pv:.6f}, "
                        f"Target = {clean_price:.6f}\n"
                    )
                    f.write(f"  Verification Error = {verification_error:.8f}\n")
                    f.write(f"  Payment schedule: {len(payment_schedule)} payments\n")

                    total_manual_pv = 0.0
                    for j, (pmt_date, pmt_amt, pmt_time) in enumerate(payment_schedule):
                        if j < len(payment_schedule) - 1:
                            rate = self.robust_interpolate_spot_rate(pmt_time)
                        else:
                            rate = spot_rate

                        df = (1.0 + rate) ** (-pmt_time) if pmt_time > 0 else 1.0
                        pv_contrib = pmt_amt * df
                        total_manual_pv += pv_contrib

                        f.write(
                            f"    Payment {j+1}: {pmt_date.strftime('%Y-%m-%d')} "
                            f"Amount={pmt_amt:.4f}, Time={pmt_time:.4f}y, "
                            f"Rate={rate*100:.4f}%, PV={pv_contrib:.6f}\n"
                        )

                    f.write(f"  Manual PV Total: {total_manual_pv:.6f}\n")
                    f.write(
                        "  Manual vs Function difference: "
                        f"{abs(total_manual_pv - verification_pv):.8f}\n"
                    )
                    f.write("-" * 60 + "\n\n")

            print(f"Comprehensive report saved to: {filename}")
        except Exception as e:
            print(f"Error creating report: {e}")
    
    def plot_results(self, save_plot=True, filename='robust_bootstrap_analysis.png'):
        """Create comprehensive visualization"""
        if self.results_df is None or not MATPLOTLIB_AVAILABLE:
            print("Cannot create plots - no results or matplotlib unavailable")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Robust Bootstrap Analysis Results', fontsize=16, fontweight='bold')
        
        # Extract data
        maturities = self.results_df['Time_to_Maturity_Years'].astype(float).values
        spot_rates = self.results_df['Bootstrap_Spot_Rate_Percent'].astype(float).values
        ytm_rates = self.results_df['Bond_YTM_Percent'].astype(float).values
        weights = self.results_df['Index_Weight_Percent'].astype(float).values
        differences = self.results_df['Spot_vs_YTM_Difference_bps'].astype(float).values
        verification_errors = self.results_df['Verification_Error'].astype(float).values
        
        # Plot 1: Spot rates vs YTM
        ax1.scatter(maturities, spot_rates, alpha=0.7, s=weights*20, c='blue', label='Bootstrap Spot Rates')
        ax1.scatter(maturities, ytm_rates, alpha=0.5, s=20, c='orange', label='Bond YTMs')
        
        if self.nss_parameters:
            curve_maturities = np.linspace(maturities.min(), maturities.max(), 200)
            params = self.nss_parameters
            curve_rates = self.nelson_siegel_svensson(
                curve_maturities, params['beta0'], params['beta1'], 
                params['beta2'], params['beta3'], params['tau1'], params['tau2']
            ) * 100
            ax1.plot(curve_maturities, curve_rates, 'red', linewidth=2, 
                    label=f'NSS Fit (R²={params["weighted_r_squared_clean"]:.3f})')
        
        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Rate (%)')
        ax1.set_title('Bootstrap Spot Rates vs YTMs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Verification errors
        colors = ['green' if e < 0.001 else 'orange' if e < 0.01 else 'red' for e in verification_errors]
        ax2.scatter(maturities, verification_errors * 100, c=colors, alpha=0.7, s=weights*15)
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='0.1 cent threshold')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1 cent threshold')
        ax2.set_xlabel('Maturity (Years)')
        ax2.set_ylabel('Verification Error (cents)')
        ax2.set_title('Bootstrap Verification Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Spot-YTM differences
        colors_diff = ['green' if abs(d) < self.outlier_threshold_bps else 'red' for d in differences]
        ax3.scatter(maturities, differences, c=colors_diff, alpha=0.7, s=weights*15)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=self.outlier_threshold_bps, color='red', linestyle='--', alpha=0.7, 
                   label=f'±{self.outlier_threshold_bps} bps threshold')
        ax3.axhline(y=-self.outlier_threshold_bps, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Maturity (Years)')
        ax3.set_ylabel('Spot - YTM (bps)')
        ax3.set_title('Bootstrap vs YTM Differences')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method distribution
        method_counts = self.results_df['Bootstrap_Method'].value_counts()
        ax4.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
        ax4.set_title('Calculation Method Distribution')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved as: {filename}")
        
        plt.show()


def main():
    """Main function for robust bootstrap analysis"""
    parser = argparse.ArgumentParser(description="Run robust zero-coupon bootstrap.")
    parser.add_argument("--csv", default="IBoxx_with_weights_310725.csv", help="CSV file name under ./Data/")
    parser.add_argument("--report_date", default="2025-07-31", help="Report date (YYYY-MM-DD)")
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

    # --- Configuration: file names only (folders handled above) ---
    CSV_FILE_NAME   = args.csv # Use command line argument
    OUTPUT_CSV_NAME = "robust_bootstrap_results.csv"
    REPORT_NAME     = "bootstrap_comprehensive_report.txt"
    PLOT_NAME       = "robust_bootstrap_analysis.png"

    REPORT_DATE = args.report_date
    OUTLIER_THRESHOLD_BPS = args.outlier_bps

    # Handle ISIN exclusion list
    exclude_isins = [s.strip() for s in args.exclude.split(',') if s.strip()]


    # Build absolute paths
    csv_path    = os.path.join(DATA_DIR, CSV_FILE_NAME)
    output_csv  = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    report_path = os.path.join(OUTPUT_DIR, REPORT_NAME)
    plot_path   = os.path.join(OUTPUT_DIR, PLOT_NAME)

    # Set up logging into Outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(OUTPUT_DIR, f'robust_bootstrap_log_{timestamp}.txt')
    logger = Logger(log_filename)
    sys.stdout = logger

    try:
        print("ROBUST BOOTSTRAP ZERO COUPON SPOT RATES CALCULATOR")
        print("=" * 55)
        print("Enhanced Features:")
        print("- High-precision numerical methods")
        print("- Comprehensive error checking and validation")
        print("- Multiple solving algorithms with fallbacks")
        print("- Detailed verification and diagnostics")
        print(f"- Configurable outlier threshold: {OUTLIER_THRESHOLD_BPS} bps")
        print("=" * 55)

        # Initialize calculator with new CSV path
        calculator = RobustBootstrapSpotRates(
            csv_file_path=csv_path,
            report_date=REPORT_DATE,
            outlier_threshold_bps=OUTLIER_THRESHOLD_BPS,
            min_mty_years=args.min_mty,
            max_mty_years=args.max_mty,
            min_clean_price=args.min_px,
            max_clean_price=args.max_px,
            min_coupon=args.min_coupon,
            max_coupon=args.max_coupon,
            min_ytm=args.min_ytm,
            max_ytm=args.max_ytm,
            exclude_isins=exclude_isins
        )

        # Step 1: Load and validate data
        print("\nSTEP 1: LOADING AND VALIDATING DATA")
        if not calculator.load_and_prepare_data():
            print("Failed to load data")
            return

        # Step 2: Run robust bootstrap
        print("\nSTEP 2: ROBUST BOOTSTRAP CALCULATION")
        if not calculator.run_robust_bootstrap():
            print("Bootstrap calculation failed")
            return

        # Step 3: NSS curve fitting
        print("\nSTEP 3: NSS CURVE FITTING")
        nss_params = calculator.fit_nss_curve_robust()
        if nss_params:
            print("NSS fitting successful")

        # Step 4: Create visualizations (saved to Outputs)
        print("\nSTEP 4: CREATING VISUALIZATIONS")
        try:
            calculator.plot_results(save_plot=True, filename=plot_path)
        except Exception as e:
            print(f"Visualization failed: {e}")

        # Step 5: Export results to Outputs
        print("\nSTEP 5: EXPORTING RESULTS")
        calculator.export_results(output_csv)
        calculator.create_comprehensive_report(report_path)

        print("\n" + "="*55)
        print("ROBUST BOOTSTRAP ANALYSIS COMPLETE!")
        print("Generated files:")
        print(f"  - {output_csv} (CSV results)")
        print(f"  - {report_path} (detailed analysis)")
        print(f"  - {plot_path} (visualization)")
        print(f"  - {log_filename} (complete execution log)")
        print("="*55)

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        print(f"\nAnalysis complete. All output saved to: {log_filename}")



if __name__ == "__main__":
    # Ensure argparse is used correctly for command line arguments
    import argparse
    main()