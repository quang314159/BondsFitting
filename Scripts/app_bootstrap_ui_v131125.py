import os
import re
import importlib.util
import streamlit as st
import pandas as pd
from datetime import date, datetime
import numpy as np

# Project folders (one level up from /scripts)
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Bond Bootstrap UI", layout="wide")
st.title("Zero-Coupon Spot-Rate Bootstrap & Curve Fitting")

# --- Initialize paths ---
CSV_FILE_NAME = "IBoxx_with_weights_310725.csv" # Placeholder name, updated by st.selectbox later
OUTPUT_CSV_NAME = "robust_bootstrap_results.csv"
REPORT_NAME     = "bootstrap_comprehensive_report.txt"
PLOT_NAME       = "robust_bootstrap_analysis.png"

# --- Calculation scripts ---
# Any Scripts/bootstrap_spot_rates_vDDMMYY.py file is offered, newest date first.
# Each must define class RobustBootstrapSpotRates with the same constructor
# arguments and methods the app calls below (REQUIRED_METHODS).
CALC_PATTERN = re.compile(r"^bootstrap_spot_rates_v(\d{6})\.py$")
REQUIRED_METHODS = ("load_and_prepare_data", "run_robust_bootstrap", "fit_nss_curve_robust",
                    "plot_results", "export_results", "create_comprehensive_report")


def list_calc_scripts():
    """Return calculation script file names, newest version date (DDMMYY) first."""
    found = []
    for f in os.listdir(SCRIPTS_DIR):
        m = CALC_PATTERN.match(f)
        if m:
            try:
                found.append((datetime.strptime(m.group(1), "%d%m%y"), f))
            except ValueError:
                continue  # six digits that are not a valid DDMMYY date
    return [f for _, f in sorted(found, reverse=True)]


@st.cache_resource
def load_calc_class(path, mtime):
    """Import the script at `path` and return its RobustBootstrapSpotRates class.

    `mtime` is part of the cache key, so an updated file is re-imported.
    """
    name = "calc_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "RobustBootstrapSpotRates", None)
    if cls is None:
        raise AttributeError(f"{os.path.basename(path)} has no class RobustBootstrapSpotRates")
    missing = [m for m in REQUIRED_METHODS if not callable(getattr(cls, m, None))]
    if missing:
        raise AttributeError(f"{os.path.basename(path)}: RobustBootstrapSpotRates lacks {', '.join(missing)}")
    return cls

# ---- Inputs (left column)
left, right = st.columns([1,2])

with left:
    st.subheader("Files & Run")
    calc_scripts = list_calc_scripts()
    calc_script = st.selectbox(
        "Calculation script",
        options=calc_scripts,
        index=0 if calc_scripts else None,
        help="Scripts/bootstrap_spot_rates_vDDMMYY.py files, newest first.",
    )
    csv_file = st.selectbox(
        "Select data CSV",
        options=[f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")],
        index=0 if any(os.listdir(DATA_DIR)) else None,
    )
    report_date = st.date_input("Report date", value=date(2025,7,31))
    outlier_bps = st.number_input("Outlier threshold (bps)", 0.0, 1000.0, 50.0, 1.0)

    st.subheader("Filters: Maturity & Price")
    min_mty = st.number_input("Min Maturity (Years)", 0.0, 10.0, 0.01, 0.01)
    max_mty = st.number_input("Max Maturity (Years)", 1.0, 200.0, 100.0, 1.0)
    min_px = st.number_input("Min Clean Price", 0.0, 200.0, 10.0, 1.0)
    max_px = st.number_input("Max Clean Price", 0.0, 300.0, 200.0, 1.0)

    st.subheader("Filters: Coupon & Yield")
    min_coupon = st.number_input("Min Coupon (%)", 0.0, 50.0, 0.0, 0.1)
    max_coupon = st.number_input("Max Coupon (%)", 0.0, 50.0, 50.0, 0.1)
    min_ytm = st.number_input("Min Annual Yield (%)", 0.0, 50.0, 0.0, 0.1)
    max_ytm = st.number_input("Max Annual Yield (%)", 0.0, 50.0, 50.0, 0.1)

    exclude_isins_input = st.text_area("ISINs to Exclude (comma separated)", value="")
    exclude_isins = [s.strip() for s in exclude_isins_input.split(',') if s.strip()]

    run_button = st.button("Run Bootstrap Analysis", type="primary", use_container_width=True)


with right:
    st.subheader("Results and Diagnostics")

    if run_button and not calc_script:
        st.error("No calculation script found: expected Scripts/bootstrap_spot_rates_vDDMMYY.py")

    if run_button and csv_file and calc_script:
        # Build absolute paths
        csv_path    = os.path.join(DATA_DIR, csv_file)
        output_csv  = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
        report_path = os.path.join(OUTPUT_DIR, REPORT_NAME)
        plot_path   = os.path.join(OUTPUT_DIR, PLOT_NAME)

        try:
            st.info(f"Starting analysis for file: {csv_file} with {calc_script}")

            # Load the selected calculation script
            calc_path = os.path.join(SCRIPTS_DIR, calc_script)
            RobustBootstrapSpotRates = load_calc_class(calc_path, os.path.getmtime(calc_path))

            # Initialize calculator
            calc = RobustBootstrapSpotRates(
                csv_file_path=csv_path,
                report_date=report_date.strftime('%Y-%m-%d'),
                outlier_threshold_bps=outlier_bps,
                min_mty_years=min_mty,
                max_mty_years=max_mty,
                min_clean_price=min_px,
                max_clean_price=max_px,
                min_coupon=min_coupon,
                max_coupon=max_coupon,
                min_ytm=min_ytm,
                max_ytm=max_ytm,
                exclude_isins=exclude_isins
            )

            if not calc.load_and_prepare_data():
                st.error("Failed to load/prepare data. Check file format or filter bounds.")
            else:
                ok = calc.run_robust_bootstrap()
                if not ok:
                    st.error("Bootstrap calculation failed.")
                else:
                    nss = calc.fit_nss_curve_robust()
                    
                    # Save outputs
                    calc.plot_results(save_plot=True, filename=plot_path)
                    calc.export_results(output_csv)
                    calc.create_comprehensive_report(report_path)

                    st.success("Analysis Complete!")
                   
                    # Plot Visualization
                    st.subheader("Analysis Visualization")
                    st.image(plot_path, caption="Spot Rates, YTMs, and NSS Fit", use_container_width=True)

                    # --- NSS Parameters Display (NEW CODE BLOCK) ---
                    if nss:
                        st.subheader("Nelson-Siegel-Svensson (NSS) Curve Parameters")
                        
                        # Prepare data for display
                        nss_data = {
                            "Parameter": [r"$\beta_0$ (Long-Term Rate)", r"$\beta_1$ (Slope)", 
                                          r"$\beta_2$ (Curvature 1)", r"$\beta_3$ (Curvature 2)", 
                                          r"$\tau_1$ (Time Constant 1)", r"$\tau_2$ (Time Constant 2)"],
                            "Value": [nss['beta0'], nss['beta1'], nss['beta2'], 
                                      nss['beta3'], nss['tau1'], nss['tau2']],
                            "Unit": ["\% (Decimal)", "\% (Decimal)", "\% (Decimal)", 
                                     "\% (Decimal)", "Years", "Years"]
                        }
                        nss_df = pd.DataFrame(nss_data)
                        
                        st.table(nss_df)
                        
                        st.markdown(f"""
                        **Goodness-of-Fit (Weighted $R^2$):**
                        * **Clean Data (Outliers Excluded):** {nss['weighted_r_squared_clean']:.6f}
                        * **All Data (Including Outliers):** {nss['weighted_r_squared_all']:.6f}
                        * **Outliers Removed:** {nss['outlier_count']} bonds (Threshold: {nss['outlier_threshold_bps']:.1f} bps)
                        """)

                        # Index statistics (only from scripts that provide them)
                        idx_stats = nss.get('index_statistics')
                        if idx_stats:
                            st.subheader("Index Statistics")
                            measures = [
                                ("Bonds", "Bonds", "{:.0f}"),
                                ("Weight in data file (%)", "Weight_in_file_%", "{:.4f}"),
                                ("GRY on fitted curve (%)", "Curve_GRY_%", "{:.4f}"),
                                ("Macaulay duration on fitted curve (years)", "Curve_Macaulay", "{:.4f}"),
                                ("Modified duration on fitted curve", "Curve_Modified", "{:.4f}"),
                                ("GRY at market prices (%)", "Market_GRY_%", "{:.4f}"),
                                ("Macaulay duration at market prices (years)", "Market_Macaulay", "{:.4f}"),
                                ("Modified duration at market prices", "Market_Modified", "{:.4f}"),
                            ]
                            idx_table = pd.DataFrame({"Measure": [m[0] for m in measures]})
                            for row in idx_stats:
                                idx_table[row["Index"]] = [fmt.format(row[key]) for _, key, fmt in measures]
                            st.table(idx_table.set_index("Measure"))
                            st.caption("Every bond in the data file, by its Index tag (a 15+ bond is also in 10+ and AA). "
                                       "Bonds held at market-value weights. GRY: annual compounding, t = days/365.25; "
                                       "durations at that GRY.")
                    # --- END NSS Parameters Display ---

                    # Download Buttons
                    st.download_button("Download results CSV", data=open(output_csv,"rb").read(),
                                       file_name="robust_bootstrap_results.csv", key="csv_dl")
                    st.download_button("Download report (txt)", data=open(report_path,"rb").read(),
                                       file_name="bootstrap_comprehensive_report.txt", key="txt_dl")


                    # Quick summary table
                    st.subheader("Method Distribution")
                    md = calc.results_df['Bootstrap_Method'].value_counts().rename_axis('Method').reset_index(name='Count')
                    st.dataframe(md, use_container_width=True)

        except Exception as e:
            import traceback
            st.error(f"Analysis Error: {e}")
            st.code(traceback.format_exc())