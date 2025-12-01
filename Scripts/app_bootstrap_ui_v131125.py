import os
import streamlit as st
import pandas as pd
from datetime import date
from bootstrap_spot_rates_v171125 import RobustBootstrapSpotRates  # import your class
import numpy as np

# Project folders (one level up from /scripts)
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

# ---- Inputs (left column)
left, right = st.columns([1,2])

with left:
    st.subheader("Files & Run")
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

    if run_button and csv_file:
        # Build absolute paths
        csv_path    = os.path.join(DATA_DIR, csv_file)
        output_csv  = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
        report_path = os.path.join(OUTPUT_DIR, REPORT_NAME)
        plot_path   = os.path.join(OUTPUT_DIR, PLOT_NAME)

        try:
            st.info(f"Starting analysis for file: {csv_file}")

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