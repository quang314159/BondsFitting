import os
import streamlit as st
from datetime import date
from bootstrap_spot_rates_v171125 import RobustBootstrapSpotRates  # import your class

# Project folders (one level up from /scripts)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Bond Bootstrap UI", layout="wide")
st.title("Zero-Coupon Spot-Rate Bootstrap • UI")

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
    min_mty = st.number_input("Min maturity (years)", 0.0, 200.0, 0.01, 0.01)
    max_mty = st.number_input("Max maturity (years)", 0.0, 200.0, 100.0, 1.0)
    min_px  = st.number_input("Min clean price", 0.0, 10000.0, 10.0, 0.5)
    max_px  = st.number_input("Max clean price", 0.0, 10000.0, 200.0, 0.5)

    st.subheader("Filters: Coupon, YTM, Freq")
    min_cpn = st.number_input("Min coupon (%)", 0.0, 100.0, 0.0, 0.1)
    max_cpn = st.number_input("Max coupon (%)", 0.0, 100.0, 50.0, 0.1)
    min_ytm = st.number_input("Min YTM (%)", -50.0, 100.0, 0.0, 0.1)
    max_ytm = st.number_input("Max YTM (%)", -50.0, 100.0, 50.0, 0.1)
    allowed_freq = st.multiselect("Allowed coupon frequencies",
                                  options=[1,2,4,12],
                                  default=[1,2,4,12])

    st.subheader("Exclude ISINs")
    excl_text = st.text_area("Enter ISINs to exclude (comma, space, or newline separated)", height=120)
    exclude_isins = {x.strip() for x in excl_text.replace(",", " ").split() if x.strip()}

    run = st.button("Run Bootstrap")

with right:
    if run:
        csv_path = os.path.join(DATA_DIR, csv_file)
        # Filenames under Outputs
        output_csv  = os.path.join(OUTPUT_DIR, "robust_bootstrap_results.csv")
        report_path = os.path.join(OUTPUT_DIR, "bootstrap_comprehensive_report.txt")
        plot_path   = os.path.join(OUTPUT_DIR, "robust_bootstrap_analysis.png")

        st.write("**Running…**")
        try:
            calc = RobustBootstrapSpotRates(
                csv_file_path=csv_path,
                report_date=str(report_date),
                outlier_threshold_bps=float(outlier_bps),
                min_mty_years=min_mty,
                max_mty_years=max_mty,
                min_clean_price=min_px,
                max_clean_price=max_px,
                min_coupon=min_cpn,
                max_coupon=max_cpn,
                allowed_freq=allowed_freq,
                min_ytm=min_ytm,
                max_ytm=max_ytm,
                exclude_isins=exclude_isins
            )

            if not calc.load_and_prepare_data():
                st.error("Failed to load/prepare data.")
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

                    st.success("Done.")
                    st.download_button("Download results CSV", data=open(output_csv,"rb").read(),
                                       file_name="robust_bootstrap_results.csv")
                    st.download_button("Download report (txt)", data=open(report_path,"rb").read(),
                                       file_name="bootstrap_comprehensive_report.txt")
                    st.image(plot_path, caption="Analysis Visualization", use_column_width=True)

                    # Quick summary table
                    st.subheader("Method distribution")
                    md = calc.results_df['Bootstrap_Method'].value_counts().rename_axis('Method').reset_index(name='Count')
                    st.dataframe(md, use_container_width=True)

        except Exception as e:
            import traceback
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
