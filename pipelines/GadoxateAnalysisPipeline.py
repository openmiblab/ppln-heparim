import os
import sys
import glob
import json
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Adjust system path to allow clean imports from the sibling 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modeling import extract_par_from_dicom, run_pharmacokinetic_fit

class GadoxateAnalysisPipeline:
    def __init__(self, root):
        self.root = root
        self.root.title("HEPARIM - Gadoxate Multi-Segment Analysis Pipeline")
        self.root.geometry("600x400")
        
        # Load environment mappings from centralized configuration safely
        config_path = os.path.join(os.path.dirname(__file__), '../config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "DEFAULT_BASE_DRIVE": "G:\\Shared drives\\HEPARIM\\Patients 001-029",
                "DEFAULT_OUTPUT_BASE": "X:\\abdominal_imaging\\Shared\\HEPARIMTS\\batch_processing",
                "DEFAULT_PATIENT_LIST": "001, 004, 005"
            }
            
        self.base_drive = tk.StringVar(value=self.config.get("DEFAULT_BASE_DRIVE", ""))
        self.output_base = tk.StringVar(value=self.config.get("DEFAULT_OUTPUT_BASE", ""))
        self.patient_list = tk.StringVar(value=self.config.get("DEFAULT_PATIENT_LIST", ""))
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Patient Database Root Location
        ttk.Label(main_frame, text="Patient Database Root Location:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f1 = ttk.Frame(main_frame); f1.pack(fill="x", pady=(0, 10))
        ttk.Entry(f1, textvariable=self.base_drive).pack(side="left", fill="x", expand=True)
        ttk.Button(f1, text="Browse", command=lambda: self.base_drive.set(filedialog.askdirectory())).pack(side="right", padx=5)

        # Execution Save Target Location
        ttk.Label(main_frame, text="Execution Save Target Location:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        f2 = ttk.Frame(main_frame); f2.pack(fill="x", pady=(0, 10))
        ttk.Entry(f2, textvariable=self.output_base).pack(side="left", fill="x", expand=True)
        ttk.Button(f2, text="Browse", command=lambda: self.output_base.set(filedialog.askdirectory())).pack(side="right", padx=5)

        # Cohort Selection
        ttk.Label(main_frame, text="Target Cohort Subject IDs:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Entry(main_frame, textvariable=self.patient_list).pack(fill="x", pady=(0, 20))
        
        # Action Button
        ttk.Button(main_frame, text="🚀 Optimize Pharmacokinetic Fit Curves", command=self.optimize_fits).pack(pady=10)

    def optimize_fits(self):
        ids = [i.strip() for i in self.patient_list.get().split(",") if i.strip()]
        base = self.base_drive.get()
        out = self.output_base.get()
        
        processed_count = 0
        delay_sec = 1200 # Standard delay offset context gap between scan blocks (20 min)
        
        for pid in ids:
            p_out = os.path.join(out, f"Patient_{pid}")
            pre_csv = os.path.join(p_out, f"patient_{pid}_pre_timecourse.csv")
            post_csv = os.path.join(p_out, f"patient_{pid}_post_timecourse.csv")
            
            if not os.path.exists(pre_csv) or not os.path.exists(post_csv):
                print(f"⚠️ Prerequisite CSV files missing for Patient {pid}. Skipping.")
                continue
            
            df_pre = pd.read_csv(pre_csv).dropna()
            df_post = pd.read_csv(post_csv).dropna()
            
            # Helper function to reliably resolve header variants
            def get_matching_column(df_cols, possibilities):
                for p in possibilities:
                    if p in df_cols:
                        return p
                normalized = {c.replace(' ', '_').replace('(', '').replace(')', ''): c for c in df_cols}
                for p in possibilities:
                    norm_p = p.replace(' ', '_').replace('(', '').replace(')', '')
                    if norm_p in normalized:
                        return normalized[norm_p]
                return None

            try:
                # 1. RESOLVE BASE TIMELINES AND HEADERS
                time_col = get_matching_column(df_pre.columns, ['Time_Seconds', 'Time (Seconds)', 'Time'])
                aorta_col = get_matching_column(df_pre.columns, ['Aorta_Signal', 'Aorta Signal', 'Aorta'])
                portal_col = get_matching_column(df_pre.columns, ['Portal_Vein_Signal', 'Portal Vein Signal', 'Portal_Vein', 'Portal Vein'])
                liver_col = get_matching_column(df_pre.columns, ['Liver_Signal', 'Liver Signal', 'Liver', 'Whole_Liver'])
                
                if not time_col or not aorta_col:
                    print(f"❌ Structural keys missing in CSV headers for Patient {pid}.")
                    continue

                t_pre = df_pre[time_col].values.astype(np.float64)
                t_post = df_post[time_col].values.astype(np.float64) + (t_pre[-1] + delay_sec)
                
                # Setup structured xdata groupings matching dcmri modeling inputs
                xdata_a = (t_pre, t_post, t_pre, t_post)
                t_combined = np.concatenate((t_pre, t_post), axis=0)
                xdata_pv = (t_combined, t_combined, t_combined)
                
                s_a_pre = df_pre[aorta_col].values.astype(np.float64)
                s_a_post = df_post[aorta_col].values.astype(np.float64)
                s_a_combined = np.concatenate((s_a_pre, s_a_post), axis=0)

                # Fetch DICOM parameters
                p_folders = glob.glob(os.path.join(base, f"Patient {pid}*"))
                if not p_folders:
                    continue
                all_dyn = [f for f in glob.glob(os.path.join(p_folders[0], "*_dyn_*")) if os.path.isdir(f)]
                pre_folders = [f for f in all_dyn if "_dyn_post" not in os.path.basename(f).lower()]
                if not pre_folders:
                    continue
                    
                par = extract_par_from_dicom(pre_folders[0])
                
                # 2. RUN WHOLE LIVER MODELING (IF LIVER DATA EXISTS)
                if liver_col:
                    s_l_pre = df_pre[liver_col].values.astype(np.float64)
                    s_l_post = df_post[liver_col].values.astype(np.float64)
                    
                    # Single Intake Aorta Model
                    ydata_a = (s_a_pre, s_a_post, s_l_pre, s_l_post)
                    run_pharmacokinetic_fit(
                        'aorta_liver', par, xdata_a, ydata_a, 
                        os.path.join(p_out, f"patient_2_Scan_Aorta_intake_{pid}_fit.png")
                    )
                    
                    # Dual Intake Model (Requires Portal Vein Data)
                    if portal_col:
                        s_pv_pre = df_pre[portal_col].values.astype(np.float64)
                        s_pv_post = df_post[portal_col].values.astype(np.float64)
                        s_pv_combined = np.concatenate((s_pv_pre, s_pv_post), axis=0)
                        s_l_combined = np.concatenate((s_l_pre, s_l_post), axis=0)
                        
                        ydata_pv = (s_a_combined, s_pv_combined, s_l_combined)
                        run_pharmacokinetic_fit(
                            'aorta_portal_liver', par, xdata_pv, ydata_pv, 
                            os.path.join(p_out, f"patient_dual_intake_{pid}_fit.png")
                        )

                # 3. DYNAMIC ITERATION OVER INDIVIDUAL LIVER SEGMENTS (S1 TO S8)
                for i in range(1, 9):
                    # Check for all plausible header namings of segment i (e.g. 'S1', 'S1_Signal', 'S1 Signal')
                    seg_col = get_matching_column(df_pre.columns, [f'S{i}', f'S{i}_Signal', f'S{i} Signal'])
                    
                    if seg_col and seg_col in df_post.columns:
                        print(f"🚀 Optimizing models for Patient {pid} segment: {seg_col}")
                        s_s_pre = df_pre[seg_col].values.astype(np.float64)
                        s_s_post = df_post[seg_col].values.astype(np.float64)
                        
                        # Process Segment through Single Intake Model
                        ydata_s_a = (s_a_pre, s_a_post, s_s_pre, s_s_post)
                        run_pharmacokinetic_fit(
                            'aorta_liver', par, xdata_a, ydata_s_a, 
                            os.path.join(p_out, f"patient_2_Scan_Aorta_intake_S{i}_{pid}_fit.png")
                        )
                        
                        # Process Segment through Dual Intake Model
                        if portal_col:
                            s_pv_pre = df_pre[portal_col].values.astype(np.float64)
                            s_pv_post = df_post[portal_col].values.astype(np.float64)
                            s_pv_combined = np.concatenate((s_pv_pre, s_pv_post), axis=0)
                            s_s_combined = np.concatenate((s_s_pre, s_s_post), axis=0)
                            
                            ydata_s_pv = (s_a_combined, s_pv_combined, s_s_combined)
                            run_pharmacokinetic_fit(
                                'aorta_portal_liver', par, xdata_pv, ydata_s_pv, 
                                os.path.join(p_out, f"patient_dual_intake_S{i}_{pid}_fit.png")
                            )
                    else:
                        print(f"ℹ️ Segment S{i} column not found for Patient {pid}, passing execution.")
                
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Execution failure in evaluation loop for Patient {pid}: {e}")
            
        messagebox.showinfo("Done", f"Multi-segment PK analysis complete. Processed {processed_count} patient(s).")

if __name__ == "__main__":
    root = tk.Tk()
    app = GadoxateAnalysisPipeline(root)
    root.mainloop()