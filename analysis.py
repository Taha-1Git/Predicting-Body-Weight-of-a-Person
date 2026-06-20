

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# ════════════════════════════════════════════════════════════════
#  TASK 1  — DATA LOADING & ENCODING
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  TASK 1: DATA LOADING & ENCODING")
print("=" * 70)

df_raw = pd.read_csv("obesity.csv")
df_raw = df_raw.dropna()
print(f"  Raw dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

df = df_raw.copy()

# Binary encoding (0/1)
df["Gender"]                        = df["Gender"].map({"Female": 0, "Male": 1})
df["family_history_with_overweight"]= df["family_history_with_overweight"].map({"no": 0, "yes": 1})
df["FAVC"]                          = df["FAVC"].map({"no": 0, "yes": 1})
df["SMOKE"]                         = df["SMOKE"].map({"no": 0, "yes": 1})
df["SCC"]                           = df["SCC"].map({"no": 0, "yes": 1})

# Ordinal encoding
ordinal = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
df["CAEC"] = df["CAEC"].map(ordinal)
df["CALC"] = df["CALC"].map(ordinal)

# Numeric encoding for transport
df["MTRANS"] = df["MTRANS"].map({
    "Walking": 0, "Bike": 1, "Motorbike": 2,
    "Public_Transportation": 3, "Automobile": 4
})

# Ordinal encoding for obesity label
df["NObeyesdad"] = df["NObeyesdad"].map({
    "Insufficient_Weight": 0, "Normal_Weight": 1,
    "Overweight_Level_I": 2,  "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,      "Obesity_Type_II": 5,
    "Obesity_Type_III": 6
})

print("\n  Encoding applied:")
print("    Binary  (0/1)  : Gender, family_history_with_overweight, FAVC, SMOKE, SCC")
print("    Ordinal (0 3)  : CAEC, CALC")
print("    Numeric (0 4)  : MTRANS")
print("    Ordinal (0 6)  : NObeyesdad")

all_17_cols = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC",
    "NCP", "CAEC", "SMOKE", "CH2O", "SCC",
    "FAF", "TUE", "CALC", "MTRANS", "NObeyesdad"
]
df_all17 = df[all_17_cols]

# Drop NObeyesdad (data leakage) before modelling
df_model = df.drop(columns=["NObeyesdad"])

Y_full = df_model["Weight"]
X_full = df_model.drop(columns=["Weight"])

X_const = sm.add_constant(X_full)
ols_full = sm.OLS(Y_full, X_const).fit()

print("\n  OLS Significance Test (α = 0.05) — All 15 candidate predictors:")
print(f"  {'Variable':<35} {'Coeff':>10} {'t-stat':>9} {'p-value':>11}  {'Sig?':>6}")
print("  " + "─" * 76)
for var in X_full.columns:
    c   = ols_full.params[var]
    t   = ols_full.tvalues[var]
    p   = ols_full.pvalues[var]
    sig = "YES ✓" if p < 0.05 else "NO  ✗  ← DROP"
    print(f"  {var:<35} {c:>10.4f} {t:>9.4f} {p:>11.6f}  {sig}")

sig_vars   = [v for v in X_full.columns if ols_full.pvalues[v] < 0.05]
insig_vars = [v for v in X_full.columns if ols_full.pvalues[v] >= 0.05]

print(f"\n  Significant   ({len(sig_vars)}/15) : {sig_vars}")
print(f"  Insignificant ({len(insig_vars)}/15) : {insig_vars}")
print(f"\n  → Dropping: NObeyesdad (DATA LEAKAGE)")
print(f"  → Dropping: {insig_vars}  (p > 0.05)")

X = df_model[sig_vars]
Y = df_model["Weight"]
ind_cols = sig_vars

print(f"\n  Final model: {len(ind_cols)} independent variables + Weight (DV)")
print(f"  Observations: {len(Y)}")

# ════════════════════════════════════════════════════════════════
#  TASK 2  — SUMMARY STATISTICS  (all 17 variables)
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  TASK 2: SUMMARY STATISTICS  (All 17 Variables)")
print("=" * 70)

encoding_notes = {
    "Gender"                       : "Female=0, Male=1",
    "family_history_with_overweight": "no=0, yes=1",
    "FAVC"                         : "no=0, yes=1",
    "SMOKE"                        : "no=0, yes=1  [DROPPED]",
    "SCC"                          : "no=0, yes=1",
    "CAEC"                         : "no=0 | Sometimes=1 | Frequently=2 | Always=3",
    "CALC"                         : "no=0 | Sometimes=1 | Frequently=2 | Always=3",
    "NCP"                          : "numeric  [DROPPED]",
    "MTRANS"                       : "Walking=0 | Bike=1 | Motorbike=2 | Public=3 | Auto=4",
    "NObeyesdad"                   : "0–6 obesity levels  [LEAKAGE — excluded]",
}

stats_rows = []
for col in all_17_cols:
    s = df_all17[col].dropna()
    mode_val = s.mode()
    stats_rows.append({
        "Variable" : col,
        "Count"    : int(s.count()),
        "Mean"     : round(s.mean(),          4),
        "Median"   : round(s.median(),        4),
        "Mode"     : round(float(mode_val[0]),4),
        "Std Dev"  : round(s.std(),           4),
        "Min"      : round(s.min(),           4),
        "Q1"       : round(s.quantile(0.25),  4),
        "Q2(Med)"  : round(s.quantile(0.50),  4),
        "Q3"       : round(s.quantile(0.75),  4),
        "Max"      : round(s.max(),           4),
        "IQR"      : round(s.quantile(0.75) - s.quantile(0.25), 4),
    })

summary_df = pd.DataFrame(stats_rows).set_index("Variable")

for col in all_17_cols:
    r      = summary_df.loc[col]
    dv_tag = "  *** DEPENDENT VARIABLE ***" if col == "Weight" else ""
    enc    = f"\n  Encoding : {encoding_notes[col]}" if col in encoding_notes else ""
    print(f"\n  {'─'*62}")
    print(f"  Variable : {col}{dv_tag}{enc}")
    print(f"  {'─'*62}")
    print(f"  Count    : {r['Count']}")
    print(f"  Mean     : {r['Mean']}")
    print(f"  Median   : {r['Median']}")
    print(f"  Mode     : {r['Mode']}")
    print(f"  Std Dev  : {r['Std Dev']}")
    print(f"  Min      : {r['Min']}")
    print(f"  Q1 (25%) : {r['Q1']}")
    print(f"  Q2 (50%) : {r['Q2(Med)']}")
    print(f"  Q3 (75%) : {r['Q3']}")
    print(f"  Max      : {r['Max']}")
    print(f"  IQR      : {r['IQR']}")

print(f"\n  {'='*70}")
print("  Full Summary Table:\n")
print(summary_df.T.to_string())
summary_df.to_csv("task2_summary_stats.csv")
print("\n  Saved: task2_summary_stats.csv")

# ════════════════════════════════════════════════════════════════
#  TASK 3  — BOX & WHISKER PLOTS
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  TASK 3: BOX & WHISKER PLOTS  (13 Significant IVs)")
print("=" * 70)

colors13 = [
    "#E63946", "#F4A261", "#2A9D8F", "#264653", "#9D4EDD",
    "#06A77D", "#F77F00", "#003049", "#D62828", "#8338EC",
    "#3A86FF", "#FB5607", "#FFBE0B",
]

short_enc = {
    "Gender"                       : "0=Female  1=Male",
    "family_history_with_overweight": "0=no  1=yes",
    "FAVC"                         : "0=no  1=yes",
    "SCC"                          : "0=no  1=yes",
    "CAEC"                         : "0=no 1=Sometimes 2=Freq 3=Always",
    "CALC"                         : "0=no 1=Sometimes 2=Freq 3=Always",
    "MTRANS"                       : "0=Walk 1=Bike 2=Moto 3=Public 4=Auto",
}

print(f"\n  {'Variable':<35} {'IQR':>8} {'Outliers':>10}")
print("  " + "─" * 56)
for col in ind_cols:
    s   = X[col]
    q1  = s.quantile(0.25); q3 = s.quantile(0.75)
    iqr = q3 - q1
    n_o = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
    print(f"  {col:<35} {iqr:>8.3f} {n_o:>10}")

fig3, axes3 = plt.subplots(4, 4, figsize=(20, 17))
fig3.patch.set_facecolor("#F5F5F5")
axes3_flat = axes3.flatten()

for i, (col, color) in enumerate(zip(ind_cols, colors13)):
    ax   = axes3_flat[i]
    data = X[col].dropna()

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.5),
        flierprops=dict(marker="o", markerfacecolor="red",
                        markersize=4, markeredgecolor="red", linestyle="none"),
        whiskerprops=dict(color=color, linewidth=1.5, linestyle="--"),
        capprops=dict(color=color, linewidth=2),
        boxprops=dict(linewidth=1.5, edgecolor=color),
    )
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.75)

    q1  = data.quantile(0.25)
    med = data.median()
    q3  = data.quantile(0.75)
    iqr = q3 - q1
    n_o = int(((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum())

    for val, lbl in [(q1, f"Q1={q1:.2f}"), (med, f"Med={med:.2f}"), (q3, f"Q3={q3:.2f}")]:
        ax.annotate(lbl, xy=(1, val), xytext=(1.12, val),
                    textcoords="data", fontsize=6.5, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.5))

    enc_label = f"\n{short_enc[col]}" if col in short_enc else ""
    ax.set_title(f"{col}{enc_label}\nOutliers: {n_o}",
                 fontsize=8.5, color=color, pad=5, linespacing=1.4)
    ax.set_ylabel("Value", fontsize=7, color="dimgray")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for j in range(len(ind_cols), 16):
    axes3_flat[j].set_visible(False)

legend3 = [mpatches.Patch(facecolor=colors13[i], alpha=0.80, label=ind_cols[i])
           for i in range(len(ind_cols))]
fig3.legend(handles=legend3, title="13 Significant Independent Variables",
            loc="lower center", ncol=7, fontsize=8, title_fontsize=9,
            frameon=True, bbox_to_anchor=(0.5, -0.03))

fig3.suptitle(
    "Task 3: Box & Whisker Plots — Obesity Dataset  (13 Significant IVs)\n"
    "Red dots = Outliers  |  Black line = Median  |  Box = Q1–Q3  |  Whiskers = 1.5×IQR",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
fig3.savefig("task3_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved: task3_boxplots.png")

# ════════════════════════════════════════════════════════════════
#  TASK 4  — SCATTER PLOT GRID
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  TASK 4: SCATTER PLOT GRID")
print("=" * 70)

fig4, axes4 = plt.subplots(4, 4, figsize=(22, 20))
fig4.patch.set_facecolor("#F5F5F5")
axes4_flat = axes4.flatten()

print(f"\n  {'Variable':<35} {'r':>8}  {'Strength & Direction'}")
print("  " + "─" * 65)

for i, (col, color) in enumerate(zip(ind_cols, colors13)):
    ax     = axes4_flat[i]
    x_vals = X[col].values.astype(float)
    y_vals = Y.values.astype(float)

    ax.scatter(x_vals, y_vals, color=color, alpha=0.30, s=12, linewidths=0, zorder=2)

    m, b, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.8, zorder=3)

    r_color   = "#1B5E20" if r_val > 0 else "#B71C1C"
    direction = ("↑ Positive" if r_val >  0.1 else
                 "↓ Negative" if r_val < -0.1 else "≈ No relationship")
    ax.text(0.05, 0.95, f"r = {r_val:.3f}", transform=ax.transAxes,
            fontsize=9, color=r_color, fontweight="bold", va="top")
    ax.text(0.05, 0.85, direction, transform=ax.transAxes,
            fontsize=7.5, color=r_color, va="top")

    enc_lbl = f"\n{short_enc[col]}" if col in short_enc else ""
    ax.set_title(f"{col}{enc_lbl}", fontsize=8.5, color=color, pad=4, linespacing=1.4)
    ax.set_xlabel(col, fontsize=7, color="dimgray")
    ax.set_ylabel("Weight (kg)", fontsize=7, color="dimgray")
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#FAFAFA")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    strength = ("Strong"   if abs(r_val) >= 0.5 else
                "Moderate" if abs(r_val) >= 0.3 else
                "Weak"     if abs(r_val) >= 0.1 else "Very weak")
    dirn = "positive" if r_val >= 0 else "negative"
    print(f"  {col:<35} {r_val:>8.4f}  {strength} {dirn}")

for j in range(len(ind_cols), 16):
    axes4_flat[j].set_visible(False)

fig4.suptitle(
    "Task 4: Scatter Plot Grid — Each IV vs Weight (DV)",
    fontsize=13, fontweight="bold", y=1.005
)
plt.tight_layout()
fig4.savefig("task4_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved: task4_scatter.png")

# ════════════════════════════════════════════════════════════════
#  TASK 5  — MODEL ESTIMATION
# ════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  TASK 5: MODEL ESTIMATION")
print("=" * 70)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"\n  Train/Test Split: 80% / 20%")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─── STEP A: OLS WITH SIGNIFICANT VARIABLES ───
print("\n  " + "─" * 65)
print("  STEP A: OLS Regression — Full Statistical Output")
print("  " + "─" * 65)

X_const13 = sm.add_constant(X)
ols13 = sm.OLS(Y, X_const13).fit()

print(f"\n  OLS R²       = {ols13.rsquared:.4f}")
print(f"  OLS Adj. R²  = {ols13.rsquared_adj:.4f}")
print(f"  F-stat p-val = {ols13.f_pvalue:.4e}")
print(f"\n  MLR Equation:")
print(f"  Ŷ = {ols13.params['const']:.4f}", end="")
for var in ind_cols:
    c    = ols13.params[var]
    sign = "+" if c >= 0 else "-"
    print(f" {sign} {abs(c):.4f}x{var}", end="")
print()

print(f"\n  {'Variable':<35} {'β̂':>12} {'SE':>10} {'t-stat':>9} {'p-value':>11}  Sig?")
print("  " + "─" * 84)
for var in ["const"] + ind_cols:
    c   = ols13.params[var]
    se  = ols13.bse[var]
    t   = ols13.tvalues[var]
    p   = ols13.pvalues[var]
    sig = "YES ✓" if p < 0.05 else "NO ✗"
    print(f"  {var:<35} {c:>12.4f} {se:>10.4f} {t:>9.4f} {p:>11.6f}  {sig}")

# Dynamic coefficient interpretation
print(f"\n  COEFFICIENT INTERPRETATIONS (auto-generated from β̂):")
print(f"  {'─'*68}")
desc = {
    "Gender": "being Male (vs Female)",
    "Age": "1-year increase in Age",
    "Height": "1-meter increase in Height",
    "family_history_with_overweight": "having family history of overweight",
    "FAVC": "consuming high-calorie food frequently",
    "FCVC": "1-unit increase in vegetable consumption frequency",
    "CAEC": "1-level increase in eating between meals",
    "CH2O": "1-unit increase in daily water intake",
    "SCC": "monitoring calorie consumption",
    "FAF": "1-unit increase in physical activity frequency",
    "TUE": "1-unit increase in technology-use time",
    "CALC": "1-level increase in alcohol consumption",
    "MTRANS": "1-step increase in transport mode code",
}
for var in ind_cols:
    coef = ols13.params[var]
    direction = "increases" if coef > 0 else "decreases"
    print(f"  β̂({var}) = {coef:+.4f}")
    print(f"     → {desc.get(var, var)} {direction} Weight by {abs(coef):.2f} kg")

# VIF check
print(f"\n  VIF (Variance Inflation Factor) Check:")
print(f"  {'Variable':<35} {'VIF':>8}")
print(f"  {'─'*46}")
for i, var in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    flag = " ← HIGH" if vif > 10 else ""
    print(f"  {var:<35} {vif:>8.3f}{flag}")



# Correlation heatmap
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
corr = X.assign(Weight=Y).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
ax_corr.set_title("Correlation Matrix — 13 IVs + Weight (DV)",
                  fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
fig_corr.savefig("task5_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved: task5_correlation_heatmap.png")


# Lecture formulas
# Y_arr  = Y.values
# Y_bar  = Y_arr.mean()
# n      = len(Y_arr)
# p      = len(ind_cols) + 1

# Y_hat_ols = ols13.fittedvalues.values
# e_ols     = Y_arr - Y_hat_ols

# SSE = float(np.dot(e_ols, e_ols))
# SST = float(np.dot(Y_arr, Y_arr) - n * Y_bar**2)
# SSR = SST - SSE
# R2  = SSR / SST
# sigma2_e = SSE / (n - p)

# print(f"\n  LECTURE FORMULAS VERIFIED:")
# print(f"  n   = {n}  |  p = {p}  |  Ȳ = {Y_bar:.4f}")
# print(f"  SSE = {SSE:,.4f}")
# print(f"  SST = {SST:,.4f}")
# print(f"  SSR = {SSR:,.4f}")
# print(f"  R²  = {R2:.6f}  ≈ {R2*100:.2f}%")
# print(f"  σ²e = {sigma2_e:.4f}")

#  STEP B: SKLEARN MLR 
print("\n  " + "─" * 65)
print("  STEP B: MLR — Train/Test Evaluation")
print("  " + "─" * 65)

mlr = LinearRegression()
mlr.fit(X_train, Y_train)
mlr_pred = mlr.predict(X_test)

mlr_mse  = mean_squared_error(Y_test, mlr_pred)
mlr_rmse = np.sqrt(mlr_mse)
mlr_r2   = r2_score(Y_test, mlr_pred)

print(f"  Intercept : {mlr.intercept_:.4f}")
print(f"  MSE  = {mlr_mse:.4f}  |  RMSE = {mlr_rmse:.4f}  |  R² = {mlr_r2:.4f}")


# STEP C: RANDOM FOREST 
print("\n  " + "─" * 65)
print("  STEP C: Random Forest Regressor")
print("  " + "─" * 65)

rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)
rf_pred  = rf.predict(X_test)

rf_mse   = mean_squared_error(Y_test, rf_pred)
rf_rmse  = np.sqrt(rf_mse)
rf_r2    = r2_score(Y_test, rf_pred)

print(f"  MSE  = {rf_mse:.4f}  |  RMSE = {rf_rmse:.4f}  |  R² = {rf_r2:.4f}")

imp_df = pd.DataFrame({
    "Variable"  : ind_cols,
    "Importance": rf.feature_importances_.round(4)
}).sort_values("Importance", ascending=False)
print(f"\n  Random Forest Feature Importances:")
for _, row in imp_df.iterrows():
    print(f"    {row['Variable']:<35} {row['Importance']:>8.4f}")



# ─── COMPARISON ───
print("\n\n" + "=" * 70)
print("  CONCLUSION: MODEL COMPARISON")
print("=" * 70)

comp_df = pd.DataFrame({
    "Metric"       : ["MSE", "RMSE (kg)", "R²"],
    "MLR"          : [round(mlr_mse, 4), round(mlr_rmse, 4), round(mlr_r2, 4)],
    "RandomForest" : [round(rf_mse,  4), round(rf_rmse,  4), round(rf_r2,  4)],
})


print(f"\n  {'Metric':<12} {'MLR':>12} {'Random Forest':>16}")
print(f"  {'─'*42}")
for _, row in comp_df.iterrows():
    print(f"  {row['Metric']:<12} {row['MLR']:>12} {row['RandomForest']:>16}")

winner = "Random Forest" if rf_mse < mlr_mse else "MLR"
print(f"\n  Winner (lower MSE): {winner}")
comp_df.to_csv("task5_comparison.csv", index=False)


sample_idx = np.random.choice(len(Y_test), 50, replace=False)
actual     = Y_test.values[sample_idx]
pred_mlr   = mlr_pred[sample_idx]
pred_rf    = rf_pred[sample_idx]
x_ax       = np.arange(50)

fig5a, ax5a = plt.subplots(figsize=(16, 6))
fig5a.patch.set_facecolor("#F5F5F5")
ax5a.set_facecolor("#FAFAFA")
ax5a.plot(x_ax, actual,   "ko-",  lw=1.5, ms=5, label="Actual Weight", alpha=0.85, zorder=3)
ax5a.plot(x_ax, pred_mlr, "b--s", lw=1.4, ms=4,
          label=f"MLR (MSE={mlr_mse:.2f})", alpha=0.80, zorder=2)
ax5a.plot(x_ax, pred_rf,  "r:^",  lw=1.4, ms=4,
          label=f"RF  (MSE={rf_mse:.2f})", alpha=0.80, zorder=2)
ax5a.set_title("Task 5: Actual vs Predicted Weight (50 Random Test Samples)",
               fontsize=13, fontweight="bold")
ax5a.set_xlabel("Sample Index", fontsize=10)
ax5a.set_ylabel("Weight (kg)",  fontsize=10)
ax5a.legend(fontsize=10)
ax5a.grid(linestyle="--", linewidth=0.5, alpha=0.6)
ax5a.spines["top"].set_visible(False)
ax5a.spines["right"].set_visible(False)
plt.tight_layout()
fig5a.savefig("task5_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()

fig5b, (ax_mlr, ax_rf) = plt.subplots(1, 2, figsize=(14, 6))
fig5b.patch.set_facecolor("#F5F5F5")
fig5b.suptitle("Task 5: Predicted vs Actual — Full Test Set",
               fontsize=13, fontweight="bold")

w_min, w_max = float(Y.min()), float(Y.max())
for ax, preds, label, color, r2, mse in [
    (ax_mlr, mlr_pred, "Multiple Linear Regression", "#3F51B5", mlr_r2, mlr_mse),
    (ax_rf,  rf_pred,  "Random Forest",              "#00897B", rf_r2,  rf_mse),
]:
    ax.scatter(Y_test, preds, alpha=0.35, s=12, color=color, linewidths=0)
    ax.plot([w_min, w_max], [w_min, w_max], "r--", lw=2, label="Perfect fit")
    ax.set_title(f"{label}\nR²={r2:.4f}  |  MSE={mse:.2f}",
                 fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Actual Weight (kg)", fontsize=9)
    ax.set_ylabel("Predicted Weight (kg)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_facecolor("#FAFAFA")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
fig5b.savefig("task5_scatter_predicted_actual.png", dpi=150, bbox_inches="tight")
plt.show()

mlr_resid = Y_test.values - mlr_pred
fig5c, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(14, 5))
fig5c.patch.set_facecolor("#F5F5F5")
fig5c.suptitle("Task 5: MLR Residual Analysis", fontsize=12, fontweight="bold")

ax_r1.scatter(mlr_pred, mlr_resid, alpha=0.40, s=12, color="#3F51B5", linewidths=0)
ax_r1.axhline(0, color="red", linestyle="--", lw=2)
ax_r1.set_title("Residuals vs Fitted", fontsize=11, fontweight="bold")
ax_r1.set_xlabel("Fitted Weight (kg)"); ax_r1.set_ylabel("Residuals (kg)")
ax_r1.set_facecolor("#FAFAFA"); ax_r1.grid(linestyle="--", alpha=0.4)
ax_r1.spines["top"].set_visible(False); ax_r1.spines["right"].set_visible(False)

ax_r2.hist(mlr_resid, bins=35, color="#3F51B5", alpha=0.75, edgecolor="white")
ax_r2.axvline(0, color="red", linestyle="--", lw=2)
ax_r2.set_title("Residual Distribution", fontsize=11, fontweight="bold")
ax_r2.set_xlabel("Residual (kg)"); ax_r2.set_ylabel("Frequency")
ax_r2.set_facecolor("#FAFAFA"); ax_r2.grid(linestyle="--", alpha=0.4)
ax_r2.spines["top"].set_visible(False); ax_r2.spines["right"].set_visible(False)
plt.tight_layout()
fig5c.savefig("task5_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

imp_sorted = imp_df.sort_values("Importance")
fig5d, ax5d = plt.subplots(figsize=(10, 7))
fig5d.patch.set_facecolor("#F5F5F5"); ax5d.set_facecolor("#FAFAFA")
bar_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(imp_sorted)))
bars = ax5d.barh(imp_sorted["Variable"], imp_sorted["Importance"],
                 color=bar_colors, edgecolor="white", height=0.65)
for bar, val in zip(bars, imp_sorted["Importance"]):
    ax5d.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
              f"{val:.4f}", va="center", fontsize=8.5, color="dimgray")
ax5d.set_xlabel("Importance Score", fontsize=10)
ax5d.set_title("Random Forest — Feature Importances",
               fontsize=12, fontweight="bold")
ax5d.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
ax5d.spines["top"].set_visible(False); ax5d.spines["right"].set_visible(False)
plt.tight_layout()
fig5d.savefig("task5_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

metrics  = ["MSE", "RMSE (kg)", "R²"]
mlr_vals = [mlr_mse, mlr_rmse, mlr_r2]
rf_vals  = [rf_mse,  rf_rmse,  rf_r2]
x_pos    = np.arange(len(metrics))

fig5e, ax5e = plt.subplots(figsize=(10, 6))
fig5e.patch.set_facecolor("#F5F5F5"); ax5e.set_facecolor("#FAFAFA")
b1 = ax5e.bar(x_pos - 0.22, mlr_vals, 0.42, label="MLR", color="#3F51B5", alpha=0.85)
b2 = ax5e.bar(x_pos + 0.22, rf_vals,  0.42, label="Random Forest", color="#00897B", alpha=0.85)
for bar in list(b1) + list(b2):
    ax5e.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
              f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    

ax5e.set_xticks(x_pos)
ax5e.set_xticklabels(metrics, fontsize=11)
ax5e.set_ylabel("Value", fontsize=11)
ax5e.set_title("MLR vs Random Forest — Performance Metrics",
               fontsize=12, fontweight="bold")
ax5e.legend(fontsize=11)
ax5e.grid(axis="y", linestyle="--", alpha=0.4)
ax5e.spines["top"].set_visible(False); ax5e.spines["right"].set_visible(False)
plt.tight_layout()
fig5e.savefig("task5_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()



print("\n\n" + "=" * 70)
print("  ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"\n  Final Performance Summary:")
print(f"    MLR  → MSE={mlr_mse:.4f}  RMSE={mlr_rmse:.4f}  R²={mlr_r2:.4f}")
print(f"    RF   → MSE={rf_mse:.4f}   RMSE={rf_rmse:.4f}   R²={rf_r2:.4f}")
print(f"    Winner: {winner}")
