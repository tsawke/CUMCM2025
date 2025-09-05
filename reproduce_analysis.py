# -*- coding: utf-8 -*-
"""
Reproduce NIPT analysis.
Usage: python reproduce_analysis.py
Requires: pandas, numpy, scikit-learn, scipy, reportlab
Input: /mnt/data/附件.xlsx with sheets '男胎检测数据' and '女胎检测数据'
Outputs: report markdown/pdf and intermediate CSVs under /mnt/data
"""
import os, re, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy import stats
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE = Path("/mnt/data")
XLSX = BASE/"附件.xlsx"

def parse_gw(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    m = re.match(r"(\\d+)\\s*w\\s*\\+\\s*(\\d+)", s, flags=re.I)
    if m: return int(m.group(1)) + int(m.group(2))/7.0
    m2 = re.match(r"(\\d+)\\s*w", s, flags=re.I)
    if m2: return float(m2.group(1))
    try:
        return float(s)
    except:
        return np.nan

def safe_spearman(x, y):
    mask = ~(pd.isna(x) | pd.isna(y))
    if mask.sum() < 10: 
        return (np.nan, np.nan)
    r, p = stats.spearmanr(x[mask], y[mask])
    return (r, p)

def draw_text_block(c, text, x=2*cm, y=A4[1]-2*cm, leading=14, font="DejaVuSans", size=10):
    width, height = A4
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        c.setFont(font, size)
    except:
        c.setFont("Helvetica", size)
    max_width = width - 4*cm
    y_pos = y
    for line in text.split("\\n"):
        if not line:
            y_pos -= leading
            if y_pos < 2*cm:
                c.showPage(); c.setFont(font, size); y_pos = height-2*cm
            continue
        chunk = ""
        for ch in line:
            test = chunk + ch
            if pdfmetrics.stringWidth(test, c._fontname, size) > max_width:
                c.drawString(x, y_pos, chunk)
                y_pos -= leading
                chunk = ch
                if y_pos < 2*cm:
                    c.showPage(); c.setFont(font, size); y_pos = height-2*cm
            else:
                chunk = test
        if chunk:
            c.drawString(x, y_pos, chunk)
            y_pos -= leading
            if y_pos < 2*cm:
                c.showPage(); c.setFont(font, size); y_pos = height-2*cm
    return y_pos

def main():
    xl = pd.ExcelFile(XLSX)
    male = xl.parse("男胎检测数据")
    female = xl.parse("女胎检测数据")

    for df in (male, female):
        if "检测孕周" in df.columns:
            df["孕周_周"] = df["检测孕周"].apply(parse_gw)
        if "身高" in df.columns and "体重" in df.columns:
            h_m = pd.to_numeric(df["身高"], errors="coerce")/100.0
            w = pd.to_numeric(df["体重"], errors="coerce")
            df["BMI"] = w/(h_m**2)

    if "Y染色体浓度" in male.columns:
        male["Y浓度"] = pd.to_numeric(male["Y染色体浓度"], errors="coerce")
    else:
        male["Y浓度"] = np.nan

    r_week, p_week = safe_spearman(male["Y浓度"], male["孕周_周"])
    r_bmi, p_bmi = safe_spearman(male["Y浓度"], male["BMI"])

    slopes = []; residuals = []
    if "孕妇代码" in male.columns:
        for pid, g in male.groupby("孕妇代码"):
            sub = g[["孕周_周","Y浓度"]].dropna()
            if len(sub) >= 2:
                x = sub["孕周_周"].values
                y = sub["Y浓度"].values
                A = np.vstack([x, np.ones_like(x)]).T
                b1, b0 = np.linalg.lstsq(A, y, rcond=None)[0]
                slopes.append(b1)
                pred = b1*x+b0
                residuals.extend((y-pred).tolist())
    avg_slope = float(np.nan if len(slopes)==0 else np.mean(slopes))
    resid_sd = float(np.nan if len(residuals)==0 else np.std(residuals, ddof=1))

    male_log = male[["Y浓度","孕周_周","BMI"]].dropna().copy()
    male_log["y_bin"] = (male_log["Y浓度"] >= 0.04).astype(int)
    auc_mean = np.nan; auc_std = np.nan
    if male_log["y_bin"].nunique() == 2:
        male_log["week_BMI"] = male_log["孕周_周"] * male_log["BMI"]
        X = male_log[["孕周_周","BMI","week_BMI"]].values
        y = male_log["y_bin"].values
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in cv.split(X, y):
            pipe.fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y[te], proba))
        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs, ddof=1))

    earliest = []
    if "孕妇代码" in male.columns:
        for pid, g in male.groupby("孕妇代码"):
            sub = g[["孕周_周","Y浓度","BMI"]].dropna().sort_values("孕周_周")
            hit = sub[sub["Y浓度"]>=0.04]
            if len(hit)>0:
                ew = hit["孕周_周"].iloc[0]
                bmi_person = sub["BMI"].median()
                earliest.append((pid, ew, bmi_person))
    earliest_df = pd.DataFrame(earliest, columns=["孕妇代码","最早达标孕周","BMI"])
    earliest_df.to_csv(BASE/"male_earliest_table.csv", index=False, encoding="utf-8-sig")

    reg_text = ""
    if len(earliest_df)>=10:
        x = earliest_df["BMI"].values
        y = earliest_df["最早达标孕周"].values
        x1 = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
        yhat = x1 @ beta
        ss_res = np.sum((y-yhat)**2)
        ss_tot = np.sum((y-y.mean())**2)
        r2 = 1-ss_res/ss_tot if ss_tot>0 else np.nan
        n = len(x)
        mse = ss_res/(n-2)
        var_b = mse*np.linalg.inv(x1.T@x1)
        se_slope = math.sqrt(var_b[1,1])
        t = beta[1]/se_slope if se_slope>0 else np.nan
        pval = 2*(1-stats.t.cdf(abs(t), df=n-2)) if not np.isnan(t) else np.nan
        reg_text = f"最早达标孕周 ≈ {beta[0]:.2f} + {beta[1]:.3f} × BMI,  R² = {r2:.3f}, p={pval:.3g}"

    # BMI grouping
    group_md = ""
    if len(earliest_df)>=10:
        df_g = earliest_df.dropna(subset=["BMI","最早达标孕周"]).copy()
        k = min(5, max(2, len(df_g)//20))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_g["cluster"] = km.fit_predict(df_g[["BMI"]])
        centers = {i: km.cluster_centers_[i,0] for i in range(k)}
        order = sorted(range(k), key=lambda i: centers[i])
        rows = []
        for cid in order:
            sub = df_g[df_g["cluster"]==cid]
            if len(sub)==0: continue
            q70 = float(np.quantile(sub["最早达标孕周"], 0.70))
            q80 = float(np.quantile(sub["最早达标孕周"], 0.80))
            rows.append({
                "BMI_min": round(sub["BMI"].min(),2),
                "BMI_max": round(sub["BMI"].max(),2),
                "n": len(sub),
                "Q70": round(q70,2),
                "Q80": round(q80,2)
            })
        grp = pd.DataFrame(rows)
        grp.to_csv(BASE/"male_bmi_groups.csv", index=False, encoding="utf-8-sig")
        group_md = grp.to_markdown(index=False)

    # Female modeling
    female_proc = female.copy()
    lab = female_proc["染色体的非整倍体"].astype(str).str.strip()
    ylab = lab.map({"是":1,"否":0})
    female_proc["label"] = ylab
    feat_cols = [c for c in ["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值","X染色体的Z值","X染色体浓度",
                             "13号染色体的GC含量","18号染色体的GC含量","21号染色体的GC含量","被过滤掉读段数的比例","BMI"]
                 if c in female_proc.columns]
    auc_text = ""
    sr_text = ""
    if len(feat_cols)>0:
        data = female_proc[feat_cols+["label"]].dropna()
        if data["label"].nunique()==2 and len(data)>20:
            # simple rule
            if {"13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"}.issubset(set(data.columns)):
                zmax = data[["13号染色体的Z值","18号染色体的Z值","21号染色体的Z值"]].abs().max(axis=1)
                pred_high = (zmax >= 3).astype(int)
                ytrue = data["label"].values
                from sklearn.metrics import precision_recall_fscore_support
                prec, rec, f1, _ = precision_recall_fscore_support(ytrue, pred_high, average="binary", zero_division=0)
                sr_text = f"简单规则 max(|Z13|,|Z18|,|Z21|)≥3: precision={prec:.2f}, recall={rec:.2f}, F1={f1:.2f}"
            # logistic AUC
            X = data[feat_cols].values
            y = data["label"].values.astype(int)
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = []
            for tr, te in cv.split(X, y):
                pipe.fit(X[tr], y[tr])
                proba = pipe.predict_proba(X[te])[:,1]
                aucs.append(roc_auc_score(y[te], proba))
            auc_text = f"Logistic AUC (5CV) = {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}"

    # Build markdown
    md = []
    md.append("# NIPT 数据分析报告（复现脚本输出）\n")
    md.append("## 男胎\n")
    md.append(f"- Spearman：Y vs 孕周 ρ={r_week:.3f}, p={p_week:.3g}；Y vs BMI ρ={r_bmi:.3f}, p={p_bmi:.3g}")
    md.append(f"- 个体内回归：平均斜率≈{avg_slope:.4f}/周；残差SD≈{resid_sd:.4f}")
    if not math.isnan(auc_mean):
        md.append(f"- Logistic(是否达标Y≥4%) AUC(5CV)≈{auc_mean:.3f} ± {auc_std:.3f}")
    md.append(f"- 最早达标样本数：{len(earliest_df)}")
    if reg_text:
        md.append(f"- 线性回归：{reg_text}")
    if group_md:
        md.append("### BMI分组Q70/Q80推荐表\n")
        md.append(group_md)
    md.append("\n## 女胎\n")
    if sr_text: md.append(sr_text)
    if auc_text: md.append(auc_text)
    md_text = "\\n".join(md)

    md_path = BASE/"NIPT_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Simple PDF
    pdf_path = BASE/"NIPT_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    draw_text_block(c, md_text)
    c.save()

    print("Saved:", md_path)
    print("Saved:", pdf_path)
    print("Also saved: male_earliest_table.csv, male_bmi_groups.csv (if created)")

if __name__ == "__main__":
    import pandas as pd
    main()
