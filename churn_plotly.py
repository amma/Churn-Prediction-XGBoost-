import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import xgboost as xgb
import shap
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# For the A/B testing step:
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')


##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################

def calculate_psi(base, current, bins=10):
    """
    Population Stability Index (PSI) to measure data drift
    between 'base' and 'current' distributions.
    """
    eps = 1e-9
    combined = np.concatenate([base, current])
    min_val, max_val = np.min(combined), np.max(combined)
    if min_val == max_val:
        # Avoid zero-division or degenerate bins
        return 0.0

    breakpoints = np.linspace(min_val, max_val, bins + 1)

    base_perc = np.histogram(base, bins=breakpoints)[0] / (len(base) + eps)
    current_perc = np.histogram(current, bins=breakpoints)[0] / (len(current) + eps)

    psi = 0.0
    for i in range(len(base_perc)):
        if base_perc[i] < eps and current_perc[i] < eps:
            continue
        ratio = (base_perc[i] + eps) / (current_perc[i] + eps)
        psi += (base_perc[i] - current_perc[i]) * np.log(ratio)
    return psi


def shap_dependence_plot_plotly(shap_matrix, X, feature_name, color_name=None,
                                title_prefix="SHAP Dependence"):
    """
    Creates a Plotly scatter plot for a single feature's SHAP dependence.
    """
    feature_names = list(X.columns)
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not in X.columns!")
    feat_idx = feature_names.index(feature_name)

    x_values = X[feature_name].values
    y_values = shap_matrix[:, feat_idx]

    color_values = None
    color_title = None
    if color_name:
        if color_name not in feature_names:
            raise ValueError(f"Color feature '{color_name}' not in X.columns!")
        color_values = X[color_name].values
        color_title = color_name

    fig = px.scatter(
        x=x_values,
        y=y_values,
        color=color_values,
        labels={'x': feature_name, 'y': f"SHAP({feature_name})", 'color': color_title},
        title=f"{title_prefix} - Feature: {feature_name}"
    )
    fig.update_layout(
        xaxis_title=feature_name,
        yaxis_title=f"SHAP value for {feature_name}"
    )
    return fig


def shap_summary_plot_plotly(shap_matrix, X, max_points=2000):
    """
    Creates a beeswarm-like Plotly strip plot of SHAP values for all features.
    """
    import plotly.express as px

    n_samples, n_features = shap_matrix.shape
    feature_names = list(X.columns)

    # If the dataset is huge, sample rows for plotting
    if n_samples > max_points:
        sample_idx = np.random.choice(n_samples, size=max_points, replace=False)
    else:
        sample_idx = np.arange(n_samples)

    records = []
    for i in sample_idx:
        for j in range(n_features):
            feat_name = feature_names[j]
            shap_val = shap_matrix[i, j]
            feat_val = X.iloc[i, j]
            records.append({
                'feature': feat_name,
                'shap_value': shap_val,
                'feature_value': feat_val
            })

    long_df = pd.DataFrame(records)

    fig = px.strip(
        long_df,
        x="shap_value",
        y="feature",
        color="feature_value",
        orientation='h',
        hover_data=['feature_value'],
        title="SHAP Summary Plot (Plotly Beeswarm)"
    )
    fig.update_layout(
        xaxis_title="SHAP Value",
        yaxis_title="Feature"
    )
    return fig


def compute_calibration_curve(y_true, y_probs, n_bins=10):
    """
    Returns a DataFrame with columns [bin_mean_pred, bin_frac_pos] for calibration.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    bin_edges = np.linspace(0, 1, n_bins+1)
    bins = np.digitize(y_probs, bin_edges) - 1

    bin_mean_pred = []
    bin_frac_pos = []
    for b in range(n_bins):
        mask = (bins == b)
        if np.sum(mask) == 0:
            bin_mean_pred.append(np.nan)
            bin_frac_pos.append(np.nan)
        else:
            bin_mean_pred.append(y_probs[mask].mean())
            bin_frac_pos.append(y_true[mask].mean())

    return pd.DataFrame({
        "bin_mean_pred": bin_mean_pred,
        "bin_frac_pos": bin_frac_pos
    })


def segmented_performance(X_test, y_test, test_preds, group_col, group_bins=None, n_calib_bins=10):
    """
    Evaluate performance metrics (average score, positive rate, AUC, PRAUC)
    and generate a calibration curve for each group defined by 'group_col'.
    
    - If group_bins is not None and the column is numeric, we apply binning.
    - Otherwise, we treat group_col as a categorical variable.
    """
    fig_calib = go.Figure()
    color_palette = px.colors.qualitative.Bold
    color_idx = 0

    segment_metrics = []

    # If bins are provided and the column is numeric, bin the data
    if group_bins is not None and np.issubdtype(X_test[group_col].dtype, np.number):
        bin_labels = []
        for (start, end) in group_bins:
            bin_labels.append(f"{start}-{end}")

        for i, (start, end) in enumerate(group_bins):
            mask = (X_test[group_col] >= start) & (X_test[group_col] < end)
            y_seg = y_test[mask]
            p_seg = test_preds[mask]

            if len(y_seg) == 0:
                continue

            avg_score = np.mean(p_seg)
            positive_rate = np.mean(y_seg)
            try:
                auc_seg = roc_auc_score(y_seg, p_seg)
            except ValueError:
                auc_seg = None
            try:
                prauc_seg = average_precision_score(y_seg, p_seg)
            except ValueError:
                prauc_seg = None

            segment_metrics.append({
                "group": f"{start}-{end}",
                "num_samples": len(y_seg),
                "avg_score": avg_score,
                "positive_rate": positive_rate,
                "auc": auc_seg,
                "prauc": prauc_seg
            })

            # calibration curve
            calib_df = compute_calibration_curve(y_seg, p_seg, n_bins=n_calib_bins)
            fig_calib.add_trace(go.Scatter(
                x=calib_df["bin_mean_pred"],
                y=calib_df["bin_frac_pos"],
                mode='lines+markers',
                name=f"{group_col}={start}-{end}",
                line=dict(color=color_palette[color_idx % len(color_palette)])
            ))
            color_idx += 1

    else:
        # treat as categorical
        unique_groups = X_test[group_col].unique()
        for g in unique_groups:
            mask = (X_test[group_col] == g)
            y_seg = y_test[mask]
            p_seg = test_preds[mask]

            if len(y_seg) == 0:
                continue

            avg_score = np.mean(p_seg)
            positive_rate = np.mean(y_seg)
            try:
                auc_seg = roc_auc_score(y_seg, p_seg)
            except ValueError:
                auc_seg = None
            try:
                prauc_seg = average_precision_score(y_seg, p_seg)
            except ValueError:
                prauc_seg = None

            segment_metrics.append({
                "group": str(g),
                "num_samples": len(y_seg),
                "avg_score": avg_score,
                "positive_rate": positive_rate,
                "auc": auc_seg,
                "prauc": prauc_seg
            })

            # calibration curve
            calib_df = compute_calibration_curve(y_seg, p_seg, n_bins=n_calib_bins)
            fig_calib.add_trace(go.Scatter(
                x=calib_df["bin_mean_pred"],
                y=calib_df["bin_frac_pos"],
                mode='lines+markers',
                name=f"{group_col}={g}",
                line=dict(color=color_palette[color_idx % len(color_palette)])
            ))
            color_idx += 1

    # Add perfect calibration line
    fig_calib.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode='lines',
        name='Perfect Calib',
        line=dict(color='black', dash='dash')
    ))
    fig_calib.update_layout(
        title=f"Calibration Plot by {group_col}",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives"
    )

    segment_df = pd.DataFrame(segment_metrics)
    return segment_df, fig_calib


##############################################################################
# 2) MAIN PIPELINE
##############################################################################

def main():
    ############################################################################
    # A) Load the Telco Customer Churn dataset
    ############################################################################
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Data shape:", df.shape)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    ############################################################################
    # B) Plotly EDA: Histograms, Boxplots, Correlation Heatmap
    ############################################################################
    numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
    for col in numeric_cols:
        # 1) Histogram
        fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}",
                           marginal="rug", opacity=0.7)
        fig.update_layout(bargap=0.1)
        fig.write_html(f"plotly_hist_{col}.html")
        fig.write_image(f"plotly_hist_{col}.png")

        # 2) Boxplot
        fig = px.box(df, y=col, points="outliers", title=f"Boxplot of {col}")
        fig.write_html(f"plotly_box_{col}.html")
        fig.write_image(f"plotly_box_{col}.png")

    # Correlation heatmap
    corr_matrix = df[numeric_cols].corr(method='spearman')
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu',
                    title="Correlation Heatmap (Spearman)")
    fig.write_html("plotly_corr_heatmap.html")
    fig.write_image("plotly_corr_heatmap.png")

    ############################################################################
    # C) Prepare Data: One-Hot Encoding for Categorical
    ############################################################################
    cat_cols = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]
    X = pd.get_dummies(df.drop(columns=['Churn']), columns=cat_cols, drop_first=True)
    y = df['Churn']

    ############################################################################
    # D) Split Data into Train/Val/Test
    ############################################################################
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)

    ############################################################################
    # E) Bayesian Optimization with Optuna for XGBoost
    ############################################################################
    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True)
        }

        model_tmp = xgb.train(
            param,
            dtrain,
            evals=[(dval, "eval")],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        preds_val = model_tmp.predict(dval)
        prauc = average_precision_score(y_val, preds_val)
        return prauc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    best_params = study.best_params
    print("Best Params:", best_params)

    # Train final model
    best_params["verbosity"] = 0
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "aucpr"
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval,"eval")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    ############################################################################
    # F) Evaluate Performance (AUC & PRAUC)
    ############################################################################
    dtrain_eval = xgb.DMatrix(X_train)
    dval_eval   = xgb.DMatrix(X_val)
    dtest_eval  = xgb.DMatrix(X_test)

    train_preds = final_model.predict(dtrain_eval)
    val_preds   = final_model.predict(dval_eval)
    test_preds  = final_model.predict(dtest_eval)

    train_auc   = roc_auc_score(y_train, train_preds)
    train_prauc = average_precision_score(y_train, train_preds)
    val_auc     = roc_auc_score(y_val, val_preds)
    val_prauc   = average_precision_score(y_val, val_preds)
    test_auc    = roc_auc_score(y_test, test_preds)
    test_prauc  = average_precision_score(y_test, test_preds)

    print(f"\nPerformance Metrics:")
    print(f"  Train AUC:   {train_auc:.4f}, Train PRAUC:   {train_prauc:.4f}")
    print(f"  Val AUC:     {val_auc:.4f},   Val PRAUC:     {val_prauc:.4f}")
    print(f"  Test AUC:    {test_auc:.4f},  Test PRAUC:    {test_prauc:.4f}")

    ############################################################################
    # F.1) PLOT ROC CURVES (Train, Val, Test)
    ############################################################################
    train_fpr, train_tpr, _ = roc_curve(y_train, train_preds)
    val_fpr, val_tpr, _     = roc_curve(y_val, val_preds)
    test_fpr, test_tpr, _   = roc_curve(y_test, test_preds)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=train_fpr, y=train_tpr,
        mode='lines', name=f"Train (AUC={train_auc:.3f})"
    ))
    fig_roc.add_trace(go.Scatter(
        x=val_fpr, y=val_tpr,
        mode='lines', name=f"Val (AUC={val_auc:.3f})"
    ))
    fig_roc.add_trace(go.Scatter(
        x=test_fpr, y=test_tpr,
        mode='lines', name=f"Test (AUC={test_auc:.3f})"
    ))
    # Diagonal line
    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                      line=dict(color='black', dash='dash'))
    fig_roc.update_layout(
        title="ROC Curves (Train/Val/Test)",
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    fig_roc.write_html("plotly_roc_curves.html")
    fig_roc.write_image("plotly_roc_curves.png")

    ############################################################################
    # F.2) PLOT PRECISION-RECALL CURVES (Train, Val, Test)
    ############################################################################
    train_precision, train_recall, _ = precision_recall_curve(y_train, train_preds)
    val_precision, val_recall, _     = precision_recall_curve(y_val, val_preds)
    test_precision, test_recall, _   = precision_recall_curve(y_test, test_preds)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=train_recall, y=train_precision,
        mode='lines', name=f"Train (PRAUC={train_prauc:.3f})"
    ))
    fig_pr.add_trace(go.Scatter(
        x=val_recall, y=val_precision,
        mode='lines', name=f"Val (PRAUC={val_prauc:.3f})"
    ))
    fig_pr.add_trace(go.Scatter(
        x=test_recall, y=test_precision,
        mode='lines', name=f"Test (PRAUC={test_prauc:.3f})"
    ))
    fig_pr.update_layout(
        title="Precision-Recall Curves (Train/Val/Test)",
        xaxis_title='Recall',
        yaxis_title='Precision'
    )
    fig_pr.write_html("plotly_pr_curves.html")
    fig_pr.write_image("plotly_pr_curves.png")

    ############################################################################
    # G) Score Distribution (Plotly) - Train vs. Test
    ############################################################################
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train_preds,
        name='Train',
        opacity=0.5,
        histnorm='probability density',
        nbinsx=50
    ))
    fig.add_trace(go.Histogram(
        x=test_preds,
        name='Test',
        opacity=0.5,
        histnorm='probability density',
        nbinsx=50
    ))
    fig.update_layout(
        title="Score Distribution - Train vs Test",
        barmode='overlay',
        xaxis_title="Predicted Probability (Churn)",
        yaxis_title="Density"
    )
    fig.write_html("plotly_score_distribution.html")
    fig.write_image("plotly_score_distribution.png")

    ############################################################################
    # H) PSI Calculation for Model Scores
    ############################################################################
    baseline_preds = np.concatenate([train_preds, val_preds])
    psi_score = calculate_psi(baseline_preds, test_preds, bins=10)
    print("\nPSI for model score:", round(psi_score,4))

    ############################################################################
    # H.1) DRIFT ASSESSMENT: Feature PSI (Test vs. Baseline)
    ############################################################################
    numeric_features = [col for col in X.columns if np.issubdtype(X[col].dtype, np.number)]
    baseline_df = pd.concat([X_train, X_val], axis=0)

    feature_psi_results = []
    for f in numeric_features:
        base_data = baseline_df[f].values
        curr_data = X_test[f].values
        psi_val = calculate_psi(base_data, curr_data, bins=10)
        feature_psi_results.append({
            "feature": f,
            "psi": psi_val
        })

    df_feature_psi = pd.DataFrame(feature_psi_results).sort_values(by="psi", ascending=False)
    print("\nFeature PSI (Test vs Baseline):")
    print(df_feature_psi.head(len(df_feature_psi)))  # Print all for clarity

    # >>>>>>>>>>>>>> NEW PLOT FOR FEATURE-LEVEL PSI <<<<<<<<<<<<<<
    fig_psi_bar = px.bar(
        df_feature_psi.sort_values("psi", ascending=False),
        x="feature",
        y="psi",
        color="psi",
        title="PSI for Numeric Features (Train+Val Baseline vs. Test)"
    )
    fig_psi_bar.update_layout(xaxis_title="Feature", yaxis_title="PSI")
    fig_psi_bar.write_html("plotly_feature_psi.html")
    fig_psi_bar.write_image("plotly_feature_psi.png")


    ############################################################################
    # I) SHAP Explainability
    ############################################################################
    explainer = shap.TreeExplainer(final_model)
    shap_values_train_exp = explainer(X_train)  # shap.Explanation
    shap_matrix_train = shap_values_train_exp.values  # (n_samples, n_features)

    # 1) SHAP Summary (beeswarm) with Plotly
    fig_summary = shap_summary_plot_plotly(shap_matrix_train, X_train, max_points=3000)
    fig_summary.write_html("plotly_shap_summary.html")
    fig_summary.write_image("plotly_shap_summary.png")

    # 2) SHAP Dependence for 'tenure', color by 'MonthlyCharges'
    if "tenure" in X_train.columns and "MonthlyCharges" in X_train.columns:
        fig_dep_tenure = shap_dependence_plot_plotly(
            shap_matrix_train,
            X_train,
            feature_name="tenure",
            color_name="MonthlyCharges",
            title_prefix="SHAP Dependence"
        )
        fig_dep_tenure.write_html("plotly_shap_dependence_tenure.html")
        fig_dep_tenure.write_image("plotly_shap_dependence_tenure.png")

    ############################################################################
    # J) K-Means on SHAP Values of Predicted Churners
    ############################################################################
    churn_threshold = 0.3
    churners_idx = np.where(test_preds >= churn_threshold)[0]
    X_test_churners = X_test.iloc[churners_idx].copy()

    shap_values_test_exp = explainer(X_test)
    shap_matrix_test = shap_values_test_exp.values
    test_churners_shap = shap_matrix_test[churners_idx, :]

    if len(test_churners_shap) > 5:
        km = KMeans(n_clusters=3, random_state=42)
        cluster_labels = km.fit_predict(test_churners_shap)
        sil = silhouette_score(test_churners_shap, cluster_labels)
        print(f"\nK-Means on SHAP (churners): Silhouette = {sil:.4f}")

        X_test_churners['cluster_id'] = cluster_labels
        print("Churners cluster distribution:\n", X_test_churners['cluster_id'].value_counts())

        shap_churners_df = pd.DataFrame(test_churners_shap, columns=X_test.columns)
        cluster_means = []
        cluster_ids = sorted(np.unique(cluster_labels))
        for cid in cluster_ids:
            row_mean = shap_churners_df[cluster_labels == cid].mean(axis=0)
            cluster_means.append(row_mean)
        cluster_mean_df = pd.DataFrame(cluster_means, index=[f"Cluster_{c}" for c in cluster_ids])
        print("\nMean SHAP values per cluster:")
        print(cluster_mean_df)
    else:
        print("\nNot enough predicted churners to perform K-Means.")

    ############################################################################
    # K) SEGMENTED PERFORMANCE: Example 1 - Tenure Range
    ############################################################################
    tenure_bins = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 999)
    ]
    seg_tenure_df, fig_calib_tenure = segmented_performance(
        X_test, y_test, test_preds, group_col="tenure", group_bins=tenure_bins, n_calib_bins=10
    )
    print("\nSegmented Performance by Tenure:")
    print(seg_tenure_df)
    fig_calib_tenure.write_html("calibration_by_tenure.html")
    fig_calib_tenure.write_image("calibration_by_tenure.png")

    ############################################################################
    # K.1) SEGMENTED PERFORMANCE: Example 2 - Gender
    ############################################################################
    # Re-attach original 'gender' to X_test for grouping
    df_test = df.loc[X_test.index].copy()  # original DF rows for test set
    group_col = 'gender'

    X_test_fair = X_test.copy()
    X_test_fair[group_col] = df_test[group_col].values  # attach gender back

    seg_gender_df, fig_calib_gender = segmented_performance(
        X_test_fair, y_test, test_preds, group_col="gender", group_bins=None, n_calib_bins=10
    )
    print("\nSegmented Performance by Gender:")
    print(seg_gender_df)
    fig_calib_gender.write_html("calibration_by_gender.html")
    fig_calib_gender.write_image("calibration_by_gender.png")

    # >>>>>>>>>>>>>> NEW BAR PLOT FOR GENDER (Avg Score vs Positive Rate) <<<<<<<<<<<<<<
    fig_gender_bar = go.Figure()
    fig_gender_bar.add_trace(go.Bar(
        x=seg_gender_df['group'],
        y=seg_gender_df['avg_score'],
        name='Avg Predicted Score'
    ))
    fig_gender_bar.add_trace(go.Bar(
        x=seg_gender_df['group'],
        y=seg_gender_df['positive_rate'],
        name='Actual Churn Rate'
    ))
    fig_gender_bar.update_layout(
        barmode='group',
        title="Average Score vs. Actual Churn Rate by Gender",
        xaxis_title='Gender',
        yaxis_title='Value'
    )
    fig_gender_bar.write_html("plotly_gender_bar.html")
    fig_gender_bar.write_image("plotly_gender_bar.png")

    ############################################################################
    # L) A/B TESTING (PILOT PHASE) EXAMPLE
    ############################################################################
    """
    In a real-world scenario, you'd deploy a new intervention to a random sample
    of your customers (the 'pilot group') for 1 month, then measure their actual
    churn outcomes vs. a control group. Below is a simplified example of how
    you might set up and analyze that with your test set.

    We'll:
      1) Randomly select ~10% of the test set as the "pilot" group.
      2) Assume we have 'actual' outcomes after the pilot.
         For illustration, we'll artificially adjust the pilot's churn rate.
      3) Compare churn rates in pilot vs. control using a difference in proportions.
    """

    np.random.seed(42)
    test_indices = X_test.index
    pilot_size = int(0.10 * len(test_indices))  # 10% for the pilot
    pilot_group_idx = np.random.choice(test_indices, size=pilot_size, replace=False)
    control_group_idx = np.setdiff1d(test_indices, pilot_group_idx)

    # Let's copy y_test for our "post-pilot" scenario
    pilot_outcomes = y_test.copy()

    # For illustration, artificially reduce churn for pilot group by 20%.
    was_churn = pilot_outcomes.loc[pilot_group_idx] == 1
    pilot_churners = pilot_outcomes.loc[pilot_group_idx][was_churn]

    # We'll randomly select 20% of these churners to become "retained" (churn=0).
    n_churners_to_flip = int(0.20 * len(pilot_churners))
    if n_churners_to_flip > 0:
        flip_idx = np.random.choice(pilot_churners.index, size=n_churners_to_flip, replace=False)
        pilot_outcomes.loc[flip_idx] = 0

    # Now let's measure the churn rates in pilot vs. control
    pilot_churn_rate = pilot_outcomes.loc[pilot_group_idx].mean()
    control_churn_rate = pilot_outcomes.loc[control_group_idx].mean()

    print("\nA/B TEST (Pilot) RESULTS:")
    print(f"  Pilot group size:   {len(pilot_group_idx)}")
    print(f"  Control group size: {len(control_group_idx)}")
    print(f"  Pilot churn rate:   {pilot_churn_rate:.3f}")
    print(f"  Control churn rate: {control_churn_rate:.3f}")

    # Significance test for difference in proportions (churn vs. not churn)
    n1 = len(pilot_group_idx)
    n2 = len(control_group_idx)
    x1 = pilot_outcomes.loc[pilot_group_idx].sum()  # number of churn=1 in pilot
    x2 = pilot_outcomes.loc[control_group_idx].sum()  # number of churn=1 in control

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / se
    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    print(f"  Difference in churn rate (pilot - control): {p1 - p2:.3f}")
    print(f"  Z-score: {z_score:.3f}, p-value: {p_value:.6f}")
    print("\n(Interpretation: If p-value is small, the difference is statistically significant.)")

    print("\nAll done! Full pipeline completed with drift, fairness/segmented analysis, "
          "and a simple A/B test example.")


if __name__ == "__main__":
    main()
