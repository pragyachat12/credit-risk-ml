"""
Credit Risk Model — Logistic Regression & Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random, time, warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix,
    precision_recall_curve, average_precision_score,
    classification_report,
)


DATA_PATH   = "data_src/credit_risk_dataset.csv"
PLOT_PATH   = "data_src/credit_risk_dashboard.png"
EXPORT_PATH = "data_src/tableau_export.csv"


#using code from lab
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]
df = df.reset_index(drop=True)
print(f"After cleaning: {df.shape}")

# FIX 1: Build data_fets BEFORE label-encoding the categorical columns.
# The string comparisons (e.g. df["loan_grade"] == "A") must happen while
# the columns still hold their original string values. After LabelEncoder
# runs below, those columns become integers and the comparisons return all False.
data_fets = np.stack([
    (df["person_home_ownership"] == "RENT"            ).astype(float),
    (df["person_home_ownership"] == "OWN"             ).astype(float),
    (df["person_home_ownership"] == "MORTGAGE"        ).astype(float),
    (df["loan_intent"] == "EDUCATION"                 ).astype(float),
    (df["loan_intent"] == "MEDICAL"                   ).astype(float),
    (df["loan_intent"] == "VENTURE"                   ).astype(float),
    (df["loan_intent"] == "PERSONAL"                  ).astype(float),
    (df["loan_intent"] == "DEBTCONSOLIDATION"         ).astype(float),
    (df["loan_grade"] == "A"                          ).astype(float),
    (df["loan_grade"] == "B"                          ).astype(float),
    (df["loan_grade"] == "C"                          ).astype(float),
    (df["loan_grade"] == "D"                          ).astype(float),
    (df["cb_person_default_on_file"] == "Y"           ).astype(float),
    df["person_age"].fillna(df["person_age"].median()).values,
    df["person_income"].values,
    df["loan_amnt"].values,
    df["loan_int_rate"].fillna(df["loan_int_rate"].median()).values,
    df["loan_percent_income"].values,
    df["person_emp_length"].fillna(df["person_emp_length"].median()).values,
    df["cb_person_cred_hist_length"].values,
], axis=1)

DT_FEATURE_NAMES = [
    "home_rent", "home_own", "home_mortgage",
    "intent_education", "intent_medical", "intent_venture",
    "intent_personal", "intent_debtcons",
    "grade_A", "grade_B", "grade_C", "grade_D",
    "prior_default",
    "age", "income", "loan_amnt", "int_rate",
    "pct_income", "emp_length", "cred_hist_len",
]

for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

FEATURE_COLS = [c for c in df.columns if c != "loan_status"]
X_raw = df[FEATURE_COLS]
y     = df["loan_status"].values

imputer = SimpleImputer(strategy="median")
X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURE_COLS)

#split into 0.8 for train and val and 0.2 for test
X_tv,  X_test,  y_tv,  y_test  = train_test_split(
    X_imp.values, y, test_size=0.20, random_state=1, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tv, y_tv, test_size=0.15, random_state=1, stratify=y_tv)

print(f"Train: {X_train.shape[0]}  Val: {X_valid.shape[0]}  Test: {X_test.shape[0]}")

# FIX 2: Split data_fets into train/val/test using the same random_state=1
# so indices align with X_train/X_valid/X_test.
# X_train_dt / X_valid_dt / X_test_dt were never defined in the original code,
# causing a NameError when the Decision Tree section ran.
X_tv_dt,  X_test_dt,  _, _  = train_test_split(
    data_fets, y, test_size=0.20, random_state=1, stratify=y)
X_train_dt, X_valid_dt, _, _ = train_test_split(
    X_tv_dt, y_tv, test_size=0.15, random_state=1, stratify=y_tv)

#normalize features
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1

X_train_norm = (X_train - mean) / std
X_valid_norm = (X_valid - mean) / std
X_test_norm  = (X_test  - mean) / std


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train_b = add_bias(X_train_norm)
X_valid_b = add_bias(X_valid_norm)
X_test_b  = add_bias(X_test_norm)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def pred(w, X):
    return sigmoid(X @ w)

def loss(w, X, t):
    """
    Numerically stable LCE loss (lab04):
        L(z, t) = t*log(1+e^{-z}) + (1-t)*log(1+e^z)
    """
    z = X @ w
    return np.mean(t * np.logaddexp(0, -z) + (1 - t) * np.logaddexp(0, z))

def grad(w, X, t):
    return (1 / len(t)) * (X.T @ (pred(w, X) - t))

def accuracy(w, X, t, thres=0.5):
    return np.mean((pred(w, X) >= thres).astype(int) == t)

#we will compare two training methods: gd and sgd.

def solve_via_gradient_descent(alpha=0.05, niter=1000,
                                X_tr=None, t_tr=None,
                                X_va=None, t_va=None, plot=False):
    X_tr = X_train_b if X_tr is None else X_tr
    t_tr = y_train   if t_tr is None else t_tr
    X_va = X_valid_b if X_va is None else X_va
    t_va = y_valid   if t_va is None else t_va

    w = np.zeros(X_tr.shape[1])
    tr_loss, va_loss, tr_acc, va_acc = [], [], [], []

    for _ in range(niter):
        w -= alpha * grad(w, X_tr, t_tr)
        if plot:
            tr_loss.append(loss(w, X_tr, t_tr))
            va_loss.append(loss(w, X_va, t_va))
            tr_acc.append(accuracy(w, X_tr, t_tr))
            va_acc.append(accuracy(w, X_va, t_va))
# plotting the training curve to compare convergence and performance.
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(tr_loss, label="Train"); axes[0].plot(va_loss, label="Val")
        axes[0].set(title="Full-Batch GD Loss", xlabel="Iter", ylabel="Loss"); axes[0].legend()
        axes[1].plot(tr_acc, label="Train"); axes[1].plot(va_acc, label="Val")
        axes[1].set(title="Full-Batch GD Accuracy", xlabel="Iter", ylabel="Accuracy"); axes[1].legend()
        plt.tight_layout()
        plt.savefig("data_src/gd_training_curve.png", dpi=120)
        plt.close()
        print(f"  GD  | Train loss {tr_loss[-1]:.4f} | Val loss {va_loss[-1]:.4f} "
              f"| Train acc {tr_acc[-1]:.4f} | Val acc {va_acc[-1]:.4f}")
    return w

def solve_via_sgd(alpha=0.05, n_epochs=40, batch_size=100,
                  X_tr=None, t_tr=None,
                  X_va=None, t_va=None, plot=False):
    X_tr = X_train_b if X_tr is None else X_tr
    t_tr = y_train   if t_tr is None else t_tr
    X_va = X_valid_b if X_va is None else X_va
    t_va = y_valid   if t_va is None else t_va

    w = np.zeros(X_tr.shape[1])
    N = X_tr.shape[0]
    indices = list(range(N))
    tr_loss, va_loss, tr_acc, va_acc = [], [], [], []

    for _ in range(n_epochs):
        random.shuffle(indices)
        for i in range(0, N, batch_size):
            if (i + batch_size) >= N:
                continue
            batch_idx = indices[i: i + batch_size]
            w -= alpha * grad(w, X_tr[batch_idx], t_tr[batch_idx])
            if plot:
                tr_loss.append(loss(w, X_tr, t_tr))
                va_loss.append(loss(w, X_va, t_va))
                tr_acc.append(accuracy(w, X_tr, t_tr))
                va_acc.append(accuracy(w, X_va, t_va))
# plotting the training curve to compare convergence and performance.
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(tr_loss, label="Train"); axes[0].plot(va_loss, label="Val")
        axes[0].set(title="SGD Loss", xlabel="Iter", ylabel="Loss"); axes[0].legend()
        axes[1].plot(tr_acc, label="Train"); axes[1].plot(va_acc, label="Val")
        axes[1].set(title="SGD Accuracy", xlabel="Iter", ylabel="Accuracy"); axes[1].legend()
        plt.tight_layout()
        plt.savefig("data_src/sgd_training_curve.png", dpi=120)
        plt.close()
        print(f"  SGD | Train loss {tr_loss[-1]:.4f} | Val loss {va_loss[-1]:.4f} "
              f"| Train acc {tr_acc[-1]:.4f} | Val acc {va_acc[-1]:.4f}")
    return w

 
print("\n── Manual Logistic Regression ──")
t0    = time.time(); w_sgd = solve_via_sgd(alpha=0.05, n_epochs=40, batch_size=100, plot=True); t_sgd = time.time()-t0
num_iter = int(40 * X_train_b.shape[0] / 100)
t0    = time.time(); w_gd  = solve_via_gradient_descent(alpha=0.05, niter=num_iter, plot=True);  t_gd  = time.time()-t0
 # compare the training time and the test AUC of the two methods
print(f"SGD time:        {t_sgd:.3f}s")
print(f"Full-Batch time: {t_gd:.3f}s")

print(f"Manual LR (SGD) test AUC: {roc_auc_score(y_test, pred(w_sgd, X_test_b)):.4f}")
print(f"Manual LR (GD)  test AUC: {roc_auc_score(y_test, pred(w_gd,  X_test_b)):.4f}")

# then we implement sklearn's LogisticRegression and compare its performance to our manual implementation. 
# We also look at the coefficients to understand which features are most influential in predicting loan defaults.

print("\n── sklearn LogisticRegression ──")
lr_sk = SklearnLR(fit_intercept=False, max_iter=1000, random_state=42, class_weight="balanced")
lr_sk.fit(X_train_b, y_train)
lr_sk_probs = lr_sk.predict_proba(X_test_b)[:, 1]
lr_preds    = lr_sk.predict(X_test_b)
lr_sk_auc   = roc_auc_score(y_test, lr_sk_probs)
lr_sk_ap    = average_precision_score(y_test, lr_sk_probs)

print(f"Train acc: {lr_sk.score(X_train_b, y_train):.4f}  "
      f"Val acc: {lr_sk.score(X_valid_b, y_valid):.4f}  "
      f"Test AUC: {lr_sk_auc:.4f}  AP: {lr_sk_ap:.4f}")
coef_df = pd.DataFrame({
    "feature": ["bias"] + FEATURE_COLS,
    "coefficient": lr_sk.coef_[0]
}).sort_values(by="coefficient", key=abs, ascending=False)
print(classification_report(y_test, lr_preds, target_names=["No Default", "Default"]))


#implementing random forest and comparing to logistic regression.
print("\n── Random Forest ──")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                             random_state=42, class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)
rf_auc   = roc_auc_score(y_test, rf_probs)
rf_ap    = average_precision_score(y_test, rf_probs)
print(f"Test AUC: {rf_auc:.4f}  AP: {rf_ap:.4f}")
print(classification_report(y_test, rf_preds, target_names=["No Default", "Default"]))

feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)


# decsion tree implementation
print("\n" + "═" * 60)
print("SECTION 5 — Decision Tree (lab02 approach)")
print("═" * 60)

# Underfit / overfit diagnosis
print("\n── Underfit / Overfit Diagnosis ──")
for label, kwargs in [
    ("Underfit  max_depth=1",      {"max_depth": 1}),
    ("Overfit   max_depth=None",   {"max_depth": None}),
    ("Underfit  min_split>N",      {"min_samples_split": X_train_dt.shape[0] + 1}),
    ("Overfit   min_split=2",      {"min_samples_split": 2}),
]:
    m = DecisionTreeClassifier(criterion="entropy", random_state=42, **kwargs)
    m.fit(X_train_dt, y_train)
    print(f"  {label:35s} | Train: {m.score(X_train_dt, y_train):.4f}  "
          f"Val: {m.score(X_valid_dt, y_valid):.4f}")


def build_all_models(max_depths, min_samples_splits, criterion,
                     X_tr=None, t_tr=None, X_va=None, t_va=None):
    """
    Build one DecisionTreeClassifier per (max_depth, min_samples_split)
    combination. Returns a dict: out[(d, s)] = {'train': ..., 'val': ...}
    Mirrors lab02 build_all_models exactly.
    """
    X_tr = X_train_dt if X_tr is None else X_tr
    t_tr = y_train    if t_tr is None else t_tr
    X_va = X_valid_dt if X_va is None else X_va
    t_va = y_valid    if t_va is None else t_va

    out = {}
    for d in max_depths:
        for s in min_samples_splits:
            model = DecisionTreeClassifier(
                criterion=criterion, max_depth=d,
                min_samples_split=s, random_state=42,
            )
            model.fit(X_tr, t_tr)
            out[(d, s)] = {
                "val":   model.score(X_va, t_va),
                "train": model.score(X_tr, t_tr),
            }
    return out


# to find the best hyperparameters for the Decision Tree, 
# we perform a grid search over a range of max_depth and min_samples_split values.
#  We evaluate each model on the validation set and select the one with the highest validation accuracy. 
# Finally, we fit the best tree on the training data and evaluate its performance on the test set.


print("\n── Grid Search ──")
criterions         = ["entropy", "gini"]
max_depths         = [3, 5, 10, 15, 20, 30, 50]
min_samples_splits = [2, 8, 32, 64, 128, 256, 512]
 
print("\n── Grid Search ──")
best_overall = {"val": -1}

for criterion in criterions:
    res = build_all_models(max_depths, min_samples_splits, criterion)
    best_val, best_params = -1, None
    for (d, s), scores in res.items():
        if scores["val"] > best_val:
            best_val, best_params = scores["val"], (d, s)
    print(f"  {criterion}: best (max_depth={best_params[0]}, "
          f"min_samples_split={best_params[1]}) → val_acc={best_val:.4f}")
    if best_val > best_overall["val"]:
        best_overall = {"val": best_val, "params": best_params, "criterion": criterion}

print(f"\n★ Best DT: criterion={best_overall['criterion']}, "
      f"max_depth={best_overall['params'][0]}, "
      f"min_samples_split={best_overall['params'][1]}, "
      f"val_acc={best_overall['val']:.4f}")

# Fit best tree
best_tree = DecisionTreeClassifier(
    criterion         = best_overall["criterion"],
    max_depth         = best_overall["params"][0],
    min_samples_split = best_overall["params"][1],
    random_state      = 42,
)
best_tree.fit(X_train_dt, y_train)
dt_probs = best_tree.predict_proba(X_test_dt)[:, 1]
dt_preds = best_tree.predict(X_test_dt)
dt_auc   = roc_auc_score(y_test, dt_probs)
dt_ap    = average_precision_score(y_test, dt_probs)

print(f"\nDT Test AUC: {dt_auc:.4f}  AP: {dt_ap:.4f}  "
      f"Acc: {best_tree.score(X_test_dt, y_test):.4f}")
print(classification_report(y_test, dt_preds, target_names=["No Default", "Default"]))

print("\nTop 10 DT Feature Importances:")
dt_feat_imp = pd.Series(best_tree.feature_importances_,
                         index=DT_FEATURE_NAMES).sort_values(ascending=False)
print(dt_feat_imp.head(10).round(4))

print("\nDecision rules (depth ≤ 3):")
print(export_text(best_tree, feature_names=DT_FEATURE_NAMES, max_depth=3))


from sklearn.tree import export_graphviz, plot_tree
# import graphviz  # Commented out - requires system graphviz installation
import matplotlib.patches as mpatches

# matplotlib visualization of the decision tree structure
fig, ax = plt.subplots(figsize=(28, 10))
plot_tree(
    best_tree,
    max_depth     = 3,
    feature_names = DT_FEATURE_NAMES,
    class_names   = ["No Default", "Default"],
    filled        = True,
    rounded       = True,
    impurity      = True,
    fontsize      = 8,
    ax            = ax,
)
ax.set_title("Credit Risk Decision Tree (depth ≤ 3)", fontsize=13, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#6baed6", label="Majority: No Default"),
    mpatches.Patch(color="#fd8d3c", label="Majority: Default"),
], loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("data_src/tree_matplotlib.png", dpi=150, bbox_inches="tight")
plt.close()

# feature importance bar chart and depth analysis line plot to visualize the importance of features in the decision tree and 
# how the model's accuracy changes with different max_depth values.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Decision Tree — Feature Importances & Depth Analysis", fontsize=13, fontweight="bold")

fi = pd.Series(best_tree.feature_importances_, index=DT_FEATURE_NAMES).sort_values(ascending=True).tail(10)
axes[0].barh(fi.index, fi.values,
             color=["#e74c3c" if v > fi.median() else "#3498db" for v in fi],
             edgecolor="white")
axes[0].set(title="Top 10 Feature Importances", xlabel="Importance")
axes[0].axvline(fi.median(), color="gray", linestyle="--", lw=1, label="median")
axes[0].legend(fontsize=9); axes[0].grid(axis="x", alpha=0.3)

depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None]
tr_accs, va_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(criterion="entropy", max_depth=d, min_samples_split=8, random_state=42)
    m.fit(X_train_dt, y_train)
    tr_accs.append(m.score(X_train_dt, y_train))
    va_accs.append(m.score(X_valid_dt, y_valid))

xlabels = [str(d) if d else "None" for d in depths]
axes[1].plot(xlabels, tr_accs, "o-", color="#e74c3c", lw=2, label="Train")
axes[1].plot(xlabels, va_accs, "s--", color="#3498db", lw=2, label="Validation")
axes[1].axvline(xlabels[depths.index(10)], color="green", linestyle=":", lw=2, label="Best (depth=10)")
axes[1].set(title="Accuracy vs max_depth  (Underfit ← → Overfit)", xlabel="max_depth", ylabel="Accuracy")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("data_src/tree_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# separate into Prime (A-B) vs Subprime (D-G) segments
test_grade = X_test[:, FEATURE_COLS.index("loan_grade")]
prime      = test_grade <= 1      # 0=A, 1=B
subprime   = test_grade >= 3      # 3=D … 6=G

PALETTE = {"LR": "#4C72B0", "RF": "#DD8452", "DT": "#55A868"}
fig = plt.figure(figsize=(22, 20))
fig.suptitle("Credit Risk Model — Full Dashboard (LR · RF · DT)",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38)

# (a) ROC — all three models
ax1 = fig.add_subplot(gs[0, 0:2])
for probs, label, col in [
    (lr_sk_probs, f"LR  AUC={lr_sk_auc:.3f}", PALETTE["LR"]),
    (rf_probs,    f"RF  AUC={rf_auc:.3f}",    PALETTE["RF"]),
    (dt_probs,    f"DT  AUC={dt_auc:.3f}",    PALETTE["DT"]),
]:
    fpr_, tpr_, _ = roc_curve(y_test, probs)
    ax1.plot(fpr_, tpr_, lw=2, label=label)
ax1.plot([0,1],[0,1],"k--",lw=1)
ax1.set(title="ROC Curves — All Models", xlabel="FPR", ylabel="TPR")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# (b) Precision-Recall — all three models
ax2 = fig.add_subplot(gs[0, 2:4])
for probs, label, col in [
    (lr_sk_probs, f"LR  AP={lr_sk_ap:.3f}", PALETTE["LR"]),
    (rf_probs,    f"RF  AP={rf_ap:.3f}",    PALETTE["RF"]),
    (dt_probs,    f"DT  AP={dt_ap:.3f}",    PALETTE["DT"]),
]:
    p_, r_, _ = precision_recall_curve(y_test, probs)
    ax2.plot(r_, p_, lw=2, label=label, color=col)
ax2.set(title="Precision-Recall — All Models", xlabel="Recall", ylabel="Precision")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# (c) Feature importance — RF
ax3 = fig.add_subplot(gs[1, 0:2])
feat_imp.head(10).sort_values().plot(kind="barh", ax=ax3, color=PALETTE["RF"])
ax3.set(title="Feature Importance — Random Forest", xlabel="Importance")
ax3.grid(axis="x", alpha=0.3)

# (d) Feature importance — LR (coefficients)
ax4 = fig.add_subplot(gs[1, 2:4])
coef_plot = coef_df[coef_df["feature"] != "bias"].head(10).sort_values("coefficient", key=abs)
ax4.barh(coef_plot["feature"], coef_plot["coefficient"], color=PALETTE["LR"])
ax4.set(title="Feature Coefficients — Logistic Regression", xlabel="Coefficient")
ax4.grid(axis="x", alpha=0.3)

# (e) Confusion matrix — LR
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt="d",
            cmap="Blues", ax=ax5,
            xticklabels=["No Default","Default"],
            yticklabels=["No Default","Default"])
ax5.set(title=f"CM — Logistic Regression\nAUC {lr_sk_auc:.3f}",
        xlabel="Predicted", ylabel="Actual")

# (f) Confusion matrix — RF
ax6 = fig.add_subplot(gs[2, 1])
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt="d",
            cmap="Oranges", ax=ax6,
            xticklabels=["No Default","Default"],
            yticklabels=["No Default","Default"])
ax6.set(title=f"CM — Random Forest\nAUC {rf_auc:.3f}",
        xlabel="Predicted", ylabel="Actual")

# (g) Confusion matrix — DT
ax7 = fig.add_subplot(gs[2, 2])
sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt="d",
            cmap="Greens", ax=ax7,
            xticklabels=["No Default","Default"],
            yticklabels=["No Default","Default"])
ax7.set(title=f"CM — Decision Tree\nAUC {dt_auc:.3f}",
        xlabel="Predicted", ylabel="Actual")

# (h) Model summary bar chart
ax8 = fig.add_subplot(gs[2, 3])
models  = ["LR", "RF", "DT"]
aucs    = [lr_sk_auc, rf_auc, dt_auc]
aps     = [lr_sk_ap,  rf_ap,  dt_ap]
x       = np.arange(len(models))
w       = 0.35
ax8.bar(x - w/2, aucs, w, label="AUC",  color=[PALETTE[m] for m in models])
ax8.bar(x + w/2, aps,  w, label="AP",   color=[PALETTE[m] for m in models], alpha=0.5)
ax8.set_xticks(x); ax8.set_xticklabels(models)
ax8.set(title="Model Comparison", ylabel="Score", ylim=[0.5, 1.0])
ax8.legend(fontsize=9); ax8.grid(axis="y", alpha=0.3)

# (i) LR probability distribution by subgroup
ax9 = fig.add_subplot(gs[3, 0:2])
ax9.hist(lr_sk_probs[prime],    bins=30, alpha=0.6, color="#3498db", label="Prime (A–B)")
ax9.hist(lr_sk_probs[subprime], bins=30, alpha=0.6, color="#e74c3c", label="Subprime (D–G)")
ax9.set(title="LR Probability Distribution by Grade Group",
        xlabel="Default Probability", ylabel="Count")
ax9.legend(fontsize=9); ax9.grid(alpha=0.3)

# (j) Subgroup ROC — LR
ax10 = fig.add_subplot(gs[3, 2:4])
for mask, label, col in [
    (prime,    "Prime A–B",    "#3498db"),
    (subprime, "Subprime D–G", "#e74c3c"),
]:
    fp_, tp_, _ = roc_curve(y_test[mask], lr_sk_probs[mask])
    ax10.plot(fp_, tp_, color=col, lw=2, label=f"{label}  AUC={auc(fp_, tp_):.3f}")
ax10.plot([0,1],[0,1],"k--",lw=1)
ax10.set(title="Subgroup ROC — Logistic Regression (by Grade)",
         xlabel="FPR", ylabel="TPR")
ax10.legend(fontsize=9); ax10.grid(alpha=0.3)

plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Dashboard saved → {PLOT_PATH}")
print("\n✓ All outputs written successfully.")