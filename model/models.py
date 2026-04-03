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

# separate into Prime (A-B) vs Subprime (D-G) segments
test_grade = X_test[:, FEATURE_COLS.index("loan_grade")]
prime      = test_grade <= 1      # 0=A, 1=B
subprime   = test_grade >= 3      # 3=D … 6=G

PALETTE = {"LR": "#4C72B0", "RF": "#DD8452"}
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Credit Risk Model — Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(4,4,figure=fig, hspace=0.45, wspace=0.38)

# (a) ROC
ax1 = fig.add_subplot(gs[0, 0])
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_sk_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
ax1.plot(lr_fpr, lr_tpr, color=PALETTE["LR"], lw=2, label=f"LR  AUC={lr_sk_auc:.3f}")
ax1.plot(rf_fpr, rf_tpr, color=PALETTE["RF"], lw=2, label=f"RF  AUC={rf_auc:.3f}")
ax1.plot([0,1],[0,1],"k--",lw=1)
ax1.set(title="ROC Curves", xlabel="FPR", ylabel="TPR"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# (b) Precision-Recall
ax2 = fig.add_subplot(gs[0, 2])
lp, lr_, _ = precision_recall_curve(y_test, lr_sk_probs)
rp, rr_, _ = precision_recall_curve(y_test, rf_probs)
ax2.plot(lr_, lp, color=PALETTE["LR"], lw=2, label=f"LR  AP={lr_sk_ap:.3f}")
ax2.plot(rr_, rp, color=PALETTE["RF"], lw=2, label=f"RF  AP={rf_ap:.3f}")
ax2.set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# (c) Feature importance (RF)
ax3 = fig.add_subplot(gs[1,0])
feat_imp.plot(kind="barh", ax=ax3, color="#4C9A2A"); ax3.invert_yaxis()
ax3.set(title="Feature Importance for Random Forest", xlabel="Importance"); ax3.grid(axis="x", alpha=0.3)

#feature importance for logistic regression
ax6 = fig.add_subplot(gs[1, 2])
coef_df.plot(kind="barh", x="feature", y="coefficient", ax=ax6, color="#4C72B0"); ax6.invert_yaxis()
ax6.set(title="Feature importance for Logistic Regression", xlabel="Coefficient"); ax6.grid(axis="x", alpha=0.3)

# Confusion matrix — LR
ax4 = fig.add_subplot(gs[2, 0])
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=["No Default","Default"], yticklabels=["No Default","Default"])
ax4.set(title=f"Confusion Matrix — Logistic Regression  (AUC {lr_sk_auc:.3f})", xlabel="Predicted", ylabel="Actual")

# Confusion matrix — RF
ax5 = fig.add_subplot(gs[2, 2])
cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges", ax=ax5,
            xticklabels=["No Default","Default"], yticklabels=["No Default","Default"])
ax5.set(title=f"Confusion Matrix — Random Forest  (AUC {rf_auc:.3f})", xlabel="Predicted", ylabel="Actual")

# logistic regression prob distribution by subgroup
ax8 = fig.add_subplot(gs[3, 0])
ax8.hist(lr_sk_probs[prime],    bins=30, alpha=0.6, color="#3498db", label="Prime (A–B)")
ax8.hist(lr_sk_probs[subprime], bins=30, alpha=0.6, color="#e74c3c", label="Subprime (D–G)")
ax8.set(title="Logistic Regression Prob Distribution by Grade Group", xlabel="Default Probability", ylabel="Count")
ax8.legend(fontsize=9); ax8.grid(alpha=0.3)

# how the logistic regression model performs across different subgroups of the data, specifically by loan grade 
ax9 = fig.add_subplot(gs[3, 2])
for mask, label, col in [(prime,    "Prime A–B",   "#3498db"),
                          (subprime, "Subprime D–G","#e74c3c")]:
    fp_, tp_, _ = roc_curve(y_test[mask], lr_sk_probs[mask])
    ax9.plot(fp_, tp_, color=col, lw=2, label=f"{label}  AUC={auc(fp_,tp_):.3f}")
ax9.plot([0,1],[0,1],"k--",lw=1)
ax9.set(title="Subgroup ROC — LR (by Grade)", xlabel="FPR", ylabel="TPR")
ax9.legend(fontsize=9); ax9.grid(alpha=0.3)

plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ All outputs written successfully.")
