# train_lightgbm.py

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss, roc_auc_score
# ────────── helper ──────────
def load_dataset(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        df_list = []
        for file in files:
            df = pd.read_csv(file)
            if "label" in df.columns:
                df_list.append(df)
        if not df_list:
            raise ValueError("❌ No valid CSVs with 'label' found in directory.")
        return pd.concat(df_list, ignore_index=True)
    elif os.path.isfile(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"{path} is not a valid file or directory.")

def get_label_mapping(target_col):
    """Get appropriate label mapping for each target column"""
    if target_col == "label":  # 15-min labels (includes TRAP)
        return {
            "DOWN": 0,
            "FLAT": 1,
            "UP": 2,
            "TRAP_UP": 3,
            "TRAP_DOWN": 4,
        }
    elif target_col == "label_30":  # 30-min labels (no TRAP)
        return {
            "DOWN": 0,
            "FLAT": 1,
            "UP": 2,
        }
    else:
        raise ValueError(f"Unknown target column: {target_col}")

def validate_labels(y, target_col, label_map):
    """Validate label distribution and check for issues"""
    print(f"\n📊 Label validation for {target_col}:")
    
    # Check for NaN values
    nan_count = np.isnan(y).sum()
    if nan_count > 0:
        raise ValueError(f"❌ Found {nan_count} NaN values in {target_col}")
    
    # Check class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    reverse_map = {v: k for k, v in label_map.items()}
    
    print(f"   Class distribution:")
    total_samples = len(y)
    for cls, count in zip(unique_classes, counts):
        class_name = reverse_map.get(cls, f"UNKNOWN_{cls}")
        percentage = (count / total_samples) * 100
        print(f"     {class_name:10} (={cls}): {count:5d} samples ({percentage:5.1f}%)")
    
    # Critical check: Must have at least 2 classes for classification
    num_classes = len(unique_classes)
    if num_classes < 2:
        raise ValueError(f"❌ Cannot train {target_col} — only {num_classes} class(es) present")
    
    # Check for missing classes
    expected_classes = set(label_map.values())
    actual_classes = set(unique_classes)
    missing_classes = expected_classes - actual_classes
    
    if missing_classes:
        missing_names = [reverse_map[cls] for cls in missing_classes]
        print(f"   ⚠️ Missing classes: {missing_names}")
    
    # Check for minimum samples per class
    min_samples_per_class = 10
    insufficient_classes = [
        (reverse_map[cls], count) 
        for cls, count in zip(unique_classes, counts) 
        if count < min_samples_per_class
    ]
    
    if insufficient_classes:
        print(f"   ⚠️ Classes with <{min_samples_per_class} samples: {insufficient_classes}")
    
    return num_classes

def train_model(df, target_col, model_path):

    # Get appropriate label mapping for this target
    label_map = get_label_mapping(target_col)

    if target_col not in df.columns:
        raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")

    # Map labels and check for invalid mappings
    y_mapped = df[target_col].map(label_map)
    
    # Check for unmapped labels (will be NaN)
    unmapped_mask = y_mapped.isna()
    if unmapped_mask.any():
        unmapped_labels = df.loc[unmapped_mask, target_col].unique()
        raise ValueError(f"❌ Unmapped labels in {target_col}: {unmapped_labels}")
    
    y = y_mapped.values.astype(int)
    
    # Validate labels and get class count
    num_classes = validate_labels(y, target_col, label_map)
     
    X = df.drop(columns=["label", "label_30"])
    #y = df[target_col].map(label_map).values
    print(f"\n🎯 Training {target_col}: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=3, 
        num_leaves=7, 
        min_child_samples=200,
        subsample=0.6,
        colsample_bytree=0.4,
        reg_lambda=10.0,
        reg_alpha=5.0, 
        objective="multiclass",
        num_class=num_classes,
        class_weight="balanced",
        force_col_wise=True,
        random_state=42
    )


    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(50)])

    # ✅ PERFORMANCE METRICS LOGGING
    print(f"\n📊 {target_col} Performance Metrics:")
    
    # Validation predictions
    y_val_pred_proba = clf.predict_proba(X_val)
    y_val_pred = clf.predict(X_val)
    
    # Log Loss
    val_log_loss = log_loss(y_val, y_val_pred_proba)
    print(f"   📉 Validation Log Loss: {val_log_loss:.4f}")
    
    # AUC (multiclass - one-vs-rest)
    try:
        val_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr', average='weighted')
        print(f"   📈 Validation AUC (weighted): {val_auc:.4f}")
    except ValueError as e:
        print(f"   ⚠️ AUC calculation failed: {e}")
    
    # Training set metrics for comparison
    y_tr_pred_proba = clf.predict_proba(X_tr)
    train_log_loss = log_loss(y_tr, y_tr_pred_proba)
    print(f"   📉 Training Log Loss: {train_log_loss:.4f}")
    
    try:
        train_auc = roc_auc_score(y_tr, y_tr_pred_proba, multi_class='ovr', average='weighted')
        print(f"   📈 Training AUC (weighted): {train_auc:.4f}")
    except ValueError:
        pass
    
    # Overfitting check
    if val_log_loss > train_log_loss * 1.2:
        print(f"   ⚠️ Potential overfitting detected (val_loss {val_log_loss/train_log_loss:.2f}x training)")

    # Add feature importance extraction
    importance_gain = clf.booster_.feature_importance(importance_type='gain')
    feature_names = clf.booster_.feature_name()

    df_gain = pd.DataFrame({
        "feature": feature_names,
        "gain": importance_gain
    }).sort_values(by="gain", ascending=False)

    print("\nTop 40 Features by Gain:")
    print(df_gain.head(40).to_string(index=False))

    unused = df_gain[df_gain.gain == 0].feature.tolist()
    print(f"🚫 {len(unused)} features had 0 gain: {unused[:5]}...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f" model saved → {model_path}")

# ────────── main ──────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LightGBM on sequence dataset")
    p.add_argument("--dataset", required=True, help="CSV file or directory of CSVs")
    p.add_argument("--model_dir", required=True, help="Directory to save .pkl models")
    args = p.parse_args()

    df = load_dataset(args.dataset)

    # Dynamic model naming based on feature configuration
    try:
        from feature_config import USE_SHAP_OPTIMIZED, get_feature_set_name
        suffix = "_shap" if USE_SHAP_OPTIMIZED else ""
        print(f"🎯 Training with {get_feature_set_name()}")
    except ImportError:
        suffix = ""
        print("🎯 Training with legacy features")

    train_model(df, target_col="label", model_path=os.path.join(args.model_dir, f"lgbm_label15{suffix}.pkl"))
    train_model(df, target_col="label_30", model_path=os.path.join(args.model_dir, f"lgbm_label30{suffix}.pkl"))
