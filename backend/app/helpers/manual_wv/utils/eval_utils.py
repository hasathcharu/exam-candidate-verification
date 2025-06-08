from datetime import datetime
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_curve,
    recall_score,
    f1_score,
    confusion_matrix,
)

from .train_utils import build_variational_autoencoder, build_global_autoencoder
from .common import load_csv, load_parquet, load_iso_forest, load_model, load_scaler
from tqdm import tqdm
from . import constants as c

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    auc_roc = auc(fpr, tpr)
    eer_threshold = thresholds[eer_threshold_idx]
    return eer, auc_roc, eer_threshold


def compute_threshold_with_fp_bias(y_true, y_scores, fp_weight=0.7):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Weighted difference to favor lower FPR
    cost = fp_weight * fpr + (1 - fp_weight) * fnr
    best_idx = np.argmin(cost)

    custom_threshold = thresholds[best_idx]
    eer = (fpr[best_idx] + fnr[best_idx]) / 2

    return eer, custom_threshold


def compute_aggregate_results(results):
    average_accuracy = results["Accuracy"].mean()
    average_precision = results["Precision"].mean()
    average_recall = results["Recall"].mean()
    average_f1 = results["F1"].mean()
    average_far = results["FAR"].mean()
    average_frr = results["FRR"].mean()
    average_eer = results["EER"].mean()
    average_roc_auc = results["AUC_ROC"].mean()
    std_accuracy = results["Accuracy"].std()
    std_precision = results["Precision"].std()
    std_recall = results["Recall"].std()
    std_f1 = results["F1"].std()
    std_far = results["FAR"].std()
    std_frr = results["FRR"].std()
    std_eer = results["EER"].std()
    std_roc_auc = results["AUC_ROC"].std()
    return (
        average_accuracy,
        min(results["Accuracy"]),
        max(results["Accuracy"]),
        average_precision,
        average_recall,
        average_f1,
        average_far,
        average_frr,
        average_eer,
        average_roc_auc,
        std_accuracy,
        std_precision,
        std_recall,
        std_f1,
        std_far,
        std_frr,
        std_eer,
        std_roc_auc,
    )


def print_results(results):
    (
        average_accuracy,
        min_accuracy,
        max_accuracy,
        average_precision,
        average_recall,
        average_f1,
        average_far,
        average_frr,
        average_eer,
        average_roc_auc,
        std_accuracy,
        std_precision,
        std_recall,
        std_f1,
        std_far,
        std_frr,
        std_eer,
        std_roc_auc,
    ) = compute_aggregate_results(results)
    print(f"Average Accuracy: {average_accuracy:.4f}")
    print(f"Minimum Accuracy: {min_accuracy:.4f}")
    print(f"Maximum Accuracy: {max_accuracy:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1: {average_f1:.4f}")
    print(f"Average FAR: {average_far:.4f}")
    print(f"Average FRR: {average_frr:.4f}")
    print(f"Average EER: {average_eer:.4f}")
    print(f"Average AUC_ROC: {average_roc_auc:.4f}")
    print(f"Std Deviation Accuracy: {std_accuracy:.4f}")
    print(f"Std Deviation Precision: {std_precision:.4f}")
    print(f"Std Deviation Recall: {std_recall:.4f}")
    print(f"Std Deviation F1: {std_f1:.4f}")
    print(f"Std Deviation FAR: {std_far:.4f}")
    print(f"Std Deviation FRR: {std_frr:.4f}")
    print(f"Std Deviation EER: {std_eer:.4f}")
    print(f"Std Deviation AUC_ROC: {std_roc_auc:.4f}")


def save_results(results, path, sample_model):
    os.makedirs(f"{path}/{TIME_STAMP}", exist_ok=True)
    results.to_csv(f"{path}/{TIME_STAMP}/results.csv", index=False)
    (
        average_accuracy,
        min_accuracy,
        max_accuracy,
        average_precision,
        average_recall,
        average_f1,
        average_far,
        average_frr,
        average_eer,
        average_roc_auc,
        std_accuracy,
        std_precision,
        std_recall,
        std_f1,
        std_far,
        std_frr,
        std_eer,
        std_roc_auc,
    ) = compute_aggregate_results(results)
    with open(f"{path}/{TIME_STAMP}/aggregates.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy:.4f}\n")
        f.write(f"Minimum Accuracy: {min_accuracy:.4f}\n")
        f.write(f"Maximum Accuracy: {max_accuracy:.4f}\n")
        f.write(f"Average Precision: {average_precision:.4f}\n")
        f.write(f"Average Recall: {average_recall:.4f}\n")
        f.write(f"Average F1: {average_f1:.4f}\n")
        f.write(f"Average FAR: {average_far:.4f}\n")
        f.write(f"Average FRR: {average_frr:.4f}\n")
        f.write(f"Average EER: {average_eer:.4f}\n")
        f.write(f"Average AUC_ROC: {average_roc_auc:.4f}\n")
        f.write(f"Std Deviation Accuracy: {std_accuracy:.4f}\n")
        f.write(f"Std Deviation Precision: {std_precision:.4f}\n")
        f.write(f"Std Deviation Recall: {std_recall:.4f}\n")
        f.write(f"Std Deviation F1: {std_f1:.4f}\n")
        f.write(f"Std Deviation FAR: {std_far:.4f}\n")
        f.write(f"Std Deviation FRR: {std_frr:.4f}\n")
        f.write(f"Std Deviation EER: {std_eer:.4f}\n")
        f.write(f"Std Deviation AUC_ROC: {std_roc_auc:.4f}\n")
    sample_model.save(f"{path}/{TIME_STAMP}/sample_model.keras")


def evaluate_autoencoders(writer_ids):

    results = []

    thresholds = load_csv(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/thresholds")

    for target_writer in tqdm(writer_ids, desc=f"Evaluating models"):
        tqdm.write(f"\nEvaluating model for writer {target_writer}...")
        scaler = load_scaler(
            f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{target_writer}_scaler"
        )
        test = load_parquet(f"{c.FEATURES}/{c.TEST}/{target_writer}")
        if test is None:
            tqdm.write(f"Skipping {target_writer}, no test data found.")
            continue

        y_true = test["label"].values
        test = test.drop(columns=["label", "sample"])

        model = build_global_autoencoder(test.shape[1])
        model.load_weights(
            f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{target_writer}_autoencoder.weights.h5"
        )

        if model is None or scaler is None:
            tqdm.write(f"Model or scaler not found for {target_writer}. Skipping...")
            continue

        test = scaler.transform(test)
        test_reconstructed = model.predict(test, verbose=0)
        y_scores = np.mean(np.square(test - test_reconstructed), axis=1)
        threshold = thresholds[thresholds["Writer"] == target_writer][
            "Threshold"
        ].values[0]
        y_pred = np.where(np.array(y_scores) <= threshold, 0, 1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        far = fp / (fp + tn + 1e-8)
        frr = fn / (fn + tp + 1e-8)
        eer, auc_roc, _ = compute_eer(y_true, y_scores)
        tqdm.write(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}, EER: {eer:.4f}, AUC_ROC: {auc_roc:.4f}"
        )

        results.append([target_writer, accuracy, precision, recall, f1, far, frr, eer, auc_roc])

    df_results = pd.DataFrame(
        results,
        columns=[
            "Writer",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "FAR",
            "FRR",
            "EER",
            "AUC_ROC",
        ],
    )
    return df_results, model


def compute_autoencoder_thresholds(writer_ids):

    thresholds = []

    for target_writer in tqdm(writer_ids, desc=f"Computing Thresholds"):
        tqdm.write(f"\nComputing thresholds for writer {target_writer}...")

        val = load_parquet(f"{c.FEATURES}/{c.VAL}/{target_writer}")

        if val is None:
            tqdm.write(f"Skipping {target_writer}, no validation data found.")
            continue
        val = val.drop(columns=["sample"])
        y_true = val["label"].values
        val = val.drop(columns=["label"])
        scaler = load_scaler(
            f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{target_writer}_scaler"
        )

        model = build_global_autoencoder(val.shape[1])
        model.load_weights(
            f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/{target_writer}_autoencoder.weights.h5"
        )

        val = scaler.transform(val)
        if model is None or scaler is None:
            tqdm.write(f"Model or scaler not found for {target_writer}. Skipping...")
            continue

        val_reconstructed = model.predict(val, verbose=0)
        y_scores = np.mean(np.square(val - val_reconstructed), axis=1)

        _, _, threshold = compute_eer(y_true, y_scores)
        thresholds.append([target_writer, threshold])

    df_results = pd.DataFrame(thresholds, columns=["Writer", "Threshold"])
    df_results.to_csv(
        f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.TRAIN}/thresholds.csv", index=False
    )
    return df_results


def compute_conv_autoencoder_thresholds(writer_ids):

    thresholds = []
    scaler_names = ["nonseq", "chain", "hod", "viz"]

    for target_writer in tqdm(writer_ids, desc=f"Computing Thresholds"):
        tqdm.write(f"\nComputing thresholds for writer {target_writer}...")

        scalers = []
        for i in scaler_names:
            scaler = load_scaler(
                f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{target_writer}_{i}_scaler"
            )
            scalers.append(scaler)

        model = load_model(
            f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{target_writer}_autoencoder"
        )

        if model is None:
            tqdm.write(f"Model not found for {target_writer}. Skipping...")
            continue

        val = load_parquet(f"{c.FEATURES}/{c.VAL}/{target_writer}")
        if val is None:
            tqdm.write(f"Skipping {target_writer}, no validation data found.")
            continue

        val = val.drop(columns=["sample"])
        y_true = val["label"].values
        val = val.drop(columns=["label"])

        val_nonseq = val.iloc[:, 0:25].values.astype(np.float32)
        val_chain = val.iloc[:, 25:33].values.astype(np.float32)
        val_hod = val.iloc[:, 33:42].values.astype(np.float32)
        val_viz = val.iloc[:, 42:].values.astype(np.float32)

        val_nonseq = scalers[0].transform(val_nonseq)
        val_chain = scalers[1].transform(val_chain)
        val_hod = scalers[2].transform(val_hod)
        val_viz = scalers[3].transform(val_viz)

        pred_nonseq, pred_chain, pred_hod, pred_viz = model.predict(
            [val_nonseq, val_chain, val_hod, val_viz], verbose=0
        )

        mse_nonseq = np.mean(np.square(val_nonseq - pred_nonseq), axis=1)
        mse_chain = np.mean(np.square(val_chain - pred_chain), axis=1)
        mse_hod = np.mean(np.square(val_hod - pred_hod), axis=1)
        mse_viz = np.mean(np.square(val_viz - pred_viz), axis=1)
        y_scores = mse_nonseq + mse_chain + mse_hod + mse_viz

        _, _, threshold = compute_eer(y_true, y_scores)
        thresholds.append([target_writer, threshold])

    df_results = pd.DataFrame(thresholds, columns=["Writer", "Threshold"])
    df_results.to_csv(
        f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/thresholds.csv", index=False
    )
    return df_results


def evaluate_conv_autoencoders(writer_ids):

    results = []
    thresholds = load_csv(
        f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/thresholds"
    )
    scaler_names = ["nonseq", "chain", "hod", "viz"]

    for target_writer in tqdm(writer_ids, desc=f"Evaluating models"):
        tqdm.write(f"\nEvaluating model for writer {target_writer}...")

        scalers = []
        for i in scaler_names:
            scaler = load_scaler(
                f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{target_writer}_{i}_scaler"
            )
            scalers.append(scaler)

        model = load_model(
            f"{c.MODEL_FOLDER}/{c.CONV_AUTOENCODERS}/{c.TRAIN}/{target_writer}_autoencoder"
        )

        if model is None:
            tqdm.write(f"Model not found for {target_writer}. Skipping...")
            continue

        test_data = load_parquet(f"{c.FEATURES}/{c.TEST}/{target_writer}")
        test_data = test_data.drop(columns=["sample"])
        y_true = test_data["label"].values
        test_data = test_data.drop(columns=["label"])

        X_nonseq = test_data.iloc[:, 0:25].values.astype(np.float32)
        X_chain = test_data.iloc[:, 25:33].values.astype(np.float32)
        X_hod = test_data.iloc[:, 33:42].values.astype(np.float32)
        X_viz = test_data.iloc[:, 42:].values.astype(np.float32)

        X_nonseq = scalers[0].transform(X_nonseq)
        X_chain = scalers[1].transform(X_chain)
        X_hod = scalers[2].transform(X_hod)
        X_viz = scalers[3].transform(X_viz)

        pred_nonseq, pred_chain, pred_hod, pred_viz = model.predict(
            [X_nonseq, X_chain, X_hod, X_viz], verbose=0
        )

        mse_nonseq = np.mean(np.square(X_nonseq - pred_nonseq), axis=1)
        mse_chain = np.mean(np.square(X_chain - pred_chain), axis=1)
        mse_hod = np.mean(np.square(X_hod - pred_hod), axis=1)
        mse_viz = np.mean(np.square(X_viz - pred_viz), axis=1)

        y_scores = mse_nonseq + mse_chain + mse_hod + mse_viz

        threshold = thresholds[thresholds["Writer"] == target_writer][
            "Threshold"
        ].values[0]

        y_pred = np.where(np.array(y_scores) <= threshold, 0, 1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        eer, auc_roc, _ = compute_eer(y_true, y_scores)
        far = fp / (fp + tn + 1e-8)
        frr = fn / (fn + tp + 1e-8)
        tqdm.write(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}, EER: {eer:.4f}, AUC_ROC: {auc_roc:.4f}"
        )

        results.append(
            [target_writer, accuracy, precision, recall, f1, far, frr, eer, auc_roc]
        )

    df_results = pd.DataFrame(
        results,
        columns=[
            "Writer",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "FAR",
            "FRR",
            "EER",
            "AUC_ROC",
        ],
    )
    return df_results, model


# def evaluate_iso_forests(
#     writer_ids, feature_folder="features", model_folder="iso_forests", sub_dir="and"
# ):
#     results = []

#     for target_writer in tqdm(writer_ids, desc=f"Evaluating {sub_dir} models"):
#         tqdm.write(
#             f"\nEvaluating '{sub_dir}' Isolation Forest for writer {target_writer}..."
#         )

#         iso_forest = load_iso_forest(target_writer, model_folder, sub_dir)
#         scaler = load_scaler(target_writer, model_folder, sub_dir)

#         y_true, y_scores = [], []

#         for test_writer in writer_ids:
#             test_data = load_parquet(test_writer, feature_folder, sub_dir)
#             if test_data is None:
#                 tqdm.write(f"Skipping {test_writer}, no data found.")
#                 continue

#             test_data = prepare_features(test_data)
#             test_data = scaler.transform(test_data)

#             scores = -iso_forest.decision_function(
#                 test_data
#             )  # Flip sign to align with anomaly score
#             y_scores.extend(scores)

#             labels = (
#                 np.zeros(len(test_data))
#                 if test_writer == target_writer
#                 else np.ones(len(test_data))
#             )
#             y_true.extend(labels)

#         eer, eer_threshold = compute_eer(y_true, y_scores)
#         y_pred = np.where(np.array(y_scores) <= eer_threshold, 0, 1)

#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
#         recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

#         tqdm.write(
#             f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, EER: {eer:.4f}"
#         )

#         results.append([target_writer, accuracy, precision, recall, eer])

#     df_results = pd.DataFrame(
#         results, columns=["Writer", "Accuracy", "Precision", "Recall", "EER"]
#     )
#     return df_results
