try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, recall_score, classification_report
    import numpy as np
    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        f1_micro = f1_score(labels, preds, labels=[1, 2, 3], average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        f1_claim = f1_score(labels, preds, labels=[1], average='micro')
        f1_evidence = f1_score(labels, preds, labels=[2], average='micro')

        return {
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'f1_claim':f1_claim,
            'f1_evidence':f1_evidence,
        }

    def pico_scores(preds, labels):
        report = classification_report(labels, preds, labels=[1,2,3,4], target_names=['None', 'Intervention', 'Outcome', 'Population'] )
        return report

    def f1_scores(y_pred, y_true, labelfilter=None):

        f1_micro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='micro')
        f1_macro_filtered = f1_score(y_true, y_pred, labels=labelfilter, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        clf_report = classification_report(y_true, y_pred)

        return {
            "eval_f1_micro_filtered": f1_micro_filtered,
            "eval_f1_macro_filtered": f1_macro_filtered,
            'eval_f1_micro': f1_micro,
            'eval_f1_macro': f1_macro,
            'clf_report': clf_report
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def compute_confusion_matrix(task_name, y_pred, y_true):

        assert len(y_pred) == len(y_true)
        if task_name == "multichoice" or task_name == "relclass":
            return confusion_matrix(y_true, y_pred)
        else:
            raise KeyError(task_name)

    def compute_metrics(task_name, preds, labels):
        """
        Compute metrics for sequence tagging tasks
        Args:
            task_name: name of the task
            preds: predictions from model
            labels: true labels
        """
        print("\nDEBUG - Metrics Calculation Start")
        print(f"Shape of predictions: {preds.shape}")
        print(f"Shape of labels: {labels.shape}")
        
        # Remove padding tokens (-100)
        mask = labels != -100
        labels_filtered = labels[mask]
        preds_filtered = preds[mask]
        
        print(f"Unique values in predictions: {np.unique(preds_filtered, return_counts=True)}")
        print(f"Unique values in true labels: {np.unique(labels_filtered, return_counts=True)}")
        
        # Calculate metrics
        results = {}
        
        # Overall metrics
        results["eval_f1_micro"] = f1_score(labels_filtered, preds_filtered, average='micro')
        results["eval_f1_macro"] = f1_score(labels_filtered, preds_filtered, average='macro')
        
        # Per-class metrics
        label_map = {0: "X", 1: "I-Claim", 2: "I-Premise", 3: "O"}
        for label_id, label_name in label_map.items():
            mask_true = (labels_filtered == label_id)
            mask_pred = (preds_filtered == label_id)
            
            # Calculate F1 score for each class
            f1 = f1_score(mask_true, mask_pred, zero_division=0)
            results[f"f1_{label_name.lower()}"] = f1
            
            # Additional debugging
            print(f"\nMetrics for {label_name}:")
            print(f"True positives: {np.sum(mask_true & mask_pred)}")
            print(f"False positives: {np.sum(mask_pred & ~mask_true)}")
            print(f"False negatives: {np.sum(mask_true & ~mask_pred)}")
        
        print("\nFinal metrics:", results)
        return results
