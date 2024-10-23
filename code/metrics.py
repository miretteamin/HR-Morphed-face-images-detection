import numpy as np 
from sklearn.metrics import confusion_matrix


# ### Basic metrics ###
def FDR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:

    y_pred_binary = (y_pred >= threshold).astype(int)

    _, fp, _, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    return fp / (fp + tp) if (fp + tp) > 0 else 0.0


def NPV(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:

    y_pred_binary = (y_pred >= threshold).astype(int)
    
    _, _, fn, tn = confusion_matrix(y_true, y_pred_binary).ravel()
    tn/(tn+fn) if (tn+fn) > 0 else 0.0
    
    return tn/(tn+fn) if (tn+fn) > 0 else 0.0



def AUC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    fpr_vals, tpr_vals = [], []
    thresholds = np.unique(y_pred)
    
    for threshold in thresholds:
        tpr, fpr = confusion_matrix(y_true, y_pred >= threshold, normalize='true').ravel()
        tpr_vals.append(tpr)
        fpr_vals.append(fpr)
    
    sorted_idx = np.argsort(fpr_vals)
    fpr_vals = np.array(fpr_vals)[sorted_idx]
    tpr_vals = np.array(tpr_vals)[sorted_idx]
    
    return np.trapz(tpr_vals, fpr_vals)



### Task Based metrics ###
def MACER(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """    
    MACER is calculated as the number of false negatives divided by the total number of morphed images in the dataset.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    y_pred = (y_pred > threshold).astype(int)

    _, _, morph_misclassified, tp = confusion_matrix.ravel()
    morph_count = morph_misclassified + tp
    return morph_misclassified / morph_count if morph_count != 0 else 0


def BPCER(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Function to calculate Bona Fide Presentation Classification Error Rate 
    
    bona fide (0) or morphed (1). 
    BPCER = bona fide images misclassified as morphs divided by the total number of bona fide images.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    y_pred = (y_pred > threshold).astype(int)
    tn, bona_fide_misclassified, _, _ = confusion_matrix(y_true, y_pred).ravel()

    bona_fide_count = tn + bona_fide_misclassified
    return bona_fide_misclassified / bona_fide_count if bona_fide_count != 0 else 0


def FAR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None) -> float:
    """
    Compute the False Acceptance Rate --> impostor_errors / total_morphs
    where:
    - impostor_errors = number of morph samples incorrectly classified as bona fides
    - total_morphs = total number of morph samples (impostors)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if threshold is not None:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    false_acceptances = np.logical_and(y_pred_binary == 1, y_true == 0)
    total_morph_attacks = np.sum(y_true == 0)
    
    return np.sum(false_acceptances) / total_morph_attacks if total_morph_attacks > 0 else 0.0


def FRR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None) -> float:
    """
    Compute the False Rejection Rate --> bona_fide_errors / total_bona_fide
    where:
    - bona_fide_errors = number of bona fide samples incorrectly classified as morphs
    - total_bona_fide = total number of bona fide samples
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if threshold is not None:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    false_rejections = np.logical_and(y_pred_binary == 0, y_true == 1)
    total_bona_fides = np.sum(y_true == 1)
    
    return np.sum(false_rejections) / total_bona_fides if total_bona_fides > 0 else 0.0


def EER(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Equal Error Rate
    The EER is the point where FAR and FRR are almost equal (the closest)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    thresholds = np.unique(y_pred)  
    min_diff = float('inf')  
    eer = None
    
    for threshold in thresholds:
        far = FAR(y_true, y_pred, threshold)
        frr = FRR(y_true, y_pred, threshold)
        
        # diff is minimized here
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2  
    
    return eer

def MCC(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute the Matthews Correlation Coefficient
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp, fp, fn, tn = confusion_matrix(y_true, y_pred_binary).ravel()
    
    numerator = tp * tn - fp * fn
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / denom if denom > 0 else 0.0


def DCF(y_true: np.ndarray, y_pred: np.ndarray, C_miss: float, C_fa: float, P_target: float, threshold: float) -> float:
    """
    Compute the Detection Cost Function
    
    C_miss: Cost of a missed detection (failed to detect morph)
    C_fa: Cost of a false alarm (falsely classifying bona fide as morph)
    P_target: Probs of a morph attack (target probs)
    """
    far = FAR(y_true, y_pred, threshold)
    frr = FRR(y_true, y_pred, threshold)
    return C_miss * P_target * frr + C_fa * (1 - P_target) * far




## FMR, FNMR, MMPMR, FMMPMR, AMPMR, APCER