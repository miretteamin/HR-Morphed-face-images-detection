import numpy as np



### Basic metrics ###

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Absolute Error (MAE) between y_true and y_pred.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean(np.abs(y_true - y_pred))


def CE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Classification Error (CE) between y_true and y_pred.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean(y_true != y_pred)

def FNR(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the False Negative Rate (FNR) between y_true and y_pred.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean((y_true == 1) & (y_pred == 0))

def TPR_FPR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple:
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for a given threshold.
    
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Recall (True Positive Rate)
    tp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 1))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # False Positive Rate
    fp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return tpr, fpr


def F1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute the F1 Score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def FDR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute the False Discovery Rate (FDR).
    
    FDR = FP / (FP + TP)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    false_positives = np.sum(np.logical_and(y_pred_binary == 1, y_true == 0))
    true_positives = np.sum(np.logical_and(y_pred_binary == 1, y_true == 1))
    
    return false_positives / (false_positives + true_positives) if (false_positives + true_positives) > 0 else 0.0


def NPV(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute the Negative Predictive Value (NPV).
    NPV = TN / (TN + FN)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    true_negatives = np.sum(np.logical_and(y_pred_binary == 0, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred_binary == 0, y_true == 1))
    
    return true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0.0



def AUC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Area Under the ROC Curve (AUC).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    fpr_values, tpr_values = [], []
    thresholds = np.unique(y_pred)
    
    for threshold in thresholds:
        tpr, fpr = TPR_FPR(y_true, y_pred, threshold)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    sorted_indices = np.argsort(fpr_values)
    fpr_values = np.array(fpr_values)[sorted_indices]
    tpr_values = np.array(tpr_values)[sorted_indices]
    
    return np.trapz(tpr_values, fpr_values)



### Task Based metrics ###

def MACER(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.5) -> float:
    """
    Compute the Mean Absolute Classification Error Rate (MACER) between y_true and y_pred.
    MACER = alpha * MAE + beta * CE + gamma * FNR
    where:
    - MAE is the Mean Absolute Error
    - CE is the Classification Error
    - FNR is the False Negative Rate
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    MAE = MAE(y_true, y_pred)
    CE = CE(y_true, y_pred)
    FNR = FNR(y_true, y_pred)
    return alpha * MAE + beta * CE + gamma * FNR


def BPCER(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None) -> float:
    """
    Compute the Bona Fide Presentation Classification Error Rate (BPCER).
    BPCER = bona_fide_errors / total_bona_fide
    where: 
    - bona_fide_errors = number of bona fide errors
    - total_bona_fide = total number of genuine samples
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if threshold is not None:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else: ## Fixed threshold
        y_pred_binary = y_pred
    
    bona_fide_errors = np.logical_and(y_pred_binary == 0, y_true == 1)
    total_bona_fide = np.sum(y_true == 1)
    
    return np.sum(bona_fide_errors) / total_bona_fide if total_bona_fide > 0 else 0.0


def FAR(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None) -> float:
    """
    Compute the False Acceptance Rate (FAR).
    FAR = impostor_errors / total_morphs
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
    Compute the False Rejection Rate (FRR).
    FRR = bona_fide_errors / total_bona_fide
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
    Compute the Equal Error Rate (EER).
    The EER is the point where FAR and FRR are equal.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    thresholds = np.unique(y_pred)  
    min_difference = float('inf')  
    eer = None
    
    for threshold in thresholds:
        far = FAR(y_true, y_pred, threshold)
        frr = FRR(y_true, y_pred, threshold)
        
        # difference is minimized
        difference = abs(far - frr)
        if difference < min_difference:
            min_difference = difference
            eer = (far + frr) / 2  
    
    return eer

def MCC(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC).
    
    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred_binary == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred_binary == 0, y_true == 1))
    
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / denominator if denominator > 0 else 0.0


def DCF(y_true: np.ndarray, y_pred: np.ndarray, C_miss: float, C_fa: float, P_target: float, threshold: float) -> float:
    """
    Compute the Detection Cost Function (DCF).
    
    :param y_true: Ground truth labels (binary, 0 or 1), numpy array.
    :param y_pred: Predicted labels or continuous scores, numpy array.
    :param C_miss: Cost of a missed detection (failed to detect morph).
    :param C_fa: Cost of a false alarm (falsely classifying bona fide as morph).
    :param P_target: Probability of a morph attack (target probability).
    :param threshold: Threshold for converting scores to binary predictions.

    :return: DCF = C_miss * P_target * FRR + C_fa * (1 - P_target) * FAR
    """
    far = FAR(y_true, y_pred, threshold)
    frr = FRR(y_true, y_pred, threshold)
    return C_miss * P_target * frr + C_fa * (1 - P_target) * far


def MMPMR(morphed_scores: list[list[float]], threshold: float) -> float:
    """
    Compute the Mated Morph Presentation Match Rate (MMPMR).
    
    :param morphed_scores: List of lists where each sublist contains comparison scores between a morphed image and contributing subjects.
    :param threshold: Score threshold for a match.
    :return: MMPMR value (float) - proportion of morphs accepted as genuine.
    """
    M = len(morphed_scores)  # Total number of morphed images
    return np.mean([min(scores) > threshold for scores in morphed_scores])


def FMMPMR(morphed_scores: list[list[float]], threshold: float) -> float:
    """
    Compute the Fully Mated Morph Presentation Match Rate (FMMPMR).
    
    :param morphed_scores: List of lists containing comparison scores for multiple subjects.
    :param threshold: Score threshold for a match.
    :return: FMMPMR value (float) - proportion of morphs matching all subjects.
    """
    P = len(morphed_scores)  # Total number of morphed images
    return np.mean([all([score > threshold for score in subject_scores]) for subject_scores in morphed_scores])


def APCER(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Attack Presentation Classification Error Rate (APCER).
    
    :param y_true: Ground truth labels (binary, 0 for attack, 1 for bona fide), numpy array.
    :param y_pred: Predicted labels (binary, 0 or 1), numpy array.
    :return: APCER value (float) - proportion of attacks misclassified as bona fide.
    """
    attack_errors = np.logical_and(y_pred == 1, y_true == 0)
    total_attacks = np.sum(y_true == 0)
    
    return np.sum(attack_errors) / total_attacks if total_attacks > 0 else 0.0


def AMPMR(face_scores: list[list[float]], MAD_scores: list[float], face_threshold: float, MAD_threshold: float) -> float:
    """
    Compute the Actual Mated Morph Presentation Match Rate (AMPMR).
    
    :param face_scores: List of lists containing face recognition scores for morphed images.
    :param MAD_scores: List of MAD (Morph Attack Detection) scores for the same images.
    :param face_threshold: Threshold for face recognition scores.
    :param MAD_threshold: Threshold for MAD scores (to detect attacks).
    :return: AMPMR value (float) - proportion of successful morphing attacks undetected by MAD.
    """
    total_images = len(face_scores)
    successful_attacks = 0
    
    for i in range(total_images):
        # Check if face recognition scores pass for all subjects and if the MAD system fails
        if all([score > face_threshold for score in face_scores[i]]) and MAD_scores[i] > MAD_threshold:
            successful_attacks += 1
    
    return successful_attacks / total_images if total_images > 0 else 0.0
