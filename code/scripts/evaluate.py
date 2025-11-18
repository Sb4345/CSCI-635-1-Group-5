from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
import warnings



def evaluate(y_true, y_pred):
    """
    Evaluates the model's predition agains the true values

    Parameters:
    y_true (array-like): The true classificaitons
    y_pred (array-like): The predictioned classifications

    Returns:
    A confussion matrix, a list of multilabel confusion matrices, a classified report, and MCC score
    """

    warnings.filterwarnings("ignore")

    cmd = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))

    mcList = []
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, cm in enumerate(mcm):
        mcList.append(ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]))
    
    cr = classification_report(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)


    return cmd, mcList, cr, mcc
