def acc_thresh(out, yb, thresh):
    return ((out > thresh).float() == yb).float().mean()

def fbeta(out, yb, thresh=0.2, beta=2, eps=1e-9):
    beta2 = beta ** 2
    y_pred = (out > thresh).float()
    y_true = yb.float()
    
    TP = (y_pred * y_true).sum(dim=1)
    
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec  = TP/(y_true.sum(dim=1)+eps)
    
    res = (1+beta2)*((prec*rec)/((prec*beta2)+rec+eps))
    return res.mean()