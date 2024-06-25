import torch 
import numpy as np


def validation(model, criterion, config):
    validset_loader = config['validset_loader']
    device = next(model.parameters()).device

    model.eval()  
    valid_loss = 0
    predList, maskList = [], []
    with torch.no_grad(): 
        for image, mask in validset_loader:
            batch_size = image.shape[0]
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            valid_loss += criterion(output, mask).item() * batch_size 
            predList += [singleBatch for singleBatch in output.cpu().numpy()]
            maskList += [singleBatch for singleBatch in mask.cpu().numpy()]
            
    valid_loss /= len(validset_loader.dataset)
    score = mean_iou_score(np.array(predList), np.array(maskList))      
    print('Validation set: Average loss = {:.4f} mIoU score = {:.2f}% \n'.format(valid_loss, 100. * score))
    
    return score


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    pred = np.argmax(pred, axis=1)

    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / ((tp_fp + tp_fn - tp))
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    return mean_iou