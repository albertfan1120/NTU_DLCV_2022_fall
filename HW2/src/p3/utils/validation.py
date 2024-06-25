import torch


def validation(config):
    model = config['DANN']
    testloader = config['t_validset_loader']
    device = config['device']
    
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            pre_label, _ = model(images)
            pred = torch.argmax(pre_label, dim=1)
            
            correct += torch.sum(pred == labels)
            total += labels.shape[0]

    score = correct / total
    print('Your accuracy is:  {:.1f}% \n'.format(100. * score))

    return score