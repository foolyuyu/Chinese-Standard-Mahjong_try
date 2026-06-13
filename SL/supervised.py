from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
import os

if __name__ == '__main__':
    logdir = 'model/'
    os.makedirs(logdir + 'checkpoint', exist_ok = True)
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
    # Train and validate
    for e in range(20):
        print('Epoch', e)
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].to(device), 'action_mask': d[1].to(device)}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().to(device))
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].to(device), 'action_mask': d[1].to(device)}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].to(device)).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
