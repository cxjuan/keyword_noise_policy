import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F
from dippo.utils import load_yaml, set_seed, ensure_dir
from dippo.data import make_loader
from dippo.classifier import SmallDigitCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg['seed']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(cfg['output_dir'])
    ccfg = cfg['classifier']
    loader = make_loader(cfg['dataset'], cfg['data_root'], int(ccfg['batch_size']), train=True, download=False)
    test_loader = make_loader(cfg['dataset'], cfg['data_root'], int(ccfg['batch_size']), train=False, download=False)
    model = SmallDigitCNN(in_channels=1, num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(ccfg['lr']))
    for epoch in range(int(ccfg['epochs'])):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.shape[0]
            correct += (logits.argmax(1) == y).sum().item()
            total += x.shape[0]
        print(f'epoch={epoch+1} train_loss={loss_sum/total:.4f} train_acc={correct/total:.4f}')
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += x.shape[0]
    print(f'test_acc={correct/total:.4f}')
    out = Path(cfg['keyword']['classifier_ckpt'])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict()}, out)
    print(f'saved classifier: {out}')


if __name__ == '__main__':
    main()
