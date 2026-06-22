import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from torchvision.utils import save_image, make_grid
from dippo.utils import load_yaml, set_seed, ensure_dir
from dippo.data import make_loader
from dippo.env_keyword import KeywordDiPPOImageEnv
from dippo.ppo import ActorCritic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num_batches', type=int, default=2)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg['seed']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(cfg['output_dir']) / 'keyword_released'
    ensure_dir(out_dir)
    env = KeywordDiPPOImageEnv(cfg, device=device)
    model = ActorCritic(cfg['ppo']['obs_dim'], len(env.action_space), cfg['ppo']['hidden_dim']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    loader = make_loader(cfg['dataset'], cfg['data_root'], cfg['ppo']['batch_size'], train=False, download=False)
    with torch.no_grad():
        for bi, (x, _) in enumerate(loader):
            if bi >= args.num_batches:
                break
            x = x.to(device)
            obs, _, _, _ = env.make_obs_from_floor(x)
            # valid_mask = env.action_space.mask(x.shape[0], device, env.max_extra_t)
            valid_mask = env.action_mask(obs)
            action, _, _, _ = model.act(obs, valid_mask)
            _, reward, info = env.step(x, action)
            released = info['released'].clamp(0, 1)
            z0 = info['z0'].clamp(0, 1)
            noised = info['noised'].clamp(0, 1)
            concept_mask = info['mask'].repeat(1, x.shape[1], 1, 1).clamp(0, 1)
            print(
                f'batch={bi}',
                f'reward={reward.mean().item():.4f}',
                f'rejected={info["rejected"].float().mean().item():.3f}',
                f'extra_keyword={info["extra_keyword"].float().mean().item():.1f}',
                f'extra_background={info["extra_background"].float().mean().item():.1f}',
                f'denoise_keyword={info["denoise_keyword"].mean().item():.3f}',
                f'denoise_background={info["denoise_background"].mean().item():.3f}',
                f'kw_score_before={info["keyword_score_before"].mean().item():.3f}',
                f'kw_score_after={info["keyword_score_after"].mean().item():.3f}',
                f'risk={info["risk"].mean().item():.4f}',
                f'thr={info["threshold"].mean().item():.4f}',
            )
            items = torch.cat([x[:32].cpu(), z0[:32].cpu(), concept_mask[:32].cpu(), noised[:32].cpu(), released[:32].cpu()], dim=0)
            save_image(make_grid(items, nrow=8), out_dir / f'compare_keyword_batch_{bi}.png')
            save_image(make_grid(released[:64].cpu(), nrow=8), out_dir / f'released_batch_{bi}.png')
    print(f'saved outputs to: {out_dir}')


if __name__ == '__main__':
    main()
