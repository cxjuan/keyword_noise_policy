import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tqdm import trange
from dippo.utils import load_yaml, set_seed, ensure_dir
from dippo.data import make_loader
from dippo.env_keyword import KeywordDiPPOImageEnv
from dippo.ppo import ActorCritic, ppo_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg['seed']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(cfg['output_dir'])
    env = KeywordDiPPOImageEnv(cfg, device=device)
    loader = make_loader(cfg['dataset'], cfg['data_root'], cfg['ppo']['batch_size'], train=True, download=False)
    data_iter = iter(loader)
    model = ActorCritic(cfg['ppo']['obs_dim'], len(env.action_space), cfg['ppo']['hidden_dim']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg['ppo']['lr']))
    for upd in trange(int(cfg['ppo']['updates']), desc='Keyword PPO updates'):
        obs_list, act_list, logp_list, rew_list, val_list, mask_list = [], [], [], [], [], []
        collected = 0
        while collected < int(cfg['ppo']['rollout_steps']):
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, _ = next(data_iter)
            x = x.to(device)
            obs, _, _, _ = env.make_obs_from_floor(x)
            # valid_mask = env.action_space.mask(x.shape[0], device, env.max_extra_t)
            valid_mask = env.action_mask(obs)
            with torch.no_grad():
                action, logp, value, _ = model.act(obs, valid_mask)
            _, reward, _ = env.step(x, action)
            obs_list.append(obs.detach()); act_list.append(action.detach()); logp_list.append(logp.detach())
            rew_list.append(reward.detach()); val_list.append(value.detach()); mask_list.append(valid_mask.detach())
            collected += x.shape[0]
        obs = torch.cat(obs_list); actions = torch.cat(act_list); logp = torch.cat(logp_list)
        rewards = torch.cat(rew_list); values = torch.cat(val_list); masks = torch.cat(mask_list)
        batch = {'obs': obs, 'actions': actions, 'logp': logp, 'returns': rewards, 'adv': rewards - values.detach(), 'masks': masks}
        loss = ppo_update(model, opt, batch, clip_eps=float(cfg['ppo']['clip_eps']), value_coef=float(cfg['ppo']['value_coef']), entropy_coef=float(cfg['ppo']['entropy_coef']), epochs=int(cfg['ppo']['epochs']))
        if (upd + 1) % 25 == 0:
            print(f'update={upd+1} reward={rewards.mean().item():.4f} loss={loss:.4f}')
    out = Path(cfg['output_dir']) / f"keyword_{cfg['keyword']['name']}_ppo.pt"
    torch.save({'model': model.state_dict(), 'cfg': cfg}, out)
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
