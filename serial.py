import torch, os


def save_ckpt(path, actor, critic=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"actor": actor.state_dict()}
    if critic is not None and hasattr(critic, 'state_dict'):
        payload["critic"] = critic.state_dict()
    torch.save(payload, path)


def load_ckpt(path, actor, critic=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    actor.load_state_dict(payload["actor"])
    if critic is not None and "critic" in payload:
        critic.load_state_dict(payload["critic"])


