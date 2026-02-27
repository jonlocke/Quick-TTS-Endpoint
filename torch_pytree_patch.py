import torch
import inspect

def _wrap_register(fn):
    """
    Wrap torch.utils._pytree._register_pytree_node so it tolerates newer kwargs
    (e.g. serialized_type_name) used by newer Transformers.
    """
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
    except Exception:
        accepted = None

    def wrapped(*args, **kwargs):
        if accepted is not None:
            # drop any kwargs older torch doesn't support
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        else:
            # fallback: explicitly drop the known problematic kwarg
            kwargs.pop("serialized_type_name", None)
        return fn(*args, **kwargs)

    return wrapped

try:
    pt = torch.utils._pytree

    # Newer libs expect pt.register_pytree_node; torch 2.1.2 has _register_pytree_node
    base = getattr(pt, "register_pytree_node", None)
    if base is None:
        base = getattr(pt, "_register_pytree_node", None)

    if base is not None:
        pt.register_pytree_node = _wrap_register(base)

except Exception:
    pass
