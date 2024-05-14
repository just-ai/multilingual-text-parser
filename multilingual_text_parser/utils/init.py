import copy
import typing as tp
import inspect
import logging
import functools

__all__ = ["init_method_from_config", "init_class_from_config"]

LOGGER = logging.getLogger("root")


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def init_method_from_config(method, cfg: tp.Dict) -> tp.Callable:

    # TODO: Avoid copying when have tensors/models in cfg
    try:
        config = copy.deepcopy(cfg)
    except RuntimeError:
        config = cfg

    config_keys = set(cfg.keys())
    config.update({"cfg": cfg, "config": cfg})

    init_params = inspect.signature(method).parameters
    params = get_default_args(method)

    init_keys = set(init_params.keys())
    if not init_keys >= config_keys and not (init_keys == {"args", "kwargs"}):
        raise ValueError(
            f"Config for {method.__name__} contains invalid or outdated parameters! {config_keys} -> {init_keys}"
        )

    for arg in init_params.keys():
        if arg in config:
            params[arg] = config[arg]

    info = f"Set params for {method.__name__}({', '.join(init_params.keys())})"
    for key, field in params.items():
        info = info.replace(key, f"{key}={field}")

    return functools.partial(method, **params)


def init_class_from_config(cls, cfg: tp.Dict) -> tp.Callable:
    config = copy.deepcopy(cfg)
    config_keys = {k for k in cfg.keys() if k not in ["type"]}
    config.update({"cfg": cfg, "config": cfg})

    init_params = inspect.signature(cls.__init__).parameters
    params = {arg: config[arg] for arg in init_params.keys() if arg in config}

    init_keys = set(init_params.keys())
    if (
        "cfg" not in init_keys
        and "pipe" not in config_keys
        and not init_keys >= config_keys
    ):
        raise ValueError(
            f"Config for {cls.__name__} contains invalid or outdated parameters! {config_keys} -> {init_keys}"
        )

    info = ", ".join(init_params.keys())
    for key, field in params.items():
        info = info.replace(key, f"{key}={field}")
    # logger.info(f"Set params for {cls.__name__}({info})")

    return functools.partial(cls, **params)
