"""Modular GPU world model of prosthetic vision (retinal + cortical)."""

_EXPORTS = {
    "generate_dataset": ("sentionaut.generate", "generate_dataset"),
    "generate_world_dataset": ("sentionaut.generate", "generate_world_dataset"),
    "plot_percept": ("sentionaut.visualize", "plot_percept"),
    "plot_world_sequence": ("sentionaut.visualize", "plot_world_sequence"),
    "get_device": ("sentionaut.core.device", "get_device"),
    "build_components": ("sentionaut.core.registry", "build_components"),
    "Config": ("sentionaut.core.config", "Config"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    module = __import__(module_name, fromlist=[attr])
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
