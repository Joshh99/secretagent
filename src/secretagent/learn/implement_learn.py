"""Implementation.Factory that loads a learned function from a training directory."""

import importlib.util
from glob import glob
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from secretagent import config
from secretagent.core import Interface, Implementation, _FACTORIES, register_factory


def _load_learned_module(learned_path):
    """Import a learned.py file and return the module."""
    spec = importlib.util.spec_from_file_location('learned', learned_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_learned_path(interface_name, learner):
    """Find the most recent learned.py matching the interface and learner."""
    train_dir = config.require('learn.train_dir')
    pattern = str(Path(train_dir) / f'*{interface_name}__{learner}' / 'learned.py')
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f'no learned implementation found matching {pattern}')
    return Path(matches[-1])


def _build_backoff_impl(interface, workdir):
    """Build a backoff implementation from source config yamls.

    Reads each yaml in workdir/source_configs/*.yaml, extracts the
    config for ptools.INTERFACE, verifies they all agree, and builds
    the implementation.
    """
    cfg_files = sorted(workdir.glob('source_configs/*.yaml'))
    if not cfg_files:
        raise FileNotFoundError(
            f'no source config files found in {workdir / "source_configs"}')
    impl_cfg = None
    for cfg_file in cfg_files:
        yaml_cfg = OmegaConf.load(cfg_file)
        ptools_cfg = OmegaConf.to_container(yaml_cfg.get('ptools', {}), resolve=True)
        iface_cfg = ptools_cfg.get(interface.name)
        if iface_cfg is None:
            raise ValueError(
                f'{cfg_file} has no ptools.{interface.name} config')
        if impl_cfg is None:
            impl_cfg = iface_cfg
        elif impl_cfg != iface_cfg:
            raise ValueError(
                f'conflicting ptools.{interface.name} configs across source yamls: '
                f'{impl_cfg} vs {iface_cfg}')
    factory_kws = dict(impl_cfg)
    method = factory_kws.pop('method')
    factory = _FACTORIES[method]
    return factory.build_implementation(interface, **factory_kws)


class LearnedFunctionFactory(Implementation.Factory):
    """Load a learned implementation from a training directory.

    At build time, finds the most recent timestamped directory matching
    TRAIN_DIR/*INTERFACE__LEARNER/learned.py, imports the function named
    INTERFACE from it, and binds that as the implementation.

    If backoff=True, also builds the original implementation from the
    source config yamls. The bound function calls the learned function
    first, falling back to the backoff implementation when the learned
    function returns None.

    Examples:
      foo.implement_via('learned', learner='rote')
      foo.implement_via('learned', learner='rote', backoff=True)
    """

    def build_fn(self, interface: Interface, learner: str,  # type: ignore[override]
                 backoff: bool = False, **_kw) -> Callable:
        learned_path = _find_learned_path(interface.name, learner)
        mod = _load_learned_module(learned_path)
        fn = getattr(mod, interface.name, None)
        if fn is None:
            raise AttributeError(
                f'{learned_path} does not define a function named {interface.name!r}')
        if not backoff:
            return fn
        workdir = learned_path.parent
        backoff_impl = _build_backoff_impl(interface, workdir)
        def learned_with_backoff(*args, **kw):
            result = fn(*args, **kw)
            if result is None:
                return backoff_impl.implementing_fn(*args, **kw)
            return result
        return learned_with_backoff

register_factory('learned', LearnedFunctionFactory())
