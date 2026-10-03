"""FasterWAM: the FastWAM world-action model with a built-in acceleration stack.

The model architecture and training path are unchanged from FastWAM. What is
new is `fasterwam.accel`, a first-class inference-acceleration stack that is on
by default on every inference path.

Submodules are imported explicitly to keep `import fasterwam` cheap:

    from fasterwam.runtime import prepare_for_inference
    from fasterwam.accel import accelerate, AccelConfig
"""

__version__ = "0.1.0"
