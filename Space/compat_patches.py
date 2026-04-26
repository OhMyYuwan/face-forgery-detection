"""
Compatibility patches for InternViT and MambaVision

Fixes:
1. NEAREST_EXACT interpolation mode for older PIL/torchvision
2. mamba_ssm compatibility with newer transformers
3. Missing attributes in custom model classes
"""

import sys


def patch_interpolation_mode():
    """Fix NEAREST_EXACT for older torchvision versions"""
    try:
        from torchvision.transforms import InterpolationMode

        # Check if NEAREST_EXACT exists
        if not hasattr(InterpolationMode, 'NEAREST_EXACT'):
            # Add NEAREST_EXACT as alias to NEAREST
            InterpolationMode.NEAREST_EXACT = InterpolationMode.NEAREST
            print("✅ Patched InterpolationMode.NEAREST_EXACT")
    except Exception as e:
        print(f"⚠️  Could not patch InterpolationMode: {e}")


def patch_mamba_ssm():
    """Fix mamba_ssm compatibility with newer transformers"""
    try:
        # Fix: transformers uses importlib.metadata to get numpy version,
        # which returns None in this env. Patch it before importing transformers.
        import importlib.metadata as _meta
        _real_version = _meta.version

        def _patched_version(pkg):
            if pkg == "numpy":
                import numpy
                return numpy.__version__
            return _real_version(pkg)

        _meta.version = _patched_version

        import transformers.generation

        if not hasattr(transformers.generation, 'GreedySearchDecoderOnlyOutput'):
            # Create dummy class
            class GreedySearchDecoderOnlyOutput:
                pass

            transformers.generation.GreedySearchDecoderOnlyOutput = GreedySearchDecoderOnlyOutput
            print("✅ Patched transformers.generation.GreedySearchDecoderOnlyOutput")

        if not hasattr(transformers.generation, 'SampleDecoderOnlyOutput'):
            class SampleDecoderOnlyOutput:
                pass

            transformers.generation.SampleDecoderOnlyOutput = SampleDecoderOnlyOutput
            print("✅ Patched transformers.generation.SampleDecoderOnlyOutput")

    except Exception as e:
        print(f"⚠️  Could not patch mamba_ssm: {e}")


def patch_pretrained_model():
    """Patch PreTrainedModel to add missing attributes before load_state_dict"""
    try:
        from transformers import PreTrainedModel

        # Save original __init__
        original_init = PreTrainedModel.__init__

        def patched_init(self, config, *args, **kwargs):
            # Call original init
            original_init(self, config, *args, **kwargs)

            # Add missing attributes - all_tied_weights_keys should be a dict, not a list!
            if not hasattr(self, 'all_tied_weights_keys'):
                self.all_tied_weights_keys = {}
            if not hasattr(self, '_tied_weights_keys'):
                self._tied_weights_keys = []

        PreTrainedModel.__init__ = patched_init
        print("✅ Patched PreTrainedModel.__init__ for missing attributes")

    except Exception as e:
        print(f"⚠️  Could not patch PreTrainedModel: {e}")


def apply_all_patches():
    """Apply all compatibility patches"""
    print("🔧 应用兼容性补丁...")
    patch_interpolation_mode()
    patch_mamba_ssm()
    patch_pretrained_model()
    print("✅ 补丁应用完成\n")


# Auto-apply patches when imported
apply_all_patches()
