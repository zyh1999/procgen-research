#!/usr/bin/env python3
import importlib.util
from pathlib import Path

path = Path(__file__).with_name('preflight_full_shared_joint2b_production_shape_recovery.py')
spec = importlib.util.spec_from_file_location('task40_preflight', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
image_size, model_shape, layout = module.resolve_production_observation_semantics((64, 64, 3))
assert image_size == 64 and model_shape == (3, 64, 64) and layout == 'HWC_to_CHW'
rejected = module.run_shape_negative_tests((64, 64, 3))
assert set(rejected) == {'missing_height_width', 'channels_first_swapped',
                         'nonproduction_spatial', 'wrong_channel_count',
                         'channel_as_image_size'}
print('TASK40_SHAPE_SEMANTICS_PASS')
