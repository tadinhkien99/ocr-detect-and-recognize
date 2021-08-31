from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

def build_post_process(config, global_config=None):
    from .db_postprocess import DBPostProcess
    from .rec_postprocess import CTCLabelDecode

    support_dict = [
        'DBPostProcess', 'CTCLabelDecode',
    ]

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class