# -*- coding utf-8 -*-

import os
import json
# from collections import OrderedDict


class Parameters:

    def __init__(self):
        pass

    def get_params(self):

        return self.__dict__

    def save_params(self, save_dir):

        # save_dir: str, e.g. './parameters.json'

        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        json.dump(self.__dict__, open(save_dir, 'w'), indent=2)