"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.add_argument('--status', type=str, default='test')
        parser.set_defaults(no_flip=True)
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        self.isTrain = False
        return parser
