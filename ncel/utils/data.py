"""Dataset handling and related yuck."""

import random
import itertools
import time
import sys
import struct

import numpy as np

class SimpleProgressBar(object):
    """ Simple Progress Bar and Timing Snippet
    """

    def __init__(self, msg=">", bar_length=80, enabled=True):
        super(SimpleProgressBar, self).__init__()
        self.enabled = enabled
        if not self.enabled:
            return

        self.begin = time.time()
        self.bar_length = bar_length
        self.msg = msg

    def step(self, i, total):
        if not self.enabled:
            return
        sys.stdout.write('\r')
        pct = (i / float(total)) * 100
        ii = i * self.bar_length // total
        fmt = "%s [%-{}s] %d%% %ds / %ds    ".format(self.bar_length)
        total_time = time.time() - self.begin
        expected = total_time / ((i + 1e-03) / float(total))
        sys.stdout.write(fmt % (self.msg, '=' * ii, pct, total_time, expected))
        sys.stdout.flush()

    def reset(self):
        if not self.enabled:
            return
        self.begin = time.time()

    def finish(self):
        if not self.enabled:
            return
        self.reset()
        sys.stdout.write('\n')