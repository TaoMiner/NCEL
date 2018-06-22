
import sys

import gflags

from ncel.models import entity_linking

from ncel.models.base import get_flags

FLAGS = gflags.FLAGS



if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    entity_linking.run(only_forward=FLAGS.expanded_eval_only_mode)
