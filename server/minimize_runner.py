import logging

import numpy as np
from optimisation.minimize import solver


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)  # silence matplotlib debug messages

    data = np.load('cut_coords.npy')
    solver(data)
