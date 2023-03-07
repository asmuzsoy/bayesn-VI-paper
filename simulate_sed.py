import os

import numpy as np
from timeit import default_timer

from model import Model

model = Model()
start = default_timer()
model.simulate_light_curve(np.arange(-8, 40, 4), 100, ['p48g', 'p48r', 'p48i'], mu='z', z=np.random.uniform(0, 0.1, 100))
end = default_timer()
print(end - start)
