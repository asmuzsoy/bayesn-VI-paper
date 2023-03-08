import numpy as np
from timeit import default_timer
from bayesn_model import SEDmodel

model = SEDmodel()
start = default_timer()
model.simulate_spectrum(np.arange(-8, 40, 4), 100, mu='z', z=np.random.uniform(0, 0.1, 100), eps=np.random.normal(0, 1, (6, 6)))
end = default_timer()
print(end - start)
