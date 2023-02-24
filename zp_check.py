import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open('data/zp/alpha_lyr_stis_008.fits') as hdu:
    head = hdu[0].header
    df = pd.DataFrame.from_records(hdu[1].data)
print(df)
plt.plot(df.WAVELENGTH, df.FLUX)
plt.xlim(9e2, 1e4)
plt.show()
