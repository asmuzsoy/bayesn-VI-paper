import os
import sncosmo

filt_map = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1', 'X': 'p48g', 'Y': 'p48r'}

for file in os.listdir('data/lcs/YSE_DR1_SNR4'):
    meta, lcdata = sncosmo.read_snana_ascii(f'data/lcs/YSE_DR1_SNR4/{file}', default_tablename='OBS')
    data = lcdata['OBS'].to_pandas()
    print(meta)
    print(meta['SPEC_CLASS'], meta['SPEC_CLASS_BROAD'])
    break
