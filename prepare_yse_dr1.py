import os

import matplotlib.pyplot as plt
import sncosmo
import glob
import dataclasses
import pandas as pd
from astropy import constants
from astropy.io import ascii
from astropy.table import Table
import numpy as np
from numpy import sin, cos, hypot, arctan2
import pickle
from bayesn_model import write_snana_lcfile

filt_map = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1', 'X': 'p48g', 'Y': 'p48r'}

# Functions from Patrick Aleo--------
REDSHIFT_UNKNOWN = -99.0

@dataclasses.dataclass
class Observation:
    MJD: float
    PASSBAND: str
    FLUX: float
    FLUXERR: float
    MAG: float
    MAGERR: float
    PHOTFLAG: str

def read_YSE_ZTF_snana_dir(dir_name, keep_ztf=True):
    """
    file_path : str
        The file path to the combined YSE+ZTF light curve SNANA-style format data file.
    keep_ztf : bool
        True: Plots including ZTF data
        False : Plots not include ZTF data

    """

    snid_list = []
    meta_list = []
    yse_ztf_fp_df_list = []

    for file_path in sorted(glob.glob(dir_name + '/*')):
        # print(file_path)

        meta = {}
        lc = []
        with open(file_path) as file:
            for line in file:
                try:
                    # SNID
                    if line.startswith('SNID: '):
                        _, snid = line.split()
                        meta['object_id'] = snid
                        meta['original_object_id'] = snid

                    # RA
                    if line.startswith('RA: '):
                        _, ra, _ = line.split()
                        meta['ra'] = float(ra)

                    # DEC
                    if line.startswith('DECL: '):
                        _, decl, _ = line.split()
                        meta['dec'] = float(decl)

                    # MWEBV
                    if line.startswith('MWEBV: '):
                        _, mwebv, _, _mwebv_error, *_ = line.split()
                        meta['mwebv'] = float(mwebv)

                    # REDSHIFT
                    if line.startswith('REDSHIFT_FINAL: '):
                        try:
                            _, redshift, _, _redshift_error, _z_type, _z_frame = line.split()
                        # 2020roe has empty redshift
                        except ValueError:
                            redshift = -99
                            redshift_err = -99
                            _z_type = 'NaN'
                            _z_frame = 'HELIO'
                        meta['redshift'] = float(redshift)
                        meta['redshift_err'] = float(_redshift_error)
                        meta['redshift_type'] = str(_z_type.split('(')[1].split(',')[0])
                        meta['redshift_frame'] = str(_z_frame.split(')')[0])

                    # PHOTO-Z
                    if line.startswith('PHOTO_Z: '):
                        try:
                            _, photoz, _, _photoz_error, _, _ = line.split()
                        except ValueError:
                            photoz = -99
                            _photoz_error = -99
                        meta['photo_z'] = float(photoz)
                        meta['photoz_err'] = float(_photoz_error)

                    # HOST INFO
                    if line.startswith('SN_OFFSET_TO_VETTED_HOST_GALAXY_CENTER: '):
                        try:
                            _, sn_offset, _ = line.split()
                        except ValueError:
                            sn_offset = -99.000
                        meta['sn_offset'] = float(sn_offset)

                    if line.startswith('VETTED_HOST_GALAXY_NAME: '):
                        try:
                            _, host_gal_name_cat, host_gal_name_id, host_gal_name_source = line.split()
                            host_gal_name = str(host_gal_name_cat) + ' ' + str(host_gal_name_id)
                        except ValueError:
                            host_gal_name = 'None (or error)'
                            host_gal_name_source = '(NED)'
                        meta['host_gal_name'] = host_gal_name
                        meta['host_gal_name_source'] = str(host_gal_name_source)

                    if line.startswith('VETTED_HOST_GALAXY_REDSHIFT: '):
                        try:
                            _, hostz, _, _hostz_error, _hostz_type, _hostz_frame = line.split()
                        except ValueError:
                            hostz = -99
                            hostz_err = -99
                            _hostz_type = 'NaN'
                            _hostz_frame = 'HELIO'
                        meta['host_gal_z'] = float(hostz)
                        meta['host_gal_z_err'] = float(_hostz_error)
                        meta['host_gal_z_type'] = str(_hostz_type.split('(')[1].split(',')[0])
                        meta['host_gal_z_frame'] = str(_hostz_frame.split(')')[0])

                    # PEAKMJD
                    if line.startswith('SEARCH_PEAKMJD: '):
                        _, pkmjd = line.split()
                        meta['peakmjd'] = search_peakmjd = float(pkmjd)

                    # HOST LOGMASS
                    if line.startswith('HOST_LOGMASS: '):
                        _, host_logmass, _, host_logmass_error = line.split()
                        meta['host_logmass'] = float(host_logmass)

                    # PEAK ABS MAG
                    if line.startswith('PEAK_ABS_MAG: '):
                        _, pkabsmag = line.split()

                        try:
                            meta['peak_abs_mag'] = peak_abs_mag = float(pkabsmag)
                        except:  # For NA
                            meta['peak_abs_mag'] = peak_abs_mag = str(pkabsmag)

                    # SPEC CLASS
                    if line.startswith('SPEC_CLASS: '):
                        try:
                            _, sn, spec_subtype = line.split()
                            meta['transient_spec_class'] = transient_spec_class = str(sn + spec_subtype)
                        except:
                            _, spec_subtype = line.split()
                            meta['transient_spec_class'] = transient_spec_class = str(spec_subtype)

                    # SPEC CLASS BROAD
                    if line.startswith('SPEC_CLASS_BROAD: '):
                        try:
                            _, sn, subtype = line.split()
                            meta['spectype_3class'] = spectype_3class = str(sn + subtype)
                        except:
                            _, subtype = line.split()
                            meta['spectype_3class'] = spectype_3class = str(subtype)

                    # PARSNIP PRED
                    if line.startswith('PARSNIP_PRED: '):
                        try:
                            _, sn, p_pred = line.split()
                            meta['parsnip_pred_class'] = parsnip_pred_class = str(sn + p_pred)
                        except:
                            _, p_pred = line.split()  # for "NA" Prediction
                            meta['parsnip_pred_class'] = parsnip_pred_class = str(p_pred)

                    # PARSNIP CONF
                    if line.startswith('PARSNIP_CONF: '):
                        _, p_conf = line.split()
                        meta['parsnip_pred_conf'] = parsnip_pred_conf = str(p_conf)

                    # PARSNIP S1
                    if line.startswith('PARSNIP_S1: '):
                        _, s1, _, s1_error = line.split()
                        try:
                            meta['parsnip_s1'] = float(s1)
                            meta['parsnip_s1_err'] = float(s1_error)
                        except:  # NA
                            meta['parsnip_s1'] = str(s1)
                            meta['parsnip_s1_err'] = str(s1_error)

                    # PARSNIP S2
                    if line.startswith('PARSNIP_S2: '):
                        _, s2, _, s2_error = line.split()
                        try:
                            meta['parsnip_s2'] = float(s2)
                            meta['parsnip_s2_err'] = float(s2_error)
                        except:  # NA
                            meta['parsnip_s2'] = str(s2)
                            meta['parsnip_s2_err'] = str(s2_error)

                    # PARSNIP S3
                    if line.startswith('PARSNIP_S3: '):
                        _, s3, _, s3_error = line.split()
                        try:
                            meta['parsnip_s3'] = float(s3)
                            meta['parsnip_s3_err'] = float(s3_error)
                        except:  # NA
                            meta['parsnip_s3'] = str(s3)
                            meta['parsnip_s3_err'] = str(s3_error)

                    # SUPERPHOT PRED
                    if line.startswith('SUPERPHOT_PRED: '):
                        try:
                            _, sn, s_pred = line.split()
                            meta['superphot_pred_class'] = superphot_pred_class = str(sn + s_pred)
                        except:
                            _, s_pred = line.split()  # for "NA" Prediction
                            meta['superphot_pred_class'] = superphot_pred_class = str(s_pred)

                    # SUPERPHOT CONF
                    if line.startswith('SUPERPHOT_CONF: '):
                        _, s_conf = line.split()
                        meta['superphot_pred_conf'] = superphot_pred_conf = str(s_conf)

                    # SUPERRAENN PRED
                    if line.startswith('SUPERRAENN_PRED: '):
                        try:
                            _, sn, sr_pred = line.split()
                            meta['superraenn_pred_class'] = superraenn_pred_class = str(sn + sr_pred)
                        except:
                            _, sr_pred = line.split()  # for "NA" Prediction
                            meta['superraenn_pred_class'] = superraenn_pred_class = str(sr_pred)

                    # SUPERRAENN CONF
                    if line.startswith('SUPERRAENN_CONF: '):
                        _, sr_conf = line.split()
                        meta['superraenn_pred_conf'] = superraenn_pred_conf = str(sr_conf)

                        # ZTF ZEROPOINT
                    if line.startswith('SET_ZTF_FP: '):
                        _, ztf_fp = line.split()
                        try:
                            meta['ztf_zeropoint'] = float(ztf_fp)
                        except:
                            meta['ztf_zeropoint'] = str(ztf_fp)

                    # PEAKMJD
                    if line.startswith('PEAK_SNR: '):
                        _, pkSNR = line.split()
                        meta['peakSNR'] = float(pkSNR)

                    # MAX MJD GAP
                    if line.startswith('MAX_MJD_GAP(days): '):
                        _, max_mjd_gap = line.split()
                        meta['max_mjd_gap'] = float(max_mjd_gap)

                    # NOBS BEFORE PEAK
                    if line.startswith('NOBS_BEFORE_PEAK: '):
                        _, nobs_before_peak = line.split()
                        meta['nobs_before_peak'] = int(nobs_before_peak)

                        # NOBS TO THE PEAK OBS (ANY BAND)
                    if line.startswith('NOBS_TO_PEAK: '):
                        _, nobs_to_peak = line.split()
                        meta['nobs_to_peak'] = int(nobs_to_peak)

                        # NOBS AFTER PEAK
                    if line.startswith('NOBS_AFTER_PEAK: '):
                        _, nobs_after_peak = line.split()
                        meta['nobs_after_peak'] = int(nobs_after_peak)

                        # PEAK MAGNITUDE
                    if line.startswith('SEARCH_PEAKMAG: '):
                        _, pkmag = line.split()
                        meta['peakmag'] = search_peakmag = float(pkmag)

                    # PEAK FILTER (PASSBAND OF OBS w/ PEAK MAG OBS)
                    if line.startswith('SEARCH_PEAKFLT: '):
                        _, pkflt = line.split()
                        meta['peakflt'] = search_peakflt = str(pkflt)

                    # PEAK MAGNITUDE YSE-r or ZTF-r (Y) band for mag lim sample!
                    if line.startswith('PEAKMAG_YSE-r/ZTF-r(Y): '):
                        _, pkmag_rY = line.split()
                        meta['peakmag_rY'] = search_peakmag_rY = float(pkmag_rY)

                    # PEAK FILTER of YSE-r or ZTF-r (Y) band peak mag
                    if line.startswith('PEAKFLT_YSE-r/ZTF-r(Y): '):
                        _, pkflt_rY = line.split()
                        meta['peakflt_rY'] = search_peakflt_rY = str(pkflt_rY)

                    # FILTERS/PASSBANDS
                    if line.startswith('FILTERS: '):
                        _, pbs = line.split()
                        meta['passbands'] = passbands = str(pbs)

                    # TOTAL OBS
                    if line.startswith('NOBS_wZTF: ') or line.startswith('NOBS_AFTER_MASK: '):
                        _, desired_nobs = line.split()
                        meta['num_points'] = int(desired_nobs)
                        continue


                except ValueError as e:
                    print(e)
                    print(meta['object_id'])
                    raise e

                if not line.startswith('OBS: '):
                    continue

                _obs, mjd, flt, _field, fluxcal, fluxcalerr, mag, magerr, _flag = line.split()
                lc.append(Observation(
                    MJD=float(mjd),
                    PASSBAND=str(flt),
                    FLUX=float(fluxcal),
                    FLUXERR=float(fluxcalerr),
                    MAG=float(mag),
                    MAGERR=float(magerr),
                    PHOTFLAG=str(_flag))
                )

        meta.setdefault('mwebv', 0.0)

        # assert len(meta) == 13, f'meta has wrong number of values,\nmeta = {meta}'
        assert len(lc) == meta['num_points']
        table = Table([dataclasses.asdict(obs) for obs in lc if keep_ztf])  # or obs.FLT not in ZTF_BANDS])

        yse_ztf_fp_df = table.to_pandas()

        snid_list.append(snid)
        meta_list.append(meta)
        yse_ztf_fp_df_list.append(yse_ztf_fp_df)

    return snid_list, meta_list, yse_ztf_fp_df_list


def get_param(meta_list, param):
    param_list = []

    for sn in meta_list:
        if param == 'peak_abs_mag':
            print(sn['object_id'], sn['transient_spec_class'])
            try:
                param_list.append(float(sn[param]))
            except:
                continue

        else:
            try:
                param_list.append(sn[param])
            except:
                print(f"WARNING: {param} not in parameter list. Check!")

    return param_list


def get_SNclass_param(meta_list, param, ifstr, ifprint=False):
    all_list, snII_list, snIa_list, snIbc_list, other_list = [], [], [], [], []

    for spec_sn in meta_list:

        if ifstr == True:
            try:
                all_list.append(str(spec_sn[param]))
            except:
                continue

            if spec_sn[param] == 'SNII':
                snII_list.append(str(spec_sn[param]))

            elif spec_sn[param] == 'SNIa':
                snIa_list.append(str(spec_sn[param]))

            elif spec_sn[param] == 'SNIbc':
                snIbc_list.append(str(spec_sn[param]))

            else:
                if ifprint == True: print(spec_sn[param])
                other_list.append(str(spec_sn[param]))

        else:
            try:
                all_list.append(float(spec_sn[param]))
            except:
                continue

            if spec_sn['spectype_3class'] == 'SNII':
                snII_list.append(float(spec_sn[param]))

            elif spec_sn['spectype_3class'] == 'SNIa':
                snIa_list.append(float(spec_sn[param]))

            elif spec_sn['spectype_3class'] == 'SNIbc':
                snIbc_list.append(float(spec_sn[param]))

            else:
                if ifprint == True: print(spec_sn['spectype_3class'])
                other_list.append(float(spec_sn[param]))

    return all_list, snII_list, snIa_list, snIbc_list, other_list


def zhel_to_zcmb(zhel, RA, Dec):
    c = 299792.458  # km/s
    v_Sun_Planck = 369.82
    d1, d2 = 264.021, 48.253  # Dipole coordinates
    RA_Sun_Planck = 167.816710  # deg
    Dec_Sun_Planck = -6.989510  # deg
    rad = np.pi / 180.0
    # using Vincenty formula because it is more accurate
    alpha = arctan2(
        hypot(
            cos(Dec_Sun_Planck * rad) * sin(np.fabs(RA - RA_Sun_Planck) * rad),
            cos(Dec * rad) * sin(Dec_Sun_Planck * rad)
            - sin(Dec * rad)
            * cos(Dec_Sun_Planck * rad)
            * cos(np.fabs(RA - RA_Sun_Planck) * rad),
        ),
        sin(Dec * rad) * sin(Dec_Sun_Planck * rad)
        + cos(Dec * rad)
        * cos(Dec_Sun_Planck * rad)
        * cos(np.fabs(RA - RA_Sun_Planck) * rad),
    )
    v_Sun_proj = v_Sun_Planck * np.cos(alpha)
    z_Sun = np.sqrt((1.0 + (-v_Sun_proj) / c) / (1.0 - (-v_Sun_proj) / c)) - 1.0
    # Full special rel. correction since it is a peculiar vel
    min_z = 0.0
    zcmb = np.where(zhel > min_z, (1 + zhel) / (1 + z_Sun) - 1, zhel)
    return zcmb
    #alpha = np.sqrt(np.power(RA - d1, 2) + np.power(Dec - d2, 2))
    #c = constants.c.value / 1e3
    #vsun = v_Sun_Planck * np.cos(alpha)
    #zsun = np.sqrt((1 - vsun / c) / (1 + vsun / c)) - 1
    #zcmb2 = ((1 + zhel) / (1 + zsun)) - 1
    #return zcmb


# ----------------
c = 299792.458
full_snid_list, full_meta_list, full_df_list = read_YSE_ZTF_snana_dir(dir_name='/Users/matt/Downloads/yse_dr1_zenodo_snr_geq_4',
                                                                      keep_ztf=True)

spec_type = np.array(get_param(meta_list=full_meta_list, param='transient_spec_class'))
Ia_inds = np.where(spec_type == 'SNIa-norm')[0]

Ia_snid_list = [full_snid_list[i] for i in Ia_inds]
Ia_meta_list = [full_meta_list[i] for i in Ia_inds]
Ia_df_list = [full_df_list[i] for i in Ia_inds]

meta_list, table_list = [], []

good = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 122, 124, 125, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 192, 193, 195, 196, 197, 198, 200, 201, 202, 204, 205, 206, 207, 209, 210, 212, 213, 215, 216, 217, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 240, 241, 243, 244, 245, 246, 247, 248, 249, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288]
bad = [47, 52, 63, 96, 101, 107, 118, 121, 123, 126, 127, 141, 144, 188, 191, 194, 199, 203, 208, 211, 214, 221, 238, 239, 242, 250, 251, 273]
zs = []

vpec_table = pd.read_csv('YSE_DR1_vpec_output.txt', delim_whitespace=True)

bad_names = ['2020tfy', '2021aaqi', '2021aaxi', '2021acza', '2021adnv', '2021ita', '2021jox', '2021kcc', '2021lfv'] # Won't converge
high_Av = ['2019pmd', '2020aatr', '2020abvg', '2020acmi', '2020aeqm',
       '2020zfn', '2021aamo', '2021gez', '2021mgc', '2021tqq', '2021van',
       '2021vwx', '2021xmq']
bad_fits = ['2020abim']
bad_names = bad_names + high_Av + bad_fits

with open(os.path.join('results', 'YSE_fit', 'chains.pkl'), 'rb') as file:
    chains = pickle.load(file)
sn_list = np.load('data/lcs/pickles/YSE_DR1/sn_list.npy', allow_pickle=True)
tmax = chains['tmax'].mean(axis=(0, 1))
"""for i in range(263):
    if tmax[i] < -8 or tmax[i] > 8:
        for n in range(4):
            plt.hist(chains['tmax'][n, :, i])
        plt.title(sn_list[i])
        plt.show()
raise ValueError('Nope')"""
tmax_dict = {sn_list[i]: float(tmax[i]) for i in range(len(tmax))}

for i in range(len(Ia_snid_list)):
    #if i in good:
    #    continue
    sn, meta, df = Ia_snid_list[i], Ia_meta_list[i], Ia_df_list[i]
    if sn in bad_names:
        continue
    #df = df[df.PASSBAND.isin(['X', 'Y'])]
    #if df.empty:
    #    continue
    df = df[~df.PASSBAND.isin(['X', 'Y'])].copy()
    if df.empty:
        continue
    FLT = df.PASSBAND.apply(lambda flt: filt_map[flt])

    colour_dict = {'X': 'g', 'Y': 'r', 'g': 'g', 'r': 'g', 'i': 'b', 'z': 'k'}
    z_helio, z_helio_err = meta['redshift'], meta['redshift_err']
    z_cmb = zhel_to_zcmb(z_helio, meta['ra'], meta['dec'])
    v_pec = vpec_table['v_pec'].values[i]
    z_pec = np.sqrt((1 + v_pec / c) / (1 - v_pec / c)) - 1
    z_hd, z_hd_err = (1 + z_cmb) / (1 + z_pec) - 1, z_helio_err

    if z_hd < 0.015:  # Cut low redshift objects
        continue

    tmax = meta['peakmjd'] # - tmax_dict[sn] * (1 + z_hd)  # Correct peak MJD based on T21 fits

    df['phase'] = (df.MJD - tmax) / (1 + z_hd)
    fit_df = df[(df.phase > -10) & (df.phase < 40)]
    if fit_df.empty:
        continue
    #print(fit_df.PASSBAND.value_counts())
    #continue

    zs.append(z_cmb)
    #for filt in df.PASSBAND.unique():
    #    filt_df = df[df.PASSBAND == filt]
    #    plt.errorbar(filt_df.MJD, filt_df.MAG, yerr=filt_df.MAGERR, fmt='x', color=colour_dict[filt])
    #plt.vlines(meta['peakmjd'], df.MAG.min(), df.MAG.max())
    #plt.title(sn)
    #plt.gca().invert_yaxis()
    #plt.show()
    #continue
    write_snana_lcfile('data/lcs/YSE_DR1', sn, df.MJD, FLT, df.MAG, df.MAGERR, tmax, z_helio, z_hd,
                       z_hd_err, meta['mwebv'], ra=meta['ra'], dec=meta['dec'])
    meta_list.append([sn, meta['peakmjd'], z_cmb, z_hd_err])
    table_list.append([sn, 'YSE_DR1', f'{sn}.snana.dat'])

meta_list, table_list = np.array(meta_list), np.array(table_list)
meta = pd.DataFrame(meta_list, columns=['SNID', 'SEARCH_PEAKMJD', 'REDSHIFT_CMB', 'REDSHIFT_CMB_ERR'])
table = pd.DataFrame(table_list)
meta.to_csv('data/lcs/meta/YSE_DR1_meta.txt', sep='\t', index=False)
table.to_csv('data/lcs/tables/YSE_DR1_table.txt', header=False, sep='\t', index=False)
