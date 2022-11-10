import argparse

default_model = 'plasticc'

l_k = [3500, 4900, 6200, 7700, 8700, 9500]

default_settings = {
    'model_version': 2,

    'input_redshift': True,

    'predict_redshift': False,
    'specz_error': 0.01,

    'min_wave': l_k[0],
    'max_wave': l_k[-1],
    'l_k': l_k,
    'spectrum_bins': 300,
    'max_redshift': 4.,
    'band_oversampling': 51,
    'time_window': 300,
    'time_pad': 100,
    'time_sigma': 20.,
    'Rv_mean': 2.610,
    'Rv_sigma': 0.0,
    'Av_mean': 0.5,
    'Av_sigma': 0.25,
    'color_sigma': 0.3,
    'magsys': 'ab',
    'error_floor': 0.01,
    'zeropoint': 25.0,

    'batch_size': 128,
    'learning_rate': 1e-3,
    'scheduler_factor': 0.5,
    'min_learning_rate': 1e-7,
    'penalty': 1e-3,
    'optimizer': 'Adam',  # 'Adam' or 'SGD'
    'sgd_momentum': 0.9,

    'latent_size': 1,
    'encode_block': 'residual',
    'encode_conv_architecture': [40, 80, 120, 160, 200, 200, 200],
    'encode_conv_dilations': [1, 2, 4, 8, 16, 32, 64],
    'encode_fc_architecture': [200],
    'encode_time_architecture': [200],
    'encode_latent_prepool_architecture': [200],
    'encode_latent_postpool_architecture': [200],
    'decode_architecture': [40, 80, 160],

    # Settings that will be filled later.
    'derived_settings_calculated': None,
    'bands': None,
    'band_mw_extinctions': None,
    'band_correct_background': None,
}


def update_settings_version(settings):
    """Update settings to a new version

    Parameters
    ----------
    settings : dict
        Old settings

    Returns
    -------
    dict
        Updates settings
    """
    # Version 2, added redshift prediction.
    if settings['model_version'] < 2:
        settings['predict_redshift'] = False
        settings['specz_error'] = 0.05

    settings['model_version'] = default_settings['model_version']

    return settings


def parse_settings(bands, settings={}, ignore_unknown_settings=False):
    """Parse the settings for a ParSNIP model

    Parameters
    ----------
    bands : List[str]
        Bands to use in the encoder model
    settings : dict, optional
        Settings to override, by default {}
    ignore_unknown_settings : bool, optional
        If False (default), raise an KeyError if there are any unknown settings.
        Otherwise, do nothing.

    Returns
    -------
    dict
        Parsed settings dictionary

    Raises
    ------
    KeyError
        If there are unknown keys in the input settings
    """
    if 'derived_settings_calculated' in settings:
        # We are loading a prebuilt-model, don't recalculate everything.
        prebuilt_model = True
    else:
        prebuilt_model = False

    use_settings = default_settings.copy()
    use_settings['bands'] = bands

    for key, value in settings.items():
        if key not in default_settings:
            if ignore_unknown_settings:
                continue
            else:
                raise KeyError(f"Unknown setting '{key}' with value '{value}'.")
        else:
            use_settings[key] = value

    if use_settings['model_version'] != default_settings['model_version']:
        # Update the settings to the latest version
        use_settings = update_settings_version(use_settings)

    return use_settings


def parse_int_list(text):
    """Parse a string into a list of integers

    For example, the string "1,2,3,4" will be parsed to [1, 2, 3, 4].

    Parameters
    ----------
    text : str
        String to parse

    Returns
    -------
    List[int]
        Parsed integer list
    """
    result = [int(i) for i in text.split(',')]
    return result


def build_default_argparse(description):
    """Build an argparse object that can handle all of the ParSNIP model settings.

    The resulting parsed namespace can be passed to parse_settings to get a ParSNIP
    settings object.

    Parameters
    ----------
    description : str
        Description for the argument parser

    Returns
    -------
    `~argparse.ArgumentParser`
        Argument parser with the ParSNIP model settings added as arguments
    """
    parser = argparse.ArgumentParser(description=description)

    for key, value in default_settings.items():
        if value is None:
            # Derived setting, not something that should be specified.
            continue

        if isinstance(value, bool):
            # Handle booleans.
            if value:
                parser.add_argument(f'--no_{key}', action='store_false', dest=key)
            else:
                parser.add_argument(f'--{key}', action='store_true', dest=key)
        elif isinstance(value, list):
            # Handle lists of integers
            parser.add_argument(f'--{key}', type=parse_int_list, default=value)
        else:
            # Handle other object types
            parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser
