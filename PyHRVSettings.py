class PyHRVDefaultSettings:
    def __init__(self):
        pass

    csv_separator = '\t'
    load_rr_column_names = 'IBI'
    # TODO: check bounds
    bands_vlf_lower_bound = 0.00001
    bands_vlf_upper_bound = 0.04
    bands_lf_upper_bound = 0.5
    bands_hf_upper_bound = 1