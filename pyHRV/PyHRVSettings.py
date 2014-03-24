class PyHRVDefaultSettings:
    def __init__(self):
        pass

    csv_separator = '\t'
    load_rr_column_names = 'IBI'
    bands_vlf_lower_bound = 0.00001
    bands_vlf_upper_bound = 0.04
    bands_lf_upper_bound = 0.15
    bands_hf_upper_bound = 0.4
    interpolation_freq_default = 4

    class TimeUnitCheck(object):
        time_unit_check_ibi = True
        time_unit_check_ibi_warn =\
            "Warning: the inter-beat-time (IBI) should be loaded in ms (milli-seconds). The mean is %d!"
        time_unit_check_ibi_mean_min = 80
        time_unit_check_ibi_mean_max = 1200
