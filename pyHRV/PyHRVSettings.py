class PyHRVDefaultSettings:
    def __init__(self):
        pass

    default_interpolation_freq = 4

    class Files:
        def __init__(self):
            pass

        csv_separator = '\t'
        load_rr_column_name = 'IBI'

    class StandardBands:
        def __init__(self):
            pass

        vlf_lower_bound = 0.00001
        vlf_upper_bound = 0.04
        lf_upper_bound = 0.15
        hf_upper_bound = 0.4

    class TimeUnitCheck:
        def __init__(self):
            pass

        time_unit_check_ibi = True
        time_unit_check_ibi_warn =\
            "Warning: the inter-beat-time (IBI) should be loaded in ms (milli-seconds). The mean is %d!"
        time_unit_check_ibi_mean_min = 80
        time_unit_check_ibi_mean_max = 1200

    class NonLinearIndexes:
        def __init__(self):
            pass

        sample_entropy_r = 0.2
        approx_entropy_r = sample_entropy_r
        correlation_dimension_len = 10
