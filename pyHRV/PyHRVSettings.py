__all__ = ['PyHRVDefaultSettings']


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
        time_unit_check_ibi_warn = \
            "Warning: the inter-beat-time (IBI) should be loaded in ms (milli-seconds). The mean is %d!"
        time_unit_check_ibi_mean_min = 80
        time_unit_check_ibi_mean_max = 1200

    class NonLinearIndexes:
        def __init__(self):
            pass

        sample_entropy_r = 0.2
        approx_entropy_r = sample_entropy_r
        correlation_dimension_len = 10
        fractal_dimension_cra = 0.005
        fractal_dimension_crb = 0.75

    class TDIndexes:
        def __init__(self):
            pass

        nnx_default_threshold = 50

    class Local:
        def __init__(self):
            pass

        indexes_not_found =\
            "Some of the indexes requested are not implemented:" \
            "maybe the name is incorrect or pyHRV version is old.\n" \
            "Indexes names: %s"

        @staticmethod
        def names(phrase, names):
            return phrase.format(", ".join(names))
