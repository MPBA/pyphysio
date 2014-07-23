##ck3
__author__ = "AleB"
__all__ = ['PyHRVSettings', 'PyHRVDefaultSettings']


class PyHRVSettings(object):
    def __init__(self):
        # Interpolation
        """
        This constructor sets the default used values for the library.
        """
        self.default_interpolation_freq = 4
        self.online_step_milliseconds = 100

        # DataImports
        self.import_bvp_filters = []
        self.import_ecg_filters = []
        self.import_ecg_delta = 1
        self.import_bvp_delta_max_min_numerator = 50

        # Files
        self.load_csv_separator = '\t'
        self.load_rr_column_name = 'IBI'
        self.load_ecg_column_name = 'ECG'
        self.load_ecg_time_column_name = 'timestamp'
        self.load_bvp_column_name = 'BVP'
        self.load_bvp_time_column_name = 'timestamp'
        self.load_windows_col_begin = 'begin'
        self.load_windows_col_end = 'end'

        # TDIndexes:
        self.nnx_default_threshold = 50
        self.cache_histogram_bins = 100

        # FDIndexes - StandardBands:
        self.vlf_band_lower_bound = 0.00001
        self.vlf_band_upper_bound = 0.04
        self.lf_band_upper_bound = 0.15
        self.hf_band_upper_bound = 0.4

        # NonLinearIndexes:
        self.sample_entropy_r = 0.2
        self.approx_entropy_r = self.sample_entropy_r
        self.correlation_dimension_len = 10
        self.fractal_dimension_cra = 0.005
        self.fractal_dimension_crb = 0.75

        # Local:
        self.indexes_not_found = \
            "Some of the indexes requested are not implemented:" \
            "maybe the name is incorrect or pyHRV version is old.\n" \
            "Indexes names: {:s}"


PyHRVDefaultSettings = PyHRVSettings()
