from astropy.stats import median_absolute_deviation, sigma_clip
import numpy as np

def turboseti_equiv(np_cc):
    # This is equivalent to what turboseti does
    # Process here (comp_stats): https://github.com/UCBerkeleySETI/turbo_seti/blob/7d9b4fde9bc98d834dc11cfc0acd2380e6676f0e/turbo_seti/find_doppler/helper_functions.py#L101
    # Applied here: https://github.com/UCBerkeleySETI/turbo_seti/blob/7d9b4fde9bc98d834dc11cfc0acd2380e6676f0e/turbo_seti/find_doppler/find_doppler.py#L663
    center_bin = np_cc.shape[1]//2
    ts_arr = np_cc.mean(axis=0)
    ts_arr[center_bin] = ts_arr[center_bin-1] + ts_arr[center_bin+1]
    low, median, high = np.percentile(ts_arr, [5, 50, 95])
    drop_high = ts_arr[ts_arr <= high]
    drop_outliers = drop_high[drop_high >= low]
    return {
            'power': np.std(drop_outliers),
            'floor': median
    }

def setigen_equiv(np_cc):
    # This is how setigen calculates SNR for signal injection purposes
    # reference: https://github.com/bbrzycki/setigen/blob/eea6844041050e874083dfa91d463eb104375faa/setigen/normalize.py#L7
    clipped_data = sigma_clip(np_cc)
    return {
            'power': np.std(clipped_data),
            'floor': np.mean(clipped_data)
    }

