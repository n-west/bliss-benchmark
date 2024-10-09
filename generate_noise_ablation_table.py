
import h5py
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
import copy

import scipy.stats as spstat
from astropy.stats import median_absolute_deviation, sigma_clip
from other_noise_estimators import turboseti_equiv, setigen_equiv
from pprint import pp

import json
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Read a JSON file and execute the program")

# Add the argument for the JSON file path
parser.add_argument("noise_files", type=str, help="Path to the JSON file")
parser.add_argument("--dev", type=str, default="cuda:1", help="device to run bliss on")
parser.add_argument("--path", type=str, default="../bliss/build/bliss/python", help="directory to append to python sys path")

# Parse the arguments
args = parser.parse_args()

try:
    import blissdedrift
except:
    print("Failed to install bliss from installed module. Attempting build directory development module")
    import sys
    sys.path.append(args.path)
    import blissdedrift

# Read and decode the JSON file
with open(args.noise_files, 'r') as file:
    test_file_md = json.load(file)


noise_ablations = {
    "baseline": {
        "latex_columns": " baseline        &            &            &            &            &"
    },
    "ts_equiv": {
        "latex_columns": " turboseti       &            &            &            &            &"
    },
    "setigen_equiv": {
        "latex_columns": " setigen         &            &            &            &            &"
    },
    "baseline_slice": {
        "latex_columns": " noise slice     &            &            &            &            &"
    },
    "bliss_unflagged": {
        "latex_columns": " bliss base      &            &            &            &            &"
    },
    "bliss_rolloff": {
        "latex_columns": "                 & \\checkmark &            &            &            &"
    },
    "bliss_sk": {
        "latex_columns": "                 &            & \\checkmark &            &            &"
    },
    "bliss_sigmaclip": {
        "latex_columns": "                 &            &            & \\checkmark &            &"
    },
    "bliss_sk_sigmaclip": {
        "latex_columns": "                 &            & \\checkmark & \\checkmark &            &"
    },
    "corrected_slice": {
        "latex_columns": " corrected slice &            &            &            & \\checkmark &"
    },
    "bliss_corrected_unflagged": {
        "latex_columns": "                 &            &            &            & \\checkmark &"
    },
    "bliss_corrected_rolloff": {
        "latex_columns": "                 & \\checkmark &            &            & \\checkmark &"
    },
    "bliss_corrected_sk": {
        "latex_columns": "                 &            & \\checkmark &            & \\checkmark &"
    },
    "bliss_corrected_sigmaclip": {
        "latex_columns": "                 &            &            & \\checkmark & \\checkmark &"
    },
    "bliss_corrected_sk_sigmaclip": {
        "latex_columns": "                 &            & \\checkmark & \\checkmark & \\checkmark &"
    },
}

for ex_key, md in test_file_md.items():
    fil_path = md["path"]
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    np_cc = np.copy(np.from_dlpack(cc.data))
    np_cc_setigen = np.copy(np.from_dlpack(cc.data))
    noise_ablations["baseline"][md["name"]] = {
            'power': np.std(np_cc),
            'floor': np.mean(np_cc)
    }
    # This is a standard measurement of a "signal free" slice of data
    np_slice = np.copy(np_cc[:,md["noise_slice"]["lower"]:md["noise_slice"]["upper"]])
    noise_ablations["baseline_slice"][md["name"]] = {
            'power': np.std(np_slice),
            'floor': np.mean(np_slice)
    }

    noise_ablations["ts_equiv"][md["name"]] = turboseti_equiv(np_cc)
    ## This is how turboseti calculates SNR for signal injection purposes
    noise_ablations["setigen_equiv"][md["name"]] = setigen_equiv(np_cc_setigen)

    # Run bliss estimator without any preprocess/flagging
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations["bliss_unflagged"][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging the rolloff
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    cc = blissdedrift.flaggers.flag_filter_rolloff(cc, .25)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_rolloff'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging SK
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    cc = blissdedrift.flaggers.flag_spectral_kurtosis(cc, .05, 25)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_sk'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging sigmaclip
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    cc = blissdedrift.flaggers.flag_sigmaclip(cc, 3, 4, 5)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_sigmaclip'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging SK + sigmaclip
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    cc = blissdedrift.flaggers.flag_spectral_kurtosis(cc, .05, 25)
    cc = blissdedrift.flaggers.flag_sigmaclip(cc, 3, 4, 5)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_sk_sigmaclip'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    #########################
    ## use pfb correction / compensation
    #########################

    # Run bliss estimator with preprocess pfb correction, no flagging
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    if "pfb_shape" in md:
        cc = blissdedrift.preprocess.equalize_passband_filter(cc, md["pfb_shape"])
    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_corrected_unflagged'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }
    np_cc = np.from_dlpack(cc.data.to("cpu"))
    np_slice = np_cc[:,md["noise_slice"]["lower"]:md["noise_slice"]["upper"]]
    noise_ablations["corrected_slice"][md["name"]] = {
            'power': np.std(np_slice),
            'floor': np.mean(np_slice)
    }

    # Run bliss estimator without any preprocess, flagging the rolloff
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    if "pfb_shape" in md:
        cc = blissdedrift.preprocess.equalize_passband_filter(cc, md["pfb_shape"])
    cc = blissdedrift.flaggers.flag_filter_rolloff(cc, .25)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_corrected_rolloff'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging SK
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    if "pfb_shape" in md:
        cc = blissdedrift.preprocess.equalize_passband_filter(cc, md["pfb_shape"])
    cc = blissdedrift.flaggers.flag_spectral_kurtosis(cc, .05, 25)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_corrected_sk'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging sigmaclip
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    if "pfb_shape" in md:
        cc = blissdedrift.preprocess.equalize_passband_filter(cc, md["pfb_shape"])
    cc = blissdedrift.flaggers.flag_sigmaclip(cc, 3, 4, 5)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_corrected_sigmaclip'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }

    # Run bliss estimator without any preprocess, flagging SK + sigmaclip
    sc = blissdedrift.scan(fil_path, md["nfpc"])
    cc = sc.read_coarse_channel(md["coarse_channel"])
    cc.set_device(args.dev)

    if "pfb_shape" in md:
        cc = blissdedrift.preprocess.equalize_passband_filter(cc, md["pfb_shape"])
    cc = blissdedrift.flaggers.flag_spectral_kurtosis(cc, .05, 25)
    cc = blissdedrift.flaggers.flag_sigmaclip(cc, 3, 4, 5)

    noise_est_options = blissdedrift.estimators.noise_power_estimate_options()
    noise_est_options.masked_estimate = True
    cc_noise_est = blissdedrift.estimators.estimate_noise_power(cc, noise_est_options)
    noise_ablations['bliss_corrected_sk_sigmaclip'][md["name"]] = {
        'power': cc_noise_est.noise_power,
        'floor': cc_noise_est.noise_floor
    }


pp(noise_ablations)

true_baseline = "corrected_slice"
normalizing_values = copy.deepcopy(noise_ablations[true_baseline])

for estimator, estimator_data in noise_ablations.items():
    for test_case in estimator_data.keys():
        if test_case != "latex_columns":
            estimator_data[test_case]["power"] /= normalizing_values[test_case]["power"]
            estimator_data[test_case]["floor"] /= normalizing_values[test_case]["floor"]


# latex_table = " &            &            &            &            &"
latex_table = " description    & rolloff    & spec kurt  & sigmaclip  & pfb correct&"
floor_latex_table = " description    & rolloff    & spec kurt  & sigmaclip  & pfb correct&"

table_keys = "ABCDEFGHIJKLMNOPQRSTUVWXY"
ind = 0
files_key = ""
for k in noise_ablations[true_baseline].keys():
    if k != "latex_columns":
        latex_table += f"& {table_keys[ind]}     "
        floor_latex_table += f"& {table_keys[ind]}     "
        files_key += f"{table_keys[ind]}: {k}\n"
        ind += 1
latex_table += "\\\\\n"
floor_latex_table += "\\\\\n"

for estimator, estimator_data in noise_ablations.items():
    latex_table += f"{estimator_data['latex_columns']}"
    floor_latex_table += f"{estimator_data['latex_columns']}"
    for test_case in estimator_data.keys():
        if test_case != "latex_columns":
            latex_table += f"& {estimator_data[test_case]['power']:.3f} "
            floor_latex_table += f"& {estimator_data[test_case]['floor']:.3f} "
    latex_table += "\\\\\n"
    floor_latex_table += "\\\\\n"

print(latex_table)
print(floor_latex_table)
print(files_key)

