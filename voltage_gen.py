# Bryan's script for generating test

import setigen as stg
import blimpy as bl
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['SETIGEN_ENABLE_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


subsample_factor = 4

sample_rate = 3e9 // subsample_factor
num_taps = 8
num_branches = 1024 // subsample_factor
print(f'Max "num_chans" is {num_branches // 2}.')

fftlength = 1048576
int_factor = 51
num_blocks = 1

digitizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                      num_bits=8)

filterbank = stg.voltage.PolyphaseFilterbank(num_taps=num_taps,
                                             num_branches=num_branches)

requantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                           num_bits=8)

antenna = stg.voltage.Antenna(sample_rate=sample_rate,
                              fch1=6*u.GHz,
                              ascending=True,
                              num_pols=2)

block_size = stg.voltage.get_block_size(num_antennas=1,
                                        tchans_per_block=1,
                                        num_bits=8,
                                        num_pols=2,
                                        num_branches=num_branches,
                                        num_chans=num_branches//2,
                                        fftlength=fftlength,
                                        int_factor=int_factor)
block_size = 134217728

rvb = stg.voltage.RawVoltageBackend(antenna,
                                    digitizer=digitizer,
                                    filterbank=filterbank,
                                    requantizer=requantizer,
                                    start_chan=1,
                                    num_chans=1,
                                    block_size=block_size,
                                    blocks_per_file=128,
                                    num_subblocks=32)

# Add noise
for stream in antenna.streams:
    stream.add_noise(v_mean=0,
                     v_std=1)

obs_length = 300
# Add signals
signal_level = stg.voltage.get_level(1000,
                                     rvb,
                                     obs_length=obs_length,
                                     length_mode='obs_length',
                                     fftlength=fftlength)

unit_drift_rate = stg.voltage.get_unit_drift_rate(rvb, fftlength, 1)
print(unit_drift_rate)
print(10*unit_drift_rate)


chan_bw = sample_rate / num_branches
df = np.abs(chan_bw / fftlength)

f_start = 6000.e6 + sample_rate/num_branches + .4e6
drift_rate = unit_drift_rate
leakage_factor = stg.voltage.get_leakage_factor(f_start, rvb, fftlength)
print(f"leakage factor is {leakage_factor}")
leakage_factor = 1
for stream in antenna.streams:
    level = stream.get_total_noise_std() * leakage_factor * signal_level
    stream.add_constant_signal(f_start=f_start,
                                drift_rate=drift_rate,
                                level=level)

DATA_DIR = '//home/nathan/datasets/siggen/raw_files/'
rvb.record(output_file_stem=f'{DATA_DIR}/doppler_smear_test_gpu_offset_1000_leakagecorrected_injected_f={f_start}_drift={drift_rate}',
           obs_length=obs_length,
           length_mode='obs_length',
           header_dict={'HELLO': 'test_value',
                        'TELESCOP': 'GBT'},
           verbose=False)
