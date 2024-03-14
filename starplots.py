import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mesaPlot
import astropy.units as u

# ELEM_BASE = ['h1', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', 'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48',
#              'cr56', 'fe52', 'fe54', 'fe56', 'ni56']
# ELEM_EXTENDED = ['s32','s33', 's34', 'cl35', 'ar36','ar38', 'k39', 'ca40', 'ca42', 'ti46', 'ti47', 'ti48', 'v49', 'v50',
#                  'v51', 'cr48', 'cr49', 'cr50', 'cr51', 'cr52', 'cr53', 'cr54', 'cr55', 'cr56', 'cr57', 'mn51', 'mn52',
#                  'mn53', 'mn54', 'mn55', 'mn56', 'fe52', 'fe53', 'fe54', 'fe55', 'fe56', 'fe57', 'fe58', 'co55', 'co56',
#                  'co57', 'co58', 'co59', 'co60', 'ni56', 'ni57', 'ni58', 'ni59', 'ni60', 'ni61', 'cu59', 'cu61', 'cu62']
# ABUN_ELEM_M127 = np.hstack([ELEM_BASE[:9], ELEM_EXTENDED])    
MIX_TYPES = ["No mixing", "Convection", "Overshooting", "Semi-convection", "Thermohaline mixing",
             "Rotational mixing", "Rayleigh-Taylor mixing", "Minimum", "Anonymous"]
MIX_HATCHES = ['',      # no mixing
               'o',     # convective
               # '.',     # softened convective
               'x',     # overshoot
               'O',     # semi-convective
               '\\',    # thermohaline
               '/',     # rotation
               '+',      # Rayleigh-Taylor
               '',      # minimum
               ''       # anonymous
               ]



# def energy_and_mixing(histfile, time_ind=-1, show_mix=False, show_mix_legend=True, raxis="star_mass", fps=10, fig=None, ax=None,
#                       show_time_label=True, time_label_loc=(), time_unit="Myr", fig_size=(5.5, 4), axis_lim=-99,
#                       axis_label="", axis_units="", show_colorbar=True, cmap=ROBS_CMAP_ENERGY, cmin=-10,
#                       cmax=10, cbar_label="", theta1=0, theta2=360, hrd_inset=True, show_hrd_ticks_and_labels=False,
#                       show_total_mass=False, show_surface=True, show_grid=False, output_fname="energy_and_mixing",
#                       anim_fmt=".mp4", time_scale_type="model_number"):


def energy_and_mixing(histfile, age, fig=None, ax=None, show_mixing=True, show_mixing_legend=True, axes_col='star_mass', axes_label='', axes_units='', fps=10, show_age_label=True,
                      age_label_loc=(), age_unit='Myr', show_colorbar=True, cmap=
    
