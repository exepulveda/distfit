import numpy as np

app_config = {
    'fit_time_template':'CPU time=%ds, Wallclock time=%ds'
}

backend_config = {
    'title_template':'Fitted distribution: %s; Probability scale: %s; Number of points: %d; nsam=%d',
}

engine_config = {
    'seed': 1684120,
    'sdev_default': 0.1,
    'primary_x_scale_points': np.array([0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]),
    'secondary_x_scale_intervals': np.array([9, 8, 10, 15, 10 , 10, 15, 10, 8, 9]),
    'tertiary_x_scale_intervals':5,
    'nwalkers': 6,
    'niter': 500,
    'ndiscard': 50,
    'nthin':5,
    'sample_size':1000,
    'opt_method':'L-BFGS-B',#'TNC', #'trust-constr',
    'gtol': 1e-10,
    'enable_opt': True,
    'enable_hessian': True,
    'enable_mcmc': True,
}

layout_config = {
    'title': 'Distribution Fitting',
    'max_file_size':100*1024,
    'max_file_rows':200,
    'max_file_cols':10,
}
