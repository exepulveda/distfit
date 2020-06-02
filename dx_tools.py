# -*- coding: utf-8 -*-
'''Library of basic/intermediate numerical tools for Python
'''
import sys
import os.path
import logging
import time

import numpy as np
import scipy.stats

import emcee
import numdifftools as nd

from typing import List

#from joblib import Memory
#memory = Memory('.cachedir', verbose=1,bytes_limit=1000*1e3)

# MCMC constants
MCMC_NITER = 600; MCMC_NDISCARD=100; MCMC_NTHIN=5; MCMC_NWALKERS=6; MCMC_PROGRESS=False

#----------------------------------

class ParameterOptimisationError(Exception):
    pass

class HessianOptimisationError(Exception):
    pass

class EmceeSamplingError(Exception):
    pass

# ----------------------

class Series_fmt(object):
    def __init__(self,**kargs):
        self.name: str = 'New series'
        self.marker: str = None
        self.colour: str = None
        self.style: str = None

        self.__dict__.update(kargs)

class Series(object):
    #FIXME: extend using 'Series_fmt'
    def __init__(self,**kargs):
        self.x: List = None
        self.y: List = None
        self.name: str = 'New series'
        self.marker: str = None
        self.colour: str = None
        self.style: str = None
        self.hover: List = None
        self.width: int = None

        self.__dict__.update(kargs)

class Axis(object):
    def __init__(self,**kargs):
        self.label = ''
        self.type = 'linear'
        self.ticks_values: List[float] = None
        self.ticks_display: List[float] = None

        self.__dict__.update(kargs)

class Chart(object):
    def __init__(self,**kargs):
        self.title = 'New Chart'
        self.series: List[Series]
        self.xaxis: Axis = None
        self.yaxis: Axis = None

        self.__dict__.update(kargs)

def parscale_lin2log(par,parscale_ix):
    # rescale parameter 'scale' from linear to log scale
    tmp = np.copy(par)
    if parscale_ix is not None:
        tmp[parscale_ix] = np.log(tmp[parscale_ix])
    return tmp

def parscale_log2lin(par,parscale_ix):
    # rescale parameter 'scale' from log to linear scale
    tmp = np.copy(par)
    if parscale_ix is not None:
        tmp[parscale_ix] = np.exp(tmp[parscale_ix])
    return tmp

def ipy(imath):
    # convert from mathematical array index to Python array index
    # should not be needed in 21 century programming, yet here we are ...
    return imath-1

def imath(ipy):
    # convert from Python array index to mathematical array index
    return ipy+1

def is_numeric(x):
    # check that 'x' is purely numeric
    return np.isfinite(x).all()
    #check_nan = not np.isnan(x).any()
    #check_inf = not np.isinf(x).any()
    #all_good = check_nan and check_inf
    #return all_good

def is_pos_def(x):
    # check that 'x' is positive-definite
    x_is_num = is_numeric(x) # pre-check for nan's and inf's ...
    if not x_is_num:
        return False

    res = np.all(np.linalg.eigvals(x) > 0) # ... as linalg hates them ...
    return res

def get_condition_number(x):
    # get condition number of 'x'
    return np.linalg.cond(x) #OJO FEO: this seems unsafe wrt nan ... inf's are ok tho

def is_well_conditioned(x):
    # check that 'x' is well-conditioned
    crit = 1.0/np.finfo(float).eps
    cn = get_condition_number(x)
    return cn < crit

def is_covar_safe(covar):
    # check that 'covar' is safe (positive-definite and well-conditioned)
    check_pd = is_pos_def(covar)
    check_cond = is_well_conditioned(covar)
    return check_pd and check_cond

def is_int(x):
    # true if 'x' is integer
    type_int = type(1)
    x_type = type(x)
    ret = x_type == type_int
    return ret

def is_float(x):
    # true if 'x' is float
    type_float = type(1.0)
    x_type = type(x)
    ret = x_type == type_float
    return ret

def is_scalar(x):
    # true of 'x' is scalar
    x_is_int = is_int(x)
    x_is_float = is_float(x)
    ret = x_is_int or x_is_float
    return ret

def is_list(x):
    # true if 'x' is list
    type_list = type([1])
    x_type = type(x)
    ret = x_type == type_list
    return ret

def shapeD(v):
    # generalises shape to work with scalars and lists
    logger = logging.getLogger('shapeD')
    v_is_scalar = is_scalar(v)
    v_is_list = is_list(v)
    if v_is_scalar:
        logger.debug('via scalar')
        ret = np.asarray([0])
    elif v_is_list:
        logger.debug('via list')
        v_len = len(v)
        logger.debug('v_len : %s',str(v_len))
        ret = np.asarray([v_len])
    else:
        logger.debug('via matrix (assumed)')
        ret = v.shape
    logger.debug('ret : %s',str(ret))
    return ret

def create_mat_diag(n=1,diag=1.0):
    # create diagonal matrix
    logger = logging.getLogger('create_mat_diag')
    v_shape = shapeD(diag)
    nd = len(v_shape)
    n1 = v_shape[0]
    logger.debug('n1 : %s',str(n1))
    err = 0
    if nd == 1:
        if n1 == n:
            logger.debug('setting via diagonal matrix')
            ret = np.diag(diag)
        elif n1 == 0:
            logger.debug('setting via scalar matrix')
            diag_vec = np.ones(n,dtype=float)
            diag_vec.fill(diag)
            ret = np.diag(diag_vec)
        else:
            err = 2; ret = 'create_mat_diag:mismatch@1'
    else:
        err = 1; ret = 'create_mat_diag:invalid@1'
    logger.debug('ret : %s',str(ret))
    return ret,err

def create_mat_square(n,v):
    # create square matrix
    logger = logging.getLogger('create_mat_square')
    logger.debug('n : %s',str(n))
    logger.debug('v : %s',str(v))
    v_shape = shapeD(v)
    logger.debug('v_shape : %s',str(v_shape))
    nd = len(v_shape)
    logger.debug('nd : %s',str(nd))
    err = 0
    if nd == 2: #FIXME: no check if 'v' is square or matches 'n'
        logger.debug('setting via full matrix')
        ret = v
    elif nd == 1:
        ret,err = create_mat_diag(n,v)
    else:
        err = 1; ret = 'create_mat_square:invalid@1'
    logger.debug('ret : %s',str(ret))
    return ret,err

def nan2inf(x,negative=False,replace=np.inf):
    # check for nan and replace by inf or -inf
    if np.ndim(x) > 0:  # an array
        nan_indices = np.where(np.isnan(x))[0]
        if len(nan_indices) > 0:
            if negative:
                x[nan_indices] = -replace
            else:
                x[nan_indices] = replace
    else:
        if np.isnan(x):
            if negative:
                x = -replace
            else:
                x = replace

    return x

def quickif(x,x_def,mask=True):
    # returns 'x' if present (not None) and 'x_def' otherwise
    # optional mask for extra control
    # usage: substitute default value when main value missing
    #        inline / on-the-fly IFs expressions including in dynamic expressions
    have_x = x is not None
    if have_x and mask:
        ret = x
    else:
        ret = x_def
    return ret

def mode_via_optim(logpdf_f,x0,bounds=None,tol=1e-6,settings={}):
    # estimate mode of a distribution using optimisation
    logger = logging.getLogger('mode_via_optim')
    logger.debug('Starting mode_via_optim')

    #maximisation using scipy optimiser
    logger.debug('calling scipy.optimize.minimize...')
    logger.debug('x0: %s',x0)
    logger.debug('bounds: %s',bounds)

    logpdf_neg = lambda t: -logpdf_f(t) # in order to use minimize algorithm
    res = scipy.optimize.minimize(logpdf_neg,x0,tol=tol,bounds=bounds,options={'disp': True,'gtol':settings.get('gtol',1.e-8),'maxls':100,'ftol':1e-20},method=settings.get('opt_method','L-BFGS-B'))

    if not res.success:
        logger.info('optimiser failed: %s',res.message)
        x_opt = x0
    else:
        x_opt = res.x
        f_opt = -res.fun
        logger.info('x opt: %s',str(x_opt))
        logger.info('f_opt: %s',str(f_opt))

        if not is_numeric(x_opt):
            try:
                exception = ParameterOptimisationError('Maximisation gave non-numeric values')
                logger.exception(exception)
                raise exception
            except ParameterOptimisationError:
                x_opt = x0

    logger.debug('calling mode_via_optim...DONE')
    return x_opt

def covar_via_hessian(logpdf_f,x0):
    # estimate covariance of a distribution using Hessian approach
    logger = logging.getLogger('covar_via_hessian')
    hess_f = nd.Hessian(logpdf_f)
    hess_val = hess_f(x0)
    #print('hess_val:',hess_val)
    covar = np.linalg.inv(-hess_val)
    #print('covar:',covar)

    if not is_numeric(covar):
        logger.error('covar = %s',covar)
        exception = HessianOptimisationError('Hessian covariance estimation gave non-numeric values')
        try:
            logger.exception(exception)
            raise HessianOptimisationError('Hessian covariance estimation gave non-numeric values')
        except HessianOptimisationError:
            return None

    logger.info('check_pd = %d',is_pos_def(covar))
    logger.info('check_cond = %d',is_well_conditioned(covar))
    return covar

def mode_covar_mcmc(logpdf_f,x0,c0,bounds,optim_settings,covar_settings,mcmc_settings):
    # estimate mode and covariance of a distribution (jacket)
    logger = logging.getLogger('mode_covar_mcmc')
    logger.info('x0: %s',str(x0))

    do_optim = optim_settings['do_optim']
    do_covar = covar_settings['do_covar']
    do_mcmc = mcmc_settings['do_mcmc']

    have_optim = False; have_covar = False

    # maximisation-based estimate of mode
    if do_optim:
        logger.info('BEFOR OPTIM::x0: %s',str(x0))

        t1 = time.time()
        x1 = mode_via_optim(logpdf_f,x0,bounds=bounds,tol=1e-6,settings=optim_settings)
        t2 = time.time()

        logger.info('AFTER OPTIM::x1: %s',str(x1))
        have_optim = True
    else:
        logger.info('optimisation skipped')
        x1 = x0

    # hessian-based estimate of covariance
    if do_covar:
        logger.debug('calling covar_via_hessian...')

        t1 = time.time()
        covar_hess = covar_via_hessian(logpdf_f,x1)
        t2 = time.time()

        logger.debug('calling covar_via_hessian...DONE')
        if covar_hess is not None:
            logger.info('covar_hess: %s',str(covar_hess))
            covar_is_safe = is_covar_safe(covar_hess)
            logger.debug('covar_is_safe: %d',covar_is_safe)
            have_covar = True
        else:
            logger.debug('covar_via_hessian returned None')
    else:
        logger.info('covar estimation skipped')
        covar_hess = None

    # mcmc sampling
    if do_mcmc:
        logger.debug('calling mcmc...')

        if have_optim:
            x0eff = x1
        else:
            x0eff = x0

        if have_covar:
            c0eff = covar_hess
        else:
            c0eff = c0

        t1 = time.time()
        samples = mcmc_sampler_2(logpdf_f,x0eff,c0eff,mcmc_settings)
        t2 = time.time()
        logger.debug('calling mcmc...DONE')

        if samples is not None:
            samples_is_num = is_numeric(samples)
            logger.debug('samples_is_num: %d',samples_is_num)
        else:
            logger.debug('mcmc returned None')
    else:
        logger.info('mcmc skipped')
        samples = None

    return x1,covar_hess,samples

def mcmc_sampler_1(logpdf_f,x0,niter=MCMC_NITER,ndiscard=MCMC_NDISCARD,nthin=MCMC_NTHIN,progress=MCMC_PROGRESS):
    # MCMC sampler
    logger = logging.getLogger('mcmc_sampler')
    nwalkers,ndim = x0.shape

    logger.debug('nwalkers: %d',nwalkers)
    logger.debug('ndim: %d',ndim)
    logger.debug('initial_positions: %s',x0.shape)

    sampler = emcee.EnsembleSampler(nwalkers,ndim,logpdf_f)
    sampler.run_mcmc(x0,niter,progress=progress)

    flat_samples = sampler.get_chain(discard=ndiscard,thin=nthin,flat=True)

    logger.debug('samples.shape:',flat_samples.shape)
    if not is_numeric(flat_samples):
        raise EmceeSamplingError('MCMC sampler gave non-numeric values')

    return flat_samples

def mcmc_sampler_2(logpdf_f,mean0,covar0,mcmc_settings):
    # MCMC sampler with auto-init of walkers
    logger = logging.getLogger('mcmc_sampler_2')

    niter    = mcmc_settings.get('niter',MCMC_NITER)
    ndiscard = mcmc_settings.get('ndiscard',MCMC_NDISCARD)
    nthin    = mcmc_settings.get('nthin',MCMC_NTHIN)
    nwalkers = mcmc_settings.get('nwalkers',MCMC_NWALKERS)
    progress = mcmc_settings.get('progress',MCMC_PROGRESS)

    # get covar0 for initial seeding
    mean0_ndim = len(mean0)
    covar_ini,err = create_mat_square(mean0_ndim,covar0)

    # initial walkers from the mutivariate according to mean0 and covar0
    x0 = scipy.stats.multivariate_normal.rvs(mean=mean0,cov=covar_ini,size=nwalkers)

    x0_min = np.min(x0,axis=0)
    x0_max = np.max(x0,axis=0)
    x0_mean = np.mean(x0,axis=0)
    x0_cov = np.cov(x0,rowvar=False)

    logger.debug('x0_min : %s',str(x0_min))
    logger.debug('x0_max : %s',str(x0_max))
    logger.debug('x0_mean: %s',str(x0_mean))
    logger.debug('x0_cov : %s',str(x0_cov))

    samples = mcmc_sampler_1(logpdf_f,x0,niter,ndiscard,nthin,progress)

    samples_min = np.min(samples,axis=0)
    samples_max = np.max(samples,axis=0)
    samples_mean = np.mean(samples,axis=0)
    samples_cov = np.cov(samples,rowvar=False)

    logger.debug('samples_min : %s',str(samples_min))
    logger.debug('samples_max : %s',str(samples_max))
    logger.debug('samples_mean: %s',str(samples_mean))
    logger.debug('samples_cov : %s',str(samples_cov))

    logger.debug('input mean0 : %s',str(mean0))
    logger.debug('input covar0: %s',str(covar0))

    return samples

def log_posterior_f(par,data,log_prior_f,log_likelihood_f):
    # Evaluate general log-posterior
    logger = logging.getLogger('log_posterior_f')
    #logger.debug('par: %s',par)
    #logger.debug('data: %s',data)
    if log_prior_f is None: # uniform
        log_prior = 0.0
    else:
        log_prior = log_prior_f(par)
    log_like = log_likelihood_f(par,data)
    #logger.debug('log_prior: %s',log_prior)
    #logger.debug('log_like: %s',log_like)
    log_post = log_prior + log_like
    return log_post

def log_posterior_f_ind(par,data,log_priors_f,log_likelihood_f):
    # Evaluate the log-posterior for the simple case of independent priors and independent likelihoods

    # Example usage
    # log_likelihood_f = lambda t,d: self.logpdf(d,t)
    # if log_priors_f is None:
    #     log_priors_f = [lambda p: 0.0]*len(par) # uniform
    # logpdf_f = lambda t: log_posterior_f_ind(t,data,log_priors_f,log_likelihood_f)

    logger = logging.getLogger('log_posterior_f')
    log_priors = np.sum([log_priors_f[i](p) for i,p in enumerate(par)])
    #logger.debug('par: %s',par)
    #logger.debug('data: %s',data)
    d_likelihood = log_likelihood_f(par,data)
    #logger.debug('d_likelihood: %s',d_likelihood)
    log_like = np.sum(d_likelihood)
    #logger.debug('log_priors: %s',log_priors)
    #logger.debug('log_like: %s',log_like)
    log_post = log_priors + log_like

    return log_post

def fit_predict_engine(logpdf,sampler,par0,bounds,optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings):
    # Given model with logpdf and sampler, fits parameters and generates predictive distribution
    logger = logging.getLogger('fit_predict_engine')

    # reset random number generator if requested
    rangen_seed = rangen_settings['seed']
    if rangen_seed is not None:
        logger.debug('resetting random.seed to %s',rangen_seed)
        np.random.seed(rangen_seed)

    # initialise covariance (default)
    sdev_default = optim_settings.get('sdev_default',0.2)
    logger.debug('sdev_default is %s',sdev_default)
    covar0 = sdev_default**2
    logger.debug('covar0 is %s',covar0)

    par_opt,par_covarH,par_samples = mode_covar_mcmc(
        logpdf,par0,covar0,bounds,optim_settings,covar_settings,mcmc_settings
    )

    # process MCMC samples
    if mcmc_settings['do_mcmc']:
        logger.info('parameter mcmc sampling done : get da moes')

        # compute and set par_mean
        par_mean = np.mean(par_samples,axis=0)
        logger.info('par_mean : %s',par_mean)

        # compute and set par_covar
        par_covar = np.cov(par_samples,rowvar=False)
        logger.info('par_covar: %s',par_covar)

    else:
        logger.info('parameter mcmc sampling skipped')

    # generate predictive distribution
    nsam_pred = pred_settings['nsam']
    pred_samples = sampler(par_samples,nsam_per_par=nsam_pred)

    return par_opt,par_covarH,par_samples,par_mean,par_covar,pred_samples

def simple_stats_1(x,):
    # Compute simple statistics of sample 'x'
    min_value = np.min(x)
    max_value = np.max(x)
    mean_value = np.mean(x)
    median_value = np.median(x)
    sdev_value = np.std(x)
    kurtosis_value = scipy.stats.kurtosis(x)
    skew_value = scipy.stats.skew(x)

    return min_value,max_value,mean_value,median_value,sdev_value,skew_value,kurtosis_value

def simple_stats_2(data,var_name,row_excluded,decimal_places=4):
    # More complex routine that returns GUI-ready statistics
    x = extract_var_from_data(data,var_name,row_excluded) #filter selected samples

    n_excluded = 0 if row_excluded is None else len(row_excluded)
    n_used  = len(x)
    n_total = 0 if data is None else len(data)

    min_value,max_value,mean_value,median_value,sdev_value,skew_value,kurtosis_value = simple_stats_1(x)

    return {
        'n_total':(n_total,0),
        'n_used':(n_used,0),
        'n_excluded':(n_excluded,0),
        'min':(min_value,decimal_places),
        'max':(max_value,decimal_places),
        'mean':(mean_value,decimal_places),
        'median':(median_value,decimal_places),
        'sdev':(sdev_value,decimal_places),
        'skewness':(skew_value,decimal_places),
        'exc kurtosis':(kurtosis_value,decimal_places),
    }

def extract_var_from_data(data,var_name,row_remove=None):
    # Extract subset of 'data' defined by 'var_name' and excluding 'row_to_remove'
    # Input data has in each row a dictionary with variable name as key
    x = [float(row[var_name]) for row in data]
    return filter(x,row_remove)

def filter(x,row_remove=None):
    # remove rows from 'x'
    x = np.asarray(x)
    if row_remove is not None and len(row_remove)>0:
        x = np.delete(x,row_remove)
    return x

def get_mask_from_rowix(n,rowix):
    # build mask=1 for row indices 'rowix'
    mask = np.zeros(n,dtype=bool) # default 0
    mask[disabled_indices] = 1    # mask = 1 for indices in rowix
    return mask

def arank(a,math=True):
    # array element ranks.
    # 'math=T' gives correct mathematical result, otherwise python's insane 0-base index nonsense.
    # https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice
    order = a.argsort()
    ranks = order.argsort()
    if math is True: # account for python array offset
        ranks = ranks + 1
    return ranks

def plotting_position(irank,n,a=0.4):
    # plotting position
    ret = (irank-a)/(n+1-2*a)
    return ret

def cp2ep(x):
    # convert cumulative probability to exceedance percentage
    ret = (1.0-x)*100.0
    return ret

def ep2cp(x):
    # convert exceedance percentage to cumulative probability
    ret = 1.0-x/100.0
    return ret

def cdf_to_ep(x,cdf):
    # convert cdf to exceedance percentage
    v = cdf(x)       # compute cdf: domain[x] -> [0,1]
    ret = cp2ep(v)   # convert to ep
    return ret

def quantile_via_icdf(ep,dist_icdf,dist_par):
    # computes quantiles corresponding to exceedance probabilities given icdf(*,par)
    #FIXME: generalise this FUNC to work with multiple 'dist_par'
    ret = [dist_icdf(ep2cp(ep_i),dist_par) for ep_i in ep]
    return ret

def parse_expr(s,sep=';',eq_symb='='):
    # parse a (possibly multi-line) string "s" with format "var1=val1; var2=val2; .."
    # returns a dictionary with key=var_i and value=val_i
    ret = {}

    if s is not None:
        try:
            lines = s.split('\n')
            for line in lines:
                exprs = line.split(sep) # separate lines with ";"
                for expr in exprs:
                    expr_parsed = expr.split(eq_symb)
                    if len(expr_parsed)>= 2: # add to the dict key=value
                        key   = expr_parsed[0].strip()
                        value = expr_parsed[1].strip()
                        ret[key]=value
        except Exception as e:
            pass

    return ret

def load_data_1(contents,filename,max_filesize=None):
    # loads dataset from file
    from base64 import b64decode
    from io import StringIO,BytesIO
    from pandas import read_csv,read_excel

    df = None; description = None; err = 0; message = 'load_data_1/ok'

    content_type,content_string = contents.split(',')

    if max_filesize is not None: # check file size
        filesize = len(content_string)
        logging.info('filesize: %d vs %d',filesize,max_filesize)
        if filesize > max_filesize:
            err = 100; message = 'f-load_data_1/filesize_exceeded'
            return df,description,err,message

    decoded = b64decode(content_string)
    try:
        if filename.endswith('csv') or filename.endswith('txt') or filename.endswith('dat'): #FIXME: get ext and use switch
            string_reader = StringIO(decoded.decode('utf-8'))
            description = string_reader.readline() #read the first line
            df = read_csv(string_reader)
        elif filename.endswith('xls'):
            df = read_excel(BytesIO(decoded))
            description = 'xls_description_currently_not_available'
        else:
            err = 200; message = 'f-load_data_1/file_ext_not_recognised'
    except Exception as e:
        err = 666; message = 'f-load_data_1/exception_unknown:source_file_not_read'

    return df,description,err,message

def load_data_2(contents,filename,max_filesize=None,max_filerows=None,max_filecols=None):
    # overlord returning results also as a pictionary

    # main call to engine
    df,description,err,message = load_data_1(contents,filename,max_filesize)

    if err is 0:
        pass
    else:
        data = None
        return df,data,description,err,message

    data = df.to_dict('records')

    # check other restrictions
    if max_filerows is not None and len(data) > max_filerows:
        err = 300; message = 'f-load_data_2/exceeded:max_filerows'
    elif max_filecols is not None and len(data) > 0 and len(data[0].keys()) > max_filecols:
        err = 400; message = 'f-load_data_2/exceeded:max_filecols'

    return df,data,description,err,message

# default tick labels for probability plots (NB: 'ep' = exceedance percentages!)
pscale_ep_ticks_default = np.array([99.9, 99.0, 97.5, 95.0, 95.0, 90.0, 75.0, 50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.1])
pscale_ep_primary_points_default = np.array([0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
pscale_ep_secondary_points_default = np.array([9, 8, 10, 15, 10 , 10, 15, 10, 8, 9])
pscale_ep_tertiary_factor_default = 5

def pscale_axis_ticks(pscale_icdf,ticks_labels=pscale_ep_ticks_default):
    # For probability scale axis: returns underlying axis tick values given desired tick labels
    ticks_values = pscale_icdf(ep2cp(ticks_labels))

    return ticks_values,ticks_labels

def qqplot_discrete_1(data,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_sorted=False):
    # Given sorted 'data', compute xy-plotting info for probability plots (QQ plots),
    # with x-scale defined by ICDF 'pxscale_icdf' (and optionally parameters 'pxscale_par')
    # Note:  if 'data' is unsorted, the plotting series will also be unsorted (but still correct)
    #FIXME: 'pxscale_fmt' controls labelling: 'ep' is default for hydrology, unusual for stats
    logger = logging.getLogger('qqplot_discrete_1')

    n = len(data)
    logger.debug('qqplot_discrete_1::n: %s',n)

    # generate two descriptors (x,y) = (x_i, y_i; i=1..n),
    # where y_i is i'th data point and x_i is its estimated EP assuming y_i comes from plotting_dist

    # compute x via plotting position and apply the ICDF
    if is_sorted is True:
        logger.debug('qqplot_discrete_1::data already sorted (quicker comp)')
        pp = [plotting_position(i,n) for i in range(1,n+1)] #OJO: I give up here ... 0-base arrays are coocoo for math
    else:
        logger.debug('qqplot_discrete_1::data unsorted - use ranks')
        dr = arank(data,math=True)
        logger.debug('qqplot_discrete_1::len(dr): %s',len(dr))
        logger.debug('qqplot_discrete_1::dr: %s',dr)
        pp = [plotting_position(dr[ipy(i)],n) for i in range(1,imath(n))] # now using python <-> math converter :-)

    logger.debug('qqplot_discrete_1::len(pp): %s',len(pp))
    logger.debug('qqplot_discrete_1::pp: %s',pp)

    x = pxscale_icdf(pp,pxscale_par)

    # hovertext depends on what we want to display on x-axis
    if pxscale_fmt is 'qq':        # classic QQ plot
        hovertext = np.array(x)
    elif pxscale_fmt is 'cp':      # CDF plot
        hovertext = np.array(pp)
    elif pxscale_fmt is 'ep':      # Exceedance percentage plot
        hovertext = cp2ep(np.array(pp))
    else:
        return 666

    logger.debug('qqplot_discrete_1::hovertext: %s',hovertext)

    return x,data,hovertext,pp

def qqplot_discrete_2(data,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_sorted=False):
    # Overlord to return packaged Series 's'.
    logger = logging.getLogger('qqplot_discrete_2')

    x,y,hovertext,pp = qqplot_discrete_1(data,pxscale_icdf,pxscale_par,pxscale_fmt,is_sorted)

    return Series(x=x,y=y,hover=hovertext),pp

def qqplot_continuous_1(dist_icdf,dist_par,pr_points,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep'):
    # Given continuous distribution with ICDF '', construct QQ plot against 'pxscale' prob scale.

    logger = logging.getLogger('qqplot_continuous_1')

    y = dist_icdf(pr_points,dist_par)
    x = pxscale_icdf(pr_points,pxscale_par)

    # hovertext depends on what we want to display on x-axis
    if pxscale_fmt is 'qq':        # classic QQ plot
        hovertext = np.array(x)
    elif pxscale_fmt is 'cp':      # CDF plot
        hovertext = np.array(pr_points)
    elif pxscale_fmt is 'ep':      # Exceedance percentage plot
        hovertext = cp2ep(pr_points)
    else:
        return 666

    logger.debug('qqplot_continuous_1::hovertext: %s',hovertext)

    return x,y,hovertext

def qqplot_continuous_2(dist,dist_par,pr_points,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep'):
    # Overlord to return packaged Series 's'.
    logger = logging.getLogger('qqplot_continuous_2')

    x,y,hovertext = qqplot_continuous_1(
        dist,dist_par,pr_points,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep'
    )

    return Series(x=x,y=y,hover=hovertext)

def qqplot_samples(samples,pr_points,pxscale_icdf,pxscale_par=None):
    # QQ plot of 'samples' from a distribution: expected quantile, median quantile and 90% CL
    # Probability axis determed by 'pxscale_icdf' and 'pxscale_par'
    logger = logging.getLogger('qqplot_samples')

    logger.debug('pr_points size: %d',len(pr_points))
    logger.debug('pr_points: %s',pr_points)

    # get sample quantiles at requested points
    samples_quantile = np.quantile(samples,pr_points,axis=1)
    logger.debug('samples_quantile: %s',samples_quantile)

    # construct probability scale data at requested quantiles
    x_all = pxscale_icdf(pr_points,pxscale_par)
    hovertext_all = cp2ep(pr_points)

    # expected quantile
    series_expected = Series(x=x_all,y=np.mean(samples_quantile,axis=1),hover = hovertext_all)

    # median quantile
    series_median = Series(x=x_all,y=np.quantile(samples_quantile,0.5,axis=1),hover=hovertext_all)

    # 90% confidence limits on the quantiles
    series_uncertainty_lower = Series(x=x_all,y=np.quantile(samples_quantile,0.05,axis=1),hover=hovertext_all)

    series_uncertainty_upper = Series(x=x_all,y=np.quantile(samples_quantile,0.95,axis=1),hover=hovertext_all)

    return series_expected,series_median,series_uncertainty_lower,series_uncertainty_upper

def qqplot_passive_1(edfv,edfp,data_passive,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_edf_sorted=False):
    # Given edf, compute xy-plotting info for data_passive to be shown on a QQ plot
    # Passive data is usually (but not necessarily) data excluded from the analysis for what-eva reason
    # This is a somewhat tricky operation that requires subtle assumptions to be made
    # 1. Passive data should not affect the QQ plot itself (otherwise its not 'passive'!)
    # 2. Passive data should be appear "close" to where it would appear had it been not passive :-)
    # 3. Passive data should "look" like it came from a QQ plot ("spread out")
    #    (otherwise we get behaviour unusual for QQ plots, with unrealistic clumping)
    # Solution: Find all points that would land in edf intervals and spread them out maximally
    #           For outliers, allow some fixed-step expansion (ranked)
    # Note: the output from this routine has the same format as qqplot_discrete_1
    logger = logging.getLogger('qqplot_passive_1')

    n_edf = len(edfv)
    logger.debug('qqplot_passive_1::n: %s',n_edf)

    n_passive = len(data_passive)
    logger.debug('qqplot_passive_1::n_passive: %s',n_passive)
    logger.debug('qqplot_passive_1::data_passive : %s',data_passive)

    # generate two descriptors (x,y) = (x_i, y_i; i=1..n),
    # where y_i is i'th data point and x_i is its estimated EP (mapped through edf and pxscale_icdf)

    # prepare edf array
    tempx = np.ones(n_edf+2,dtype=float)
    tempy = np.ones(n_edf+2,dtype=float)
    tempx[0]=-np.inf; tempx[n_edf+1]=np.inf
    tempy[0]=0; tempy[n_edf+1]=1

    if is_edf_sorted is True:
        logger.debug('qqplot_passive_1::edf already sorted (quicker comp)')
        tempx[1:n_edf+1] = edfv
        tempy[1:n_edf+1] = edfp
    else:
        logger.debug('qqplot_passive_1::edf unsorted - use order')
        order=edfv.argsort()
        logger.debug('qqplot_passive_1::len(order) : %s',len(order))
        logger.debug('qqplot_passive_1::order : %s',order)
        tempx[1:n_edf+1] = [edfv[order[i]] for i in range(0,n_edf)]
        tempy[1:n_edf+1] = [edfp[order[i]] for i in range(0,n_edf)] # this relies on edf monotonicity!

    logger.debug('qqplot_passive_1::tempx : %s',tempx)
    logger.debug('qqplot_passive_1::tempy : %s',tempy)

    n_processed = 0
    pp = np.ones(n_passive,dtype=float)

    for i in range(0,n_edf+1): # loop over edf intervals and find passive points in each interval
        logger.debug('qqplot_passive_1::i : %s',i)
        logger.debug('qqplot_passive_1::tempx[i]   : %s',tempx[i])
        logger.debug('qqplot_passive_1::tempx[i+1] : %s',tempx[i+1])
        index_sub = np.where((tempx[i]<data_passive) & (data_passive<=tempx[i+1]))
        passive_sub = data_passive[index_sub]
        logger.debug('qqplot_passive_1::passive_sub : %s',passive_sub)
        n_sub = len(passive_sub)
        logger.debug('qqplot_passive_1::n_sub : %s',n_sub)
        order_sub = arank(passive_sub,math=False)
        logger.debug('qqplot_passive_1::order_sub : %s',order_sub)
        logger.debug('qqplot_passive_1::tempy[i]   : %s',tempy[i])
        logger.debug('qqplot_passive_1::tempy[i+1] : %s',tempy[i+1])
        pp_new = np.linspace(tempy[i],tempy[i+1],n_sub+1,endpoint=False)
        pp_new = pp_new[1:] # not sure why - as endpoint=F - but python ...
        logger.debug('qqplot_passive_1::pp_new : %s',pp_new)
        pp_new = pp_new[order_sub] # here we assign pp's in ascending order
        logger.debug('qqplot_passive_1::pp_new : %s',pp_new)
        n_processed += n_sub
        pp[index_sub] = pp_new # put new pp's into correct slots in original array (unsorted!)
        logger.debug('qqplot_passive_1::n_processed : %s',n_processed)
        logger.debug('qqplot_passive_1::pp : %s',pp)

    all_found = n_processed == n_passive
    logger.debug('qqplot_passive_1::all_found : %s',all_found)
    if not all_found:
        logger.debug('qqplot_passive_1::ERROR!')

    logger.debug('qqplot_passive_1::pp : %s',pp)
    #quit()

    x = pxscale_icdf(pp,pxscale_par)  # note: this code is the same for all QQ plots once we know 'pp'
    logger.debug('qqplot_passive_1::x  : %s',x)

    # hovertext depends on what we want to display on x-axis
    if pxscale_fmt is 'qq':        # classic QQ plot
        hovertext = np.array(x)
    elif pxscale_fmt is 'cp':      # CDF plot
        hovertext = np.array(pp)
    elif pxscale_fmt is 'ep':      # Exceedance percentage plot
        hovertext = cp2ep(np.array(pp))
    else:
        return 666

    logger.debug('qqplot_passive_1::hovertext: %s',hovertext)

    return x,data_passive,hovertext,pp

def qqplot_passive_2(edfv,edfp,data_passive,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_edf_sorted=False):
    # Overlord to return packaged Series 's'.
    logger = logging.getLogger('qqplot_passive_2')

    s = Series()
    s.x,s.y,s.hovertext,pp = qqplot_passive_1(edfv,edfp,data_passive,pxscale_icdf,pxscale_par,pxscale_fmt,is_edf_sorted)

    return s,pp

def qqplot_passive_edf_1(edfv,edfp,data_passive,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_edf_sorted=False):
    # Given edf, compute xy-plotting info for data_passive to be shown on a QQ plot
    # Passive data is usually (but not necessarily) data excluded from the analysis for what-eva reason
    # This is a somewhat tricky operation that requires subtle assumptions to be made
    # This algorithm implements DK's "old" approach, with following principles:
    # 1. Passive data should not affect the QQ plot itself (otherwise its not 'passive'!)
    # 2. Passive data should be appear "close" to where it would appear had it been not passive :-)
    # 3. Passive data should appear at locations that are independent of other passive data
    #    (otherwise we get behaviour unusual for plotting data, some kind of graphic commutation violation)
    # Solution: compute the pp of data_passive using the edf (using safeguards when extrapolating)
    # Note: the output from this routine has the same format as qqplot_discrete_1
    # Limitations: This approach is not unreasonable, but does not recognize that QQ plots "spread" the data
    #              It can hence produce "clumps" of points unlike real QQ plots.
    #              Outliers are awkward - can end up on a vertical line defined by the safeguard.
    logger = logging.getLogger('qqplot_passive_edf_1')

    n_edf = len(edfv)
    logger.debug('qqplot_passive_edf_1::n: %s',n_edf)

    n_passive = len(data_passive)
    logger.debug('qqplot_passive_edf_1::n_passive: %s',n_passive)
    logger.debug('qqplot_passive_edf_1::data_passive : %s',data_passive)

    # generate two descriptors (x,y) = (x_i, y_i; i=1..n),
    # where y_i is i'th data point and x_i is its estimated EP (mapped through edf and pxscale_icdf)

    # prepare edf array
    if is_edf_sorted is True:
        logger.debug('qqplot_passive_edf_1::edf already sorted (quicker comp)')
        tempx = edfv
        tempy = edfp
    else:
        logger.debug('qqplot_passive_edf_1::edf unsorted - use order')
        order=edfv.argsort()
        logger.debug('qqplot_passive_edf_1::len(order) : %s',len(order))
        logger.debug('qqplot_passive_edf_1::order : %s',order)
        tempx = [edfv[order[i]] for i in range(0,n_edf)]
        tempy = [edfp[order[i]] for i in range(0,n_edf)] # this relies on edf monotonicity!

    logger.debug('qqplot_passive_edf_1::tempx : %s',tempx)
    logger.debug('qqplot_passive_edf_1::tempy : %s',tempy)

    dppL = 0.5*tempy[0]; dppU = 0.5*(1.0-tempy[n_edf-1]) # limit proximity to cdf endpoints
    pp_min = dppL; pp_max = 1.0-dppU

    logger.debug('qqplot_passive_edf_1::pp_min : %s',pp_min)
    logger.debug('qqplot_passive_edf_1::pp_max : %s',pp_max)

    pp = interp_edf_2(   # note this procedure assumes expedential tails asymptoting to (0,1) outside tabulated edf ...
        x=data_passive,xp=tempx,fp=tempy,
        left=pp_min,right=pp_max  # ... but still impose safeguards on estimated 'pp'
    )
    logger.debug('qqplot_passive_edf_1::pp : %s',pp)
    #quit()

    x = pxscale_icdf(pp,pxscale_par)  # note: this code is the same for all QQ plots once we know 'pp'
    logger.debug('qqplot_passive_edf_1::x : %s',x)

    # hovertext depends on what we want to display on x-axis
    if pxscale_fmt is 'qq':        # classic QQ plot
        hovertext = np.array(x)
    elif pxscale_fmt is 'cp':      # CDF plot
        hovertext = np.array(pp)
    elif pxscale_fmt is 'ep':      # Exceedance percentage plot
        hovertext = cp2ep(np.array(pp))
    else:
        return 666

    logger.debug('qqplot_passive_edf_1::hovertext: %s',hovertext)

    return x,data_passive,hovertext,pp

def qqplot_passive_edf_2(edfv,edfp,data_passive,pxscale_icdf,pxscale_par=None,pxscale_fmt='ep',is_edf_sorted=False):
    # Overlord to return packaged Series 's'.
    logger = logging.getLogger('qqplot_passive_2')

    s = Series()
    s.x,s.y,s.hovertext,pp = qqplot_passive_edf_1(edfv,edfp,data_passive,pxscale_icdf,pxscale_par,pxscale_fmt,is_edf_sorted)

    return s,pp

def chart_init(title,\
               xaxis_title,x_ticks_values,x_ticks_texts,\
               yaxis_title,yaxis_log=False,\
               series_list=[],\
               tag=None):
    # create default chart
    logger = logging.getLogger('chart_init')

    # pack axis
    xaxis = Axis(label=xaxis_title,ticks_values=x_ticks_values,ticks_display=x_ticks_texts)
    yaxis = Axis(label=yaxis_title,type='log' if yaxis_log else 'linear')

    chart = Chart(
        title  = title,
        series = series_list,
        xaxis  = xaxis,
        yaxis  = yaxis,
        tag = tag, # create new attribute to support multi-plot
    )

    return chart

def chart_qq_init(title,plotting_dist_icdf,yaxis_log,xaxis_title,yaxis_title,series_list=[],tag=None):
    # create default QQ chart
    logger = logging.getLogger('chart_qq_init')

    # ticks values - currently same for all prob scales
    x_ticks_values,x_ticks_texts = pscale_axis_ticks(plotting_dist_icdf)

    chart = chart_init(title,xaxis_title,x_ticks_values,x_ticks_texts,yaxis_title,yaxis_log,series_list,tag)
    return chart

def qqplot_series_discrete(plotting_dist_icdf,data,data_passive,is_data_sorted,series_fmt=None):
    # QQ plot series based on 'data' and 'data_passive' with probability scale given by 'plotting_dist'
    logger = logging.getLogger('qqplot_series_discrete')

    # active discrete data - usually observations
    series_data,sd_pp = qqplot_discrete_2(data,plotting_dist_icdf,is_sorted=is_data_sorted)

    if series_fmt is not None:
        series_data.name   = series_fmt[0].name
        series_data.marker = series_fmt[0].marker
        series_data.colour = series_fmt[0].colour
        series_data.style  = series_fmt[0].style

    series_list = [series_data]

    # passive discrete data - usually excluded observations
    n_data_passive = len(data_passive)
    if n_data_passive > 0: # attempt to place passive points on the plot

        series_data_passive,dum_pp = qqplot_passive_2( # use some expensive magic
            edfv=series_data.y,
            edfp=sd_pp,
            data_passive=data_passive,
            pxscale_icdf=plotting_dist_icdf,
            is_edf_sorted=is_data_sorted
        )

        logger.debug('series_data_passive.y : %s',series_data_passive.y)
        logger.debug('series_data_passive.x : %s',series_data_passive.x)
        logger.debug('series_data_passive.hover : %s',series_data_passive.hover)
        logger.debug('dum_pp : %s',dum_pp)

        if series_fmt is not None:
            series_data_passive.name   = series_fmt[1].name
            series_data_passive.marker = series_fmt[1].marker
            series_data_passive.colour = series_fmt[1].colour
            series_data_passive.style  = series_fmt[1].style

        series_list += [series_data_passive]

    return series_list

def qqplot_series_continuous(plotting_dist_icdf,dist_icdf,dist_par,pscal_base,series_fmt):
    # QQ plot series based on 'dist' with probability scale given by 'plotting_dist'
    logger = logging.getLogger('qqplot_series_continuous')

    x_points = pscale_axis_points(pscal_base[0],pscal_base[1],pscal_base[2])

    # series obtained from dist(par)
    series_dist = qqplot_continuous_2(dist_icdf,dist_par,x_points,plotting_dist_icdf)

    if series_fmt is not None:
        series_dist.name   = series_fmt.name
        series_dist.marker = series_fmt.marker
        series_dist.colour = series_fmt.colour
        series_dist.style  = series_fmt.style

    series_list = [series_dist]

    return series_list

def qqplot_series_samples(plotting_dist_icdf,dist_samples,pscal_base,series_fmt=None):
    # QQ plot series based on 'dist_samples' with probability scale given by 'plotting_dist'
    logger = logging.getLogger('qqplot_series_samples')

    x_points = pscale_axis_points(pscal_base[0],pscal_base[1],pscal_base[2])

    # series obtained from 'predictive distribution' samples: expected and median
    series_expected,series_median,series_uncertainty_lower,series_uncertainty_upper = \
        qqplot_samples(dist_samples,x_points,plotting_dist_icdf)

    if series_fmt is not None:
        series_expected.name   = series_fmt[0].name
        series_expected.marker = series_fmt[0].marker
        series_expected.colour = series_fmt[0].colour
        series_expected.style  = series_fmt[0].style

        series_median.name   = series_fmt[1].name
        series_median.marker = series_fmt[1].marker
        series_median.colour = series_fmt[1].colour
        series_median.style  = series_fmt[1].style

        series_uncertainty_lower.name   = series_fmt[2].name
        series_uncertainty_lower.marker = series_fmt[2].marker
        series_uncertainty_lower.colour = series_fmt[2].colour
        series_uncertainty_lower.style  = series_fmt[2].style

        series_uncertainty_upper.name   = series_fmt[3].name
        series_uncertainty_upper.marker = series_fmt[3].marker
        series_uncertainty_upper.colour = series_fmt[3].colour
        series_uncertainty_upper.style  = series_fmt[3].style

    series_list = [
        series_expected,
        series_median,
        series_uncertainty_lower,
        series_uncertainty_upper
    ]

    return series_list

def chart_qq(title,xaxis_title,yaxis_title,plotting_dist_icdf,yaxis_log=False,\
             data=None,data_passive=None,is_data_sorted=None,data_sfmt=None,\
             dist_icdf=None,dist_par=None,dist_pscal_base=None,dist_sfmt=None,\
             samples=None,samples_pscal_base=None,samples_sfmt=None,\
             tag=None):
    # data for complete qq plot with given probability scale
    # supports omitting 'data', 'dist_icdf' and 'samples'
    logger = logging.getLogger('chart_qq')

    cs = []

    if data is not None:
        cs_data = qqplot_series_discrete  (plotting_dist_icdf,data,data_passive,is_data_sorted,data_sfmt)
        cs += cs_data

    if dist_icdf is not None:
        cs_dist = qqplot_series_continuous(plotting_dist_icdf,dist_icdf,dist_par,dist_pscal_base,dist_sfmt)
        cs += cs_dist

    if samples is not None:
        cs_samp = qqplot_series_samples   (plotting_dist_icdf,samples,samples_pscal_base,samples_sfmt)
        cs += cs_samp

    # pack into chart object
    chart = chart_qq_init(title,plotting_dist_icdf,yaxis_log,xaxis_title,yaxis_title,cs,tag)

    return chart

def chart_qq_series_fmt(distpar_name=None):
    # create QQ series formatting (standard layout)
    logger = logging.getLogger('chart_qq_series_fmt')

    # observations and passive
    sfmt_data_obs      = Series_fmt(name = 'Observations', marker = 'circle'     , colour = 'blue')
    sfmt_data_excluded = Series_fmt(name = 'Excluded'    , marker = 'square-open', colour = 'blue')
    sfmt_data = [sfmt_data_obs, sfmt_data_excluded]

    # quantile from given distribution (optional)
    if distpar_name is not None:
        sfmt_dist = Series_fmt(name = distpar_name, colour = 'red')
    else:
        sfmt_dist = None

    # quantiles from samples
    sfmt_eq = Series_fmt(name = 'Expected quantile', colour = 'green')
    sfmt_mq = Series_fmt(name = 'Median quantile  ', colour = 'orange')
    sfmt_lb = Series_fmt(name = '90% conf limits L', colour = 'gray', style = 'dash')
    sfmt_ub = Series_fmt(name = '90% conf limits U', colour = 'gray', style = 'dash')
    sfmt_samp = [sfmt_eq, sfmt_mq, sfmt_lb, sfmt_ub]

    return sfmt_data, sfmt_dist, sfmt_samp

def pscale_axis_points(
        primary_points=pscale_ep_primary_points_default,
        secondary_intervals=pscale_ep_secondary_points_default,
        tertiary_factor=pscale_ep_tertiary_factor_default,
):
    # generate points for constructing probability scale plots
    logger = logging.getLogger('pscale_axis_points')

    all_ticks = np.array(primary_points)

    for i in range(min(len(secondary_intervals),len(primary_points)-1)):
        # points between primary points:
        ticks = np.linspace(primary_points[i],primary_points[i+1],secondary_intervals[i]*tertiary_factor,endpoint=False)
        ticks = ticks[1:] # discard first secondary point because it coincides with the primary point
        all_ticks = np.r_[all_ticks,ticks]

    all_ticks = ep2cp(all_ticks) # convert from exceedence percentage to cumulative probability
    all_ticks.sort()
    all_ticks = all_ticks[::-1] # descending order for consistency with EP format convention

    return all_ticks

def linterp(x,xa,xb,fa,fb):
    # linear trend prediction allowing for extrapolation
    wa = (xb-x)/(xb-xa); wb = 1.0-(x-xa)/(xb-xa); w = 0.5*(wa+wb)
    f = fa*w + fb*(1.0-w)
    return f

def expedentialD(x,xa,xb,fa,fb,top_asympt=None,bot_asympt=None):
    # expedential trenD (linear in log-space)
    # use for extrapolation where asymptotes are required (eg, edf interpolation)
    # by default, will generate an asymptote at 0 from above/right (eg, left-side of edf)
    # top_asympt requests a reflection to asymptote towards 'top_asympt' from below/left (eg, 1 also edf-related)
    # bot_asympt specifies a different bottom asymptote (0 by default)
    # note: only single asymptote can be provided!
    bot = bot_asympt is not None
    top = top_asympt is not None
    if bot and top: return 'f-expedentialD/illegal_usage'
    if bot:
        faL = fa-bot_asympt; fbL = fb-bot_asympt
    elif top:
        faL = top_asympt-fa; fbL = top_asympt-fb
    else:
        faL = fa; fbL = fb
    log_fa = np.log(faL); log_fb = np.log(fbL)
    t = linterp(x,xa,xb,log_fa,log_fb)
    ret = np.exp(t)
    if bot:
        ret = ret+bot_asympt
    elif top:
        ret = top_asympt-ret
    return ret

#-----------------------

# For testing expedential extrapolation approaches
#tempx=[0,1]; tempf=[0.011952191,0.03187251]; tempff=[0.96812749,0.988047809]

#res1=dx_tools.interp(x=-1.0,xp=tempx,fp=tempf,left='Exp',right='Exp',edf=True)
#print("res1=",res1)
#res2=dx_tools.interp(x=2.0,xp=tempx,fp=tempff,left='Exp',right='Exp',edf=True)
#print("res2=",res2)
#quit()

#-----------------------

def interp_1(x,xp,fp,left=None,right=None,left_x=None,right_x=None):
    # linear interpolation including outside tabulated endpoints
    # left_x limits x on the left; right_x limits x on the right (ignored when left/right is numeric)
    # left if numeric gives value on the outside left; right if numeric gives value outside on the right
    # with these conventions, 'interp_1' behaviour reduces to numpy.interp when 'left' and/or 'right' are numeric
    # Warning : make sure you know what you are doing when using this routine! :-)
    n = len(xp)
    leftM = left is 'Lin'
    rightM = right is 'Lin'
    if x < xp[0] and leftM:       # left extrapolation
        xL = x; xa = xp[0]; xb = xp[1]; fa = fp[0]; fb = fp[1]
        if (left_x is not None) and (xL < left_x): xL = left_x # left constraints
        ret = linterp(xL,xa,xb,fa,fb)
    elif x > xp[n-1] and rightM:  # right extrapolation
        xL = x; xa = xp[n-2]; xb = xp[n-1]; fa = fp[n-2]; fb = fp[n-1]
        if (right_x is not None) and (xL > right_x): xL = right_x # right constraint
        ret = linterp(xL,xa,xb,fa,fb)
    else:                         # revert to basic 'interp' if no fun options provided
        L = None if leftM else left
        R = None if rightM else right
        ret = np.interp(x,xp,fp,L,R) # left and right still get here when numeric
    return ret

def interp_edf_1(x,xp,fp,left=None,right=None,left_x=None,right_x=None):
    # linear interpolation of edf assuming exponential tails outside tabulated endpoints
    # left_x and right_x: limit x on left and right
    # left and right: limit fx on left and right (assuming monotonic up!)
    # Warning : make sure you know what you are doing when using this routine! :-)
    n = len(xp)
    if x < xp[0]:       # left extrapolation
        xL = x; xa = xp[0]; xb = xp[1]; fa = fp[0]; fb = fp[1]
        if (left_x is not None) and (xL < left_x): xL = left_x # left constraint on x
        ret = expedentialD(xL,xa,xb,fa,fb)
        if (left is not None) and ret < left: ret = left # right constraint on fx
    elif x > xp[n-1]:   # right extrapolation
        xL = x; xa = xp[n-2]; xb = xp[n-1]; fa = fp[n-2]; fb = fp[n-1]
        if (right_x is not None) and (xL > right_x): xL = right_x # right constraint on x
        ret = expedentialD(xL,xa,xb,fa,fb,top_asympt=1.0)
        if (right is not None) and x > right: ret = right # right constraint on fx
    else:               # revert to basic 'interp' when within table
        ret = np.interp(x,xp,fp) # left,right not provided here
    return ret

def interp_2(x,xp,fp,left=None,right=None,left_x=None,right_x=None):
    # overlord for vector 'x'
    nx = len(x)
    ret = np.ones(nx,dtype=float)
    for i in range(0,nx): ret[i] = interp_1(x[i],xp,fp,left,right,left_x,right_x,edf)
    return ret

def interp_edf_2(x,xp,fp,left=None,right=None,left_x=None,right_x=None):
    # overlord for vector 'x'
    nx = len(x)
    ret = np.ones(nx,dtype=float)
    for i in range(0,nx): ret[i] = interp_edf_1(x[i],xp,fp,left,right,left_x,right_x)
    return ret

def process_settings_engines(settings={},advanced_settings={}):
    # processes 'settings' and 'advanced_settings' to produce engine-specific settings

    optim_settings = {
        'do_optim': settings.get('enable_optim',True),
        'premeth': advanced_settings.get('premeth',settings.get('premeth','native_py')),
        'sdev_default': float(advanced_settings.get('sdev_default',settings.get('sdev_default'))),
        'gtol': settings.get('gtol',1.e-8),
        'opt_method': settings.get('opt_method','L-BFGS-B'),
    }

    covar_settings = {
        'do_covar': settings.get('enable_hessian',True),
    }

    mcmc_settings = {
        'do_mcmc': settings.get('enable_mcmc',True),
        'nwalkers': int(advanced_settings.get('nwalkers',settings.get('nwalkers'))),
        'niter': int(advanced_settings.get('niter',settings.get('niter'))),
        'ndiscard': int(advanced_settings.get('ndiscard',settings.get('ndiscard'))),
        'nthin': int(advanced_settings.get('nthin',settings.get('nthin'))),
    }

    pred_settings = {
        'nsam': settings.get('sample_size',1000),
    }

    seed = quickif(advanced_settings.get('rseed'),settings.get('seed',None))
    if seed is not None: seed = int(seed)
    rangen_settings = {
        'seed': seed,
    }

    return optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings

def get_svn_revision():
    # retrieve the SVN repo revision number - distributor of versions
    ret = get_svn_revision_1()
    return ret

def get_svn_revision_1():
    # retrieve the SVN repo revision number - using 'svn info' command in subprocess
    import subprocess
    try:   #execute command svn info and get revision from output
        output = subprocess.check_output(["svn", "info"])#, capture_output=True)
        #print('output:',output)
        revision = 'Undetermined'
        for line in output.splitlines():
            line = line.decode('ascii')
            if line.startswith('Revision:'):
                revision = line.split(':')
                if len(revision)>1:
                    revision = revision[1]
                else:
                    revision = revision[1]
        svn_revision = revision.strip()
    except:
        svn_revision = get_svn_revision_2()
    return svn_revision

def get_svn_revision_2():
    # retrieve the SVN repo revision number - using 'svn' package
    try:
        import svn.local
        rev = svn.local.LocalClient('.')
        info = rev.info()
        svn_revision = info['entry_revision'] #commit_revision']
    except:
        svn_revision = 'Undetermined'
    return svn_revision

def get_svn_revision_3():
    # retrieve the SVN repo revision number - using Bash
# #!/bin/bash
# svn $1
# svn info | grep Revision | awk '{print $2}' > ../dash_app/static/svn_revision.txt
    try:
        filename = 'static/svn_revision.txt'
        svn_revision = 'unknown'
        if os.path.exists(filename):
            try:
                with open(filename,'r') as fin:
                    svn_revision = fin.readline()
                    print('svn_revision:',svn_revision)
            except Exception as e:
                pass
        svn_revision = svn_revision.strip()
        #strip '\n'
        if svn_revision.endswith('\n'):
            svn_revision = svn_revision[:-1]
    except:
        svn_revision = 'Undetermined'
    return svn_revision

def get_setting(settings,key,default=None,raise_keyerror=True):
    '''we retrieve from settings using the key,
    but we try first ket_%OS% based on the running platform'''
    import platform
    running_os = platform.system()

    if running_os is not None:
        #try key_%OS%
        os_key ='%s%s%s'%(key,delimiter,running_os)
        return get_safe(settings,os_key,default=default,raise_keyerror=raise_keyerror)
    else:
        return get_safe(settings,key,default=default,raise_keyerror=raise_keyerror)

def get_safe(settings,key,default=None,raise_keyerror=True):
    '''we retrieve from settings using the key'''
    if key in settings:
        return settings[key]
    elif raise_keyerror:
        raise KeyError('key %s not found'%key)
    else:
        return default

# ----------------------

class ProbabilityModel(object):
    def __init__(self,dist):
        self.par_opt = None
        self.par_mean = None
        self.par_covar = None
        self.npar = None
        self.dist = dist
        self.parscale_ix = None #position of scale parameter in the parameters list
        self.bounds = None
        self.par_default = None

    def get_par(self):
        if self.par_mean is not None:
            return self.par_mean
        else:
            return self.par_opt

    def set_par(self,par):
        self.par_opt = par

    def set_par_default(self):
        raise "abstract method"

    def fit_predict(self,data,optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings):
        # Fits model parameters and generates predictive distribution
        logger = logging.getLogger('ProbabilityModel::fit_predict')

        premeth = optim_settings.get('premeth','native_py') #premeth = pre-method
        logger.debug('premeth is %s',premeth)

        par0 = None

        if premeth=='native_py': # some kind of 'Native Python' baked into 'dist.fit'
            t1 = time.time()
            par0_fit = self.dist.fit(data)
            par0 = np.asarray(par0_fit)
            t2 = time.time()

            logger.debug('par0 (raw) is %s',par0)
            par0 = parscale_lin2log(par0,self.parscale_ix)
            logger.debug('par0 (transformed) is %s',par0)
        elif (premeth=='null' or premeth is None):
            par0 = self.par_default
            logger.debug('par0 (via def) is %s',par0)
        else:
            raise Exception('premeth %s not yet supported'%premeth)

        self.par_opt = par0
        self.par_covar = None

        logpdf_f = lambda t: self.log_posterior_f(t,data)
        par_opt,par_covarH,par_samples,par_mean,par_covar,pred_samples = fit_predict_engine(
            logpdf_f,self.rvsb,par0,self.bounds,
            optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings
        )

        self.par_opt = par_opt
        if par_covarH is not None:
            self.par_covar = par_covarH   #FIXME: or use MCMC covar?

        self.par_mean = par_mean
        if par_covar is not None:
            self.par_covar = par_covar

        return par_opt,par_covarH,par_samples,par_mean,par_covar,pred_samples #FIXME: we are also changing 'self' !

    def log_prior_f(self,par):
        par_temp = self.mypar(par) # no unlogging
        log_prior = 0.0 # uniform prior
        return log_prior

    def log_likelihood_f(self,par,data):
        par_temp = self.mypar(par) # no unlogging
        log_like_term = self.logpdf(x=data,par=par_temp)
        #logger.debug('log_like_term: %s',log_like_term)
        log_like = np.sum(log_like_term)
        return log_like

    def log_posterior_f(self,par,data):
        log_prior = self.log_prior_f(par)
        log_like = self.log_likelihood_f(par,data) # statistical independence "assumed"
        log_post = log_prior + log_like
        return log_post

    def logpdf(self,x,par=None):
        #logger = logging.getLogger('ProbabilityModel::logpdf')
        #logger.debug('init par: %s',par)
        par_temp = self.mypar(par,unlog_parscale=True)
        ret = self.dist.logpdf(x,*par_temp)
        return ret

    def icdf(self,x,par=None):
        par_temp = self.mypar(par,unlog_parscale=True)
        ret = self.dist.ppf(x,*par_temp)
        return ret

    def cdf(self,x,par=None):
        par_temp = self.mypar(par,unlog_parscale=True)
        ret = self.dist.cdf(x,*par_temp)
        return ret

    def rvs(self,par,size=1):
        par_temp = self.mypar(par,unlog_parscale=True)
        ret = self.dist.rvs(*par_temp,size=size)
        return ret

    def rvsb(self,par,nsam_per_par=50): # batch version of 'rvs'
        #logger = logging.getLogger('rvsb')
        assert par.shape[1] == self.npar
        #nbatch_par = len(par)
        nbatch_par = par.shape[0]  #DK-FIXME: Is 'shape' safer than 'len' here?
        samples = np.empty((nbatch_par,nsam_per_par))

        #for i in range(nbatch_par):
        #    logger.debug('par at %d: %s',i,par[i])

        for i in range(nbatch_par):
            samp = self.rvs(par[i],size=nsam_per_par)
            #logger.debug('samples at %d: min=%f, mean=%f, max=%f',
            #    i,
            #    np.min(par[i]),
            #    np.mean(par[i]),
            #    np.max(par[i]),
            #)
            samples[i,:] = samp

        return samples

    def mypar(self,par,unlog_parscale=False):
        if par is None:
            par_temp = self.get_par().copy()
        else:
            par_temp = np.asarray(par)
        if unlog_parscale:
            #logger.debug('par_temp before log2lin: %s',par_temp)
            par_temp = parscale_log2lin(par_temp,self.parscale_ix)
            #logger.debug('par_temp after log2lin: %s',par_temp)
        return par_temp

class Normal(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.norm)
        self.npar = 2
        self.parscale_ix = 1 #second :-)
        self.bounds = [(None,None),(-10,10)]
        self.par_default = np.asarray([0,np.log(1)])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

class Exponential(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.expon)
        self.npar = 2
        self.parscale_ix = 1
        self.bounds = [(None,None),(-10,10)]
        self.par_default = np.asarray([0,np.log(1)])

    def fit_predict(self,data,*args): #DK[2020-04-18] : EVA r122 and ticket N99_C6 : simplifies DK maintenance, though Exe un-impressed :-(
        self.bounds = [(np.min(data),None),(-10,10)]
        ret = super().fit_predict(data,*args)
        return ret

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

class Pearson3(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.pearson3)
        self.npar = 3
        self.parscale_ix = 2 #third :-)
        self.bounds = [(None,None),(None,None),(-10,10)]
        self.par_default = np.asarray([0,0,np.log(1)])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

    def cdf(self,x,par=None):
        par_temp = self.mypar(par,unlog_parscale=True)
        ret = self.dist.cdf(x,*par_temp)

        # workaround for scipy Pearson-3 bug with shape<0
        if par_temp[0] < 0:
            ret = 1.0-ret

        return ret

    def icdf(self,x,par=None):
        x = np.asarray(x)
        par_temp = self.mypar(par,unlog_parscale=True)

        # workaround for scipy Pearson-3 bug with shape<0
        if par_temp[0] < 0:
            xtemp = 1.0-x
        else:
            xtemp = x

        ret = self.dist.ppf(xtemp,*par_temp)
        return ret

    def logpdf(self,x,par=None):
        logger = logging.getLogger('Pearson3::logpdf')
        temp = super().logpdf(x,par=par)
        ret = nan2inf(temp,negative=True)
        return ret

class GEV(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.genextreme)
        self.npar = 3
        self.parscale_ix = 2
        self.bounds = [(None,None),(None,None),(-10,10)]
        self.par_default = np.asarray([0,0,np.log(1)])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

class Gumbel(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.gumbel_r)
        self.npar = 2
        self.parscale_ix = 1
        self.bounds = [(None,None),(-10,10)]
        self.par_default = np.asarray([0,np.log(1)])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

class LinearScale(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.uniform)
        self.npar = 2
        self.parscale_ix = 1
        self.bounds = [(None,None)]*self.npar
        self.par_default = np.asarray([0,1])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

class GenPareto(ProbabilityModel):
    def __init__(self):
        super().__init__(scipy.stats.genpareto)
        self.npar = 3
        self.parscale_ix = 2
        self.bounds = [(None,None),(None,None),(-10,10)]
        self.par_default = np.asarray([0.1,0,np.log(1)])

    def set_par_default(self):
        par = self.par_default
        return super().set_par(par)

'''Distributions to support,
current implementations are based on scipy.stats package
'''

distributions = {
    'normal': Normal,
    'exp':Exponential,
    'pearson3': Pearson3,
    'genextreme': GEV,
    'gumbel': Gumbel,
    'genpareto': GenPareto,
    'linear': LinearScale,
}
