# -*- coding: utf-8 -*-
import datetime
import uuid
import sys
import os.path
import logging

import numpy as np
import scipy.stats

import plotly.graph_objects as go

import dx_tools
from dx_tools import distributions
from dx_tools import Series, Series_fmt, Axis, Chart

'''The values of each item is a list of:
      distribution name (label)
      parameter template for display
      flag indicating if the distribution is displayed for fitting
      flag indicating if the distribution is displayed for plotting
'''

dist_par_template = {
    'native': ('Native','loc={0:0.3f}, log-scale={1:0.3f}',True,True),
    'normal': ('Normal','loc={0:0.3f}, log-scale={1:0.3f}',True,True),
    'exp': ('Exponential','loc={0:0.3f}, log-scale={1:0.3f}',True,True),
    'pearson3': ('Pearson3','shape={0:0.3f}, loc={1:0.3f}, log-scale={2:0.3f}',True,False),
    'genextreme': ('GEV','shape={0:0.3f}, loc={1:0.3f}, log-scale={2:0.3f}',True,False),
    'gumbel': ('Gumbel','loc={0:0.3f}, log-scale={1:0.3f}',True,True),
    'linear': ('Linear','loc={0:0.3f}, log-scale={1:0.3f}',False,True),
    'genpareto': ('GenPareto','shape={0:0.3f}, loc={1:0.3f}, log-scale={2:0.3f}',True,False),
}

def fit_dist(dist_name,data,configs=None,advanced_settings=None):
    # Fit 'dist_name' to 'data' - wrapper for 'fit_predict'
    logger = logging.getLogger('fit_dist')

    # get distributions by name
    dist = distributions[dist_name]()

    # process settings and advanced_settings
    optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings = \
        dx_tools.process_settings_engines(configs,advanced_settings)

    # fit parameters and sample from (fitted) distribution
    par_opt,par_covarH,par_samples,par_mean,par_covar,dist_samples = dist.fit_predict(data,
        optim_settings,covar_settings,mcmc_settings,pred_settings,rangen_settings
    )

    return dist,dist_samples

def fit_plot(data_input,series_name,rows_excluded,apply_log,fitted_dist_name,plotting_dist_name,is_data_sorted=False,configs=None,configs_engine=None,advanced_settings=None):
    # Create qqplot from 'data_input', interface with numerical engines
    # Notes
    # 1. ARG 'data_input' has multiple variables: only 'series_name' is requested.
    # 2. Rows maybe excluded via ARG 'rows_excluded'
    logger = logging.getLogger('fit_plot')

    logger.debug('Serie name: %s',series_name)
    logger.debug('rows_excluded: %s',rows_excluded)
    logger.debug('apply_log: %s',apply_log)
    logger.debug('Distribution to fit: %s',fitted_dist_name)
    logger.debug('Plotting distribution: %s',plotting_dist_name)
    logger.debug('is_data_sorted: %s',is_data_sorted)
    logger.debug('configs: %s',configs)
    logger.debug('configs_engine: %s',configs_engine)
    logger.debug('advanced_settings: %s',advanced_settings)

    # extract data from the table
    data_original = np.array([float(row[series_name]) for row in data_input])
    n_data_original = len(data_original)

    # filter data - prepare mask
    if rows_excluded is None: # no excluded data
        indices_excluded = []
        n_data_excluded = 0
    else:                     # some data is excluded
        indices_excluded = rows_excluded
        n_data_excluded = len(indices_excluded)

        mask_sel = np.ones(n_data_original,dtype=bool); mask_sel[indices_excluded] = 0
        data_excluded = data_original[~mask_sel]

        logger.debug('data_excluded.y : %s',data_excluded)

    # filter data - apply mask
    data = dx_tools.filter(data_original,indices_excluded)
    n_data = len(data)

    # information from fitting (parameters, quantiles, etc)
    fit_info = {}

    # parse advanced_settings
    advanced_settings = dx_tools.parse_expr(advanced_settings)

    # log-transform data if requested (only needed for fitting model)
    if apply_log:
        data_transformed = np.log(data)
    else:
        data_transformed = data

    # fit selected model
    fitted_dist,fitted_dist_samples = fit_dist(fitted_dist_name,data_transformed,
        configs_engine,advanced_settings
    )

    data_transformed = None #FIXME: does this do proper garbage collection?

    logger.debug('fitted_dist:par_opt: %s',fitted_dist.par_opt)

    # key info from fiting
    fit_info['par_opt'] = fitted_dist.par_opt
    fit_info['par_mean'] = fitted_dist.par_mean

    fit_info['nsam'] = len(fitted_dist_samples)
    fit_info['npoints'] = n_data

    # compute percentiles for the requested exceedence percentages
    ep_reportgui     = np.array([10.0, 1.0]) #FIXME: exceedence percentages should come from configs
    perc_reportgui_optimalpar  = dx_tools.quantile_via_icdf(ep_reportgui,fitted_dist.icdf,fitted_dist.par_opt)
    perc_reportgui_expectedpar = dx_tools.quantile_via_icdf(ep_reportgui,fitted_dist.icdf,fitted_dist.par_mean)
    #FIXME: also ideally get perc_expected (NB: but from quantile samples, not icdf!)
    #FIXME: types of parameters to use here should also come from configs, use lists to store them

    logger.debug('perc_reportgui_optimalpar : %s',perc_reportgui_optimalpar)
    logger.debug('perc_reportgui_expectedpar: %s',perc_reportgui_expectedpar)

    # back transform
    if apply_log:
        fitted_dist_samples        = np.exp(fitted_dist_samples)
        perc_reportgui_optimalpar  = np.exp(perc_reportgui_optimalpar)
        perc_reportgui_expectedpar = np.exp(perc_reportgui_expectedpar)

    # more info
    fit_info['ep_reportgui'] = ep_reportgui
    fit_info['perc_reportgui_optimalpar'] = perc_reportgui_optimalpar
    fit_info['perc_reportgui_expectedpar'] = perc_reportgui_expectedpar

    # series formatting
    data_sfmt, dist_sfmt, samples_sfmt = dx_tools.chart_qq_series_fmt(distpar_name='Expected par quantile')

    # p-axis points used for continuous data - currently same for all plotting_dist
    pscal_base_1 = configs_engine.get('primary_x_scale_points',dx_tools.pscale_ep_primary_points_default)
    pscal_base_2 = configs_engine.get('secondary_x_scale_intervals',dx_tools.pscale_ep_secondary_points_default)
    pscal_base_3 = configs_engine.get('tertiary_x_scale_intervals',dx_tools.pscale_ep_tertiary_factor_default)

    pscal_base = [pscal_base_1, pscal_base_2, pscal_base_3]

    logger.debug('primary_x_scale_points : %s',pscal_base[0])
    logger.debug('secondary_x_scale_intervals : %s',pscal_base[1])
    logger.debug('tertiary_x_scale_factor : %s',pscal_base[2])

    # now generate all series for all plotting distributions
    prob_chart_list = []
    for plotting_dist_name,plotting_info in dist_par_template.items():
        plotting_dist_label,_,fit_flag,plot_flag = plotting_info
        if not plot_flag: continue

        if plotting_dist_name == 'native':
            plotting_dist = distributions[fitted_dist_name]()
            plotting_dist.set_par(fitted_dist.par_opt) #FIXME: need option to use 'par_mean', etc
        else:
            plotting_dist = distributions[plotting_dist_name]()
            plotting_dist.set_par_default()

        # title and axes labels
        prob_chart_title = gen_prob_chart_title(fitted_dist_name,plotting_dist_name,fit_info,configs)
        prob_chart_xaxis_title = 'Exceedance probability'
        prob_chart_yaxis_title = series_name

        # construct single chart
        prob_chart = dx_tools.chart_qq(prob_chart_title,\
            prob_chart_xaxis_title,prob_chart_yaxis_title,\
            plotting_dist.icdf,apply_log,\
            data,data_excluded,is_data_sorted,data_sfmt,\
            fitted_dist.icdf,fitted_dist.par_mean,pscal_base,dist_sfmt,\
            fitted_dist_samples,pscal_base,samples_sfmt,\
            plotting_dist_label)

        # account for log operation
        if apply_log:
            ii = 1 if n_data_excluded == 0 else 2
            prob_chart.series[ii].y = np.exp(prob_chart.series[ii].y)

        # add to the total list of charts (multiple probability scales)
        prob_chart_list += [prob_chart]

    return prob_chart_list,fit_info

def gen_prob_chart_title(dist_name,plotting_dist_name,fit_info,configs):
    # constructs chart title
    logger = logging.getLogger('gen_prob_chart_title')

    dist_label = dist_par_template[dist_name][0]
    if plotting_dist_name == 'native':
        plotting_dist_label = 'Native'
    else:
        plotting_dist_label = dist_par_template[plotting_dist_name][0]

    title_args = (dist_label,plotting_dist_label,fit_info['npoints'],fit_info['nsam'])
    title  = configs['title_template']%title_args

    return title

def generate_figure_ploty(prob_chart_list,chart_layout,configs=None,main_index=0):
    main_prob_chart = prob_chart_list[main_index]

    new_layout = go.Layout(**chart_layout)
    new_layout.yaxis.title = main_prob_chart.yaxis.label
    new_layout.yaxis.type = main_prob_chart.yaxis.type
    if main_prob_chart.xaxis.ticks_values is not None:
        new_layout.xaxis.tickmode = 'array'
        new_layout.xaxis.tickvals = main_prob_chart.xaxis.ticks_values
        new_layout.xaxis.ticktext = np.round(main_prob_chart.xaxis.ticks_display,1)

    new_layout.title = main_prob_chart.title

    data = []
    for series in main_prob_chart.series:
        scatter = go.Scatter(x=series.x,y=series.y,name=series.name)
        if series.marker is not None:
            scatter.mode='markers'
            scatter.marker={'symbol': series.marker}
        #generate line porps
        #line={'color':'grey','width':2,'dash':'dash'}
        line={}
        if series.colour is not None:
            line['color'] = series.colour
        if series.style is not None:
            line['dash'] = series.style
        scatter.line = line
        if series.hover is not None:
            hovertext_data = ["%0.3f,%0.3f"%(a,b) for a,b in zip(series.hover,series.y)]
            scatter.hovertext = hovertext_data

        data += [scatter]

    new_figure = go.Figure(
        data=data,
        layout=new_layout,
    )

    #generate all plotting scales for the dropdown:
    plotting_dist_scales = []

    for chart in prob_chart_list:
        plotting_dist_scales_x = []
        plotting_dist_scales_y = []
        plotting_dist_scales_mode = []
        plotting_dist_scales_marker = []
        plotting_dist_scales_line = []

        for s in chart.series:
            plotting_dist_scales_x += [s.x]
            plotting_dist_scales_y += [s.y]
            plotting_dist_scales_mode += ['markers' if s.marker is not None else None]
            plotting_dist_scales_marker += [{'symbol':s.marker} if s.marker is not None else None]
            plotting_dist_scales_line += [{'color':s.colour,'width':s.width,'dash':s.style}]

        layout_update_args = {
            "xaxis.tickmode": 'array',
            "xaxis.tickvals": chart.xaxis.ticks_values,
            "xaxis.ticktext": np.round(chart.xaxis.ticks_display,1),
            "yaxis.type": chart.yaxis.type,
            "title": chart.title
        }

        data_update_args = {
            'x': plotting_dist_scales_x,
            'y': plotting_dist_scales_y,
            'mode': plotting_dist_scales_mode,
            'markers': plotting_dist_scales_marker,
            'line': plotting_dist_scales_line,
        }

        plotting_dist_scales += [
            {
                'label':"X-axis %s"%chart.tag,
                'method':"update",
                'args':(data_update_args,layout_update_args)
            }
        ]

    new_figure.update_layout(
        autosize=True,
        margin=dict(t=60, b=0, l=0, r=0),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive= True,
                active = 1 if main_prob_chart.yaxis.type == 'linear' else 0,
                x=1.025,
                y=0.7,
                xanchor='left',
                buttons=list([
                    dict(label="Y-axis log",
                         method="relayout",
                         args=[{"yaxis.type":'log'}]),
                    dict(label="Y-axis linear",
                         method="relayout",
                         args=["yaxis.type",'linear']),
                ]),
            ),
            dict(
                type="dropdown",
                direction="down",
                showactive= True,
                active = 0,
                x=1.025,
                y=0.5,
                xanchor='left',
                buttons=plotting_dist_scales,
            ),
        ]
    )

    return new_figure
