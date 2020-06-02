# -*- coding: utf-8 -*-
import datetime
import time
import os
import os.path
import numpy as np

import dash
from dash.dependencies import Input, Output, State

import dx_tools
import backend
import layouts

import logging
import logging.config
import importlib

import configurations

configs = configurations.app_config
#DONE dx_tools.configs = configurations.engine_config
#DONE backend.configs = configurations.backend_config
layouts.configs = configurations.layout_config


# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
]

# external CSS stylesheets
external_stylesheets = [
    'https://use.fontawesome.com/releases/v5.8.2/css/all.css',
    'https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.0/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.0/css/mdb.min.css',
]


app = dash.Dash(__name__,external_scripts=external_scripts,external_stylesheets=external_stylesheets) #



server = app.server

cell_format = "%0.4f"

app.layout = layouts.server_layout(None)

def build_columns_content(df):
    options = []
    for i,col in enumerate(df.columns.values):
        #print(col)
        options += [{'label': col, 'value': col}]

    #dd.value = df.columns.values[0]

    return options

@app.callback(
    Output('error-display-js', 'run'),
    [
        Input('error-display', 'data'),
    ])
def show_error(error_msg):
    if error_msg and len(error_msg) > 0:
        js_code = "alert('%s')"%error_msg
    else:
        js_code = ''

    return js_code

@app.callback(
    [
        Output('data-filename', 'children'),
        Output('data-description', 'children'),
        Output('selected-series', 'options'),
        Output('selected-series', 'value'),
        Output('attribute-table', 'columns'),
        Output('attribute-table', 'data'),
        Output('attribute-table','selected_rows'),
        Output('table-panel','style'),
        Output('output-panel','style'),
        Output('error-display','data')
    ],
    [
        Input('upload-data', 'contents')
    ],
    [
        State('upload-data', 'filename'),
        State('output-panel','style'),
        State('table-panel','style'),
    ])
def upload_file(file_content,filename,output_panel_style,table_panel_style):
    logger = logging.getLogger('upload_file')

    if output_panel_style is None: output_panel_style = {}
    output_panel_style['visibility'] ='hidden'

    if table_panel_style is None: table_panel_style = {}
    table_panel_style['visibility'] ='hidden'

    default_ret = [dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,table_panel_style,output_panel_style,'']

    if file_content is not None:

        max_file_size = layouts.configs.get('max_file_size',100*1024)
        max_file_rows = layouts.configs['max_file_rows']
        max_file_cols = layouts.configs['max_file_cols']

        df,data,description,err,message = dx_tools.load_data_2(
            file_content, filename, max_file_size, max_file_rows, max_file_cols
        )

        if err is 0:  # alles gut
            pass
        elif err is 100:
            default_ret[-1] = 'File size exceeds the maximum allowed'
            return default_ret
        elif err is 200:
            default_ret[-1] = 'File extension not recognised (supported: csv, txt, dat)'
            return default_ret
        elif err is 300:
            default_ret[-1] = 'Number of rows exceeds the maximum allowed'
            return default_ret
        elif err is 400:
            default_ret[-1] = 'Number of columns exceeds the maximum allowed'
            return default_ret
        else:         # some other error
            default_ret[-1] = message
            return default_ret

        options = build_columns_content(df)
        value = df.columns.values[0]
        columns = [{"name": col, "id": col,'type': 'numeric'} for col in df.columns.values]

        formatted_data = []
        for row in data:
            fd = {}
            for k,v in row.items():
                try:
                    fd[k]= cell_format%v
                except:
                    fd[k]= v

            formatted_data += [fd]

        output_panel_style['display'] ='inline'
        output_panel_style['visibility'] ='visible'
        table_panel_style['display'] ='inline'
        table_panel_style['visibility'] ='visible'

        sel_rows = [] #start with no selected rows

        return filename,description,options,value,columns,formatted_data,sel_rows,table_panel_style,output_panel_style,''
    else:
        return default_ret

@app.callback(
    Output('series-characteristics', 'value'),
    [
        Input('selected-series', 'value'),
        Input('attribute-table','selected_rows'),
    ],
    [
        State('attribute-table', 'data')
    ])
def series_selection(var_name,rows_sel,data_input):

    if data_input is None:
        return dash.no_update

    if (len(data_input) - 0 if rows_sel is None else len(rows_sel))>= 0:
        stats = dx_tools.simple_stats_2(data_input,var_name,rows_sel)

        ret_text = ''
        for k,v in stats.items():
            val,decimal_places = v
            fmt = '%0.{0}f'.format(decimal_places)
            ret_text += ('%s: '+fmt+', ')%(k,val)

        return ret_text
    else:
        return dash.no_update

@app.callback(
    Output('chart-unupdated-js', 'run'),
    [
        Input('selected-series', 'value'),
        Input('fitted-distributions', 'value'),
        Input('apply-log-transform','value'),
        Input('attribute-table','selected_rows'),
        Input('attribute-table', 'data'),
    ])
def graph_old(var_name,dist_name,apply_log,rows_sel,data_input):
    js_code = '''
    var graphDiv = document.getElementById('graph-refresh');

        graphDiv.innerHTML = '<strong>Press FIT to update chart</strong>';

    '''

    return js_code

@app.callback(
    Output('chart-updated-js', 'run'),
    [
    Input('graph-refresh-hidden', 'data'),
    ]
    )
def graph_old(new_text):
    js_code = '''
    var labelDiv = document.getElementById('graph-refresh');
    labelDiv.innerHTML = '%s';
    '''%new_text

    return js_code

@app.callback(
    [
        Output('graph', 'figure'),
        Output('fitted-par', 'value'),
        Output('estimated-quantiles', 'value'),
        Output('graph', 'style'),
        Output('graph-refresh-hidden','data'),
        Output('message-panel','children'),

    ],
    [
        Input('fit-button', 'n_clicks'),
    ],
    [
        State('selected-series', 'value'),
        State('fitted-distributions', 'value'),
        State('apply-log-transform','value'),
        State('attribute-table','selected_rows'),
        State('attribute-table', 'data'),
        State('graph', 'style'),
    ])
def fit_and_plot(fit_clicks,var_name,dist_name,apply_log,rows_sel,data_input,graph_style):
    if fit_clicks <= 0:
        raise dash.exceptions.PreventUpdate

    logger = logging.getLogger('fit_and_plot')
    logger.debug('fit_clicks: %s',fit_clicks)
    logger.debug('var_name: %s',var_name)
    logger.debug('dist_name: %s',dist_name)
    logger.debug('apply_log: %s',apply_log)

    default_ret = [dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,'']

    plotting_dist_name = 'native'

    if data_input is None or len(data_input) <= 0:
        default_ret[-1] = 'data_input is None or empty'
        return default_ret

    if apply_log is None or len(apply_log) == 0:
        apply_log = False
    else:
        apply_log = apply_log[0]=='logtransform'

    try:
        t1 = time.process_time()
        c1 = time.perf_counter()

        #TODO
        #speed up with the use of cache
        #from joblib import Memory
        #memory = Memory('.cachedir', verbose=1,bytes_limit=1000*1e3)
        #mode_production = False

        # fit distribution and generate chart data
        prob_chart_list,fit_info = backend.fit_plot(
            data_input,var_name,rows_sel,apply_log,
            dist_name,plotting_dist_name,
            configs=configurations.backend_config,
            configs_engine=configurations.engine_config
        )

        # unpack fitting info
        par_opt = fit_info['par_opt']
        par_mean = fit_info['par_mean']

        ep_reportgui = fit_info['ep_reportgui']
        perc_reportgui_optimalpar = fit_info['perc_reportgui_optimalpar']
        perc_reportgui_expectedpar = fit_info['perc_reportgui_expectedpar']

        new_figure = backend.generate_figure_ploty(
            prob_chart_list,
            layouts.chart_layout,
            configs=configurations.backend_config,
        )

        # construct output strings with values of par and quantiles
        quantiles_text = ''
        for label,perc_vals in [('Optimal par: ',perc_reportgui_optimalpar),('Expected par: ',perc_reportgui_expectedpar)]:
            quantiles_text = quantiles_text + label
            for i in range(len(perc_vals)):
                v1 = ep_reportgui[i]
                v2 = perc_vals[i]
                if i > 0:
                    quantiles_text += ', '
                quantiles_text += 'value at {0:0.0f}%: {1:0.3f}'.format(v1,v2)
            quantiles_text = quantiles_text + '; '

        paropt_str = backend.dist_par_template[dist_name][1].format(*par_opt)
        parexpected_str = backend.dist_par_template[dist_name][1].format(*par_mean)

        estimated_par_desc = "Optimal par: {}; Expected par: {}".format(paropt_str,parexpected_str)

        if graph_style is None:
            graph_style = {}

        graph_style['display'] = 'block'
        graph_style['visible'] = 'visible'
        graph_style['height'] = '550px'

        t2 = time.process_time()
        c2 = time.perf_counter()

        elapsed_time = t2-t1
        cpu_time = c2-c1
        msg_text = configs['fit_time_template']%(cpu_time,elapsed_time)

        default_ret = new_figure,estimated_par_desc,quantiles_text,graph_style,msg_text,''
    except dx_tools.ParameterOptimisationError as e:
        logger.exception(e)
        default_ret[-1] = 'Error in parameters optimisation'
        default_ret[0] = {}
    except dx_tools.HessianOptimisationError as e:
        logger.exception(e)
        default_ret[-1] = 'Error in Hessian covariance estimation'
        default_ret[0] = {}
    except dx_tools.EmceeSamplingError as e:
        logger.exception(e)
        default_ret[-1] = 'Error in MCMC sampling'
        default_ret[0] = {}
    except Exception as e:
        logger.exception(e)
        default_ret[-1] = 'Unexpected error'
        print('Unexpected error')

    return default_ret

if __name__ == '__main__':

    #in production we can speed up with cache and logging only errors.
    backend.mode_production = True
    app.run_server(debug=False)
