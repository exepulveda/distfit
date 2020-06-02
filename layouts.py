# -*- coding: utf-8 -*-
import os.path
import dash

import uuid

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import visdcc
import dash_reusable_components as drc

import dx_tools

from backend import dist_par_template

configs = {}

#default layout of the chart
chart_layout = {
    'hovermode':"closest",
    #'legend'=dict(x=0, y=-0.01, orientation="h"),
    'autosize':True,
    'margin':dict(l=0, r=0, t=0, b=0),
    #'plot_bgcolor': "white", #"#282b38",
    #'paper_bgcolor': "white", #"#282b38",
    #'font':{"color": "black"}, #"#a5b1cd"},
    'template': 'plotly_white',
    'title': {
        'text':'Plot Title',
        'font': {
            #'family': 'Courier New, monospace',
            'size': 12
        },
        'xref': 'paper',
        'x': 0.05,
    },
    #'title_fontsize': 10,
    'xaxis': {
        'title':'Exceedance Probability',
        #'autorange': 'reversed',
        'type':'linear',
        #'gridcolor':'lightgrey',
        'tickcolor':'grey',
        'gridcolor':'grey',
        'gridwidth':1,
        'showgrid':True,
        'zeroline':False,
    },
    'yaxis': {
        'showgrid':True,
        'zeroline':False,
        'tickcolor':'grey',
        'gridwidth':1,
        'gridcolor':'grey'
    }, #'gridcolor':'lightgrey',}
}

graph_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'height': 500*2,
        'width': 800*2,
      }
}

def server_layout(mode=None):
    #return the layout of the GUI
    session_id = str(uuid.uuid4())

    #variable, distribution selection panel
    selection_panel = html.Div(id='selection-panel',children=[
        html.Div(id='selected-column-panel',children=[
            html.Div(id='selected-column-sec1-panel',children=[
                drc.NamedDropdown(
                    name='Select series',
                    id='selected-series',
                    searchable=False,
                    clearable=False
                ),
                dcc.Checklist(
                    id='apply-log-transform',
                    options=[{'label':' Apply log transform','value':'logtransform'}
                ]),
            ]),
            drc.NamedTextarea(
                id='series-characteristics',
                name='Series characteristics',
                cols='50',
                rows='2',
                readOnly='readOnly'
            ),
        ]),
        html.Div(id='apply-panel',children=[
            html.Div(id='apply-sec1-panel',children=[
                drc.NamedDropdown(
                    id='fitted-distributions',
                    name='Fitted distribution',
                    options=[
                        {'label': ' '+ v[0], 'value': k}
                        for k,v in dist_par_template.items() if k != 'native' and v[2]
                    ],
                    value='normal',
                    searchable=False,
                    clearable=False
                ),
            ]),
            # html.Div(id='apply-sec1b-panel',children=[
            #     drc.NamedDropdown(
            #         id='plotting-distributions',
            #         name='Probability scale',
            #         options=[
            #             {'label': ' '+ v[0], 'value': k}
            #             for k,v in dist_par_template.items() if v[2]
            #         ],
            #         value='native',
            #         searchable=False,
            #         clearable=False
            #     ),
            # ]),
            html.Div(id='apply-sec2a-panel',children=[
                html.Button('Fit',id='fit-button',n_clicks=0),
            ]),
            #dcc.Checklist(
            #    id='show-y-log',
            #    options=[{'label':' Show y-axis in log scale','value':'true'}
            #]),
        ]),
        html.Label(id='graph-refresh'),
        dcc.Store(
            id='graph-refresh-hidden',
            data=0
        ),
        html.Div(id='message-panel',children=[""]),

    ])

    #variable, distribution selection panel
    par_panel = html.Div(id='fitted-panel',children=[
        html.Div(id='fitted-sec1-panel',children=[
            drc.NamedTextarea(
                id='fitted-par',
                name='Estimated distribution parameters',
                readOnly='readOnly',
                cols= 40,
                rows= 3
            ),
            drc.NamedTextarea(
                id='estimated-quantiles',
                name='Estimated quantiles',
                readOnly='readOnly',
                cols= 40,
                rows= 3
            ),
        ])
    ])

    upload_panel = html.Div(id='last-card',children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
            }
        ),
        html.Label('''File upload limited to < %0.0f Kb, containing no more than %d rows and %d columns'''%(configs['max_file_size']/1024,configs['max_file_rows'],configs['max_file_cols'])),
        html.Label('',id='data-filename'),
        html.Label('',id='data-description'),
        html.Div(id='table-panel',children=[
            dash_table.DataTable(
                id='attribute-table',
                columns=[],
                row_selectable='multi',
                editable=False,
                style_header={
                    'textAlign': 'center',
                    'fontWeight': 'bold',
                    'color': 'black',
                },
                style_cell={
                    'padding': '2px',
                    '--selected-background': 'grey',
                    'color': 'black',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                #locale_format= '0.3',
            )
        ],style={'visibility':'none','display':'none'}),
    ])

    chart_panel = html.Div(id='chart-panel',children=[
        dcc.Loading(id = "loading-icon", children=[
            dcc.Graph(
                id='graph',
                figure={
                },
                responsive=True,
                config=graph_config,
                style={'display':'none'}
            )
        ]),
    ],
    style={'visibility':'none'})


    layout = html.Div(
        id="body",
        className="container scalable",
        children=[
            visdcc.Run_js(id = 'chart-updated-js'),
            visdcc.Run_js(id = 'chart-unupdated-js'),
            visdcc.Run_js(id = 'error-display-js'),
            dcc.Store(id='error-display',data=''),
            html.Title(title=configs['title']),
            #hidden session id
            #row 1 for title
            html.Div(id='top-panel',children=[
                html.H5(id='app-title',children=
                    '''Distribution Fitting and Analysis Tool.
                    This tool is only for demostration porpuses''',
                    style={"margin-bottom": "0px"},
                ),
            ],className="row flex-display"),
            #second row layout
            html.Div(id='main-panel',children=[
                html.Div(id='left-panel',children=[
                    upload_panel
                ]),
                html.Div(id='output-panel',children=[
                    selection_panel,
                    par_panel,
                    chart_panel
                ],style={'visibility':'none','display':'none'}),
            ]),
            #row 4 is the footer
            html.Div(id='footer-panel',children=[
                '''
                    A web application built on top of Dash (v%s) (framework for Python) by
                    Exequiel Sepúlveda and Dmitri Kavetski.'''%(dash.__version__)
            ]),
        ],
    )
    return layout
