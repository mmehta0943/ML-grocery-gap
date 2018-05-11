import plotly.figure_factory as ff
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import base64

## PART 1. DATA AND IMAGES
# Create the diabetes figure
df_diabetes = pd.read_csv('predict_proba.csv')
df_diabetes['FIPS'] = df_diabetes['FIPS'].apply(lambda x: str(x).zfill(5))

endpts = list(np.linspace(1, 100, 12))
fips = df_diabetes['FIPS'].tolist()
values = df_diabetes['Probability'].tolist()
fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,

    county_outline={'color': 'rgb(255,255,255)', 'width': 0.1},
    round_legend_values=True,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title="Predicted probability that the county experiences a diabetes rate higher than the national average",
    legend_title='Predicted Probability'
)

# Create the grocery store figure
df_stores = pd.read_csv('output_file2.csv', names=['Probability', 'FIPS'], header=0)
df_stores['FIPS']=df_stores['FIPS'].astype(int)
df_stores['FIPS'] = df_stores['FIPS'].apply(lambda x: str(x).zfill(5))
df_stores['Probability']=round(df_stores['Probability']*100, 1)

endpts = list(np.linspace(1, 100, 16))
fips = df_stores['FIPS'].tolist()
values = df_stores['Probability'].tolist()
fig2 = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,

    county_outline={'color': 'rgb(255,255,255)', 'width': 0.1},
    round_legend_values=True,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title="Predicted probability that the county experiences a decrease in grocery stores over next 5 years",
    legend_title='Predicted Probability'
)


# Model images
rocauc_image = base64.b64encode(open('pics/rocauc.png', 'rb').read())
rocauc_2 = base64.b64encode(open('pics/ROC-AUC curve grocery.png', 'rb').read())
feature_import = base64.b64encode(open('pics/Diabetes Model Feature Importance.png', 'rb').read())
# Home page images
deloitte = base64.b64encode(open('pics/Deloitte_Logo.png', 'rb').read())
austin = base64.b64encode(open('pics/Austin.png', 'rb').read())
kaushik = base64.b64encode(open('pics/Kaushik.png', 'rb').read())
milonee = base64.b64encode(open('pics/Milonee_Mehta.png', 'rb').read())
swapna = base64.b64encode(open('pics/Swapna Batta.png', 'rb').read())
rohit = base64.b64encode(open('pics/Rohit.png', 'rb').read())
# Project description images
fig301 = base64.b64encode(open('pics/OutnumberHunger.png', 'rb').read())
fig302 = base64.b64encode(open('pics/Food Insecurity Spiral.png', 'rb').read())
fig303 = base64.b64encode(open('pics/Model Development.png', 'rb').read())
#





## PART 2. BUILDING OUT THE DASHBOARD

app = dash.Dash()
server = app.server
app.config.suppress_callback_exceptions=True

# Home page
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')

])
index_page = html.Div([
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(deloitte.decode()), style={'width': '300px'})
        ]),
        html.H1(children='Predictive Health Dashboard for USDA'),
    html.Div('Machine learning models to predict county-level health outcomes. Submission for the Machine Learning Hackathon, May 2018'),

    dcc.Link('Go to Model 1: Predicting Diabetes using Random Forest', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Model 2: Predicting Grocery Store Decline using DNN', href='/page-2'),
    html.Br(),
    dcc.Link('Go to Project Description', href='/page-3'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([

        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(milonee.decode()), style={'width': '100px'}),
            html.Div('Milonee Mehta, Business Technology Analyst, Consulting (Arlington, VA)'),
            ], className="two columns"),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(austin.decode()), style={'width': '100px'}),
            html.Div('Austin Lasseter, Senior Data Scientist, Consulting (Arlington, VA)'),
            ], className="two columns"),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(rohit.decode()), style={'width': '100px'}),
            html.Div('Rohit Shah, Consultant, Consulting (Mumbai, India)'),
            ], className="two columns"),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(kaushik.decode()), style={'width': '100px'}),
            html.Div('Kaushik Moudgalya, Advisory Analyst, Risk & Financial Advisory (Bengaluru, India)'),
            ], className="two columns"),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(swapna.decode()), style={'width': '100px'}),
            html.Div('Swapna Batta, Specialist Senior, Audit & Assurance, (Detroit, MI)'),
            ], className="two columns"),
        ], className="Row"),

])


# Page 1
page_1_layout = html.Div([

        html.H1(children='Model 1: Predicting Diabetes using Random Forest'),
        html.Br(),
        dcc.Graph(
            id='my-graph',
            figure=fig
        ),
        html.Div([
            html.P('Diabetes Model: This model uses a random forest classifier to determine the probability that a \
            county in the United States experiences a diabetes rate greater than the national average of 9.9%. \
            It is trained on features from 2009 to predict the diabetes rate in 2009. It has a ROC-AUC score of 90%, and an accuracy of 80%.'),
        ], style={'margin':20}),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(rocauc_image.decode()), style={'width': '400px'}),
            html.Img(src='data:image/png;base64,{}'.format(feature_import.decode()), style={'width': '600px'})
        ], className="Row"),

        html.H4(children='Navigation'),
        html.Div([
            dcc.Link('Go to Model 2', href='/page-2'),
            html.Br(),
            dcc.Link('Go to Project Description', href='/page-3'),
            html.Br(),
            dcc.Link('Go back to home', href='/'),
            html.Br(),
            html.A('Model 1 Code on Github', href='https://github.com/austinlasseter/predicting_diabetes_2008'),
        ], className="three columns")

]),


# Page 2
page_2_layout = html.Div([
    html.H1(children='Model 2: Predicting Grocery Store Decline'),
    html.Br(),
    dcc.Graph(
        id='my-graph2',
        figure=fig2
    ),
    html.Div([
    html.P('This model uses a neural network to determine the probability that the number of grocery stores in \
    a county will decrease over the next 5 years. It is trained on data from 2009 to predict if there will be a \
    decrease in stores between 2009 and 2014. It has a ROC-AUC score of 72%, and an accuracy of 75%.')
    ], style={'margin':20}),
    html.Div([
            html.Img(src='data:image/png;base64,{}'.format(rocauc_2 .decode()), style={'width': '500px'})
        ]),
    html.H4(children='Navigation'),
    dcc.Link('Go to Model 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Project Description', href='/page-3'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.A('Code on Github', href='https://github.com/mmehta0943/ML-grocery-gap'),
])



# Page 3 layout
page_3_layout = html.Div([
    html.H1(children='About This Dashboard'),
    html.Img(src='data:image/png;base64,{}'.format(fig303.decode()), style={'width': '700px'}),
    html.Div([
        html.P('The purpose of this dashboard is to help state governments find a way to \
        adequately allocate their limited resources for fighting food insecurity and diet-related illnesses.'),
        html.P('The first model takes non-health related data for every county, and outputs the probability that the \
        county experiences a diabetes rate greater than the national average. The power of this model is that \
        it allows state governments to collect meaningful health data without having to conduct costly/\
        time-consuming health surveys. Furthermore, it allows states to determine health statistics without \
        dealing with Protected Health Information (PHI), which can be a pain due to the strict privacy protocols \
        necessary for handling this type of information.'),
        html.P('The second model takes data regarding various factors such as food access, food insecurity, and \
        socioeconomic statistics, and outputs the probability that a county within a state will experience a \
        decrease in its number of grocery stores over the next 5 years. This model would be helpful to state \
        governments who are trying to take action on reducing insecurity. Whenever a grocery store closes down, \
        the neighborhood surrounding it potentially loses its primary source of fruits, vegetables, and whole grains, \
        which could further exacerbate the problem of food insecurity in that county. Therefore, it is in the state’s \
        best interest to stop stores from closing down. By being able to predict into the future, the government could\
         take preventative action to keep stores from closing down, by offering tax breaks or some other incentive.'),
        html.P('The two models should be used in conjunction to direct government resources towards the counties \
        which need them most. The first model will diagnose where the problem exists, and the second model will offer \
        insight into where preventative action can be taken.'),
        ], style={'margin':20}),

    html.H2(children='Background'),
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(fig301.decode()), style={'width': '300px'}),
        html.Img(src='data:image/png;base64,{}'.format(fig302.decode()), style={'width': '300px'}),

    ], className="Row"),
    html.H2(children='Data Sources'),
    html.Div([
        html.P(html.A('USDA Food Atlas', href='https://www.ers.usda.gov/data-products/food-environment-atlas.aspx', target="_blank")),
        html.P(html.A('Feeding America: Understand Food Insecurity', href='https://hungerandhealth.feedingamerica.org/understand-food-insecurity/', target="_blank")),
        html.P(html.A('Daily Herald: Poor neighborhoods struggle without supermarkets', href='http://www.dailyherald.com/article/20151206/entlife/151209164/', target="_blank")),
        html.P(html.A('CDC: National Diabetes Statistics Report', href='https://www.cdc.gov/diabetes/data/statistics-report/deaths-cost.html', target="_blank")),
        html.P(html.A('Mashable: Supermarket chains avoid low-income neighborhoods', href='https://mashable.com/2015/12/08/supermarkets-food-deserts/#WTjDxQF.qSq0', target="_blank")),
        html.P(html.A('Feeding America: Western Michigan', href='https://www.feedwm.org/freshstart/', target="_blank")),
    ], style={'margin':20}),


    html.Div(id='page-3-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])



# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page



app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server()
