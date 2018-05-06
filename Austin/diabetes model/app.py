import plotly.figure_factory as ff
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import base64

# Create the figure
df_sample = pd.read_csv('predict_proba.csv')
df_sample['FIPS'] = df_sample['FIPS'].apply(lambda x: str(x).zfill(5))
colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
endpts = list(np.linspace(1, 100, len(colorscale) - 1))
fips = df_sample['FIPS'].tolist()
values = df_sample['Probability'].tolist()
fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=False,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title="County's diabetes rate (2008) surpasses natural average of 9.9%",
    legend_title='Predicted Probability'
)

# Load the images
rocauc_image = base64.b64encode(open('rocauc.png', 'rb').read())
feature_import = base64.b64encode(open('feature_import.png', 'rb').read())
#

# Host the figure on Dash
app = dash.Dash()
server = app.server
app.layout = html.Div(children=[
    html.H1(children='Which USA counties are most likely to have high diabetes?'),

    html.Br(),


    dcc.Graph(
        id='my-graph',
        figure=fig
    ),

    html.Div([
        html.Div([
            html.H4('ROC-AUC Score'),
            html.Img(src='data:image/png;base64,{}'.format(rocauc_image.decode()), style={'width': '400px'})
            ], className="six columns"),
        html.Div([
            html.H4('Feature Importance'),
            html.Img(src='data:image/png;base64,{}'.format(feature_import.decode()), style={'width': '600px'})
            ], className="six columns"),
        ], className="Row"),

    html.Div(' Data Source: USDA Food Atlas \
                (https://www.ers.usda.gov/data-products/food-environment-atlas/data-access-and-documentation-downloads) '),
    html.Div('Dashboard built using Dash for Python (Austin Lasseter, 2018)'),


])

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server(debug=True)
