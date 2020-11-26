import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pickle

FNAME_STRING = '_every_3rd_day_top_1_sub_top_comments_only_method_comments_cleaned_standardized_vader_flair.csv'

LIST_OF_SUBREDDITS = ['politics', 'leagueoflegends', 'law', 'askreddit', 'plants', 'movies',
                      'globaloffensive', 'wallstreetbets', 'investing', 'upliftingnews',
                      '2meirl4meirl', 'me_irl', 'roastme', 'toastme', 'funny', 'aww']


def get_list_of_subreddit_values(listofsubreddits):
    """given a list of subreddit strings, return a dictionary of proper format for dash dropdown"""
    menu_dictionary_list = []

    for sr in listofsubreddits:
        file_name_string = sr + FNAME_STRING
        temp_dict = {'label': sr, 'value': file_name_string}
        menu_dictionary_list.append(temp_dict)

    return menu_dictionary_list


def test_get_list_of_subreddit_values():
    """prints out the dictionary of the list of subreddits"""
    sr_dict = get_list_of_subreddit_values(LIST_OF_SUBREDDITS)
    print(sr_dict)


def get_html_chart(subreddit_name, type_of_graph):
    """returns the path to the html graph"""
    pass


fname = 'C:/Users/Cooper/Documents/GitHub/SubredditSentimentTracker/backend/histograms_for_date_posted/' \
        'politics_date_posted_histogram_every_3rd_day_top_1_sub_top_comments_only_method.html'
#with open(fname, 'rb') as f:
#    fig = pickle.load(f)



# basic stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


# App layout
app.layout = html.Div([

    dcc.Dropdown(
        id='subreddit-dropdown',
        options=get_list_of_subreddit_values(LIST_OF_SUBREDDITS),
        placeholder="Select subreddit to view"
    ),

    html.Iframe(id='graph',
                src=fname)


])


#
"""
@app.callback(
    Output(component_id='item-dropdown', component_property='options'),
    Input('shop-dropdown', 'value')
)
def update_dropdown_option(shop_id_from_dropdown):
    temp_list = get_valid_item_list(train_grouped_month, shop_id_from_dropdown)
    return convert_list_to_options_dict(temp_list)


# Connect the Plotly graphs with Dash drop down Components
@app.callback(
    [Output(component_id='3d-scatter', component_property='figure'),
     Output(component_id='item_name', component_property='children')],
    [Input('submit-button-state', 'n_clicks')],
    [State("shop-dropdown", "value"),
     State("item-dropdown", "value")]
)
def update_graph(n_clicks, input_shop_id, input_item_id):
    update_df = create_one_shop_one_item_df(input_item_id, input_shop_id, train_grouped_month)
    # update_df = create_one_shop_df(input_shop_id, train_grouped_month)
    return create_3d_scatter_fig(update_df), 'Item Name: '.join(get_translated_name(input_item_id))

"""
# runs the whole thing
if __name__ == '__main__':
    app.run_server(debug=True)

