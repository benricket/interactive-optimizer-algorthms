import dash
from dash import dcc, html, Input, Output, State
import dash_extensions as de
import plotly.graph_objs as go
import numpy as np

# Create the Dash app
app = dash.Dash(__name__)

# Layout for the app
app.layout = html.Div([
    de.EventListener(
        id='listener',
        events=[{'event': 'keydown', 'props':['key']}],  # Listen for keydown events
        style={'height': '100%'}
    ),
    dcc.Graph(id='surface-plot', style={'height': '90vh'}),
    dcc.Store(id='saved-points', data=[]),
    dcc.Store(id='hovered-point', data=None),
    html.Div(id='output'),
    html.H1("Interactive 3D Function Plotter"),
    html.Label("Enter a function (in terms of x and y):"),
    dcc.Input(id='function-input', value='(x-1)**2 + (y-2)**2 + 3', type='text', style={'width': '80%'}),
    html.Button('Update Plot', id='submit-button', n_clicks=0)
])

# Callback to update the graph when the function or button is changed
@app.callback(
    Output('surface-plot', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('function-input', 'value'),
     State('saved-points', 'data')]
)
def update_graph(n_clicks, function_string, saved_points):
    x = np.linspace(-2, 4, 50)
    y = np.linspace(-1, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    try:
        Z = eval(function_string, {"x": X, "y": Y, "np": np})
    except:
        Z = np.zeros_like(X)
    
    surface = go.Surface(z=Z, x=x, y=y)

    points_x = []
    points_y = []
    points_z = []

    for point in saved_points:
        points_x.append(point[0])
        points_y.append(point[1])
        points_z.append(point[2])

    points = go.Scatter3d(
        x=points_x, 
        y=points_y, 
        z=points_z, 
        mode='markers+lines',
        marker=dict(size=5, color='red'),
        line=dict(width=2, color='blue')
    )

    fig = go.Figure(data=[surface, points])
    fig.update_layout(title="Custom Surface Plot with Points and Lines",
                      scene=dict(
                          xaxis_title='X-axis',
                          yaxis_title='Y-axis',
                          zaxis_title='Z-axis'),
                      height=600)

    return fig

@app.callback(
    Output('hovered-point', 'data'),
    Input('surface-plot', 'hoverData')
)
def capture_hovered_point(hoverData):
    if hoverData and 'points' in hoverData and len(hoverData['points']) > 0:
        point = hoverData['points'][0]
        return [point['x'], point['y'], point['z']]
    return dash.no_update

@app.callback(
    Output('saved-points', 'data'),
    Input('listener', 'event'),
    State('hovered-point', 'data'),
    State('saved-points', 'data')
)
def save_point_on_space(event, hovered_point, saved_points):
    # Initialize saved_points as an empty list if it's None
    if saved_points is None:
        saved_points = []
    # Check if event is a space key press and there's a hovered_point
    if event and event.get('key') == ' ' and hovered_point:
        saved_points.append([hovered_point[0],hovered_point[1]])
        print(saved_points)
        return saved_points  # Return updated list of saved points
    
    return dash.no_update





# Run the app
if __name__ == '__main__':
    app.run(debug=True)
