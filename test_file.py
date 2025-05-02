import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_extensions as de
import plotly.graph_objs as go
import numpy as np
from optimize import Optimizer

# Create the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# Algorithm descriptions
algorithm_descriptions = {
    'gradient_descent': 'Gradient Descent is a first-order optimization algorithm that iteratively moves in the direction of steepest descent to find the minimum of a function.',
    'newton': 'Newton\'s method uses both first and second derivatives (Hessian) to find the minimum of a function. It typically converges faster than gradient descent but is more computationally intensive.',
    'BFGS': 'BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method that approximates the Hessian matrix to avoid costly second derivative calculations.'
}

# Layout for the app
app.layout = html.Div([

    # Title and toggle view button
    html.Div([
        html.H1("Interactive 3D Function Plotter", style={'margin': '0'}),
        html.Button("Switch View", id='toggle-view', n_clicks=0)
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),

    dcc.Store(id='view-mode', data='surface'),

    # Event listener and graph for first view

    html.Div([
        html.Div([
            de.EventListener(
                id='listener',
                events=[{'event': 'keydown', 'props': ['key']}],
                style={'height': '100%'}
            ),
            dcc.Graph(id='surface-plot', style={'height': '80vh', 'width': '100%'}),
        ], id='first-view', style={'height': '80vh', 'width': '70%', 'position': 'relative', 'z-index': 2, 'display': 'inline-block'}),  # initially visible

        html.Div([
            html.H3("Optimization Settings", style={'textAlign': 'center'}),
            html.Label("Select Algorithm:"),
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=[
                    {'label': 'Gradient Descent', 'value': 'gradient_descent'},
                    {'label': 'Newton\'s Method', 'value': 'newton'},
                    {'label': 'BFGS', 'value': 'BFGS'}
                ],
                value='BFGS',
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            html.Div([
                html.Label("Algorithm Description:"),
                html.Div(id='algorithm-description', 
                         style={'border': '1px solid #ddd', 'padding': '10px', 'minHeight': '100px', 
                                'marginBottom': '10px', 'backgroundColor': '#f9f9f9'})
            ]),
            html.Div([
                html.Label("Optimization Parameters:"),
                html.Label("Max Iterations:"),
                dcc.Input(id='max-iters-input', value='100', type='number', 
                          style={'width': '100%', 'marginBottom': '10px'}),
                html.Label("Tolerance:"),
                dcc.Input(id='tolerance-input', value='1e-6', type='text', 
                          style={'width': '100%', 'marginBottom': '10px'}),
            ]),
            html.Div([
                html.H4("Status:"),
                html.Div(id='input-feedback', style={
                    'border': '1px solid #ddd', 
                    'padding': '10px', 
                    'marginBottom': '10px',
                    'backgroundColor': '#f0f8ff',
                    'borderRadius': '5px',
                    'minHeight': '40px'
                }),
                dcc.Loading(
                    id="loading-optimization",
                    type="circle",
                    children=html.Div(id='optimization-status', style={
                        'border': '1px solid #ddd', 
                        'padding': '10px', 
                        'marginBottom': '10px',
                        'backgroundColor': '#f9f9f9',
                        'minHeight': '100px'
                    })
                )
            ])
        ], style={'height': '80vh', 'width': '30%', 'position': 'relative', 'z-index': 2, 
                  'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 20px'})
    ], id='first-view-controls', style={'display': 'flex','flexDirection':'row','justifyContent': 'flex-start', 'width': '100%'}),

        html.Div(id='second-view', children=[
            html.Div([
                html.Label("Number of Dimensions:"),
                dcc.Input(id='num-dimensions', type='number', value=10, min=1),
                html.Label("Select Function:"),
                dcc.Dropdown(
                    id='function-selector',
                    options=[
                        {'label': 'Rosenbrock', 'value': 'rosenbrock'},
                        {'label': 'Sphere', 'value': 'sphere'},
                        {'label': 'Rastrigin', 'value': 'rastrigin'}
                    ],
                    value='rosenbrock'
                ),
                html.Label("Tolerance:"),
                dcc.Input(id='tolerance', type='number', value=1e-6, step=1e-7),

                html.Label("Max Iterations:"),
                dcc.Input(id='max-iterations', type='number', value=100, min=1),

                html.Button("Run Optimizer", id='run-optimizer', n_clicks=0)
            ], style={'marginBottom': '1em'}),

                html.Div("Optimizer differences really start to matter with high-dimensional functions, \
                         but it's hard to view those as easily for the purpose of demonstration. Here, we've \
                         visualized how each variable converges towards its optimal value. By clicking \"Run Optimization\", \
                         the selected optimization algorithm will run on the selected test function in the given number of dimensions. \
                         Every column of the heatmap below represents a variable, and every row represents a single guess for. Lower \
                         values indicate the value is closer to its ideal value at the nearest local optimum.", style={"fontSize": "14px","text-align":"center"}),

            html.Div([
                dcc.Graph(id='optimizer-progress-1', style={'width': '50%'}),
                dcc.Graph(id='optimizer-progress-2', style={'width': '50%'})
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ], style={'display': 'none'}),  # Hidden by default

    dcc.Store(id='saved-points', data=[]),
    dcc.Store(id='optimization-results', data=None),
    dcc.Store(id='hovered-point', data=None),
    dcc.Store(id='is-optimizing', data=False),
    dcc.Store(id='current-function', data="(x-1)**2 + (y-2)**2 + 3"),  # Store the current function
    html.Div(id='output'),

    html.Div([
        html.H3("Function Parameters", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.Label("Function (x,y):"),
                dcc.Input(id='function-input', value='(x-1)**2 + (y-2)**2 + 3', type='text', 
                          style={'width': '90%', 'position': 'relative', 'z-index': 3, 'margin': 'auto', 'display': 'block'}),
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '0 10px'}),
            html.Div([
                html.Label("X range (min,max):"),
                dcc.Input(id='x-range-input', value='-2,4', type='text', 
                          style={'width': '90%', 'position': 'relative', 'z-index': 3, 'margin': 'auto', 'display': 'block'}),
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '0 10px'}),
            html.Div([
                html.Label("Y range (min,max):"),
                dcc.Input(id='y-range-input', value='-1,5', type='text', 
                          style={'width': '90%', 'position': 'relative', 'z-index': 3, 'margin': 'auto', 'display': 'block'}),
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '0 10px'}),
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
        html.Button('Update Plot', id='submit-button', n_clicks=0, 
                    style={'position': 'relative', 'z-index': 3, 'margin': '10px auto', 'display': 'block'})
    ], id='first-view-options',style={'position': 'relative', 'z-index': 3, 'textAlign': 'center', 'margin': '20px auto', 'width': '80%'}),


    ])

# Store the current function for reuse in callbacks
@app.callback(
    Output('current-function', 'data'),
    [Input('submit-button', 'n_clicks')],
    [State('function-input', 'value')]
)
def update_current_function(n_clicks, function_string):
    return function_string

# Callback to update the input feedback status
@app.callback(
    Output('input-feedback', 'children'),
    [Input('listener', 'event'),
     Input('function-input', 'value'),
     Input('x-range-input', 'value'),
     Input('y-range-input', 'value'),
     Input('submit-button', 'n_clicks'),
     Input('hovered-point', 'data'),
     Input('is-optimizing', 'data'),
     Input('surface-plot', 'hoverData')]
)
def update_input_feedback(event, function, x_range, y_range, n_clicks, 
                          hovered_point, is_optimizing, hover_data):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Is the user currently hovering?
    is_currently_hovering = hover_data is not None and 'points' in hover_data and len(hover_data['points']) > 0
    
    if is_optimizing:
        return html.Div([
            html.P("Optimization in progress...", style={'color': 'blue', 'fontWeight': 'bold'}),
            html.P("Please wait while the algorithm runs. This may take a few seconds.")
        ])
    
    if trigger_id == 'listener' and event and event.get('key') == 'w':
        if hovered_point:
            return html.Div([
                html.P("Space key pressed! ✓", style={'color': 'green'}),
                html.P(f"Point added at: ({hovered_point[0]:.2f}, {hovered_point[1]:.2f}, {hovered_point[2]:.2f})"),
                html.P("Starting optimization from this point...")
            ])
        else:
            return html.P("Space key pressed, but no point is currently hovered.", style={'color': 'orange'})
    
    elif trigger_id == 'submit-button':
        return html.Div([
            html.P("Update Plot button clicked! ✓", style={'color': 'green'}),
            html.P("Updating surface and re-running optimization if points exist...")
        ])
    
    elif trigger_id in ['function-input', 'x-range-input', 'y-range-input']:
        return html.Div([
            html.P(f"{trigger_id.replace('-input', '').title()} changed! ✓", style={'color': 'green'}),
            html.P("Click 'Update Plot' to apply changes.")
        ])
    
    # Show hover information
    elif is_currently_hovering and hovered_point:
        return html.Div([
            html.P("Hovering over surface", style={'color': 'blue'}),
            html.P(f"Current position: ({hovered_point[0]:.2f}, {hovered_point[1]:.2f}, {hovered_point[2]:.2f})"),
            html.P("Press SPACE to add this point and start optimization.")
        ])
    else:
        return html.Div([
            html.P("Not hovering over surface", style={'color': 'gray'}),
            html.P("Hover over the surface to select a point.")
        ])

# Callback to update algorithm description
@app.callback(
    Output('algorithm-description', 'children'),
    Input('algorithm-dropdown', 'value')
)
def update_algorithm_description(algorithm):
    return algorithm_descriptions.get(algorithm, "No description available.")

# Function to interpolate points along optimization path
def interpolate_path_on_surface(path, function_string, num_points=10):
    """
    Generate smooth path that follows the surface by interpolating between optimization points.
    
    Args:
        path: List of [x, y] points from optimization
        function_string: String representation of the function to evaluate z values
        num_points: Number of intermediate points to generate between each path point
        
    Returns:
        List of [x, y, z] points that follow the surface
    """
    if not path or len(path) < 2:
        return []
    
    smooth_path = []
    
    for i in range(len(path) - 1):
        # Get consecutive points
        p1 = path[i]
        p2 = path[i+1]
        
        # Generate interpolated points
        for j in range(num_points + 1):
            # Linear interpolation parameter
            t = j / num_points
            
            # Interpolate x and y
            x = p1[0] * (1 - t) + p2[0] * t
            y = p1[1] * (1 - t) + p2[1] * t
            
            # Calculate z value from function
            try:
                z = eval(function_string, {"x": x, "y": y, "np": np})
                smooth_path.append([x, y, z])
            except:
                # Skip this point if function evaluation fails
                continue
                
    return smooth_path

# Callback to update the graph based on inputs or saved points changes
@app.callback(
    Output('surface-plot', 'figure'),
    [Input('submit-button', 'n_clicks'),
     Input('saved-points', 'data'),
     Input('optimization-results', 'data'),
     Input('x-range-input', 'value'),
     Input('y-range-input', 'value'),
     Input('current-function', 'data')]
)
def update_graph(n_clicks, saved_points, optimization_results, x_range_str, y_range_str, function_string):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    print(f"update_graph triggered by: {trigger_id}") # Debug print
    print(f"  - Received saved_points: {saved_points}") # DEBUG: Check received points
    
    # Initialize saved_points if None
    if saved_points is None:
        saved_points = []
    
    # Initialize optimization_results if None
    if optimization_results is None:
        optimization_results = []
    
    try:
        x_min, x_max = map(float, x_range_str.split(','))
        y_min, y_max = map(float, y_range_str.split(','))
    except:
        x_min, x_max = -2, 4
        y_min, y_max = -1, 5
    
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    
    try:
        Z = eval(function_string, {"x": X, "y": Y, "np": np})
    except:
        Z = np.zeros_like(X)
    
    # Filter out points that are outside the current range
    filtered_points = []
    for point in saved_points:
        if (x_min <= point[0] <= x_max and y_min <= point[1] <= y_max):
            # Recalculate z-value for the point based on current function
            try:
                z_val = eval(function_string, {"x": point[0], "y": point[1], "np": np})
                filtered_points.append([point[0], point[1], z_val])
            except:
                # If there's an error evaluating the function, skip this point
                continue
    print(f"  - Filtered points: {filtered_points}") # DEBUG: Check points after filtering
    
    # Create the surface plot
    surface = go.Surface(
        z=Z, 
        x=x, 
        y=y, 
        colorscale='Viridis',
        lighting=dict(ambient=0.7, diffuse=0.5, roughness=0.5, specular=0.2),
        lightposition=dict(x=0, y=0, z=100000),
        opacity=0.8
    )

    # Initialize empty lists for points
    points_x = []
    points_y = []
    points_z = []
    
    data = [surface]
    
    # Process the user-selected points
    if filtered_points and len(filtered_points) > 0:
        # Extract points for scatter plot
        for point in filtered_points:
            points_x.append(point[0])
            points_y.append(point[1])
            points_z.append(point[2])
        
        # Create user points
        user_points = go.Scatter3d(
            x=points_x, 
            y=points_y, 
            z=points_z, 
            mode='markers',
            marker=dict(
                size=10, 
                color='red',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='User selected points',
            showlegend=True
        )
        data.append(user_points)
    
    # Process optimization results if they exist
    if optimization_results and len(optimization_results) > 0:
        # Convert optimization path to follow the surface geometry
        # First create a list of just the 2D coordinates from optimization results
        path_2d = [point for point in optimization_results]
        
        # Generate smooth path that follows the surface
        smooth_path = interpolate_path_on_surface(path_2d, function_string, num_points=20)
        
        if smooth_path:
            # Extract coordinates for plotting
            opt_x = [p[0] for p in smooth_path]
            opt_y = [p[1] for p in smooth_path]
            opt_z = [p[2] for p in smooth_path]
            
            # Create optimization path line that follows the surface
            if len(opt_x) > 1:
                opt_line = go.Scatter3d(
                    x=opt_x,
                    y=opt_y,
                    z=opt_z,
                    mode='lines',
                    line=dict(width=5, color='green'),
                    name='Optimization path',
                    showlegend=True
                )
                data.append(opt_line)
                
                # Add optimization points as markers
                opt_points = []
                for point in optimization_results:
                    if len(point) >= 2:  # Ensure we have at least x and y
                        x_val, y_val = point[0], point[1]
                        # Check if point is within our current plot range
                        if x_min <= x_val <= x_max and y_min <= y_val <= y_max:
                            try:
                                # Calculate the z-value based on the current function
                                z_val = eval(function_string, {"x": x_val, "y": y_val, "np": np})
                                opt_points.append([x_val, y_val, z_val])
                            except:
                                continue
                
                # Extract coordinates for plotting iteration points
                iter_x = [p[0] for p in opt_points]
                iter_y = [p[1] for p in opt_points]
                iter_z = [p[2] for p in opt_points]
                
                # Add iteration points as markers
                iter_markers = go.Scatter3d(
                    x=iter_x,
                    y=iter_y,
                    z=iter_z,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color='blue',
                        symbol='circle',
                    ),
                    name='Optimization iterations',
                    showlegend=True
                )
                data.append(iter_markers)
                
                # Add the final point as a special marker
                final_point = go.Scatter3d(
                    x=[iter_x[-1]],
                    y=[iter_y[-1]],
                    z=[iter_z[-1]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='yellow',
                        symbol='diamond',
                        line=dict(width=2, color='black')
                    ),
                    name='Optimization result',
                    showlegend=True
                )
                data.append(final_point)

    # Create figure
    fig = go.Figure(data=data)
    
    fig.update_layout(
        title="Interactive Surface Plot with Optimization Path",
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            xaxis=dict(range=[x_min, x_max], autorange=False),
            yaxis=dict(range=[y_min, y_max], autorange=False),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        uirevision='constant',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        autosize=False,
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

@app.callback(
    Output('hovered-point', 'data'),
    [Input('surface-plot', 'hoverData'),
     Input('surface-plot', 'clickData')]
)
def capture_hovered_point(hoverData, clickData):
    # If user is hovering, update the point
    if hoverData and 'points' in hoverData and len(hoverData['points']) > 0:
        point = hoverData['points'][0]
        return [point['x'], point['y'], point['z']]

    # If no hover data, explicitly return None
    return None

# Callback to save points and run optimization when space is pressed
@app.callback(
    [Output('saved-points', 'data'),
     Output('optimization-results', 'data'),
     Output('optimization-status', 'children'),
     Output('is-optimizing', 'data')],  # Removed allow_duplicate=True to fix double press issue
    [Input('listener', 'event'),
     Input('submit-button', 'n_clicks')],
    [State('hovered-point', 'data'),
     State('saved-points', 'data'),
     State('optimization-results', 'data'),
     State('algorithm-dropdown', 'value'),
     State('max-iters-input', 'value'),
     State('tolerance-input', 'value'),
     State('function-input', 'value'),  # Use current function input rather than stored function
     State('x-range-input', 'value'),
     State('y-range-input', 'value')],
    prevent_initial_call=True
)
def save_point_and_optimize(event, n_clicks, hovered_point, saved_points, 
                           optimization_results, algorithm, max_iters, tolerance, 
                           function_string, x_range, y_range):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    print(f"save_point_and_optimize triggered by: {trigger_id}")  # Debug print
    
    # Initialize saved_points as an empty list if it's None
    if saved_points is None:
        saved_points = []
    
    # Handle Update Plot button click to rerun optimization with the current function
    if trigger_id == 'submit-button':
        print("Update Plot clicked, re-running optimization with current function if possible")
        
        # Clear previous points if we have hovered point or existing points
        if hovered_point or (saved_points and len(saved_points) > 0):
            # Reset saved points to be empty
            new_saved_points = []
            
            # If we have hovered point, use that as the starting point
            if hovered_point:
                start_point = [hovered_point[0], hovered_point[1]]
                # Add the hovered point as the only saved point
                new_saved_points.append(hovered_point)
            # Otherwise use the last saved point if any exist
            elif saved_points and len(saved_points) > 0:
                start_point = [saved_points[-1][0], saved_points[-1][1]]
                # Add the last saved point as the only saved point
                new_saved_points.append(saved_points[-1])
            else:
                # If somehow we get here without any points, return empty
                return [], None, "", False
                
            try:
                # Create a function that only depends on x,y (2D vector)
                def cost_function(xy):
                    x, y = xy
                    return eval(function_string, {"x": x, "y": y, "np": np})
                
                # Set parameters for optimization
                try:
                    max_iters_val = int(max_iters)
                except:
                    max_iters_val = 100
                
                try:
                    tol_val = float(tolerance)
                except:
                    tol_val = 1e-6
                
                params = {
                    "method": algorithm,
                    "max_iters": max_iters_val,
                    "tol": tol_val
                }
                
                # Run optimization
                optimizer = Optimizer()
                optimizer.optimize(cost_function, start_point, params)
                
                # Get the history of points visited
                new_optimization_results = optimizer.x_history
                
                # Create status message
                if len(new_optimization_results) > 0:
                    final_point = new_optimization_results[-1]
                    final_value = cost_function(final_point)
                    iterations = len(new_optimization_results) - 1
                    status_message = html.Div([
                        html.P(f"✅ Optimization complete!", style={'color': 'green', 'fontWeight': 'bold'}),
                        html.P(f"Algorithm: {algorithm}"),
                        html.P(f"Iterations: {iterations}"),
                        html.P(f"Starting point: ({start_point[0]:.4f}, {start_point[1]:.4f})"),
                        html.P(f"Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})"),
                        html.P(f"Function value: {final_value:.6f}")
                    ])
                    return new_saved_points, new_optimization_results, status_message, False
                else:
                    status_message = html.P("❌ Optimization failed to produce results", style={'color': 'red'})
                    return new_saved_points, None, status_message, False
            except Exception as e:
                print(f"Error re-running optimization: {str(e)}")
                status_message = html.Div([
                    html.P(f"❌ Error re-running optimization:", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P(str(e))
                ])
                return new_saved_points, None, status_message, False
        
        # If no points to work with, just return current state
        return saved_points, optimization_results, dash.no_update, False
    
    # Handle space key press to save point and run optimization
    if trigger_id == 'listener' and event and event.get('key') == 'w' and hovered_point:
        print("Space key pressed and hovered point exists")  # Debug print
        # Set is_optimizing to True immediately
        is_optimizing = True
        
        # Make sure we have all three coordinates
        if len(hovered_point) == 3:
            # Create a new list with ONLY the current point (clearing previous points)
            new_saved_points = [hovered_point]
            print(f"Saved point: {hovered_point}")  # Debug print
            
            # Run optimization from this point
            try:
                # Create a function that only depends on x,y (2D vector)
                def cost_function(xy):
                    x, y = xy
                    return eval(function_string, {"x": x, "y": y, "np": np})
                
                # Get starting point (x,y)
                start_point = [hovered_point[0], hovered_point[1]]
                
                # Configure optimizer
                optimizer = Optimizer()
                
                # Set parameters for optimization
                try:
                    max_iters_val = int(max_iters)
                except:
                    max_iters_val = 100
                
                try:
                    tol_val = float(tolerance)
                except:
                    tol_val = 1e-6
                
                params = {
                    "method": algorithm,
                    "max_iters": max_iters_val,
                    "tol": tol_val
                }
                
                # Run optimization
                optimizer.optimize(cost_function, start_point, params)
                
                # Get the history of points visited
                optimization_results = optimizer.x_history
                
                # Create status message
                if len(optimization_results) > 0:
                    final_point = optimization_results[-1]
                    final_value = cost_function(final_point)
                    iterations = len(optimization_results) - 1
                    status_message = html.Div([
                        html.P(f"✅ Optimization complete!", style={'color': 'green', 'fontWeight': 'bold'}),
                        html.P(f"Algorithm: {algorithm}"),
                        html.P(f"Iterations: {iterations}"),
                        html.P(f"Starting point: ({start_point[0]:.4f}, {start_point[1]:.4f})"),
                        html.P(f"Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})"),
                        html.P(f"Function value: {final_value:.6f}")
                    ])
                else:
                    status_message = html.P("❌ Optimization failed to produce results", style={'color': 'red'})
                
                return new_saved_points, optimization_results, status_message, False
                
            except Exception as e:
                print(f"Optimization error: {str(e)}")
                status_message = html.Div([
                    html.P(f"❌ Error in optimization:", style={'color': 'red', 'fontWeight': 'bold'}),
                    html.P(str(e))
                ])
                return new_saved_points, [], status_message, False
    
    # If no trigger matched, return current values
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('optimizer-progress-1', 'figure'),
     Output('optimizer-progress-2', 'figure')],
    Input('submit-button', 'n_clicks'),
    State('function-input', 'value')
)
def update_optimizer_views(n_clicks, function_str):
    # Simulate optimizer guesses
    guesses = np.random.rand(30, 10)
    optimum = np.linspace(0, 1, 10)
    distances = np.abs(guesses - optimum)
    normalized = distances / np.max(distances)

    fig1 = go.Figure(data=[go.Heatmap(z=normalized, colorscale='Viridis')])
    fig1.update_layout(title="Distance to Optimum per Variable")

    errors = np.sum(distances, axis=1)
    fig2 = go.Figure(data=[go.Scatter(y=errors, mode='lines+markers')])
    fig2.update_layout(title="Total Error per Step")

    return fig1, fig2


@app.callback(
    [Output('first-view', 'style'),
     Output('second-view', 'style'),
     Output('view-mode', 'data'),
     Output('first-view-controls', 'style'),
     Output('first-view-options', 'style')],
    Input('toggle-view', 'n_clicks'),
    State('view-mode', 'data')
)
def toggle_views(n_clicks, current_mode):
    if current_mode == 'surface':
        return {'display': 'none'}, {'display': 'block'}, 'optimizer', {'display': 'none'}, {'display': 'none'}
    else:
        return {'display': 'block'}, {'display': 'none'}, 'surface', {'display': 'flex','flexDirection':'row','justifyContent': 'flex-start', 'width': '100%'}, {'display': 'flex'}


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8050,debug=True)