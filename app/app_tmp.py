import os
import io
import re
import base64
import random
from uuid import uuid4

import dash
from dash import dcc, html, dash_table, Patch
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import requests
from label_studio_sdk import LabelStudio
import httpx
import time


# -----------------
# Config & constants
# -----------------
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080")
LABEL_STUDIO_API_KEY = os.getenv(
    "LABEL_STUDIO_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDk2NTE4MiwiaWF0IjoxNzYzNzY1MTgyLCJqdGkiOiIyMjc1ZDhjZjE5MGU0Y2M0YmMzYWJiN2VkYjRhMDEyMSIsInVzZXJfaWQiOjF9.pAMcDVKI7yCDvkYvP6mxJtoCN8GCeOAPGzd_i2fb-tc",
)
API_URL = os.getenv("API_URL", "http://api:5075/predict")

PROJECT_ID = 1
DEBUG = True
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_IOU_THRESHOLD = 0.5

ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

max_retries = 10
for i in range(max_retries):
    try:
        print(f"Attempting to connect to Label Studio (Try {i+1}/{max_retries})...")
        CLASSES = ls.projects.get(PROJECT_ID).parsed_label_config['label']['labels']
        print("Connected successfully!")
        break
        
    except (httpx.ConnectError, Exception) as e:
        # If connection fails, wait and try again
        print(f"Connection failed: {e}")
        if i < max_retries - 1:
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
        else:
            print("Max retries reached. Exiting.")
            raise e # Crash if it still fails after 50 seconds


COLUMNS = ["Class", "X0", "Y0", "X1", "Y1", "Score"]


# -----------------
# Helpers
# -----------------

def debug_print(*args):
    if DEBUG:
        print(*args, flush=True)


def generate_colors(n, seed=1337):
    random.seed(seed)
    colors = ["rgb(255,0,0)"]  # keep first deterministic red
    seen = {colors[0]}
    while len(colors) < n:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        col = f"rgb({r},{g},{b})"
        if col not in seen:
            colors.append(col)
            seen.add(col)
    return colors


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return "rgb(128,128,128)"
    try:
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgb({r},{g},{b})"
    except ValueError:
        return "rgb(128,128,128)"


def format_float(val):
    return f"{float(val):.2f}"


def corrd_to_tab_column(coord):
    return coord.upper()


def shape_to_row(shape, type_lookup):
    color = shape["line"]["color"]
    original_color = color
    if color.startswith("#"):
        color = hex_to_rgb(color)
    
    # Debug: check if color is in lookup
    if color not in type_lookup:
        debug_print(f"WARNING: Color '{original_color}' (converted: '{color}') not found in type_dict!")
        debug_print(f"Available colors in type_dict: {list(type_lookup.keys())[:5]}...")
    
    return {
        "Class": type_lookup.get(color, CLASSES[0]),
        "X0": format_float(shape["x0"]),
        "Y0": format_float(shape["y0"]),
        "X1": format_float(shape["x1"]),
        "Y1": format_float(shape["y1"]),
        "Score": format_float(1.0),
    }

def annotations_table_shape_resize(annotations_table_data, fig_data):
    """ Extract the shape that was resized (its index) and store the resized coordinates"""
    
    for key in fig_data.keys():
        shape_nb, coord = key.split(".")
        # shape_nb is for example "shapes[2].x0": this extract the number
        shape_nb = shape_nb.split(".")[0].split("[")[-1].split("]")[0]
        # this should correspond to the same row in the data table
        # we have to format the float here because this is exactly the entry in the table
        annotations_table_data[int(shape_nb)][corrd_to_tab_column(coord)] = format_float(fig_data[key])
    
    return annotations_table_data




def empty_fig():
    fig = go.Figure()
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="drawrect",
        newshape=dict(line=dict(color="rgb(255,0,0)", width=2), fillcolor="rgba(255,0,0,0.2)"),
    )
    return fig


# -----------------
# Paths & color maps
# -----------------
file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = file_dir
source_storage_dir = os.path.join(project_dir, "data/ls_data/source_storage", str(PROJECT_ID))
ANNOTATION_COLORMAP = generate_colors(len(CLASSES))
color_dict = dict(zip(CLASSES, ANNOTATION_COLORMAP))
type_dict = {v: k for k, v in color_dict.items()}


# -----------------
# App & layout
# -----------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, requests_pathname_prefix="/")
from werkzeug.middleware.proxy_fix import ProxyFix

app.server.wsgi_app = ProxyFix(app.server.wsgi_app, x_proto=1, x_host=1)

iou_bar = dcc.Slider(
    id="iou-threshold",
    min=0,
    max=1,
    step=0.001,
    value=DEFAULT_IOU_THRESHOLD,
    marks={0: "0.0", 1: "1.0"},
    vertical=True,
    verticalHeight=150,
    tooltip={"always_visible": True, "placement": "right"},
)

score_bar = dcc.Slider(
    id="score-threshold",
    min=0,
    max=1,
    step=0.001,
    value=DEFAULT_SCORE_THRESHOLD,
    marks={0: "0.0", 1: "1.0"},
    vertical=True,
    verticalHeight=150,
    tooltip={"always_visible": True, "placement": "right"},
)

advanced_settings = dbc.Card(
    [
        dbc.CardHeader("⚙️ Advanced Settings"),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                [html.H6("IOU Threshold", className="text-center"), iou_bar],
                                className="mb-4 d-flex flex-column align-items-center",
                            ),
                            html.Hr(style={"width": "80%", "margin": "20px auto"}),
                            html.Div(
                                [html.H6("Score Threshold", className="text-center"), score_bar],
                                className="d-flex flex-column align-items-center",
                            ),
                        ],
                        className="d-flex flex-column align-items-center",
                    )
                )
            )
        ),
    ],
    style={"width": "220px", "margin": "10px"},
)

image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H2("Annotation area")),
        dbc.CardBody(
            [
                html.Div(
                    id="upload-div",
                    children=dcc.Upload(
                        id="upload-image",
                        children=html.Div(
                            ["Drag & Drop or Click to Select Image"],
                            style={
                                "textAlign": "center",
                                "height": "400px",
                                "padding": "40px",
                                "border": "2px dashed #ccc",
                                "borderRadius": "12px",
                                "cursor": "pointer",
                            },
                        ),
                        multiple=False,
                        style={"marginBottom": "20px"},
                    ),
                ),
                html.Div(
                    id="graph-container",
                    children=[
                        dcc.Graph(
                            id="graph",
                            figure=empty_fig(),
                            style={"display": "none"},
                            config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                        ),
                    ],
                ),
                dcc.Store(id="uploaded-image-store", data=None),
                dcc.Store(id="graph-update-source", data=None),  # guards bounce
                dcc.Store(id="inf-container", data=None),  # store image info for LS submission
            ]
        ),
        dbc.CardFooter(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Process",
                            id="get-pred-btn",
                            outline=True,
                            size="lg",
                            color="primary",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Remove Image",
                            id="remove-image",
                            outline=True,
                            color="danger",
                            size="lg",
                            style={"width": "100%"},
                        ),
                        width=6,
                    ),
                ],
                className="g-2",
                justify="center",
            )
        ),
    ],
)

annotation_table_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Annotations data")),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H3("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation="dropdown" if n == "Class" else "input",
                                    )
                                    for n in COLUMNS
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={"textOverflow": "ellipsis", "maxWidth": 0},
                                dropdown=dict(
                                    Class=dict(
                                        options=[dict(label=o, value=o) for o in CLASSES],
                                        clearable=False,
                                    )
                                ),
                                style_cell_conditional=[{"if": {"column_id": "Class"}, "textAlign": "left"}],
                                style_table={
                                    "height": "300px",
                                    "overflowY": "auto",
                                    "border": "1px solid #ccc",
                                },
                                fill_width=True,
                            ),
                        ]
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.H3("Create new annotation for"),
                            dcc.Dropdown(
                                id="annotation-type-dropdown",
                                options=[{"label": t, "value": t} for t in CLASSES],
                                value=CLASSES[0],
                                searchable=True,
                                clearable=False,
                            ),
                        ],
                        align="center",
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.A(id="download", download="annotations.json", style={"display": "none"}),
                            dbc.Button("Download annotations", id="download-button", outline=True, color="primary", className="me-2"),
                            dbc.Button("Submit correction", id="submit-to-ls", outline=True, color="success"),
                            html.Div(id="dummy", style={"display": "none"}),
                            html.Div(id="submit-ls-status", style={"display": "none"}),
                            dbc.Tooltip(
                                "You can download the annotated data in a .json format by clicking this button.",
                                target="download-button",
                            ),
                        ],
                        className="d-flex justify-content-center",
                    ),
                    width="auto",
                ),
                justify="center",
            ),
        ),
    ]
)


# -----------------
# Callbacks (one-way flow)
# -----------------
@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "style", allow_duplicate=True),
    Output("upload-div", "style", allow_duplicate=True),
    Output("uploaded-image-store", "data", allow_duplicate=True),
    Output("graph-update-source", "data", allow_duplicate=True),
    Output("inf-container", "data", allow_duplicate=True),
    Input("upload-image", "contents"),
    prevent_initial_call=True,
)
def display_uploaded_image(contents):
    debug_print("uploaded_image callback")
    print(f"Upload callback triggered, contents={'present' if contents else 'None'}", flush=True)
    if contents is None:
        return empty_fig(), {"display": "none"}, {"display": "block"}, None, None
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        pil_image = Image.open(io.BytesIO(decoded))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        image_name = f"{uuid4().hex[:8]}.png"
        os.makedirs(source_storage_dir, exist_ok=True)
        pil_image.save(os.path.join(source_storage_dir, image_name))
        task = ls.tasks.create(
            project=PROJECT_ID,
            data={
                "image": f"/data/local-files/?d=source_storage/{PROJECT_ID}/{image_name}"
            }
        )
        img_array = np.array(pil_image)
        inf = {"id": task.id, "height": img_array.shape[0], "width": img_array.shape[1]}
        fig = go.Figure()
        fig.add_trace(go.Image(z=img_array))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="drawrect",
            newshape=dict(line=dict(color=color_dict[CLASSES[0]], width=2), fillcolor=color_dict[CLASSES[0]].replace("rgb", "rgba").replace(")", ",0.2)")),
        )
        debug_print("Image processed successfully")
        return fig, {"display": "block"}, {"display": "none"}, content_string, "upload", inf
    except Exception as e:
        import traceback
        traceback.print_exc()
        debug_print(f"Error processing image: {e}")
        return empty_fig(), {"display": "none"}, {"display": "block"}, None, None, None


@app.callback(
    Output("graph", "figure"),
    Output("graph", "style"),
    Output("upload-div", "style"),
    Output("uploaded-image-store", "data"),
    Output("annotations-table", "data"),
    Output("annotation-type-dropdown", "value"),
    Output("upload-image", "contents"),
    Output("graph-update-source", "data"),
    Input("remove-image", "n_clicks"),
    prevent_initial_call=True,
)
def remove_image(_):
    debug_print("remove image callback")
    return (
        empty_fig(),
        {"display": "none"},
        {"display": "block"},
        None,
        None,
        CLASSES[0],
        None,
        None,
    )


@app.callback(
    Output("annotations-table", "data", allow_duplicate=True),
    Input("get-pred-btn", "n_clicks"),
    State("uploaded-image-store", "data"),
    State("iou-threshold", "value"),
    State("score-threshold", "value"),
    prevent_initial_call=True,
)
def get_prediction(n_clicks, img_base64_string, iou_threshold, score_threshold):
    debug_print("get_prediction callback")
    if not n_clicks:
        raise PreventUpdate
    if img_base64_string is None:
        raise PreventUpdate
    image_bytes = base64.b64decode(img_base64_string)
    resp = requests.post(
        API_URL,
        files={"file": ("image.png", image_bytes)},
        data={"iou_threshold": iou_threshold, "score_threshold": score_threshold},
    )
    targets = resp.json()
    rows = []
    for box, score, label in zip(targets.get("boxes", []), targets.get("scores", []), targets.get("classes", [])):
        x0, y0, x1, y1 = box
        rows.append(
            {
                "Class": label,
                "X0": format_float(x0),
                "Y0": format_float(y0),
                "X1": format_float(x1),
                "Y1": format_float(y1),
                "Score": format_float(score),
            }
        )
    return rows


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("annotations-table", "data"),
    Input("annotation-type-dropdown", "value"),
    State("graph", "figure"),
    State("graph-update-source", "data"),
    prevent_initial_call=True,
)
def table_to_graph(table_data, current_class, figdict, source):
    debug_print("table_to_graph callback")
    ctx = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    debug_print(f"Callback triggered by {ctx}")
    
    if figdict is None:
        debug_print("No figure, returning empty")
        return empty_fig()

    # Use Patch() for efficient partial updates
    patched_figure = Patch()
    new_color = color_dict[current_class]
    
    # Update newshape color when dropdown changes
    if ctx == "annotation-type-dropdown.value":
        debug_print(f"Dropdown changed - current_class: {current_class}, color {new_color}")
        patched_figure["layout"]["newshape"] = {
            "line": {"color": new_color, "width": 2},
            "fillcolor": new_color.replace("rgb", "rgba").replace(")", ",0.2)")
        }
        return patched_figure

    # Table data changed - update shapes
    if not table_data:
        debug_print("Table is empty, clearing shapes")
        patched_figure["layout"]["shapes"] = []
        patched_figure["layout"]["newshape"] = {
            "line": {"color": new_color, "width": 2},
            "fillcolor": new_color.replace("rgb", "rgba").replace(")", ",0.2)")
        }
        return patched_figure

    shapes = []
    for row in table_data:
        try:
            x0, y0, x1, y1 = map(float, (row["X0"], row["Y0"], row["X1"], row["Y1"]))
            label = row["Class"]
            col = color_dict.get(label, color_dict[CLASSES[0]])
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color=col, width=2),
                    fillcolor=col.replace("rgb", "rgba").replace(")", ",0.2)"),
                    editable=True,
                )
            )
        except Exception as e:
            debug_print(f"Error processing row: {e}")
            continue
    
    debug_print(f"Table updated - {len(shapes)} shapes, current_class: {current_class}, color {new_color}")
    patched_figure["layout"]["shapes"] = shapes
    patched_figure["layout"]["newshape"] = {
        "line": {"color": new_color, "width": 2},
        "fillcolor": new_color.replace("rgb", "rgba").replace(")", ",0.2)")
    }
    return patched_figure


@app.callback(
    Output("annotations-table", "data", allow_duplicate=True),
    Output("graph-update-source", "data", allow_duplicate=True),
    Input("graph", "relayoutData"),
    State("annotations-table", "data"),
    prevent_initial_call=True,
)
def graph_to_table(relayout_data, table_data):
    debug_print("graph_to_table callback")
    debug_print(f"relayout_data keys: {list(relayout_data.keys()) if relayout_data else None}")
    
    if relayout_data is None:
        raise PreventUpdate
    if set(relayout_data.keys()) <= {"autosize", "margin", "xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"}:
        raise PreventUpdate

    if "shapes" in relayout_data:
        debug_print(f"Processing {len(relayout_data['shapes'])} shapes")
        for i, sh in enumerate(relayout_data["shapes"]):
            debug_print(f"Shape {i}: color={sh['line']['color']}")
        table = [shape_to_row(sh, type_dict) for sh in relayout_data["shapes"]]
        debug_print(f"Converted to {len(table)} table rows")
        for i, row in enumerate(table):
            debug_print(f"Row {i}: Class={row['Class']}")
    elif re.match(r"shapes\[(\d+)\].(\w+)", list(relayout_data.keys())[0]):
        if table_data is None:
            raise PreventUpdate
        table = table_data.copy()
        table = annotations_table_shape_resize(table, relayout_data)
    else:
        raise PreventUpdate

    return table, "draw"


@app.callback(
    Input("submit-to-ls", "n_clicks"),
    State("inf-container", "data"),
    State("annotations-table","data"),
    prevent_initial_call=True
)
def submit_to_ls(n_clicks, inf, annotations):
    if  (inf is not None) and (annotations is not None):
        task_id, img_height, img_width = inf.values()
        result = []
        for row in annotations:
            x0, y0, x1, y1 = float(row["X0"]), float(row["Y0"]), float(row["X1"]), float(row["Y1"])
            result.append({
                "from_name": "label",
                "to_name": "image",
                "source": "$image",
                "type": "rectanglelabels",
                "value": {
                    "height":(y1 - y0)*100 / img_height,
                    "width": (x1 - x0)*100 / img_width,
                    "x": x0 * 100 / img_width,
                    "y": y0 * 100 / img_height,
                    "rectanglelabels": [row["Class"]]
                }
            })
        ls.predictions.create(
            model_version="user_correction",
            result=result,
            task=task_id
        )

app.clientside_callback(
    """
function(tableData) {
    const s = JSON.stringify(tableData || []);
    const b = new Blob([s], {type: 'application/json'});
    return URL.createObjectURL(b);
}
""",
    Output("download", "href"),
    Input("annotations-table", "data"),
)

app.clientside_callback(
    """
function(n){
    const a = document.getElementById('download');
    if(a){ a.click(); }
    return '';
}
""",
    Output("dummy", "children"),
    Input("download-button", "n_clicks"),
)


app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(advanced_settings, width="auto"),
            dbc.Col(dbc.Row([dbc.Col(image_annotation_card, width=7), dbc.Col(annotation_table_card, width=5)])),
        ],
        align="start",
    ),
    fluid=True,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, dev_tools_hot_reload=False)