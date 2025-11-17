#%%
import sys, os
sys.path.append("..")
from training import LoggingConfig
from dataset import VOCDataset
from NNModels import FasterRCNNMobile
import dash
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import re
import time
import io
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import base64
import torch
import colorsys
import random
from dash.exceptions import PreventUpdate
from uuid import uuid4

from label_studio_sdk import LabelStudio

LABEL_STUDIO_URL="http://localhost:8080"
LABEL_STUDIO_API_KEY= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDUxNzQyNywiaWF0IjoxNzYzMzE3NDI3LCJqdGkiOiI4ZTE0MWI1ZjAxZjA0NTE4ODcxMjgxZDE1YjFlYmJiNSIsInVzZXJfaWQiOjF9.398QTAIU5nMK21YiHxbASw5R9_MTtSCM9Bqe1IoTrDU"
PROJECT_ID = 1

ls = LabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY
)

def generate_colors(n, seed=1337):
    random.seed(seed)
    colors = set()
    while len(colors) < n:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.add(f"rgb({r},{g},{b})")
    return list(colors)

def hex_to_rgb(hex_color):
    """Convert hex ('#RRGGBB') to 'rgb(r,g,b)' or 'rgba(r,g,b,a)'."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgb({r},{g},{b})"

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(file_dir, "..")
source_storage_dir = os.path.join(project_dir, "data/ls_data/source_storage", str(PROJECT_ID))
exp_dir = os.path.join(project_dir,"exp/object_detection")

#Model configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
LOGGER = LoggingConfig(project_dir=exp_dir,
                       exp_name="VOC_fasterrcnn_mobilenet_v3_large_320_fpn_2000")
LOGGER.monitor_metric = "avg_loss"
LOGGER.monitor_mode = "min"

DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_IOU_THRESHOLD = 0.5
MODEL = FasterRCNNMobile(score_threshold=DEFAULT_SCORE_THRESHOLD,
                         iou_threshold=DEFAULT_IOU_THRESHOLD,
                         device=DEVICE)

# Load checkpoint
state = LOGGER.load_checkpoint()
MODEL.model.load_state_dict(state['model_state_dict'])


MODEL.eval()


# App configuration
DEBUG = True
NUM_ATYPES = 15
DEFAULT_FIG_MODE = "layout"
CLASSES = VOCDataset.voc_cls
DEFAULT_CLASS = CLASSES[0]
CLASS_TO_ID = VOCDataset.cls_to_id
# ANNOTATION_COLORMAP = px.colors.qualitative.Light24
ANNOTATION_COLORMAP = generate_colors(len(CLASSES))


# prepare bijective type<->color mapping
typ_col_pairs = list(zip(CLASSES, ANNOTATION_COLORMAP))
# types to colors
color_dict = {typ: col for typ, col in typ_col_pairs}
type_dict  = {col: typ for typ, col in typ_col_pairs}
#%%
COLUMNS = ["Class", "X0", "Y0", "X1", "Y1", "Score"]

def debug_print(*args):
    if DEBUG:
        print(*args)

def corrd_to_tab_column(coord):
    return coord.upper()


def format_float(f):
    return "%.2f" %(float(f),)

def shape_to_table_row(sh):
    return{
        "Class": type_dict[sh["line"]["color"]],
        "X0": format_float(sh["x0"]),
        "Y0": format_float(sh["y0"]),
        "X1": format_float(sh["x1"]),
        "Y1": format_float(sh["y1"]),
        "Score": format_float(1)   
    }



def annotations_table_shape_resize(annotations_table_data, fig_data):
    """ Extract the shape that was resized (its index) and store the resized coordinates"""
    debug_print("fig_data", fig_data)
    debug_print("table_data", annotations_table_data)
    
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
        newshape=dict(line_color="red"),
    )
    return fig



externel_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=externel_stylesheets)

server = app.server


# Avance setting: iou threshold and score threshold
iou_bar = dcc.Slider(
                id="iou-threshold",
                min=0,
                max=1,
                step=0.001,
                value=DEFAULT_IOU_THRESHOLD,
                marks={0:"0.0", 1:"1.0"},
                vertical=True,
                verticalHeight=150,
                tooltip={"always_visible": True, "placement":"right"},
            )

score_bar = dcc.Slider(
                id="score-threshold",
                min=0,
                max=1,
                step=0.001,
                value=DEFAULT_SCORE_THRESHOLD,
                marks={0:"0.0", 1:"1.0"},
                vertical=True,
                verticalHeight=150,
                tooltip={"always_visible": True, "placement":"right"},
            )


advanced_settings = dbc.Card(
    [
        dbc.CardHeader("⚙️ Advanced Settings"),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div([html.H6("IOU Threshold", className="text-center"), iou_bar], 
                                     className="mb-4 d-flex flex-column align-items-center"),
                            html.Hr(style={"width": "80%", "margin": "20px auto"}),
                            html.Div([html.H6("Score Threshold", className="text-center"), score_bar],
                                     className="d-flex flex-column align-items-center"),
                        ],
                        className="d-flex flex-column align-items-center",
                    )
                )
            )
        ),
    ],
    style={"width": "220px", "margin": "10px"},
)


# --- Image figure ---
image_annotation_card = dbc.Card(
    id="imagebox",
    children=[
        dbc.CardHeader(html.H2("Annotation area")),
        dbc.CardBody(
            [
                # Upload area
                html.Div(
                    id="upload-div",
                    children=dcc.Upload(
                        id="upload-image",
                        children=html.Div(
                            ["Drag & Drop or Click to Select Image"],
                            style={
                                "textAlign": "center",
                                "height":"400px",
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
                # Graph + remove icon container
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
                    # style={"position": "relative"},
                ),
                dcc.Store(id="uploaded-image-store", data=None),
                dcc.Store(id="inf-container", data=None),
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
                            style={"width": "100%"},  # hidden initially
                        ),
                        width=6,
                    ),
                ],
                className="g-2",  # gap between buttons
                justify="center",
            )
        ),
    ]
)

# Annotation table
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
                                    dict(name=n,
                                         id=n,
                                         presentation="dropdown" if n == "Class" else "input",
                                    )
                                    for n in COLUMNS
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_cell={
                                    # "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown=dict(
                                    Class = dict(
                                        options = [
                                            dict(label = o, value = o)
                                            for o in CLASSES
                                        ],
                                        # searchable=True,
                                        clearable=False,
                                    )
                                ),
                                style_cell_conditional=[
                                    {"if": {"column_id": "Class"}, "textAlign": "left",},
                                ],
                                style_table = {
                                    "height": "300px",
                                    "overflowY": "auto",
                                    "border": "1px solid #ccc"
                                },
                                fill_width=True,
                            ),
                            dcc.Store(
                                id="annotations-store",
                                data={"bboxes": []}
                            ),
                        ],
                    ),
                ),
                dbc.Row(
                    dbc.Col(
                        [
                            html.H3("Create new annotation for"),
                            dcc.Dropdown(
                                id="annotation-type-dropdown",
                                options=[
                                    {"label":t, "value":t} for t in CLASSES
                                ],
                                searchable=True,
                                value=DEFAULT_CLASS,
                                clearable=False,
                            ),
                        ],
                        align="center"
                    )
                ),
            ]
        ),
        dbc.CardFooter(
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.A(
                                id="download",
                                download="annotations.json",
                                style={"display": "none"},
                            ),

                            dbc.Button(
                                "Download annotations",
                                id="download-button",
                                outline=True,
                                color="primary",
                                className="me-2",
                            ),

                            dbc.Button(
                                "Submit correction",
                                id="submit-to-ls",
                                outline=True,
                                color="success",
                            ),

                            html.Div(id="dummy", style={"display":"none"}),
                            html.Div(id="submit-ls-status", style={"display":"none"}),

                            dbc.Tooltip(
                                "You can download the annotated data in a .json format by clicking this button.",
                                target="download-button"
                            ),
                        ],
                        className="d-flex justify-content-center"
                    ),
                    width="auto"     # <-- THIS MAKES THE FOOTER CONTENT SHRINK TO THE BUTTONS
                ),
                justify="center"     # <-- center only the column, not the whole container
            ),
        )
    ]
)


# --- Callbacks ---

# Display image callback
@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "style", allow_duplicate=True),
    Output("upload-div", "style", allow_duplicate=True),
    Output("uploaded-image-store", "data", allow_duplicate=True),
    Output("inf-container", "data", allow_duplicate=True),
    Input("upload-image", "contents"),
    prevent_initial_call=True,
)
def display_uploaded_image(contents):
    debug_print("uploaded_image callback")
    if contents is not None:
        # Decode uploaded image
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        pil_image = Image.open(io.BytesIO(decoded))

        # Save image to source storage to import to ls after
        image_name = f"{str(uuid4())[:8]}.png"
        pil_image.save(os.path.join(source_storage_dir, image_name))
        task = ls.tasks.create(
            project=PROJECT_ID,
            data={
                "image": f"/data/local-files/?d=source_storage/{PROJECT_ID}/{image_name}"
            }
        )

        img_array = np.array(pil_image)
        inf = {"id": task.id, "height": img_array.shape[0], "width": img_array.shape[1]}

        # Create figure with image
        fig = go.Figure()
        fig.add_trace(go.Image(z=img_array))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode="drawrect",
            newshape=dict(line_color="red"),
        )

        # Show graph and remove button, hide upload
        graph_style = {"display": "block"}
        upload_style = {"display": "none"}

        return fig, graph_style, upload_style, img_array, inf
    

    # Nothing uploaded
    return empty_fig(), {"display": "none"}, {"display": "block"}, dash.no_update, dash.no_update

# remove image callback
@app.callback(
    Output("graph", "style"),
    Output("upload-div", "style"),
    Output("uploaded-image-store", "data"),
    Output("annotations-table","data", allow_duplicate=True),
    Output("annotations-store", "data", allow_duplicate=True),
    Output("inf-container", "data"),
    Input("remove-image", "n_clicks"),
    prevent_initial_call=True,
)
def remove_image(n_clicks):
    debug_print("remove image callback")
    return {"display": "none"}, None, {"display": "block"}, None, None, None

# get prediction callback
@app.callback(
    [
        Output("annotations-table","data"),
    ],
    [Input("get-pred-btn", "n_clicks")],
    [
        State("iou-threshold", "value"),
        State("score-threshold", "value"),
        State("uploaded-image-store", "data"),
        State("graph", "figure"),
        State("annotation-type-dropdown", "value"),
    ],
    prevent_initial_call = True
)
def get_prediction(n_clicks, iou_threshold, score_threshold, image_array, figdict, new_shape_cls):
    debug_print("get_prediction callback")
    if image_array is None:
        raise PreventUpdate
    image_numpy = np.array(image_array, dtype=np.float32)
    image_torch = torch.from_numpy(image_numpy).permute(2,0,1).float()
    max_pixel = image_torch.max()
    image_torch = image_torch/max_pixel
    with torch.no_grad():
        targets_pred_single = MODEL.predict(image_torch, iou_threshold, score_threshold)[0]
    rows = []

    for box, label_id, score in zip(*targets_pred_single.values()):
        x0, y0, x1, y1 = box.tolist()
        label_id = int(label_id.item())
        score = score.item()
        label = CLASSES[label_id - 1]
        
        rows.append({"Class": str(label),
                    "X0": format_float(x0),
                    "Y0": format_float(y0),
                    "X1": format_float(x1),
                    "Y1": format_float(y1),
                    "Score": format_float(score)
                    })
        
    return (rows,)
    

@app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("annotation-type-dropdown", "value"),
        Input("annotations-table", "data"),
        State("graph", "figure"),
        prevent_initial_call=True,
)
def table_to_graph(current_class, table_data, figdict):
    debug_print("table_to_graph callback")
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    debug_print(f"Callback is triggered by {cbcontext}")
    
    if cbcontext == "annotation-type-dropdown.value":
        if figdict is None:
            PreventUpdate
        shapes = figdict.get("layout", {}).get("shapes", [])
        fig = go.Figure(figdict)
        fig.update_layout(
            shapes=shapes,
            dragmode="drawrect",
            newshape=dict(line_color=color_dict[current_class],
                        line_width = 2,
                        fillcolor=color_dict[current_class].replace("rgb", "rgba").replace(")", ",0.2)"),
                        )
        )    
        return fig
    
    if (figdict is None) or (table_data is None):
        raise PreventUpdate
    
    fig = go.Figure(figdict)
    shapes = []
    for row in table_data:
        x0, y0, x1, y1 = float(row["X0"]), float(row["Y0"]), float(row["X1"]), float(row["Y1"])
        label = row['Class']
        shapes.append(
            dict(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color_dict[label], width=2),
                fillcolor=color_dict[label].replace("rgb", "rgba").replace(")", ",0.2)"),
                editable=True
            )
        )
    fig.update_layout(shapes=shapes)
    return fig


@app.callback(
        Output("annotations-table", "data", allow_duplicate=True),
        Input("graph", "relayoutData"),
        State("annotations-table", "data"),
        State("graph", "figure"),
        prevent_initial_call=True,
)
def graph_to_table(relayout_data, table_data, figdict):
    debug_print("graph_to_table callback")
    print(f"relayout_data:",relayout_data)
    
    if relayout_data is None:
        raise PreventUpdate
    if set(relayout_data.keys()) <= {"autosize", "margin", "xaxis.range[0]", "xaxis.range[1]", "yaxis.range[0]", "yaxis.range[1]"}:
        raise PreventUpdate
    if "shapes" in relayout_data.keys():
        table_data = [
            shape_to_table_row(sh) for sh in relayout_data["shapes"]
        ]
    elif re.match(r"shapes\[(\d+)\].(\w+)", list(relayout_data.keys())[0]):
        table_data = annotations_table_shape_resize(
            table_data, relayout_data
        )

    if table_data is None:
        raise PreventUpdate
    else:
        return table_data
    
    
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("annotations-table", "data")],
)

# click on download link via button
app.clientside_callback(
    """
function(download_button_n_clicks)
{
    let download_a=document.getElementById("download");
    download_a.click();
    return '';
}
""",
    Output("dummy", "children"),
    [Input("download-button", "n_clicks")],
)

@app.callback(
    Input("submit-to-ls", "n_clicks"),
    State("inf-container", "data"),
    State("annotations-table","data"),
    prevent_initila_call=True
)
def submit_to_ls(n_clicks, inf, annotations):
    if  (inf is not None) and (annotations is not None):
        task_id, img_height, img_width = inf.values()
        result = []
        for row in annotations:
            print(row["Class"])
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
    
    



app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(advanced_settings, width="auto"),
            dbc.Col(
                dbc.Row(
                    [
                        dbc.Col(image_annotation_card, width=7),
                        dbc.Col(annotation_table_card, width=5)
                    ]
                )
            ),

        ],
        align="start"
    ),
    fluid=True,
)



if __name__ == "__main__":
    app.run(debug=True)
