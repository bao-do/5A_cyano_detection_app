import base64
import io
import numpy as np
from PIL import Image
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H3("🖼️ Drag & Drop Image Upload + Display"),
        dcc.Upload(
            id="upload-image",
            children=html.Div(["Drag and Drop or ", html.A("Select an Image")]),
            style={
                "width": "80%",
                "height": "150px",
                "lineHeight": "150px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "margin": "20px auto",
                "background-color": "#fafafa",
            },
            multiple=False,
        ),
        dcc.Graph(
            id="image-display",
            style={"height": "600px", "width": "90%", "margin": "auto"},
            config={"modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
        ),
    ]
)


@app.callback(
    Output("image-display", "figure"),
    Input("upload-image", "contents"),
)
def update_output(contents):
    if contents is None:
        # Empty graph before any image is uploaded
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return fig

    # Decode uploaded image
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    pil_image = Image.open(io.BytesIO(decoded))
    img_array = np.array(pil_image)

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Image(z=img_array))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="drawrect",  # Allow rectangle annotation
        newshape=dict(line_color="red"),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
