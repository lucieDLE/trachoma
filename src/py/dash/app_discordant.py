import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

pio.renderers.default = "chrome"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


# csv_path_stacks = "/work/jprieto/data/remote/EGower/jprieto/trachoma_bsl_mtss_besrat_field_Ungraded.csv"
# test_df = pd.read_csv(csv_path_stacks).replace("hinashah/", "", regex=True)
# test_df['tt sev'] = (test_df['tt sev'] >= 1).astype(int)

sev_col = 'tt sev'
id_col = 'cid'
# csv_path = "/work/jprieto/data/remote/EGower/hinashah/Analysis_Set_20220326/trachoma_bsl_mtss_besrat_field_test_20220326.csv"

csv_path = "hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT.csv"


test_df = pd.read_csv(csv_path).replace("hinashah/", "", regex=True)
test_df[sev_col] = (test_df[sev_col] >= 1).astype(int)

# csv_path_stacks = csv_path.replace(".csv", "_stacks.csv")
# with open(csv_path_stacks.replace(".csv", "_25042022_prediction.pickle"), 'rb') as f:
    # results_epi = pickle.load(f)

with open(csv_path.replace(".csv", "_model01042022_prediction.pickle"), 'rb') as f:
    results_epi = pickle.load(f)

x, x_a, x_s, x_v, x_v_p = results_epi

pca_epi = PCA(n_components=2)
pca_epi_fit = pca_epi.fit_transform(x_a)
test_df["pca_0"] = pca_epi_fit[:,0]
test_df["pca_1"] = pca_epi_fit[:,1]
test_df["pred"] = np.argmax(x, axis=1)

@app.callback(
    Output('studies-img', 'figure'),    
    Input('studies-img', 'clickData'),
    Input('studies-img', 'figure'),
    Input('colorby-dropdown', 'value'),
    Input("studies-search", "value"))
def studies_img(dict_points, fig, color_by, studies_search):

    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df["pca_0"], y=test_df["pca_1"], mode='markers', showlegend=False, marker=dict(size=(test_df["pred"] + 1)*5, colorscale='sunset', showscale=True, opacity=1, line=dict(color='red', width=1)
        )))
        fig.add_trace(go.Scatter(mode='markers', showlegend=False, marker=dict(line=dict(color='red', width=1))))
    else:
        fig = go.Figure(fig)

    if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:        

        idx = dict_points["points"][0]["pointIndex"]

        fig.data[1]['x'] = [test_df.loc[idx]["pca_0"]]
        fig.data[1]['y'] = [test_df.loc[idx]["pca_1"]]

        if(color_by == 'class'):
            color = test_df[sev_col]
        elif(color_by == 'max'):            
            color = np.max(x_s, axis=1).reshape(-1)

        fig.data[0]['marker']['color'] = color
        fig.data[0]['text'] = ['s: {:f}, c: {:f}, p: {:f}'.format(s, c, p) for s, c, p in zip(np.max(x_s, axis=1).reshape(-1), test_df[sev_col], test_df['pred'])]        
    
    if studies_search is not None and studies_search != '':
        fig.data[0]['marker']['opacity'] = (test_df[id_col].astype(str).str.match(studies_search).astype(float) + 0.1).clip(0,1)
    else:
        fig.data[0]['marker']['opacity'] = 1

    fig.update_layout(autosize=True)
    return fig



@app.callback(
    Output('study-level', 'figure'),    
    Input('studies-img', 'clickData'),
    Input('study-level', 'figure'))
def update_study(dict_points, fig_study):    

    if fig_study is None:    
        fig_study = go.Figure()
        fig_study.add_trace(
            go.Scatter(mode='markers', showlegend=False, marker=dict(showscale=True, size=10, cmin=np.min(x_v_p), cmax=np.max(x_v_p), colorscale='sunset', line=dict(color='red', width=1)))
            )

        fig_study.add_trace(go.Scatter(mode='markers', marker=dict(size=10, line=dict(color='magenta', width=2)), showlegend=False))
    else: 
        fig_study = go.Figure(fig_study)
    
    if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:        
        
        idx = dict_points["points"][0]["pointIndex"]
        
        x_feat_idx = np.array(x_v[idx]).reshape(-1, 256)
        x_feat_idx_pca = pca_epi.transform(x_feat_idx)

        df_idx = pd.DataFrame({
            "pca_0": x_feat_idx_pca[:,0],
            "pca_1": x_feat_idx_pca[:,1],
            "pred": np.array(np.argmax(x_v_p[idx], axis=1)).reshape(-1), 
            "score": np.array(x_s[idx]).reshape(-1)
            })
        
        fig_study.data[0]['x'] = df_idx["pca_0"]
        fig_study.data[0]['y'] = df_idx["pca_1"]
        fig_study.data[0]['text'] = ['s: {:f}, p: {:f}'.format(s, p) for s, p in zip(df_idx['score'], df_idx['pred'])]
        fig_study.data[0]['marker']['color'] = df_idx["score"]
        fig_study.data[0]['marker']['cmin'] = np.min(x_s)
        fig_study.data[0]['marker']['cmax'] = np.max(x_s)
    
    fig_study.update_layout(
        autosize=True
    )
    return fig_study


@app.callback(
    Output('study-index', 'children'),
    Output('study-id', 'children'),
    Output('study-img', 'figure'),    
    Input('studies-img', 'clickData'),
    Input('img-opacity', 'value'),
    Input('img-size', 'value'))
def update_img(dict_points, opacity, size):
    
    fig_img = go.Figure()
    study_id = ""
    idx = -1
    if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:
        
        idx = dict_points["points"][0]["pointIndex"]
        
        study_id = test_df.loc[idx]["image"]
        img_path = os.path.join("/work/jprieto/data/remote/EGower/jprieto/", study_id)
        

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)

        fig_img = px.imshow(img_np, binary_string=True, binary_compression_level=5, binary_backend='pil')

        # img_path = os.path.join("/work/jprieto/data/remote/EGower/hinashah/Analyses_Set_20220321_Images_seg/", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))        
        img_path = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_seg", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))        
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)
        
        fig_img.add_trace(go.Heatmap(z=img_np, opacity=opacity, colorscale='rdbu'))

        fig_img.update_layout(
            autosize=False,
            width=size,
            height=size
        )

    return ["idx: " + str(idx), "id: " + study_id, fig_img]


@app.callback(
    Output('study-frames', 'figure'),
    Input('study-index', 'children'),
    Input('study-level', 'clickData'),
    Input('frame-id', 'value'),
    Input('explain-dropdown', 'value'),
    Input('explain-opacity', 'value'))
def update_frames(idx, dict_points, frame_id, explain_class, opacity):
    idx = int(idx.replace("idx: ", ""))         
    
    if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:
        frame_id = dict_points["points"][0]["pointIndex"]

    if idx >= 0:
        # img_path = os.path.join("/work/jprieto/data/remote/EGower/hinashah/Analyses_Set_20220321_Images_stacks/", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))        
        img_path = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_stacks", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))        
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)        
        
        t, xs, ys, _ = img_np.shape
        xo = (xs - 448)//2
        yo = (ys - 448)//2
        img_np = img_np[:,xo:xo + 448, yo:yo + 448,:]

        fig_frames = px.imshow(img_np[frame_id], binary_string=True, binary_compression_level=5)

        img_path_explain = os.path.join("hinashah/hinashah_applist/jimma_tis_may_2022_discordant_NOTT_rise_explainability", test_df.loc[idx]["image"].replace(".jpg", ".nrrd"))
        
        if os.path.exists(img_path_explain):
            img_explain_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path_explain))
            print(img_explain_np.shape)
            fig_frames.add_trace(go.Heatmap(z=img_explain_np[frame_id,:,:,explain_class], zmin=0, zmax=1, opacity=opacity, colorscale='rdbu'))

        # sliders = [dict(
        #     active=frame_idx
        # )]

        # fig_frames.update_layout(
        #     sliders=sliders
        # )
        
        return fig_frames
    else: 
        return go.Figure()

    
@app.callback(
    Output('conf-matrix', 'figure'),
    Input('conf-matrix', 'figure')
    )
def conf_matrix(fig):
    
    cnf_matrix_stacks = confusion_matrix(test_df[sev_col], test_df["pred"])
    cnf_matrix_norm_stacks = cnf_matrix_stacks.astype('float') / cnf_matrix_stacks.sum(axis=1)[:, np.newaxis]
    
    fig = px.imshow(cnf_matrix_norm_stacks,
        labels=dict(x="Predicted", y="True"),
        x=['Healthy', 'TT'],
        y=['Healthy', 'TT'],
        color_continuous_scale='blues',
        text_auto='.2f'
        )

    return fig

@app.callback(
    Output('table-class-report', 'data'),
    Input('table-class-report', 'data')
    )
def conf_matrix(data):
    
    c_report = classification_report(test_df[sev_col], test_df["pred"], output_dict=True)

    data = []

    for k in c_report:
        obj = c_report[k]
        if type(obj) is dict:
            obj['row'] = k
            data.append(obj)    
        elif k == 'accuracy':
            data.append({"f1-score": obj, "row": "accuracy"})

    return data

app.layout = html.Div(children=[
    html.H1(children='Trachomatous Trichiasis Web App'),
    html.Div([
        html.Div([
            html.Div(
                [
                html.Div([dcc.Input(id="studies-search", type="search", placeholder="Search study")]),
                html.Div([dcc.Dropdown(options=['class', 'max'], value='class', id='colorby-dropdown')]),
                html.Div([dcc.Graph(id='studies-img')])],
                className='six columns'
            ),            
            html.Div(
                [html.Div([
                    html.Div(html.H2('', id='study-index'), className='two columns'),
                    html.Div(html.H2('id:', id='study-id'), className='ten columns')                    
                    ], className='row'),
                dcc.Graph(id='study-img'),
                dcc.Slider(0, 1500, 10, value=450, id='img-size', marks={ 0: {'label': '0'}, 1500: {'label': '1500'}}),
                dcc.Slider(0, 1, 0.1, value=0.2, id='img-opacity')],
                className='six columns'
                )
        ], className='row'),
        html.Div([            
            html.Div(
                [dcc.Graph(id='study-level')],
                className='six columns'),
            html.Div(
                [dcc.Dropdown(options=[0, 1], value=0, id='explain-dropdown'),
                dcc.Slider(0, 1, 0.1, value=0.2, id='explain-opacity'),
                dcc.Graph(id='study-frames'),
                dcc.Slider(0, 16, 1, value=0, id='frame-id')],
                className='six columns'
                )
        ], className='row'),
        html.Div([            
            html.Div(
                [dcc.Graph(id='conf-matrix')],
                className='six columns'),
            html.Div(
                [dash_table.DataTable(id='table-class-report', columns=[
                    {"name": "", "id": "row"}, 
                    {"name": "precision", "id": "precision", "type":'numeric', "format":Format(precision=2)}, 
                    {"name": "recall", "id": "recall", "type":'numeric', "format":Format(precision=2)}, 
                    {"name": "f1-score", "id": "f1-score", "type":'numeric', "format":Format(precision=2)}, 
                    {"name": "support", "id": "support", "type":'numeric'}])],
                className='six columns'
                )
        ], className='row')
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)