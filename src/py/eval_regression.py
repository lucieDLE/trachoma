import numpy as np
import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def main(args):
    y_true_arr = [] 
    y_pred_arr = []

    path_to_csv = os.path.join(args.mount_point, args.csv)

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df = pd.read_csv(path_to_csv)
    else:        
        df = pd.read_parquet(path_to_csv)

    fig = make_subplots(rows=1, cols=2)

    all_abs = []
    all_error = []
    for i in range(2):
      y_true_arr = [] 
      y_pred_arr = []

      y_true_arr = df[args.k1k2_column[i]]
      y_pred_arr = df[args.k1k2_prediction_column[i]]
      df['abs'] = np.abs(y_true_arr - y_pred_arr)
      df['error'] = y_true_arr - y_pred_arr
      # fig = px.violin(df, y="abs", color="class", box=True)
      # abs_filename = os.path.splitext(path_to_csv)[0] + "_abs.png"
      # fig.write_image(abs_filename)

      # fig = px.violin(df, y="error", color="class", box=True)
      # error_filename = os.path.splitext(path_to_csv)[0] + "_error.png"
      # fig.write_image(error_filename)

      # df["e"] = df[args.k1k2_column] - df[args.k1k2_prediction_column]
      fig.add_trace(go.Scatter(x=df[args.k1k2_column[i]], y=df[args.k1k2_prediction_column[i]],mode='markers'), row=1, col=i+1)
      # fig = px.scatter(df, x=, y=, trendline="ols")

      # # Calculate the errors
      mae = np.mean(df['abs'])
      mse_abs = np.mean(df['abs']**2)
      rmse_abs = np.sqrt(mse_abs)
      mean_error = np.mean(df['error'])
      mse_error = np.mean(df['error']**2)
      rmse_error = np.sqrt(mse_error)

      # Create a DataFrame with the calculated errors
      errors_df = pd.DataFrame({
          'Metric': ['MAE', 'MSE_ABS', 'RMSE_ABS', 
                    'ME', 'MSE', 'RMSE'],
          'Value': [mae, mse_abs, rmse_abs, mean_error, mse_error, rmse_error]
      })
      errors_filename = f"{os.path.splitext(path_to_csv)[0]}_k{i+1}_errors.csv"
      errors_df.to_csv(errors_filename, index=False)


    scatter_filename = os.path.splitext(path_to_csv)[0] + "_scatter.png"
    fig.update_layout(height=600, width=800, title_text="K1-K2 regression")
    fig.write_image(scatter_filename)


def get_argparse():
  # Function to parse arguments for the evaluation script
  parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--csv', type=str, help='CSV file', required=True)
  
  parser.add_argument('--k1k2_column', type=str, help='Which column to do the stats on',nargs='+', default=None)
  parser.add_argument('--k1k2_prediction_column', type=str, help='csv true class',nargs='+', default=None)

  parser.add_argument('--title', type=str, help='Title for the image', default='Confusion matrix')
  parser.add_argument('--figsize', type=str, nargs='+', help='Figure size', default=(6.4, 4.8))
  parser.add_argument('--surf_id', type=str, help='Name of array in point data for the labels', default='UniversalID')
  parser.add_argument('--pred_id', type=str, help='Name of array in point data for the predicted labels', default='PredictedID')
  parser.add_argument('--eval_metric', type=str, help='Score you want to choose for picking the best model : F1, AUC, MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) or ME (Mean Error)', default='F1', choices=['F1', 'AUC','MAE', 'RMSE', 'ME'])
  parser.add_argument('--mount_point', type=str, help='Mount point for the data', default='./')

  return parser

if __name__ == '__main__':
  parser = get_argparse()
  initial_args, unknownargs = parser.parse_known_args()
  args = parser.parse_args()
  main(args)