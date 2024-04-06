import pandas as pd
import numpy as np
from scipy.optimize import minimize

def load_data(input_data_path):
    # Load data from SageMaker's input channel
    data = pd.read_csv(input_data_path)
    return data

def train_model(data):
    # Your model training code using kalman_filter and kalman_smoother functions
    Y = data['Adj Close'].values
    S = Y.shape[0]
    param0 = np.array([0.85, 0.90, np.var(Y) / 45, np.var(Y) / 45])
    results = minimize(kalman_filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
    param_star = results.x
    path = kalman_smoother(param_star, Y, S)
    
    # Save the trained model
    model_path = '/opt/ml/model'
    np.save(f'{model_path}/model.npy', path)
    
# Your model functions
def kalman_filter(param,*args):
    # initialize params
    Z = param[0]
    T = param[1]
    H = param[2]
    Q = param[3]
    # initialize vector values:
    u_predict,  u_update,  P_predict, P_update, v, F = {},{},{},{},{},{}
    u_update[0] = Y[0]
    u_predict[0] = u_update[0]
    P_update[0] = np.var(Y)/4
    P_predict[0] =  T*P_update[0]*np.transpose(T)+Q
    Likelihood = 0
    for s in range(1, S):
        F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
        v[s]= Y[s-1]-Z*u_predict[s-1]
        u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
        u_predict[s] = T*u_update[s]
        P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
        P_predict[s] = T*P_update[s]*np.transpose(T)+Q
        Likelihood += (1/2)*np.log(2*np.pi)+(1/2)*np.log(abs(F[s]))+(1/2)*np.transpose(v[s])*(1/F[s])*v[s]

    return Likelihood


def kalman_smoother(params, *args):
    # initialize params
    Z = params[0]
    T = params[1]
    H = params[2]
    Q = params[3]
    # initialize vector values:
    u_predict,  u_update,  P_predict, P_update, v, F = {},{},{},{},{},{}
    u_update[0] = Y[0]
    u_predict[0] = u_update[0]
    P_update[0] = np.var(Y)/4
    P_predict[0] =  T*P_update[0]*np.transpose(T)+Q
    for s in range(1, S):
        F[s] = Z*P_predict[s-1]*np.transpose(Z)+H
        v[s]=Y[s-1]-Z*u_predict[s-1]
        u_update[s] = u_predict[s-1]+P_predict[s-1]*np.transpose(Z)*(1/F[s])*v[s]
        u_predict[s] = T*u_update[s]
        P_update[s] = P_predict[s-1]-P_predict[s-1]*np.transpose(Z)*(1/F[s])*Z*P_predict[s-1]
        P_predict[s] = T*P_update[s]*np.transpose(T)+Q

    u_smooth, P_smooth = {}, {}
    u_smooth[S-1] = u_update[S-1]
    P_smooth[S-1] = P_update[S-1]
    for t in range(S-1, 0, -1):
        u_smooth[t-1] = u_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(u_smooth[t]-T*u_update[s])
        P_smooth[t-1] = P_update[t] + P_update[t]*np.transpose(T)/P_predict[t]*(P_smooth[t]-P_predict[t])/P_predict[t]*T*P_update[t]

    # del u_update[-1]
    smooth_path = u_smooth
    return smooth_path

if __name__ == '__main__':
    input_data_path = '/opt/ml/input/data/data.csv'
    data = load_data(input_data_path)
    train_model(data)