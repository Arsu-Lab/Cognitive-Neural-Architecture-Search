import glob
import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr, ttest_1samp, sem
import warnings


from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np

def aggregate_df_by_layer(df):
    """
    Aggregates a single DataFrame by layer, averaging R values and computing combined significance,
    ensuring scalar values for each column.
    """
    aggregated_data = []

    for layer, group in df.groupby('Layer'):
        mean_r = group['R'].mean()
        t_stat, significance = ttest_1samp(group['R'], 0)
        
        # Assuming all rows within a single DataFrame have the same ROI
        common_roi_name = find_common_roi_name(group['ROI'].tolist())

        layer_data = {
            'ROI': common_roi_name,
            'Layer': layer,
            'Model': group['Model'].iloc[0],
            'R': mean_r,  # Use scalar value
            '%R2': mean_r ** 2,  # Use scalar value for %R2, computed from mean_r
            'Significance': significance,  # Use scalar value
            'SEM': group['R'].sem(),  # Use scalar value for SEM, if needed
            'LNC': np.nan,  # Placeholder for LNC, adjust as needed
            'UNC': np.nan  # Placeholder for UNC, adjust as needed
        }

        aggregated_data.append(layer_data)

    return pd.DataFrame(aggregated_data)


def find_common_roi_name(names):
    """
    Identifies the common ROI name within a single DataFrame.
    """
    if len(names) == 1:
        return names[0]  # Directly return the name if there's only one

    split_names = [name.split('_') for name in names]
    common_parts = set(split_names[0]).intersection(*split_names[1:])
    common_roi_name = '_'.join(common_parts)
    return common_roi_name

def aggregate_layers(dataframes):
    """
    Processes each DataFrame independently to aggregate by layer, then combines the results.
    Each DataFrame represents its own ROI, maintaining a single aggregated value per layer.
    """
    # Ensure dataframes is a list
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    aggregated_dfs = []

    for df in dataframes:
        aggregated_df = aggregate_df_by_layer(df)
        aggregated_dfs.append(aggregated_df)

    # Combine aggregated results from all DataFrames
    final_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    return final_df


def get_layers_ncondns(features):
    """
    Extracts layer information from the features dictionary.

    Parameters:
    - features (dict): Dictionary containing model features with layer names as keys

    Returns:
    - num_layers (int): Number of layers
    - layer_list (list): List of layer names
    - num_condns (int): Number of conditions/samples
    """
    layer_list = list(features.keys())
    num_layers = len(layer_list)
    
    # Get number of conditions from first layer
    first_layer = features[layer_list[0]]
    num_condns = len(first_layer)
    
    return num_layers, layer_list, num_condns

def encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, features):
    """
    Encodes layer activations using IncrementalPCA for training and test sets.

    Parameters:
    - layer_id (str): Layer name
    - n_components (int): Number of PCA components
    - batch_size (int): Batch size for IncrementalPCA
    - trn_Idx (list): Training indices
    - tst_Idx (list): Test indices
    - features (dict): Dictionary containing model features

    Returns:
    - trn, tst: Training and test features
    """
    layer_features = features[layer_id]
    feature_keys = list(layer_features.keys())
    
    # Get sample feature to check dimensions
    sample_feat = layer_features[feature_keys[0]]
    processed_sample_feat = np.mean(sample_feat, axis=1).flatten() if sample_feat.ndim > 2 else sample_feat.flatten()
    
    use_pca = False  # Simplified PCA logic as before

    if use_pca:
        # PCA implementation remains similar but uses dictionary access
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        activations = []
        
        for idx in trn_Idx:
            feat = layer_features[feature_keys[idx]]
            activations.append(np.mean(feat, axis=1).flatten() if feat.ndim > 2 else feat.flatten())
            
            if ((len(activations) % batch_size) == 0) or (len(activations) == len(trn_Idx)):
                pca.partial_fit(np.stack(activations[-batch_size:], axis=0))
        
        trn = pca.transform(np.stack(activations, axis=0))
        
        # Transform test set
        tst = pca.transform(np.stack([
            np.mean(layer_features[feature_keys[idx]], axis=1).flatten() 
            if layer_features[feature_keys[idx]].ndim > 2 
            else layer_features[feature_keys[idx]].flatten() 
            for idx in tst_Idx
        ], axis=0))
    else:
        # Direct feature extraction without PCA
        trn = np.stack([
            np.mean(layer_features[feature_keys[idx]], axis=1).flatten() 
            if layer_features[feature_keys[idx]].ndim > 2 
            else layer_features[feature_keys[idx]].flatten() 
            for idx in trn_Idx
        ])
        
        tst = np.stack([
            np.mean(layer_features[feature_keys[idx]], axis=1).flatten() 
            if layer_features[feature_keys[idx]].ndim > 2 
            else layer_features[feature_keys[idx]].flatten() 
            for idx in tst_Idx
        ])
    
    return trn, tst


def train_Ridgeregression_per_ROI(trn_x,tst_x,trn_y,tst_y,alpha):
    """
    Train a linear regression model for each ROI and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """
    #reg = LinearRegression().fit(trn_x, trn_y)
    reg = Ridge(alpha=alpha)
    
    #print('for training:', trn_x.shape)
    #print('for training:', trn_y.shape)
    
    reg.fit(trn_x, trn_y)
    y_prd = reg.predict(tst_x)
    correlation_lst = np.zeros(y_prd.shape[1])
    for v in range(y_prd.shape[1]):
        correlation_lst[v] = pearsonr(y_prd[:,v], tst_y[:,v])[0]
    return correlation_lst

def train_RidgeCV_per_ROI(X, y, alpha, cv):
    """
    Train a RidgeCV regression model for each ROI and compute correlation coefficients.

    Args:
        X (numpy.ndarray): Encoded features.
        y (numpy.ndarray): fMRI data.
        alpha (float): Alpha value for ridge regression.
        cv (int): Number of cross-validation folds.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """
    ridgecv = RidgeCV(alphas=[alpha], cv=cv)  # Use a single alpha value
    ridgecv.fit(X, y)
    y_pred = ridgecv.predict(X)
    correlation_lst = np.array([pearsonr(y_pred[:, v], y[:, v])[0] for v in range(y.shape[1])])
    return correlation_lst

def RidgeCV_Encoding(features, roi_path, model_name, alpha, trn_tst_split=0.8, n_folds=3, 
                     n_components=100, batch_size=100, just_corr=True, 
                     return_correlations=False, random_state=14, 
                     save_path="RidgeCV_Encoding_Results"):
    """
    Modified to accept features dictionary instead of feat_path.
    Rest of the parameters remain the same.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    num_layers, layer_list, num_samples = get_layers_ncondns(features)
    feature_keys = list(features[layer_list[0]].keys())  # Get keys from first layer
    
    trn_Idx, tst_Idx = train_test_split(
        range(num_samples), 
        test_size=(1-trn_tst_split), 
        train_size=trn_tst_split, 
        random_state=random_state
    )
    
    all_results = []
    corr_dict = {} if return_correlations else None
    
    for layer_id in layer_list:
        X_train, X_test = encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, features)
        
        if return_correlations:
            corr_dict[layer_id] = {}
        
        for roi_path in roi_paths:
            roi_files = glob.glob(os.path.join(roi_path, '*.npy')) if os.path.isdir(roi_path) else [roi_path]
            
            for roi_file in roi_files:
                roi_name = os.path.basename(roi_file)[:-4]
                fmri_data = np.load(roi_file)
                y_train, y_test = fmri_data[trn_Idx], fmri_data[tst_Idx]
                
                ridgecv = RidgeCV(alphas=[alpha], cv=n_folds)
                ridgecv.fit(X_train, y_train)
                y_pred = ridgecv.predict(X_test)
                
                correlations = np.array([pearsonr(y_pred[:, v], y_test[:, v])[0] for v in range(y_test.shape[1])])
                r = np.mean(correlations)
                
                result = {
                    "ROI": roi_name,
                    "Layer": layer_id,
                    "Model": model_name,
                    "R": r,
                    "%R2": r ** 2,
                    "Significance": ttest_1samp(correlations, 0)[1],
                    "SEM": sem(correlations),
                    "Alpha": alpha,
                    "LNC": np.nan,
                    "UNC": np.nan
                }
                all_results.append(result)
                
                if return_correlations:
                    corr_dict[layer_id][roi_name] = correlations

    all_rois_df = pd.DataFrame(all_results)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_file_path = f"{save_path}/{model_name}_RidgeCV.csv"
    all_rois_df.to_csv(csv_file_path, index=False)
    
    if return_correlations:
        return all_rois_df, corr_dict
    
    return all_rois_df


def Ridge_Encoding(feat_path, roi_path, model_name, alpha, trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14, save_path="Linear_Encoding_Results"):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
        
            If we have a list of folders, each folders content will be summarized into one value. This is important if one folder contains data for the same ROI for different subjects
            
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Check if its a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    list_dataframes = []
    
    # Iterate through all folder paths
    for roi_path in roi_paths:
        result_dataframe = _ridge_encoding(feat_path, 
                                            roi_path, 
                                            model_name,
                                            alpha,
                                            trn_tst_split=trn_tst_split, 
                                            n_folds=n_folds, 
                                            n_components=n_components, 
                                            batch_size=batch_size,
                                            just_corr=just_corr, 
                                            return_correlations=return_correlations,
                                            random_state=random_state)
        

        # Collect dataframes in list
        list_dataframes.append(result_dataframe[0])
    
    # If just one dataframe, return it as it is
    if len(list_dataframes) == 1:
        final_df = list_dataframes[0]
    else:
        final_df = aggregate_layers(list_dataframes)
        
    # Create the output folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    csv_file_path = f"{save_path}/{model_name}.csv"
    final_df.to_csv(csv_file_path, index=False)
    
    return final_df
        
        
        
        
def ridge_encoding(*args, **kwargs):
    warnings.warn(
        "The 'linear_encoding' function is deprecated and has been replaced by 'Linear_Encoding'. "
        "Please update your code to use the new function name, as this alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    return Ridge_Encoding(*args, **kwargs)   
        
    

def _ridge_encoding(feat_path, roi_path, model_name, alpha, trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested
    
    # Check if roi_path is a list, if not, make it a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path+'/*.npz')
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    
    # Create a tqdm object with an initial description
    #pbar = tqdm(range(n_folds), desc="Initializing folds")
    
    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):
        #pbar.set_description(f"Processing fold {fold_ii + 1}/{n_folds}")
        
        # Set random seeds for reproducibility
        np.random.seed(fold_ii+random_state)
        random.seed(fold_ii+random_state)
        
        # Split the data indices into training and testing sets
        trn_Idx,tst_Idx = train_test_split(range(len(feat_files)),test_size=(1-trn_tst_split),train_size=trn_tst_split,random_state=fold_ii+random_state)
        
        # Process each layer of model activations
        for layer_id in layer_list:
            if layer_id not in fold_dict.keys():
                fold_dict[layer_id] = {}
                corr_dict[layer_id] = {}
            
            # Encode the current layer using PCA and split into training and testing sets
            pca_trn,pca_tst = encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, feat_path)

            for roi_path in roi_paths:
                roi_files = []
            
                 # Check if the roi_path is a file or a directory
                if os.path.isfile(roi_path) and roi_path.endswith('.npy'):
                    # If it's a file, directly use it
                    roi_files.append(roi_path)
                elif os.path.isdir(roi_path):
                    # If it's a directory, list all .npy files within it
                    roi_files.extend(glob.glob(os.path.join(roi_path, '*.npy')))
                else:
                    print(f"Invalid ROI path: {roi_path}")
                    continue  # Skip this roi_path if it's neither a valid file nor a directory

                # Process each ROI's fMRI data
                if not roi_files:
                    print(f"No roi_files found in {roi_path}")
                    continue  # Skip to the next roi_path if no ROI files were found

                for roi_file in roi_files:
                    roi_name = os.path.basename(roi_file)[:-4]
                    if roi_name not in fold_dict[layer_id].keys():
                        fold_dict[layer_id][roi_name] = []
                        corr_dict[layer_id][roi_name] = []
                        
                    # Load fMRI data for the current ROI and split into training and testing sets
                    fmri_data = np.load(os.path.join(roi_file))
                    fmri_trn,fmri_tst = fmri_data[trn_Idx],fmri_data[tst_Idx]
                    
                    # Train a linear regression model and compute correlations for the current ROI
                    r_lst = train_Ridgeregression_per_ROI(pca_trn,pca_tst,fmri_trn,fmri_tst,alpha)
                    r = np.mean(r_lst) # Mean of all train test splits
                    
                    # Store correlation results
                    if return_correlations:                   
                        corr_dict[layer_id][roi_name].append(r_lst)
                        if fold_ii == n_folds-1:
                            corr_dict[layer_id][roi_name] = np.mean(np.array(corr_dict[layer_id][roi_name], dtype=np.float16),axis=0)
                    fold_dict[layer_id][roi_name].append(r)
                
    # Compile all results into a DataFrame for easy analysis
    # Define the column types explicitly
    column_types = {
        'ROI': str,
        'Layer': str,
        'Model': str,
        'R': float,
        '%R2': float,
        'Significance': float,
        'SEM': float,
        'LNC': float,
        'UNC': float
    }

    # Initialize an empty list to store individual DataFrames
    all_rois_list = []

    for layer_id, layer_dict in fold_dict.items():
        for roi_name, r_lst in layer_dict.items():
            # Compute statistical significance of the correlations
            r_lst_array = np.array(r_lst)  # Convert the list to a NumPy array
            significance = ttest_1samp(r_lst_array, 0)[1]
            R = np.mean(r_lst_array)
            
            output_dict = {
                "ROI": roi_name,
                "Layer": layer_id,
                "Model": model_name,
                "R": R,
                "%R2": R ** 2,
                "Significance": significance,
                "SEM": sem(r_lst_array),
                "LNC": 0,  # or np.nan if you prefer
                "UNC": 0   # or np.nan if you prefer
            }
            
            # Create a DataFrame for this layer/ROI combination
            layer_df = pd.DataFrame([output_dict])
            all_rois_list.append(layer_df)

    # Concatenate all DataFrames at once
    all_rois_df = pd.concat(all_rois_list, ignore_index=True)

    # Ensure all columns have the correct data type
    all_rois_df = all_rois_df.astype(column_types)
            
    if return_correlations:
        return all_rois_df,corr_dict
    
    return all_rois_df
