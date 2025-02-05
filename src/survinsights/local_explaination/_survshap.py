import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap

from survinsights.prediction import predict

sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")


def survshap(explainer, new_data, sample_id=0):
    """
    Compute SurvSHAP


    Parameters
    ----------
    explainer : Python object
            Instance of the explainer class for the survival model.
    new_data : pd.DataFrame or np.ndarray
            Data for new observations whose predictions need to be explained.
    sample_id : int, optional
            Index of the observation in new_data to explain. Defaults to the first observation (index 0).

    Returns
    -------
    pd.DataFrame
           DataFrame containing survshap values for each feature and time point.
    """

    def preprocess_for_shap(feats_df):
        preprocessed_feats_df = pd.DataFrame()
        decoded_feat_names = explainer.feat_names
        for idx, f_name in enumerate(decoded_feat_names):
            if f_name in explainer.numeric_feat_names:
                preprocessed_feats_df[f_name] = feats_df[:, idx].flatten()
            else:
                feat_encoder = explainer.encoders[f_name]
                feat_col_sel = feat_encoder.get_feature_names_out([f_name]).tolist()
                preprocessed_feats_df[feat_col_sel] = feat_encoder.transform(
                    feats_df[:, idx].reshape((-1, 1))
                ).toarray()
                
        # sort the dataframe based on the order of encoded feature name in trained model
        preprocessed_feats_df = preprocessed_feats_df[explainer.features_df.columns.tolist()]

        predictions_df = predict(explainer, preprocessed_feats_df, prediction_type="survival")
        return predictions_df.pred.values.reshape((feats_df.shape[0], -1))

    if sample_id is None:
        sample_id = 0

    sel_sample_df = new_data.iloc[[sample_id]]

    decoded_features_df = pd.DataFrame(columns=explainer.feat_names)
    decoded_sel_sample_df = pd.DataFrame(columns=explainer.feat_names)
    decoded_feat_names = explainer.feat_names

    for feat_name in explainer.feat_names:
        if feat_name in explainer.numeric_feat_names:
            decoded_features_df[feat_name] = explainer.features_df[feat_name]
            decoded_sel_sample_df[feat_name] = sel_sample_df[feat_name]
        else:
            encoder = explainer.encoders[feat_name]
            encoded_feat_name = encoder.get_feature_names_out([feat_name])
            decoded_features_df[feat_name] = encoder.inverse_transform(
                explainer.features_df[encoded_feat_name]
            ).flatten()
            decoded_sel_sample_df[feat_name] = encoder.inverse_transform(sel_sample_df[encoded_feat_name]).flatten()

    # Support KernelSHAP
    shap_explainer = shap.KernelExplainer(preprocess_for_shap, decoded_features_df)
    shap_values = shap_explainer(decoded_sel_sample_df)
    survshap_result_df = pd.DataFrame(data=shap_values[0].values.T, columns=decoded_feat_names)
    survshap_result_df["times"] = explainer.times

    return survshap_result_df


def plot_survshap(survshap_result_df, sample_id=0):
    """
    Visualize SurvSHAP values for a given observation over time.

    Parameters
    ----------
    survshap_result_df : `pd.Dataframe`
            survshap result to be visualized
    sample_id : int, optional
           ID of the observation being visualized. Default is 0.
    """

    _, ax = plt.subplots(figsize=(9, 5))
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor("black")

    melted_survshap_df = survshap_result_df.melt("times", var_name="features", value_name="values")
    sns.lineplot(data=melted_survshap_df, x="times", y="values", hue="features", ax=ax)

    plt.legend(prop={"size": 12})
    plt.xlabel("Time")
    plt.ylabel("SurvSHAP(t)")
    plt.title(f"SurvSHAP for observation ID = {sample_id}")
    plt.show()
