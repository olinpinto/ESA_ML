import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Dropping unused data columns
def preprocess_dataframe(df, type_value):
    columns_to_drop = ['HTXS_njets30', 'weight_lumiScaled', 'centralObjectWeight', 
                       'HTXS_Higgs_pt', 'HTXS_Higgs_y', 'genWeight', 'leadPhotonPfRelIsoAll', 
                       'leadPhotonPfRelIsoChg', 'leadPhotonR9', 'leadPhotonSieie', 
                       'leadPhotonElectronVeto', 'leadPhotonPixelSeed', 'subleadPhotonPfRelIsoAll', 
                       'subleadPhotonPfRelIsoChg', 'subleadPhotonR9', 'subleadPhotonSieie', 
                       'subleadPhotonElectronVeto', 'subleadPhotonPixelSeed', 'subsubleadPhotonIDMVA', 
                       'subsubleadPhotonPfRelIsoAll', 'subsubleadPhotonPfRelIsoChg', 'subsubleadPhotonR9', 
                       'subsubleadPhotonSieie', 'subsubleadPhotonElectronVeto', 'subsubleadPhotonPixelSeed', 
                       'leadJetID', 'leadJetPUJID', 'subleadJetID', 'subleadJetPUJID', 'subsubleadJetID', 
                       'subsubleadJetPUJID', 'leadElectronMvaFall17V2Iso', 'leadElectronConvVeto', 
                       'subleadElectronMvaFall17V2Iso', 'subleadElectronConvVeto', 'subsubleadElectronMvaFall17V2Iso', 
                       'subsubleadElectronConvVeto', 'leadMuonPfRelIso04', 'subleadMuonPfRelIso04', 
                       'subsubleadMuonPfRelIso04', 'leadPhotonPtOvM', 'subleadPhotonPtOvM', 
                       'subsubleadElectronEn', 'subsubleadElectronEta', 'subsubleadElectronPhi', 
                       'subsubleadElectronPt', 'subsubleadElectronCharge', 'subsubleadElectronMass', 
                       'leadMuonMass', 'subleadMuonMass', 'subsubleadMuonMass']

    df_red = df.drop(columns_to_drop, axis=1)
    df_red['Type'] = type_value

    return df_red

# Assigning type values according to the HTXS Stage1_2 data
def preprocess_signal_dataframe(df):
    type_1_values = list(range(101, 117))  # ggH
    type_2_values = list(range(206, 211))  # VBF-like
    type_3_values = [204]                  # VH-like
    type_4_values = list(range(601, 606))  # ttH
    type_5_values = list(range(301, 306))  # WH-lep
    type_6_values = list(range(401, 406))  # ZH-lep

    df['Type'] = '3'  # Default to '3' for all
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_1_values), 'Type'] = '0'
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_2_values), 'Type'] = '1'
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_3_values), 'Type'] = '3'
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_4_values), 'Type'] = '2'
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_5_values), 'Type'] = '3'
    df.loc[df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_6_values), 'Type'] = '3'
    df.loc[~df['HTXS_stage1_2_cat_pTjet30GeV'].isin(type_1_values + type_2_values + type_3_values + type_4_values + type_5_values + type_6_values), 'Type'] = '3'

    df = df.drop(['HTXS_stage1_2_cat_pTjet30GeV'], axis=1)

    return df

# Combining and shuffling dataframes
def combine_and_preprocess_dfs(main_dfs, background_dfs, first_shuffle_state=5, final_shuffle_state=3):
    preprocessed_main_dfs = [preprocess_signal_dataframe(df) for df in main_dfs]

    df_combined_signal = pd.concat(preprocessed_main_dfs, ignore_index=True)
    df_combined_signal = shuffle(df_combined_signal, random_state=first_shuffle_state)

    df_combined = pd.concat([df_combined_signal] + background_dfs, ignore_index=True)
    df_combined = shuffle(df_combined, random_state=final_shuffle_state)
    df_combined = df_combined.replace(np.nan, 0)

    # Adding the PtoM column
    df_combined['leadPhotonPtoM'] = df_combined['leadPhotonPt'] / df_combined['diphotonMass']
    df_combined['subleadPhotonPtoM'] = df_combined['subleadPhotonPt'] / df_combined['diphotonMass']

    # Columns to be kept for furhter analysis
    columns_to_keep = [
        'leadPhotonPtoM', 'leadPhotonEta', 
        'subleadPhotonPtoM', 'subleadPhotonEta', 
        'leadJetPt', 'leadJetEta', 'leadJetPhi', 'leadJetQGL', 'leadJetBTagScore', 
        'subleadJetPt', 'subleadJetEta', 'subleadJetPhi', 'subleadJetQGL', 'subleadJetBTagScore', 
        'subsubleadJetPt', 'subsubleadJetEta', 'subsubleadJetPhi', 'subsubleadJetQGL', 'subsubleadJetBTagScore', 
        'leadElectronPt', 'leadElectronEta', 'leadElectronPhi', 'leadElectronCharge', 
        'subleadElectronPt', 'subleadElectronEta', 'subleadElectronPhi', 'subleadElectronCharge', 
        'leadMuonPt', 'leadMuonEta', 'leadMuonPhi', 'leadMuonCharge', 
        'subleadMuonPt', 'subleadMuonEta', 'subleadMuonPhi', 'subleadMuonCharge', 
        'dijetMass', 'dijetPt', 'dijetEta', 'dijetPhi', 'dijetAbsDEta', 'dijetDPhi', 'dijetMinDRJetPho', 'dijetCentrality', 'dijetDiphoAbsDPhiTrunc', 'dijetDiphoAbsDEta', 
        'leadJetDiphoDPhi', 'leadJetDiphoDEta', 'subleadJetDiphoDPhi', 'subleadJetDiphoDEta', 
        'nSoftJets', 
        'metPt', 'metPhi', 'metSumET', 'metSignificance', 'Type', 'weight', 'diphotonMass', 'leadPhotonIDMVA', 'subleadPhotonIDMVA'
    ]

    df_combined = df_combined[columns_to_keep]

    # Utilising one-hot enoding for jet tagging, replacing -999 and NaN values
    for col in df_combined.columns:
        if 'Eta' in col:
            new_col = f"Jet_{col}"
            df_combined[new_col] = df_combined[col].apply(lambda x: 0 if x != -999 else 1)

    df_combined = df_combined.replace(-999, 0)
    df_combined = df_combined.replace(np.nan, 0)

    return df_combined

# Function for adding the diphotonMass and original weights back to the dataframe, should there be a need
def add_diphotonMass_to_results(df_test_results, original_df):
    df_test_results = df_test_results.reset_index(drop=True)
    original_df = original_df.reset_index(drop=True)

    df_test_results_with_mass_and_weights = df_test_results.copy()
    df_test_results_with_mass_and_weights['diphotonMass'] = original_df.loc[df_test_results.index, 'diphotonMass']
    df_test_results_with_mass_and_weights['original_weight'] = original_df.loc[df_test_results.index, 'weight']
    df_test_results_with_mass_and_weights['scaled_original_weight'] = original_df.loc[df_test_results.index, 'weight'] * 137000

    return df_test_results_with_mass_and_weights


