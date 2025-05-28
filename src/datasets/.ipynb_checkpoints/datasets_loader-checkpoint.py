import pandas as pd

def zinc_process(zinc_path="250k_rndm_zinc_drugs_clean_3.csv", num=1000, min_length=0, max_length=20):
    """
    extract top-num smiles_list & logP_list from zinc
    """
    df = pd.read_csv(zinc_path)
    df['smiles'] = df['smiles'].str.replace('\n', '')
    df['smiles'] = df['smiles'].str.replace('\r', '')
    df['length'] = df['smiles'].apply(len)
    df_ = df.sort_values(by='logP', ascending=False).reset_index()
    df_ = df_[(df_['length'] >= min_length) & (df_['length'] <= max_length)].head(num).reset_index()
    smiles_list = df_['smiles'].to_list()
    logP_list = df_['logP'].to_list()
    return smiles_list, logP_list