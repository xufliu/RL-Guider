import torch
from rdkit import Chem

smarts_library = {
    # ---------------------------
    # 原子 (Atoms)
    # ---------------------------
    "Carbon Atom": "[C]",
    "Chlorine Atom": "[Cl]",
    "Sulfur Atom": "[S]",
    "Oxygen Atom": "[O]",
    "Bromine Atom": "[Br]",
    "Fluorine Atom": "[F]",
    "Nitrogen Atom": "[N]",
    "Iodine": "[I]",
    "Phosphorus": "[P]",
    "Hydrogen": "[H]",
    
    # ---------------------------
    # 官能团 (Functional Groups)
    # ---------------------------
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Tertiary Amine": "[#6][N]([#6])[#6]",
    "Fluoroalkane": "[#6][F]",
    "Chloroalkane": "[#6][Cl]",
    "Secondary Amine": "[NX3;H2,H1;!$(NC=O)]",
    "Aliphatic Alcohol": "[#6;!a][O;H1]",
    "Carboxylic Acid": "[C](=[O])[O;H1]",
    "Aromatic Aldehyde": "[#6;a][C](=[O])",
    "Aromatic Alcohol": "[#6;a][O;H1]",
    "Sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
    "Aromatic Amine": "[#6;a][N;H2]",
    "Aliphatic Amine": "[#6;!a][N;H2]",
    "Ester": "[#6][O][C](=[O])[#6]",
    "Alkene": "C=C",
    "Ketone": "[#6][C](=[O])[#6]",
    "Di-alkyl urea": "[#6][N]([#6])C(=O)N([#6])[#6]",
    "Thioalkyl ether": "[#6]-[#16]-[#6]",
    "Cyano": "[C]#[N]",
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
    "Bromoalkane": "[CX4;!$(C=*)]Br",
    "Sulfonate": "[S;$(S(=O)(=O)[O])]",
    "Guanidine": "[C](=[N])([N])[N]",
    "Nitro Group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "Acetylide": "C#C",
    "Acetohydroxamate": "[C;$(C(=O)[N]-[O])]",
    "Dimethoxymethyl": "[C][O][C][O][C]",
    "Dialkylaminomethyleneimine": "[N]-[C]=[N]",
    "N,N-Dimethylacrylamide": "C=C-[C;$(C(=O)-[N;$(N(-C)-C)])]",
    "Iodoalkane": "[C](I)",
    
    "CH2": "[CH2]", 
    "CF3": "C(F)(F)F",  # 三氟甲基
    "Phosphate": "[PX4](=O)([O-])([O-])[O-]",  # 磷酸基
    "Cyclopropane": "C1CC1",  # 环丙烷
    "Piperidine": "N1CCCCC1",  # 哌啶环
    "Trifluoromethyl Group": "C(F)(F)F",  # 同CF3
    "Ethylene Glycol": "C(O)CO",  # 乙二醇片段
    "Long Alkyl Chain": "[CX4][CX4][CX4][CX4][CX4]",  # 长链烷基（5+碳）
    "Aromatic Ether": "[#6;a][O][#6;a]"  # 芳香醚
}

def check_smiles_for_groups(smiles, groups):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    results = {}
    for name, smarts in groups.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS pattern for {name}")
        
        matches = mol.GetSubstructMatches(query)
        results[name] = len(matches) > 0
    
    return results

def check_smiles_for_group(smiles, group):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    query = Chem.MolFromSmarts(smarts_library[smiles])
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern for {name}")

    matches = mol.GetSubstructMatches(query)
    result = len(matches) > 0
    return result


def get_peptide_general_action_list(peptide):
    aminos = {
        "alanine (A)": "A",
        "arginine (R)": "R",
        "asparagine (N)": "N",
        "aspartate (D)": "D",
        "cysteine (C)": "C",
        "glutamate (E)": "E",
        "glutamine (Q)": "Q",
        "glycine (G)": "G",
        "histidine (H)": "H",
        "isoleucine (I)": "I",
        "leucine (L)": "L",
        "lysine (K)": "K",
        "methionine (M)": "M",
        "phenylalanine (F)": "F",
        "proline (P)": "P",
        "serine (S)": "S",
        "threonine (T)": "T",
        "tryptophan (W)": "W",
        "tyrosine (Y)": "Y",
        "valine (V)": "V"
    }
    amino_list = list(aminos.keys())

    action_list = []

    # add
    for amino in amino_list:
        action_list.append(f"Insert a {amino}. ")

    # remove
    for amino in amino_list:
        action_list.append(f"Remove a {amino}. ")


    # Replace
    for amino_1 in amino_list:
        for amino_2 in amino_list:
            if amino_1 == amino_2:
                continue
            if aminos[amino_1] in peptide:
                action_list.append(f"Replace {amino_1} with {amino_2}. ")
            else:
                continue

    return action_list

def get_peptide_all_general_action_list():
    aminos = {
        "alanine (A)": "A",
        "arginine (R)": "R",
        "asparagine (N)": "N",
        "aspartate (D)": "D",
        "cysteine (C)": "C",
        "glutamate (E)": "E",
        "glutamine (Q)": "Q",
        "glycine (G)": "G",
        "histidine (H)": "H",
        "isoleucine (I)": "I",
        "leucine (L)": "L",
        "lysine (K)": "K",
        "methionine (M)": "M",
        "phenylalanine (F)": "F",
        "proline (P)": "P",
        "serine (S)": "S",
        "threonine (T)": "T",
        "tryptophan (W)": "W",
        "tyrosine (Y)": "Y",
        "valine (V)": "V"
    }
    amino_list = list(aminos.keys())

    action_list = []

    # add
    for amino in amino_list:
        action_list.append(f"Insert a {amino}. ")

    # remove
    for amino in amino_list:
        action_list.append(f"Remove a {amino}. ")


    # Replace
    for amino_1 in amino_list:
        for amino_2 in amino_list:
            if amino_1 == amino_2:
                continue
            action_list.append(f"Replace {amino_1} with {amino_2}. ")

    return action_list


def get_protein_general_action_list(protein):
    aminos = {
        "alanine (A)": "A",
        "arginine (R)": "R",
        "asparagine (N)": "N",
        "aspartate (D)": "D",
        "cysteine (C)": "C",
        "glutamate (E)": "E",
        "glutamine (Q)": "Q",
        "glycine (G)": "G",
        "histidine (H)": "H",
        "isoleucine (I)": "I",
        "leucine (L)": "L",
        "lysine (K)": "K",
        "methionine (M)": "M",
        "phenylalanine (F)": "F",
        "proline (P)": "P",
        "serine (S)": "S",
        "threonine (T)": "T",
        "tryptophan (W)": "W",
        "tyrosine (Y)": "Y",
        "valine (V)": "V"
    }
    amino_list = list(aminos.keys())

    action_list = []

    # add
    for amino in amino_list:
        action_list.append(f"Insert a {amino}. ")

    # remove
    for amino in amino_list:
        action_list.append(f"Remove a {amino}. ")


    # Replace
    for amino_1 in amino_list:
        for amino_2 in amino_list:
            if amino_1 == amino_2:
                continue
            if aminos[amino_1] in protein:
                action_list.append(f"Replace {amino_1} with {amino_2}. ")
            else:
                continue

    return action_list

def get_protein_all_general_action_list():
    aminos = {
        "alanine (A)": "A",
        "arginine (R)": "R",
        "asparagine (N)": "N",
        "aspartate (D)": "D",
        "cysteine (C)": "C",
        "glutamate (E)": "E",
        "glutamine (Q)": "Q",
        "glycine (G)": "G",
        "histidine (H)": "H",
        "isoleucine (I)": "I",
        "leucine (L)": "L",
        "lysine (K)": "K",
        "methionine (M)": "M",
        "phenylalanine (F)": "F",
        "proline (P)": "P",
        "serine (S)": "S",
        "threonine (T)": "T",
        "tryptophan (W)": "W",
        "tyrosine (Y)": "Y",
        "valine (V)": "V"
    }
    amino_list = list(aminos.keys())

    action_list = []

    # add
    for amino in amino_list:
        action_list.append(f"Insert a {amino}. ")

    # remove
    for amino in amino_list:
        action_list.append(f"Remove a {amino}. ")


    # Replace
    for amino_1 in amino_list:
        for amino_2 in amino_list:
            if amino_1 == amino_2:
                continue
            action_list.append(f"Replace {amino_1} with {amino_2}. ")
            

    return action_list


def get_smiles_general_action_list(smiles):
    atoms = {
        "Carbon Atom": "C",
        "Chlorine Atom": "Cl",
        "Sulfur Atom": "S",
        "Oxygen Atom": "O",
        "Bromine Atom": "Br",
        "Fluorine Atom": "F",
        "Nitrogen Atom": "N",
        "Iodine": "I",
        "Phosphorus": "P",
    }
    atom_list = list(atoms.keys())

    functional_groups = {
        "Amide": "[NX3][CX3](=[OX1])[#6]",
        "Ether": "[OD2]([#6])[#6]",
        "Tertiary Amine": "[#6][N]([#6])[#6]",
        "Fluoroalkane": "[#6][F]",
        "Chloroalkane": "[#6][Cl]",
        "Secondary Amine": "[NX3;H2,H1;!$(NC=O)]",
        "Aliphatic Alcohol": "[#6;!a][O;H1]",
        "Carboxylic Acid": "[C](=[O])[O;H1]",
        "Aromatic Aldehyde": "[#6;a][C](=[O])",
        "Aromatic Alcohol": "[#6;a][O;H1]",
        "Sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "Aromatic Amine": "[#6;a][N;H2]",
        "Aliphatic Amine": "[#6;!a][N;H2]",
        "Ester": "[#6][O][C](=[O])[#6]",
        "Alkene": "C=C",
        "Ketone": "[#6][C](=[O])[#6]",
        "Di-alkyl urea": "[#6][N]([#6])C(=O)N([#6])[#6]",
        "Thioalkyl ether": "[#6]-[#16]-[#6]",
        "Cyano": "[C]#[N]",
        "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "Bromoalkane": "[CX4;!$(C=*)]Br",
        "Sulfonate": "[S;$(S(=O)(=O)[O])]",
        "Guanidine": "[C](=[N])([N])[N]",
        "Nitro Group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "Acetylide": "C#C",
        "Acetohydroxamate": "[C;$(C(=O)[N]-[O])]",
        "Dimethoxymethyl": "[C][O][C][O][C]",
        "Dialkylaminomethyleneimine": "[N]-[C]=[N]",
        "N,N-Dimethylacrylamide": "C=C-[C;$(C(=O)-[N;$(N(-C)-C)])]",
        "Iodoalkane": "[C](I)",
    }
    functional_group_list = list(functional_groups.keys())

    atom_results = check_smiles_for_groups(smiles, atoms)
    functional_group_results = check_smiles_for_groups(smiles, functional_groups)

    action_list = []

    # add
    for group in atom_list:
        action_list.append(f"Add a {group}. ")

    for group in functional_group_list:
        action_list.append(f"Add a {group}. ")

    # remove
    for group in atom_list:
        if atom_results[group]:
            action_list.append(f"Remove a {group}. ")
    
    for group in functional_group_list:
        if functional_group_results[group]:
            action_list.append(f"Remove a {group}. ")

    # replace
    for group_1 in atom_list:
        for group_2 in functional_group_list:
            if atom_results[group_1]:
                action_list.append(f"Replace {group_1} with {group_2}. ")
    
    for group_1 in atom_list:
        for group_2 in atom_list:
            if group_1 == group_2:
                continue
            if atom_results[group_1]:
                action_list.append(f"Replace {group_1} with {group_2}. ")

    for group_1 in functional_group_list:
        for group_2 in functional_group_list:
            if group_1 == group_2:
                continue
            if functional_group_results[group_1]:
                action_list.append(f"Replace {group_1} with {group_2}. ")
    
    for group_1 in functional_group_list:
        for group_2 in atom_list:
            if functional_group_results[group_1]:
                action_list.append(f"Replace {group_1} with {group_2}. ")

    return action_list

# def get_smiles_general_action_list(smiles):

#     substitution_actions = [
#         ("Iodoalkane", "Fluoroalkane"),
#         ("Bromoalkane", "Chloroalkane"),
#         ("Aromatic Aldehyde", "Ketone"),
#         ("Thioalkyl ether", "Ether"),
#         ("Sulfonate", "Carboxylic Acid"),
#         ("Nitro Group", "Cyano"),
#         ("Tertiary Amine", "Secondary Amine"),
#         ("Aromatic Alcohol", "Aliphatic Alcohol"),
#         ("Ester", "Amide"),
#         ("Cyano", "Carbon Atom"),
#         ("Iodine", "Hydrogen"),  # 假设Hydrogen在您的atom_list中存在
#         ("Sulfonamide", "Di-alkyl urea"),
#         ("Acetohydroxamate", "Ester"),
#         ("Aromatic Amine", "Aliphatic Amine"),
#         ("Dialkylaminomethyleneimine", "Amide"),
#         ("Guanidine", "Tertiary Amine"),
#         ("Di-alkyl urea", "Carbamate"),
#         ("Phosphorus", "Sulfur Atom"),
#         ("N,N-Dimethylacrylamide", "Amide"),
#         ("Nitro Group", "Fluorine Atom"),
#         ("Chloroalkane", "Ether"),
#         ("Ketone", "CH2"),  # 假设CH2由Carbon Atom组合
#         ("Alkene-containing amide", "Amide"),  # 需自定义
#         ("Bromine", "Carbon Atom"),
#         ("Aromatic Ether", "Ether"),  # 需自定义
#         ("Cyano", "CF3"),  # 需组合Carbon/Fluorine
#         ("Sulfonate", "Phosphate"),  # 需自定义
#         ("Iodoalkane", "Acetylide"),
#         ("Secondary Amine", "Tertiary Amine"),
#         ("Sulfur Atom", "Oxygen Atom")
#     ]        
    
#     # 增加操作（Additions, 10个）
#     addition_actions = [
#         "Fluorine Atom",         # 策略31
#         "Ether",                 # 策略32
#         "Tertiary Amine",        # 策略33
#         "Ether",                 # 策略34
#         "Amide",                 # 策略35
#         "Cyclopropane",          # 策略36（需Carbon组合）
#         "Chlorine Atom",         # 策略37
#         "Piperidine",            # 策略38（需Carbon/Nitrogen组合）
#         "Trifluoromethyl Group", # 策略39（需Carbon/Fluorine组合）
#         "Ethylene Glycol"        # 策略40（需Oxygen/Carbon组合）
#     ]
    
#     # 删除操作（Removals, 10个）
#     removal_actions = [
#         "Iodine",               # 策略41
#         "Nitro Group",          # 策略42
#         "Sulfonate",            # 策略43
#         "Guanidine",            # 策略44
#         "Aliphatic Alcohol",    # 策略45（需指定多个羟基）
#         "Long Alkyl Chain",     # 策略46（需Carbon组合）
#         "Alkene",               # 策略47
#         "Aromatic Amine",       # 策略48
#         "Sulfur Atom",          # 策略49
#         "Bromoalkane"           # 策略50
#     ]

#     for substitution_action in substitution_actions:
#         if not check_smiles_for_group(smiles, substitution_action[0]):
#             substitution_actions.remove(substitution_action)

#     for removal_action in removal_actions:
#         if not check_smiles_for_group(smiles, removal_action):
#             removal_actions.remove(removal_action)

#     action_list = []

#     # add
#     for addition_action in addition_actions:
#         action_list.append(f"Add a {addition_action}. ")

#     # remove
#     for removal_action in removal_actions:
#         action_list.append(f"Remove a {removal_action}. ")
    
#     # replace
#     for substitution_action in substitution_actions:
#         action_list.append(f"Replace {substitution_action[0]} with {substitution_action[1]}. ")
    
#     return action_list


def get_smiles_all_general_action_list():
    atoms = {
        "Carbon Atom": "C",
        "Chlorine Atom": "Cl",
        "Sulfur Atom": "S",
        "Oxygen Atom": "O",
        "Bromine Atom": "Br",
        "Fluorine Atom": "F",
        "Nitrogen Atom": "N",
        "Iodine": "I",
        "Phosphorus": "P",
    }
    atom_list = list(atoms.keys())

    functional_groups = {
        "Amide": "[NX3][CX3](=[OX1])[#6]",
        "Ether": "[OD2]([#6])[#6]",
        "Tertiary Amine": "[#6][N]([#6])[#6]",
        "Fluoroalkane": "[#6][F]",
        "Chloroalkane": "[#6][Cl]",
        "Secondary Amine": "[NX3;H2,H1;!$(NC=O)]",
        "Aliphatic Alcohol": "[#6;!a][O;H1]",
        "Carboxylic Acid": "[C](=[O])[O;H1]",
        "Aromatic Aldehyde": "[#6;a][C](=[O])",
        "Aromatic Alcohol": "[#6;a][O;H1]",
        "Sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "Aromatic Amine": "[#6;a][N;H2]",
        "Aliphatic Amine": "[#6;!a][N;H2]",
        "Ester": "[#6][O][C](=[O])[#6]",
        "Alkene": "C=C",
        "Ketone": "[#6][C](=[O])[#6]",
        "Di-alkyl urea": "[#6][N]([#6])C(=O)N([#6])[#6]",
        "Thioalkyl ether": "[#6]-[#16]-[#6]",
        "Cyano": "[C]#[N]",
        "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "Bromoalkane": "[CX4;!$(C=*)]Br",
        "Sulfonate": "[S;$(S(=O)(=O)[O])]",
        "Guanidine": "[C](=[N])([N])[N]",
        "Nitro Group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "Acetylide": "C#C",
        "Acetohydroxamate": "[C;$(C(=O)[N]-[O])]",
        "Dimethoxymethyl": "[C][O][C][O][C]",
        "Dialkylaminomethyleneimine": "[N]-[C]=[N]",
        "N,N-Dimethylacrylamide": "C=C-[C;$(C(=O)-[N;$(N(-C)-C)])]",
        "Iodoalkane": "[C](I)",
    }
    functional_group_list = list(functional_groups.keys())

    action_list = []

    # add
    for group in atom_list:
        action_list.append(f"Add a {group}. ")

    for group in functional_group_list:
        action_list.append(f"Add a {group}. ")

    # remove
    for group in atom_list:
        action_list.append(f"Remove a {group}. ")
    
    for group in functional_group_list:
        action_list.append(f"Remove a {group}. ")

    # replace
    for group_1 in atom_list:
        for group_2 in functional_group_list:
            action_list.append(f"Replace {group_1} with {group_2}. ")
    
    for group_1 in atom_list:
        for group_2 in atom_list:
            if group_1 == group_2:
                continue
            action_list.append(f"Replace {group_1} with {group_2}. ")

    for group_1 in functional_group_list:
        for group_2 in functional_group_list:
            if group_1 == group_2:
                continue
            action_list.append(f"Replace {group_1} with {group_2}. ")
    
    for group_1 in functional_group_list:
        for group_2 in atom_list:
            action_list.append(f"Replace {group_1} with {group_2}. ")
    
    return action_list

# def get_smiles_all_general_action_list():

#     substitution_actions = [
#         ("Iodoalkane", "Fluoroalkane"),
#         ("Bromoalkane", "Chloroalkane"),
#         ("Aromatic Aldehyde", "Ketone"),
#         ("Thioalkyl ether", "Ether"),
#         ("Sulfonate", "Carboxylic Acid"),
#         ("Nitro Group", "Cyano"),
#         ("Tertiary Amine", "Secondary Amine"),
#         ("Aromatic Alcohol", "Aliphatic Alcohol"),
#         ("Ester", "Amide"),
#         ("Cyano", "Carbon Atom"),
#         ("Iodine", "Hydrogen"),  # 假设Hydrogen在您的atom_list中存在
#         ("Sulfonamide", "Di-alkyl urea"),
#         ("Acetohydroxamate", "Ester"),
#         ("Aromatic Amine", "Aliphatic Amine"),
#         ("Dialkylaminomethyleneimine", "Amide"),
#         ("Guanidine", "Tertiary Amine"),
#         ("Di-alkyl urea", "Carbamate"),
#         ("Phosphorus", "Sulfur Atom"),
#         ("N,N-Dimethylacrylamide", "Amide"),
#         ("Nitro Group", "Fluorine Atom"),
#         ("Chloroalkane", "Ether"),
#         ("Ketone", "CH2"),  # 假设CH2由Carbon Atom组合
#         ("Alkene-containing amide", "Amide"),  # 需自定义
#         ("Bromine", "Carbon Atom"),
#         ("Aromatic Ether", "Ether"),  # 需自定义
#         ("Cyano", "CF3"),  # 需组合Carbon/Fluorine
#         ("Sulfonate", "Phosphate"),  # 需自定义
#         ("Iodoalkane", "Acetylide"),
#         ("Secondary Amine", "Tertiary Amine"),
#         ("Sulfur Atom", "Oxygen Atom")
#     ]
    
#     # 增加操作（Additions, 10个）
#     addition_actions = [
#         "Fluorine Atom",         # 策略31
#         "Ether",                 # 策略32
#         "Tertiary Amine",        # 策略33
#         "Ether",                 # 策略34
#         "Amide",                 # 策略35
#         "Cyclopropane",          # 策略36（需Carbon组合）
#         "Chlorine Atom",         # 策略37
#         "Piperidine",            # 策略38（需Carbon/Nitrogen组合）
#         "Trifluoromethyl Group", # 策略39（需Carbon/Fluorine组合）
#         "Ethylene Glycol"        # 策略40（需Oxygen/Carbon组合）
#     ]
    
#     # 删除操作（Removals, 10个）
#     removal_actions = [
#         "Iodine",               # 策略41
#         "Nitro Group",          # 策略42
#         "Sulfonate",            # 策略43
#         "Guanidine",            # 策略44
#         "Aliphatic Alcohol",    # 策略45（需指定多个羟基）
#         "Long Alkyl Chain",     # 策略46（需Carbon组合）
#         "Alkene",               # 策略47
#         "Aromatic Amine",       # 策略48
#         "Sulfur Atom",          # 策略49
#         "Bromoalkane"           # 策略50
#     ]

#     action_list = []

#     # add
#     for addition_action in addition_actions:
#         action_list.append(f"Add a {addition_action}. ")

#     # remove
#     for removal_action in removal_actions:
#         action_list.append(f"Remove a {removal_action}. ")
    
#     # replace
#     for substitution_action in substitution_actions:
#         action_list.append(f"Replace {substitution_action[0]} with {substitution_action[1]}. ")
    
#     return action_list

def str_2_emb(smiles_list, tokenizer, model, collator):
    inputs = collator(tokenizer(smiles_list))
    outputs = model(**inputs, output_hidden_states=True)
    full_embeddings = outputs[1][-1]
    mask = inputs['attention_mask']
    embeddings = torch.tensor(((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)), requires_grad=False)
    return embeddings


from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)



def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[:-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list



