import os
import re
import sys
import json
import lmdb
import numpy as np
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
import Levenshtein
from functools import partial
from torch.utils.data import Dataset
from modelscope import snapshot_download
from mhcflurry import Class1PresentationPredictor
from transformers import BertTokenizerFast
import torch
import torch.nn.functional as F
import logging

sys.path.append("src")
from llm.deepseek_interface import run_deepseek_prompts
from llm.chatgpt_interface import run_openai_prompts
from llm.llama_interface import run_llama_prompts
from llm.galactica_interface import run_gala_prompts
from llm.chemdfm_interface import run_dfm_prompts
from utils import sascorer
from model.tape_benchmark_models import BertForTokenClassification2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)

def load_ProteinDT_model(input_model_path, chache_dir, mean_output, num_labels):

    model = BertForTokenClassification2.from_pretrained(
        "Rostlab/prot_bert_bfd",
        cache_dir=chache_dir,
        mean_output=mean_output,
        num_labels=num_labels,
    )

    # load model from checkpoint
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing keys: {}".format(missing_keys))
    print("unexpected keys: {}".format(unexpected_keys))
    
    return model

def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class ProteinListDataset(Dataset):
    def __init__(self, protein_sequence_list, tokenizer, task_id):
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.protein_sequence_list = protein_sequence_list
        return

    def __len__(self):
        return len(self.protein_sequence_list)

    def __getitem__(self, index: int):
        protein_sequence = self.protein_sequence_list[index]

        token_ids = self.tokenizer(list(protein_sequence), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)

        return token_ids, input_mask

    def collate_fn(self, batch):
        input_ids, input_mask = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))

        output = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return output


class ProteinSecondaryStructureDataset(Dataset):
    def __init__(self, data_path, tokenizer, target='ss3'):
        self.tokenizer = tokenizer
        self.target = target
        self.ignore_index = -100

        env = lmdb.open(data_path, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        
        self.protein_sequence_list = []
        self.ss3_labels_list = []
        self.ss8_labels_list = []

        for index in range(num_examples):
            with env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
            # print(item.keys())
            protein_sequence = item["primary"]
            ss3_labels = item["ss3"]
            ss8_labels = item["ss8"]
            protein_length = item["protein_length"]

            if len(protein_sequence) > 1024:
                protein_sequence = protein_sequence[:1024]
                ss3_labels = ss3_labels[:1024]
                ss8_labels = ss8_labels[:1024]
                
            self.protein_sequence_list.append(protein_sequence)
            self.ss3_labels_list.append(ss3_labels)
            self.ss8_labels_list.append(ss8_labels)
        
        if self.target == "ss3":
            self.labels_list = self.ss3_labels_list
            self.num_labels = 3
        else:
            self.labels_list = self.ss8_labels_list
            self.num_labels = 8
        return

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index: int):
        protein_sequence = self.protein_sequence_list[index]
        labels = self.labels_list[index]

        token_ids = self.tokenizer(list(protein_sequence), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)
        
        # pad with -1s because of cls/sep tokens
        labels = np.asarray(labels, np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)

        return token_ids, input_mask, labels

    def collate_fn(self, batch):
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        return output

model_pretrained_checkpoint = "/root/ChatDrug/data/peptide/models_class1_presentation/models" # path to model pretrained checkpoint
MHC_peptide_predictor = Class1PresentationPredictor.load(model_pretrained_checkpoint)
EPS = 1e-10


#########################################################################################################################
device = "cuda"
chache_dir = "/root/ChatDrug/data/protein/temp_pretrained_ProteinDT" # cache dir
input_model_path = "/root/ChatDrug/data/protein/pytorch_model_ss3.bin" # input model path
protein_model = load_ProteinDT_model(input_model_path, chache_dir, mean_output=True, num_labels=3)
protein_model = protein_model.to(device)
protein_tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)
#########################################################################################################################


props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func


task2threshold_list = {
    101: [[{'logP': 0}], [{'logP': 0.5}]],
    102: [[{'logP': 0}], [{'logP': 0.5}]],
    103: [[{'QED': 0}], [{'QED': 0.1}]],
    104: [[{'QED': 0}], [{'QED': 0.1}]],
    105: [[{'tPSA': 0}], [{'tPSA': 10}]],
    106: [[{'tPSA': 0}], [{'tPSA': 10}]],
    107: [[{'HBA': 0}], [{'HBA': 1}]],
    108: [[{'HBD': 0}], [{'HBD': 1}]],

    201: [[{'logP': 0, 'HBA': 0}], [{'logP': 0.5, 'HBA': 1}]],
    202: [[{'logP': 0, 'HBA': 0}], [{'logP': 0.5, 'HBA': 1}]],
    203: [[{'logP': 0, 'HBD': 0}], [{'logP': 0.5, 'HBD': 1}]],
    204: [[{'logP': 0, 'HBD': 0}], [{'logP': 0.5, 'HBD': 1}]],
    205: [[{'logP': 0, 'tPSA': 0}], [{'logP': 0.5, 'tPSA': 10}]],
    206: [[{'logP': 0, 'tPSA': 0}], [{'logP': 0.5, 'tPSA': 10}]],
}

task_specification_dict_peptide = {
    301: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*16:01", "HLA-B*44:02"
    ],
    302: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-B*08:01", "HLA-C*03:03"
    ],
    303: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*12:02", "HLA-B*40:01"
    ],
    304: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*11:01", "HLA-B*08:01"
    ],
    305: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*24:02", "HLA-B*08:01"
    ],
    306: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*12:02", "HLA-B*40:02"
    ],

    401: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*29:02", "HLA-B*08:01", "HLA-C*15:02"
    ],
    402: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*03:01", "HLA-B*40:02", "HLA-C*14:02"
    ],
    403: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*14:02", "HLA-B*08:01", "HLA-A*11:01"
    ],
}


def load_thredhold(drug_type):
    if drug_type == 'peptide':
        f_threshold = open("/root/ChatDrug/data/peptide/peptide_editing_threshold.json", 'r')
        threshold_dict = json.load(f_threshold)
        for k, v in threshold_dict.items():
            threshold_dict[k] = v/2
        f_threshold.close()
    else:
        threshold_dict = None
    return threshold_dict


def load_dataset(task_id):
    if task_id < 300:
        # with open("/root/ChatDrug/data/small_molecule/small_molecule_editing.txt") as f:
        #     test_data = f.read().splitlines()
        test_data = []
        with open('/root/code/Data/junction_tree_smiles.txt', 'r') as f:
            test_data = [line.strip() for line in f.readlines()]
    elif (task_id > 300 and task_id < 500):
        if task_id < 400:
            _, source_allele_type, _ = task_specification_dict_peptide[task_id]
        else:
            _, source_allele_type, _, _ = task_specification_dict_peptide[task_id]
        #################################################################################################################################
        f = open("/root/ChatDrug/data/peptide/peptide_editing.json", "r") # path to peptide dataset
        #################################################################################################################################
        data = json.load(f)
        test_data = data[source_allele_type]
    elif task_id > 500:
        #################################################################################################################################
        data_dir = "/root/ChatDrug/data/protein/downstream_datasets"
        chache_dir = "/root/ChatDrug/data/protein/temp_pretrained_ProteinDT"
        #################################################################################################################################
        data_file = os.path.join(data_dir, "secondary_structure", "secondary_structure_cb513.lmdb")
        tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)
        dataset = ProteinSecondaryStructureDataset(data_file, tokenizer)
        test_data = dataset.protein_sequence_list
    else:
        raise NotImplementedError
    return test_data

def examine_complete(state_list, task_id, constraint):
    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint, task_id)
    if task_id < 300:
        for i, s in enumerate(state_list):
            mol = s.mol
            root_mol = s.root_mol
            if not is_valid_smiles(mol):
                continue
            elif (task_id == 101):
                prop_nm = 'logP'
                root_prop = cal_logP(root_mol)
                prop = cal_logP(mol)
                return (prop + threshold[prop_nm] < root_prop)
            elif (task_id == 102):
                prop_nm = 'logP'
                root_prop = cal_logP(root_mol)
                prop = cal_logP(mol)
                return (prop > root_prop + threshold[prop_nm])
            elif (task_id == 103):
                prop_nm = 'QED'
                root_prop = cal_QED(root_mol)
                prop = cal_QED(mol)
                return (prop > root_prop + threshold[prop_nm])
            elif (task_id == 104):
                prop_nm = 'QED'
                root_prop = cal_QED(root_mol)
                prop = cal_QED(mol)
                return (prop + threshold[prop_nm] < root_prop)
            elif (task_id == 105):
                prop_nm = 'tPSA'
                root_prop = cal_tPSA(root_mol)
                prop = cal_tPSA(mol)
                return (prop + threshold[prop_nm] < root_prop)
            elif (task_id == 106):
                prop_nm = 'tPSA'
                root_prop = cal_tPSA(root_mol)
                prop = cal_tPSA(mol)
                return (prop > root_prop + threshold[prop_nm])
            elif (task_id == 107):
                prop_nm = 'HBA'
                root_prop = cal_HBA(root_mol)
                prop = cal_HBA(mol)
                return (prop > root_prop + threshold[prop_nm])
            elif (task_id == 108):
                prop_nm = 'HBD'
                root_prop = cal_HBD(root_mol)
                prop = cal_HBD(mol)
                return (prop > root_prop + threshold[prop_nm])
            elif (task_id == 201) and (examine_complete(state_list, 101, constraint) and (examine_complete(state_list, 107, constraint))):
                return True
            elif (task_id == 202) and (examine_complete(state_list, 102, constraint) and (examine_complete(state_list, 107, constraint))):
                return True
            elif (task_id == 203) and (examine_complete(state_list, 101, constraint) and (examine_complete(state_list, 108, constraint))):
                return True
            elif (task_id == 204) and (examine_complete(state_list, 102, constraint) and (examine_complete(state_list, 108, constraint))):
                return True
            elif (task_id == 205) and (examine_complete(state_list, 101, constraint) and (examine_complete(state_list, 105, constraint))):
                return True
            elif (task_id == 206) and (examine_complete(state_list, 101, constraint) and (examine_complete(state_list, 106, constraint))):
                return True
            else:
                continue
    elif (task_id > 300 and task_id < 500):
        is_complete = False
        for i, s in enumerate(state_list):
            drug = s.mol
            root_drug = s.root_mol
            for prop_nm in prop_name:
                if "similarity" in prop_nm:
                    continue
                target_allele_type = prop_nm[-11:]
                root_binding_affinity = cal_binding_affinity(root_drug, target_allele_type)
                new_binding_affinity = cal_binding_affinity(drug, target_allele_type)
                if not np.logical_and((new_binding_affinity > root_binding_affinity + EPS), (new_binding_affinity > threshold[prop_nm])):
                    return False
            is_complete = True
        return is_complete
                
    elif (task_id > 500):
        for i, s in enumerate(state_list):
            drug = s.mol
            root_drug = s.root_mol
            root_secondary_structures = cal_secondary_structures(drug, task_id)
            new_secondary_structures = cal_secondary_structures(drug, task_id)
            if (new_secondary_structures > root_secondary_structures):
                return True
    return False

def examine_complete_fast_protein(state_list, fast_dict, task_id):
    for i, s in enumerate(state_list):
        drug = s.mol
        root_drug = s.root_mol
        root_count = fast_dict[root_drug]
        new_secondary_structures = cal_secondary_structures(drug, task_id)
        if (new_secondary_structures > root_count):
            return True
    return False
    
def get_track(tree, depth, reward_fn):
    prop_track = [None] * (depth+1)
    for i, layer_nodes in enumerate(tree.nodes):
        reward = 0
        prop = None
        sim = None
        for _, s in enumerate(layer_nodes):
            mol = s.mol
            if s.drug_type == "small_molecule":
                if not is_valid_smiles(mol):
                    continue
            mol_reward = reward_fn([s])[0]
            if mol_reward >= reward:
                reward = mol_reward
                prop = s.prop
        prop_track[i] = prop
        
    def fill_none_with_previous(track):
        previous_value = None
        for i in range(len(track)):
            if track[i] is None:
                if previous_value is not None:
                    track[i] = previous_value
            else:
                previous_value = track[i]
    
    if prop_track[0] is None:
        prop_track[0] = {}
    fill_none_with_previous(prop_track)
    
    return prop_track
    

def get_llm_function(llm_name):
    assert isinstance(llm_name, str)
    if llm_name == "deepseek":
        llm_function = run_deepseek_prompts
    elif llm_name == "chatgpt":
        llm_function = run_openai_prompts
    elif llm_name == "llama":
        llm_function = partial(run_llama_prompts, model_id=snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir='/root/autodl-tmp/Llama-3.1-8B-Instruct-modelscope'))
    elif llm_name == "galactica":
        llm_function = run_gala_prompts
    elif llm_name == "dfm":
        llm_function = run_dfm_prompts
    else:
        raise ValueError(f"Unknown LLM {llm_name}.")
    return llm_function

# return prop_name, opt_direction, threshold
def get_task_info(constraint, task_id):
    drug_type = None
    prop_name = None
    opt_direction = None
    task_objective = None
    threshold = None
    if task_id < 300:
        drug_type = "small_molecule"
    elif (task_id > 300 and task_id < 500):
        drug_type = "peptide"
    elif task_id > 500:
        drug_type = "protein"
    if constraint == 'loose':
        threshold_idx = 0
    elif constraint == 'strict':
        threshold_idx = 1
    if task_id < 300:
        threshold = task2threshold_list[task_id][threshold_idx][0]
    elif (task_id > 300 and task_id < 500):
        threshold_dict = load_thredhold(drug_type)
    if task_id == 101:
        prop_name = ["logP", "tanimoto_similarity"]
        opt_direction = {"logP": "decrease", "tanimoto_similarity": "increase"}
        task_objective = "more soluble in water"
    elif task_id == 102:
        prop_name = ["logP", "tanimoto_similarity"]
        opt_direction = {"logP": "increase", "tanimoto_similarity": "increase"}
        task_objective = "less soluble in water"
    elif task_id == 103:
        prop_name = ["QED", "tanimoto_similarity"] 
        opt_direction = {"QED": "increase", "tanimoto_similarity": "increase"}
        task_objective = "more like a drug"
    elif task_id == 104:
        prop_name = ["QED", "tanimoto_similarity"]
        opt_direction = {"QED": "decrease", "tanimoto_similarity": "increase"}
        task_objective = "less like a drug"
    elif task_id == 105:
        prop_name = ["tPSA", "tanimoto_similarity"]
        opt_direction = {"tPSA": "decrease", "tanimoto_similarity": "increase"}
        task_objective = "higher permeability"
    elif task_id == 106:
        prop_name = ["tPSA", "tanimoto_similarity"]
        opt_direction = {"tPSA": "increase", "tanimoto_similarity": "increase"}
        task_objective = "lower permeability"
    elif task_id == 107:
        prop_name = ["HBA", "tanimoto_similarity"]
        opt_direction = {"HBA": "increase", "tanimoto_similarity": "increase"}
        task_objective = "with more hydrogen bond acceptors"
    elif task_id == 108:
        prop_name = ["HBD", "tanimoto_similarity"]
        opt_direction = {"HBD": "increase", "tanimoto_similarity": "increase"}
        task_objective = "with more hydrogen bond donors"
    elif task_id == 201:
        prop_name = ["logP", "HBA", "tanimoto_similarity"]
        opt_direction = {"logP": "decrease", "HBA": "increase", "tanimoto_similarity": "increase"}
        task_objective = "more soluble in water and more hydrogen bond acceptors"
    elif task_id == 202:
        prop_name = ["logP", "HBA", "tanimoto_similarity"]
        opt_direction = {"logP": "increase", "HBA": "increase", "tanimoto_similarity": "increase"}
        task_objective = "less soluble in water and more hydrogen bond acceptors"
    elif task_id == 203:
        prop_name = ["logP", "HBD", "tanimoto_similarity"]
        opt_direction = {"logP": "decrease", "HBD": "increase", "tanimoto_similarity": "increase"}
        task_objective = "more soluble in water and more hydrogen bond donors"
    elif task_id == 204:
        prop_name = ["logP", "HBD", "tanimoto_similarity"]
        opt_direction = {"logP": "increase", "HBD": "increase", "tanimoto_similarity": "increase"}
        task_objective = "less soluble in water and more hydrogen bond donors"
    elif task_id == 205:
        prop_name = ["logP", "tPSA", "tanimoto_similarity"]
        opt_direction = {"logP": "decrease", "tPSA": "decrease", "tanimoto_similarity": "increase"}
        task_objective = "more soluble in water and higher permeability"
    elif task_id == 206:
        prop_name = ["logP", "tPSA", "tanimoto_similarity"]
        opt_direction = {"logP": "decrease", "tPSA": "increase", "tanimoto_similarity": "increase"}
        task_objective = "more soluble in water and lower permeability"
    elif task_id == 301:
        prop_name = ["binding_affinity_HLA-B*44:02", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*44:02": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*44:02"
    elif task_id == 302:
        prop_name = ["binding_affinity_HLA-C*03:03", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-C*03:03": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-C*03:03"
    elif task_id == 303:
        prop_name = ["binding_affinity_HLA-B*40:01", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*40:01": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*40:01"
    elif task_id == 304:
        prop_name = ["binding_affinity_HLA-B*08:01", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*08:01": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*08:01"
    elif task_id == 305:
        prop_name = ["binding_affinity_HLA-B*08:01", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*08:01": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*08:01"
    elif task_id == 306:
        prop_name = ["binding_affinity_HLA-B*40:02", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*40:02": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*40:02"
    elif task_id == 401:
        prop_name = ["binding_affinity_HLA-B*08:01", "binding_affinity_HLA-C*15:02", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*08:01": "increase", "binding_affinity_HLA-C*15:02": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*08:01 and HLA-C*15:02"
    elif task_id == 402:
        prop_name = ["binding_affinity_HLA-B*40:02", "binding_affinity_HLA-C*14:02", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*40:02": "increase", "binding_affinity_HLA-C*14:02": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*40:02 and HLA-C*14:02"
    elif task_id == 403:
        prop_name = ["binding_affinity_HLA-B*08:01", "binding_affinity_HLA-A*11:01", "levenshtein_similarity"]
        opt_direction = {"binding_affinity_HLA-B*08:01": "increase", "binding_affinity_HLA-A*11:01": "increase", "levenshtein_similarity": "increase"}
        threshold = {}
        for prop_nm in prop_name:
            if "similarity" in prop_nm:
                continue
            threshold[prop_nm] = threshold_dict[prop_nm[-11:]]
        task_objective = "binds to HLA-B*08:01 and HLA-A*11:01"
    elif task_id == 501:
        prop_name = ["secondary_structures_501", "levenshtein_similarity"]
        opt_direction = {"secondary_structures_501": "increase", "levenshtein_similarity": "increase"}
        task_objective = "making more amino acids into the helix structure (secondary structure)"
        threshold = {}
    elif task_id == 502:
        prop_name = ["secondary_structures_502", "levenshtein_similarity"]
        opt_direction = {"secondary_structures_502": "increase", "levenshtein_similarity": "increase"}
        task_objective = "making more amino acids into the strand structure (secondary structure)"
        threshold = {}
        
    return drug_type, prop_name, opt_direction, task_objective, threshold


def get_prop_function():
    prop_function = {}
    prop_function["tanimoto_similarity"] = calculate_tanimoto_similarity
    prop_function["levenshtein_similarity"] = sim_levenshtein
    prop_function["logP"] = cal_logP
    prop_function["QED"] = cal_QED
    prop_function["tPSA"] = cal_tPSA
    prop_function["HBA"] = cal_HBA
    prop_function["HBD"] = cal_HBD
    prop_function["binding_affinity_HLA-B*44:02"] = partial(cal_binding_affinity, target_allele_type="HLA-B*44:02")
    prop_function["binding_affinity_HLA-C*03:03"] = partial(cal_binding_affinity, target_allele_type="HLA-C*03:03")
    prop_function["binding_affinity_HLA-B*40:01"] = partial(cal_binding_affinity, target_allele_type="HLA-B*40:01")
    prop_function["binding_affinity_HLA-B*08:01"] = partial(cal_binding_affinity, target_allele_type="HLA-B*08:01")
    prop_function["binding_affinity_HLA-B*40:02"] = partial(cal_binding_affinity, target_allele_type="HLA-B*40:02")
    prop_function["binding_affinity_HLA-C*15:02"] = partial(cal_binding_affinity, target_allele_type="HLA-C*15:02")
    prop_function["binding_affinity_HLA-C*14:02"] = partial(cal_binding_affinity, target_allele_type="HLA-C*14:02")
    prop_function["binding_affinity_HLA-A*11:01"] = partial(cal_binding_affinity, target_allele_type="HLA-A*11:01")
    
    prop_function["secondary_structures_501"] = partial(cal_secondary_structures, task_id=501)
    prop_function["secondary_structures_502"] = partial(cal_secondary_structures, task_id=502)
    
    return prop_function


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f"""f'''{fstring_text}'''""", vals)
    return ret_val

def is_valid_smiles(smiles):
    """
    Check if the SMILES valid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        logging.error(f"Error: {e}")
        return False
    
# def cal_logP(smiles):
#     assert is_valid_smiles(smiles)
#     mol = Chem.MolFromSmiles(smiles)
#     logP = prop2func["MolLogP"](mol)
#     # logP = Descriptors.MolLogP(mol)
#     return logP

def cal_logP(smiles):
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)
    return score

def cal_QED(smiles):
    assert is_valid_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    qed = prop2func["qed"](mol)
    # qed = QED.qed(mol)
    return qed

def cal_tPSA(smiles):
    assert is_valid_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    tpsa = prop2func["TPSA"](mol)
    # tpsa = Descriptors.TPSA(mol)
    return tpsa

def cal_HBA(smiles):
    assert is_valid_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    hba_count = prop2func["NumHAcceptors"](mol)
    # hba_count = rdMolDescriptors.CalcNumHBA(mol)
    return hba_count

def cal_HBD(smiles):
    assert is_valid_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    hbd_count = prop2func["NumHDonors"](mol)
    hbd_count = rdMolDescriptors.CalcNumHBD(mol)
    return hbd_count
    

def calculate_tanimoto_similarity(smiles1, smiles2):
    """
    Calculate the tanimoto similarity between two SMILES strings.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        logging.error("Invalid SMILES string(s).")
        raise ValueError("Invalid SMILES string(s).")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def sim_levenshtein(seq1, seq2):
    sim = Levenshtein.ratio(seq1, seq2)
    return sim

def cal_binding_affinity(drug, target_allele_type):
    df = MHC_peptide_predictor.predict(peptides=[drug], alleles=[target_allele_type], verbose=False)
    value = df["presentation_score"].to_list()
    value = np.array(value)
    return value[0]

@torch.no_grad()
def cal_secondary_structures(drug, task_id, device="cuda"):
    protein_list = [drug]
    from torch.utils.data import DataLoader

    batch_size = 1
    protein_dataset = ProteinListDataset(protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    protein_dataloader = DataLoader(protein_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=protein_dataset.collate_fn)

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    def get_target_label_count_list(dataloader, target_label):
        count_list = []
        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            logits = output.logits  # [B, seq_length, 3]
            predicted_labels = F.softmax(logits, dim=-1)  # [B, seq_length, 3]
            predicted_labels = predicted_labels.argmax(dim=-1)  # [B, seq_length]

            temp_count_list = ((predicted_labels == target_label) * attention_mask)
            temp_count_list = temp_count_list.sum(dim=1)  # [B]
            count_list.append(temp_count_list.detach().cpu().numpy())
        
        count_list = np.concatenate(count_list)
        return count_list

    count_list = get_target_label_count_list(protein_dataloader, target_label)

    return count_list[0]


@torch.no_grad()
def evaluate_fast_protein_dict(input_protein_list, task_id, device="cuda"):
    from torch.utils.data import DataLoader

    batch_size = 128
    input_dataset = ProteinListDataset(input_protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    input_dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=input_dataset.collate_fn)

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    def get_target_label_count_list(dataloader, target_label):
        count_list = []
        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            logits = output.logits  # [B, seq_length, 3]
            predicted_labels = F.softmax(logits, dim=-1)  # [B, seq_length, 3]
            predicted_labels = predicted_labels.argmax(dim=-1)  # [B, seq_length]

            temp_count_list = ((predicted_labels == target_label) * attention_mask)
            temp_count_list = temp_count_list.sum(dim=1)  # [B]
            count_list.append(temp_count_list.detach().cpu().numpy())
        
        count_list = np.concatenate(count_list)
        print("count_list", count_list.shape)
        return count_list

    input_count_list = get_target_label_count_list(input_dataloader, target_label)
    return input_count_list
    
def get_fast_protein_dict(task_id, input_drug_list, saved_file):
    fast_protein_list = []
    for input_drug in input_drug_list:
        fast_protein_list.append(input_drug)
    fast_protein_count_list = evaluate_fast_protein_dict(fast_protein_list, task_id)
    fast_protein_example_dict = dict(zip(fast_protein_list, fast_protein_count_list))
    np.save(save_file + '/fast_protein_dict_' + str(task_id) + '.npy', fast_protein_example_dict)
    return fast_protein_example_dict

def parse_answer(answer: str, num_answers=None):
    """parse an answer to a list of molecules"""
    try:
        final_answer_location = answer.lower().find("final_answer")
        if final_answer_location == -1:
            final_answer_location = answer.lower().find("final answer")
        if final_answer_location == -1:
            final_answer_location = answer.lower().find("final")
        if final_answer_location == -1:
            final_answer_location = 0
            
        list_start = answer.find("[", final_answer_location)
        list_end = answer.find("]", list_start)
        substring = answer[list_start+1:]
        if '[' in substring:
            num = substring.count('[')
            list_start_ = list_start
            for _ in range(num):
                list_start_ = answer.find("[", list_start_+1)
                list_end = answer.find("]", list_end+1)
                substring = answer[list_start_+1:]
        try:
            answer_list = literal_eval(answer[list_start : list_end + 1])
        except Exception:
            answer_list = answer[list_start + 1 : list_end]
            answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
        return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]
    except:
        return []
    

def parse_molecule(response):
    pattern = re.compile(r'[0-9BCOHNSOPrIFlanocs@+\.\-\[\]\(\)\\\/%=#$]{10,}')
    output_sequence_list = pattern.findall(response)
    return output_sequence_list

# def parse_peptide(input_peptide, raw_text, retrieval_sequence):
#     pattern = re.compile('[A-Z]{5,}')
#     output_peptide_list = pattern.findall(raw_text)
#     while input_peptide in output_peptide_list:
#         output_peptide_list.remove(input_peptide)

#     if retrieval_sequence!=None:
#         while retrieval_sequence in output_peptide_list:
#             output_peptide_list.remove(retrieval_sequence)

#     if len(output_peptide_list) > 0:
#         output_peptide = output_peptide_list[0]
#         if len(output_peptide) < 16 and "X" not in output_peptide: 
#             output_peptide = [output_peptide]
#         else: 
#             output_peptide = None
#     else:
#         output_peptide=[]
#     return output_peptide

# def parse_protein(input_protein, raw_text, retrieval_sequence):
#     pattern = re.compile('[A-Z]{5,}')
#     output_protein_list = pattern.findall(raw_text)
#     while input_protein in output_protein_list:
#         output_protein_list.remove(input_protein)

#     if retrieval_sequence!=None:
#         while retrieval_sequence in output_protein_list:
#             output_protein_list.remove(retrieval_sequence)

#     if len(output_protein_list) > 0:
#         output_protein = output_protein_list[0][:1024]
#         return [output_protein]
#     else:
#         return []


def parse_peptide(response):
    pattern = re.compile('[A-Z]{5,}')
    output_peptide_list = pattern.findall(response)

    new_output_peptide_list = []
    for output_peptide in output_peptide_list:
        if len(output_peptide) < 16 and "X" not in output_peptide:
            new_output_peptide_list.append(output_peptide)
    output_peptide_list = new_output_peptide_list
    return output_peptide_list

def parse_protein(response):
    pattern = re.compile('[A-Z]{5,}')
    output_protein_list = pattern.findall(response)
    new_output_protein_list = []
    for output_protein in output_protein_list:
        new_output_protein_list.append(output_protein)
    output_protein_list = new_output_protein_list
    return output_protein_list
