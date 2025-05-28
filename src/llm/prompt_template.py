def get_generation_prompt_template(task_id, conversation_type, conversational_LLM):
    if task_id < 300:
        if conversational_LLM == 'galactica':
            if conversation_type == 'single':
                prompt = [generation_mol_prompt_single_gala, generation_mol_prompt_single_gala]
            elif conversation_type == 'multi':
                prompt = [generation_mol_prompt_multi_round1_gala, generation_mol_prompt_multi_roundx_gala]
        else:
            if conversation_type == 'single':
                prompt = [generation_mol_prompt_single, generation_mol_prompt_single]
            elif conversation_type == 'multi':
                prompt = [generation_mol_prompt_multi_round1, generation_mol_prompt_multi_roundx]
    elif (task_id > 300 and task_id < 500):
        if conversational_LLM == 'galactica':
            if conversation_type == 'single':
                prompt = [generation_pep_prompt_single_gala, generation_pep_prompt_single_gala]
            elif conversation_type == 'multi':
                prompt = [generation_pep_prompt_multi_round1_gala, generation_pep_prompt_multi_roundx_gala]
        else:
            if conversation_type == 'single':
                prompt = [generation_pep_prompt_single, generation_pep_prompt_single]
            elif conversation_type == 'multi':
                prompt = [generation_pep_prompt_multi_round1, generation_pep_prompt_multi_roundx]
    elif (task_id > 500):
        if conversational_LLM == 'galactica':
            if conversation_type == 'single':
                prompt = [generation_pro_prompt_single_gala, generation_pro_prompt_single_gala]
            elif conversation_type == 'multi':
                prompt = [generation_pro_prompt_multi_round1_gala, generation_pro_prompt_multi_roundx_gala]
        else:
            if conversation_type == 'single':
                prompt = [generation_pro_prompt_single, generation_pro_prompt_single]
            elif conversation_type == 'multi':
                prompt = [generation_pro_prompt_multi_round1, generation_pro_prompt_multi_roundx]
    return prompt


system_prompt = (
    "You are a helpful chemistry expert with extensive knowledge of drug design. "
)

#### small molecule ####

generation_mol_prompt_multi_round1 = (
    "Can you make molecule {root_mol} {task_objective}? "
    "The output molecule should be similar to the input molecule. "
    "{suggestion} "
    "Give me five molecules in SMILES only and list them using bullet points. "
    "{reasoning_instruction} "
)

generation_mol_prompt_multi_round1_gala = (
    "Question: Can you make molecule [START_I_SMILES]{root_mol}[END_I_SMILES] {task_objective}? The output molecule should be similar to the input molecule.\n"
)

generation_mol_prompt_multi_roundx = (
    "Your provided sequence {prev_wrong_mol} could not achieve goal. "
    "You are suggested to edit the molecule according to the suggestion: {suggestion}. "
    "Can you give me new molecules?"
)

generation_mol_prompt_multi_roundx_gala = (
    "Question: Your provided sequence [START_I_SMILES]{prev_wrong_mol}[END_I_SMILES] could not achieve goal. "
    "You are suggested to edit the molecule according to the suggestion: {suggestion}. "
    "Can you give me new molecules?\n"
)

# generation_mol_prompt_single = (
#     "Can you make root_molecule = {root_mol} {task_objective} and {threshold_specific_prompt}? "
#     "The output molecules should be similar to the input molecule. "
#     "You are suggested to edit root_molecule according to the suggestion: {suggestion} "
#     "Give me five new molecules different from root_molecule in SMILES only and list them using bullet points. "
#     "{reasoning_instruction} "
# )

generation_mol_prompt_single = (
    "Can you make root_molecule = {root_mol} {task_objective}? "
    "The output molecules should be similar to the input molecule. "
    "You are suggested to edit root_molecule according to the suggestion: {suggestion} "
    "Give me five new molecules different from root_molecule in SMILES only and list them using bullet points. "
    "{reasoning_instruction} "
)

generation_mol_prompt_single_gala = (
    "Question: Can you make molecule [START_I_SMILES]{root_mol}[END_I_SMILES] {task_objective} and {threshold_specific_prompt}? You are suggested to edit the molecule according to the suggestion: {suggestion}. The output molecule should be similar to the input molecule.\n"
)

#### peptide ####

generation_pep_prompt_multi_round1 = (
    "We want a peptide that {task_objective}. "
    "We have a peptide {root_mol} that binds to {source_allele_type}, can you help modify it? "
    "The output peptide should be similar to input peptide. "
    "{suggestion} "
    "Please provide the possible modified peptide sequence only. "
    "{reasoning_instruction} "
)

generation_pep_prompt_multi_round1_gala = (
    "Question: We want a peptide that {task_objective}. We have a peptide {root_mol} that binds to {source_allele_type}, can you help modify it? The output peptWeide should be similar to input peptide.\n"
)

generation_pep_prompt_multi_roundx = (
    "Your provided sequence {prev_wrong_mol} is not correct. "
    "{suggestion} "
    "Can you give me a new peptide?"
)

generation_pep_prompt_multi_roundx_gala = (
    "Question: Your provided sequence {prev_wrong_mol} is not correct. "
    "You are suggested to modify the peptide according to the suggestion: {suggestion}. "
    "Can you give me a new peptide?\n"
)

generation_pep_prompt_single = (
    "We want a peptide that {task_objective}. "
    "We have a peptide {root_mol} that binds to {source_allele_type}, can you help modify it? "
    "The output peptide should be similar to input peptide. "
    "You are suggested to modify the peptide according to the suggestion: {suggestion}. "
    "Please provide the possible modified peptide sequence only. "
    "{reasoning_instruction} "
)

generation_pep_prompt_single_gala = (
    "Question: We want a peptide that {task_objective}. We have a peptide {root_mol} that binds to {source_allele_type}, can you help modify it? You are suggested to modify the peptide according to the suggestion: {suggestion}. The output peptide should be similar to input peptide.\n"
)

#### protein ####

generation_pro_prompt_multi_round1 = (
    "We have a protein {root_mol}. "
    "Can you update modify it by {task_objective}? "
    "The input and output protein sequences should be similar but different. "
    "{suggestion} "
    "{reasoning_instruction} "
)

generation_pro_prompt_multi_round1_gala = (
    "Question: We have a protein [START_AMINO]{root_mol}[END_AMINO]. Can you update modify it by {task_objective}? The input and output protein sequences should be similar but different.\n"
)

generation_pro_prompt_multi_roundx = (
    "Your provided sequence {prev_wrong_mol} is not correct. "
    "You are suggested to modify the protein according to the suggestion: {suggestion}. "
    "Can you give me a new protein?"
)

generation_pro_prompt_multi_roundx_gala = (
    "Question: Your provided sequence [START_AMINO]{prev_wrong_mol}[END_AMINO] is not correct. "
    "You are suggested to modify the protein according to the suggestion: {suggestion}. "
    "Can you give me a new protein?\n"
)

generation_pro_prompt_single = (
    "We have a protein {root_mol}. "
    "Can you update modify it by {task_objective}? "
    "The input and output protein sequences should be similar but different. "
    "You are suggested to modify the protein according to the suggestion: {suggestion}. "
    "{reasoning_instruction} "
)

generation_pro_prompt_single_gala = (
    "Question: We have a protein [START_AMINO]{root_mol}[END_AMINO]. Can you update modify it by {task_objective}? You are suggested to modify the protein according to the suggestion: {suggestion}. The input and output protein sequences should be similar but different.\n"
)


#### planner ####

priors_template = """
    $root_question: {root_prompt}

    $root_property: {root_property}

    $threshold: {threshold}

    {previous_prompt_answer}
    Consider the {current_conditions}. Your task is to suggest possible actions that could achieve the intent of the $root_question.

    $search_state: current_best_mol in the message

    $action_space: Add, delete, replace an atom or functional group.

    {guidelines}

    {final_task}
"""

prev_prompt_template = (
    "Start the modification from molecule {prev_mol}."
)




