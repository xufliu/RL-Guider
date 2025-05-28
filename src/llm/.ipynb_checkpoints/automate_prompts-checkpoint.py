"""Functions for automate prompts"""
import sys
sys.path.append("src")
# from search.state.reasoner_state import ReasonerState, ReasonerState_
from search.state.molreasoner_state import ReasonerState


def get_template(query, chain_of_thought):
    """Get the template for the given query"""
    template = query
    assert chain_of_thought is True
    template += (
            " {include_statement}{exclude_statement}"
            "Provide scientific explanations for each of the drug molecule. "
            "Finally, return a python list named final_answer which contains the top-5 drug molecule. "
            "{candidate_list_statement}"
            r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
        )
    return template


# def get_initial_state_mol(
#     query,
#     prediction_model,
#     reward_model,
#     simulation_reward=False,
#     chain_of_thought=True,
# ):
#     """Get initial state for LLM query"""
#     molecule_name = query['smiles']
#     query_ = query['prompt'].replace("###", query['smiles']).replace("$$$", query['property'])
#     template = get_template(query=query_, chain_of_thought=chain_of_thought)
#     starting_state = ReasonerState(
#         template=template
#     )
#     return starting_state

def get_initial_state(
    task_id,
    drug_type,
    prompt,
    prop_name,
    opt_direction,
    task_objective,
    threshold,
    mol,
    prop,
    root_mol,
    root_prop,
    conversation_type,
    conversational_LLM,
    cot,
    root,
):
    """Get initial state for small molecule editing"""
    starting_state = ReasonerState(
        task_id=task_id,
        drug_type=drug_type,
        template=prompt,
        prop_name=prop_name,
        opt_direction=opt_direction,
        task_objective=task_objective,
        threshold=threshold,
        mol=mol,
        prop=prop,
        root_mol=root_mol,
        root_prop=root_prop,
        conversation_type=conversation_type,
        conversational_LLM=conversational_LLM,
        cot=cot,
        root=root,
    )
    return starting_state


# def get_initial_state_pep(
#     prompt,
    
# ):
#     """Get initial state for peptide editing"""
#     starting_state = pepReasonerState(
#         template=prompt,
#         prop_name=prop_name,
#         opt_direction=opt_direction,
#         task_objective=task_objective,
#         threshold=threshold,
#         mol=mol,
#         prop=prop,
#         root_mol=root_mol,
#         root_prop=root_prop,
#         root_sim=root_sim,
#         conversation_type=conversation_type,
#         conversational_LLM=conversational_LLM,
#         cot=cot,
#         root=root,
#     )
#     return starting_state


# def get_initial_state_pro(
#     prompt,
    
# ):
#     """Get initial state for protein editing"""
#     starting_state = proReasonerState(
#         template=prompt,
#         prop_name=prop_name,
#         opt_direction=opt_direction,
#         task_objective=task_objective,
#         threshold=threshold,
#         mol=mol,
#         prop=prop,
#         root_mol=root_mol,
#         root_prop=root_prop,
#         root_sim=root_sim,
#         conversation_type=conversation_type,
#         conversational_LLM=conversational_LLM,
#         cot=cot,
#         root=root,
#     )
#     return starting_state



