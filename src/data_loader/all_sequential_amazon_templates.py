'''
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
Zeroshot Tasks -- 1 Prompt Family (Z)
'''

all_tasks = {}

# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

'''
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be purchased by the user?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following purchase history of user_{} : \n {} \n predict next possible item to be purchased by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}
'''
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?
 
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the purchase history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me decide ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}
'''
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user
 
Target template:
{{item [item_id]}}
 
 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the purchase history list of user_{} : \n {} \n try to recommend next item to the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template




template = {}
'''

Input template:
The {{user_id}} has recently purchased the following items: {{history item list of {{item_id}}}}
What item should come next in the sequence?

Target template:
{{item [item_id]}}

 
Metrics:
HR, NDCG, MRR
'''
template['source'] = "The user_{} has recently purchased the following items: \n {} \n What item should come next in the sequence?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-14"

task_subgroup_2["2-14"] = template



template = {}
'''

Input template:
Based on the interaction history of the {{user_id}}: {{history item list of {{item_id}}}}
Which item is most likely to be bought next?

Target template:
{{item [item_id]}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the interaction history of user_{}: \n {} \n Which item is most likely to be bought next?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-15"

task_subgroup_2["2-15"] = template


template = {}
'''

Input template:
From this list of past purchases by the {{user_id}}: {{history item list of {{item_id}}}}
Predict the next likely purchase.

Target template:
{{item [item_id]}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "From this list of past purchases by user_{}: \n {} \n Predict the next likely purchase."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-16"

task_subgroup_2["2-16"] = template


template = {}
'''

Input template:
Here’s what the {{user_id}} has bought so far: {{history item list of {{item_id}}}}
What’s the most logical next item?

Target template:
{{item [item_id]}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here’s what user_{} has bought so far: \n {} \n What’s the most logical next item?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-17"

task_subgroup_2["2-17"] = template


template = {}
'''

Input template:
Considering the {{user_id}}’s purchase history: {{history item list of {{item_id}}}}
Please suggest the next item they might be interested in.

Target template:
{{item [item_id]}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Considering user_{}’s purchase history: \n {} \n Please suggest the next item they might be interested in."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-18"

task_subgroup_2["2-18"] = template


template = {}
'''

Input template:
Following this sequence of purchases from the {{user_id}}: {{history item list of {{item_id}}}}
What would be your recommendation for the next item?

Target template:
{{item [item_id]}}

Metrics:
HR, NDCG, MRR
'''
template['source'] = "Following this sequence of purchases from user_{}: \n {} \n What would be your recommendation for the next item?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'purchase_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-19"

task_subgroup_2["2-19"] = template

