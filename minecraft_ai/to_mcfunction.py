import os
import pickle
from collections import OrderedDict

import numpy as np
import six
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from minecraft_ai.mc_types import Action


def fit_tree(states, actions):
    """
    fits a random forest model to the action outputs of a policy
    """

    # rf = RandomForestClassifier(n_estimators=100)
    # #rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    #
    # rf.fit(states, actions)
    #
    # action_probs = rf.predict_proba(states)

    # Train small tree to match RF
    tr = DecisionTreeClassifier()
    tr.fit(states, actions)

    #acc = accuracy_score(actions, rf.predict(states))
    acc2 = accuracy_score(actions, tr.predict(states))

    print(acc2)
    return tr

def build_inference_module(
    module_name: str,
    entity_name: str = "policy"
):
    path = f"data/datapack_modules/{module_name}/{entity_name}/inference"

    os.makedirs(path, exist_ok=True)
    with open(f"data/models/{module_name}/{entity_name}.pkl", "rb") as f:
        tree_dict = pickle.load(f)

    def build_node(node):
        if node["is_leaf"]:
            print(node.keys())
            body = f"""
scoreboard players set @s action {Action[node['prediction']].value}
"""

        else:
            body = f"""
# prepare scores for comparison
execute store result score #tmp action run data get storage ai {node['feature']} 100000
scoreboard players set #threshdold action {round(node['threshold'] * 100000)}

# branch decisions
# if
execute if score #tmp action {node['op']} #threshold action run return run function ai:ai_modules/{module_name}/{entity_name}/inference/node_{node['left']['id']}
# else
function ai:ai_modules/{module_name}/{entity_name}/inference/node_{node['right']['id']}
"""

        # create file
        with open(f"{path}/node_{node['id']}.mcfunction", "w") as f:
            f.write(body)

        if node["is_leaf"]:
            return

        build_node(node["left"])
        build_node(node["right"])

    build_node(tree_dict)


def tree_to_dict(dt_model, target_names, feature_names=None):
    tree_ = dt_model.tree_
    tree_dict = OrderedDict()

    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"
        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]
        json_value = ', '.join([str(x) for x in value])
        if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            is_leaf = True
            #             return {"id": int(node_id), "is_leaf": is_leaf, "criterion": criterion, "impurity": float(tree.impurity[node_id]), "samples": float(tree.n_node_samples[node_id]), "prediction": str(target_names[np.argmax(tree_.value[node_id][0])]), "samplesDistribution": json_value}
            return {"id": int(node_id), "is_leaf": is_leaf, "prediction": str(target_names[np.argmax(tree_.value[node_id][0])])}
        else:
            is_leaf = False
            if feature_names is not None:
                feature = str(feature_names[tree.feature[node_id]])
            else:
                feature = tree.feature[node_id]
        if "=" in feature:
            rule_type = "="
            rule_value = "false"
        else:
            rule_type = "<="
            rule_value = tree.threshold[node_id]

        return {"id": int(node_id), "is_leaf": is_leaf, "feature": feature, "op": rule_type, "threshold": rule_value}

    def add_node(tree, node_id, criterion, parent=None, depth=0):
        tree_dict = OrderedDict()
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        tree_dict.update(node_to_str(tree, node_id, criterion))
        if left_child != sklearn.tree._tree.TREE_LEAF:
            tree_dict.update({"left": add_node(tree, left_child, criterion=criterion, parent=node_id, depth=depth + 1)})
            tree_dict.update({"right": add_node(tree, right_child, criterion=criterion, parent=node_id, depth=depth + 1)})
        return tree_dict

    if isinstance(dt_model, sklearn.tree._tree.Tree):
        tree_dict.update(add_node(dt_model, 0, criterion="gini"))
    else:
        tree_dict.update(add_node(dt_model.tree_, 0, criterion=dt_model.criterion))

    return tree_dict


if __name__ == '__main__':
    module_name = "test"
    entity_name = "test"

    states, actions = make_classification(n_samples=1000, n_features=16,
                           n_informative=8, n_redundant=0,
                           n_classes=len(Action),
                           random_state=0, shuffle=False)

    tr = fit_tree(states, actions)
    fn = list("abcdefghijklmnop")
    cn = [act.name for act in Action]

    d = tree_to_dict(tr, cn, fn)
    print(d)

    path = f"data/models/{module_name}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{entity_name}.pkl", "wb") as f:
        pickle.dump(d, f)

    build_inference_module(module_name, entity_name)






    print(d)

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    # tree.plot_tree(tr,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig.savefig('rf_individualtree.png')
