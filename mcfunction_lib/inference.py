import os
import pickle
from collections import OrderedDict
from typing import NamedTuple, List

import numpy as np
import six
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class PolicyDataset(NamedTuple):
    size: int
    aid: str
    num_actions: int
    actions: List[np.ndarray] = []
    observations: List[np.ndarray] = []
    observation_paths: List[str] = []

    def current_size(self):
        num_obs = sum(obs.shape[0] for obs in self.observations)
        num_act = sum(a.shape[0] for a in self.actions)
        assert num_obs == num_act, "Malformed dataset"
        return num_obs

    def is_full(self):
        return self.size == self.current_size()

    def fill_proportion(self):
        return self.current_size() / self.size

    def put(
            self,
            actions,
            observations,
            observation_paths,
    ):
        if self.is_full():
            return

        if len(self.observation_paths) == 0:
            self.observation_paths.extend(list(observation_paths))

        remaining = self.size - self.current_size()
        self.actions.append(np.array(actions[:remaining]))
        self.observations.append(np.array(observations[:remaining]))

    def save(self, path):
        print(f'Saving dataset for {self.aid}: {self.observation_paths}')
        np.savez_compressed(
            path,
            size=self.size,
            aid=self.aid,
            num_actions=self.num_actions,
            actions=np.uint8(np.concatenate(self.actions, axis=0)),
            observations=np.concatenate(self.observations, axis=0, dtype=np.float32),
            observation_paths=np.array(self.observation_paths, dtype=object)
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls(**data)


def fit_tree(dataset: PolicyDataset, max_depth):
    dt = DecisionTreeClassifier(max_depth=max_depth, splitter="best", criterion="entropy")
    #dt = CalibratedClassifierCV(dt)
    dt.fit(dataset.observations, dataset.actions)
    acc = accuracy_score(dataset.actions, dt.predict(dataset.observations))
    print(f"Accuracy of the generated decision tree model ({dt.get_n_leaves()} nodes): {acc}")
    return dt


def tree_to_dict(dt_model, feature_names, dataset):
    tree_ = dt_model.tree_
    tree_dict = OrderedDict()

    def node_to_str(tree, node_id, criterion):
        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]
        json_value = ', '.join([str(x) for x in value])
        if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            is_leaf = True
            counts = tree_.value[node_id][0]
            p = counts / counts.sum()
            probs = [0] * dataset.num_actions.item()
            for i, cls in enumerate(dt_model.classes_):
                probs[cls] = p[i]
            return {"id": int(node_id), "is_leaf": is_leaf, "probabilities": probs}
        else:
            is_leaf = False
            if feature_names is not None:
                feature = str(feature_names[tree.feature[node_id]])
            else:
                feature = tree.feature[node_id]
        if "=" in feature:
            rule_type = "matches {value}"
            rule_value = "false"
        else:
            rule_type = "matches ..{value}"
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


def build_inference_module(
    module_name: str,
    mob_name: str,
    dataset: PolicyDataset,
    inference_tree_depth: int
):
    print(dataset.observation_paths, dataset.num_actions.item())
    print(np.unique(dataset.actions, return_counts=True))

    tree_dict = tree_to_dict(
        fit_tree(dataset, inference_tree_depth),
        feature_names=dataset.observation_paths,
        dataset=dataset
    )
    os.makedirs(f"data/datapack_modules/{module_name}/{mob_name}/inference", exist_ok=True)

    def build_node(node, no_write=False):
        if node["is_leaf"]:
            most = np.argmax(node["probabilities"])
            body = "" #f"""say {node['id']}\n"""
            if node["probabilities"][most] < 1.:
                body += "execute store result score #tmp random run random value 0..999\n"
            cumulated = 0
            #print(node['probabilities'])
            for action, prob in enumerate(node['probabilities']):
                if prob == 0:
                    continue
                cumulated += prob
                if np.isclose(cumulated, 1):
                    body += f"return run scoreboard players set @s action {action}\n"
                    break
                else:
                    body += f"execute if score #tmp random matches ..{round(cumulated * 1000) - 1} run return run scoreboard players set @s action {action}\n"
        else:
            body = ""
            if node['id'] == 0 :
                body += "execute store result score #tmp random run random value 0..999\n"
                for action in range(dataset.num_actions):
                    body += f"execute if score #tmp random matches ..{round((1+action) * 50 / dataset.num_actions) - 1} run return run scoreboard players set @s action {action}\n"

            body += f"""
execute store result score #tmp action run data get storage ai {node['feature']} 100000

# if
execute if score #tmp action {node['op'].format(value=round(node['threshold'] * 100000))} run return run function ai:ai_modules/{module_name}/{mob_name}/inference/node_{node['left']['id']}

# else
{build_node(node['right'], no_write=True)}
"""
            # function ai:ai_modules/{module_name}/{mob_name}/inference/node_{node['right']['id']}

        # create file
        if not no_write:
            with open(f"data/datapack_modules/{module_name}/{mob_name}/inference/node_{node['id']}.mcfunction", "w") as f:
                f.write(body)

        if node["is_leaf"]:
            return body

        build_node(node["left"])

        return body

    build_node(tree_dict)