import pickle
from typing import List, Tuple, Dict, NamedTuple

from mcfunction_lib.compile import mcfunction, compile
from mcfunction_lib.inference import PolicyDataset, build_inference_module


@mcfunction
def step(
        module_name:str,
        mobs,
):
    body = ""
    for mob in mobs:
        if "player" in mob.name:
            continue
        body += f"$execute as $(uuid) at @s if entity @s[tag={mob.name}] run return run function ai:ai_modules/{module_name}/{mob.name}/step\n"

    return body

@mcfunction
def fetch(
        module_name: str,
        mobs,
):
    body = ""
    for mob in mobs:
        if "player" in mob.name:
            body +=f"$execute as $(uuid) at @s if entity @s[type=player] run return run function ai:ai_modules/{module_name}/{mob.name}/fetch\n"
        else:
            body += f"$execute as $(uuid) at @s if entity @s[tag={mob.name}] run return run function ai:ai_modules/{module_name}/{mob.name}/fetch\n"

    return body

@mcfunction
def summon(
    module_name: str,
    mobs,
):
    body = ""
    for mob in mobs:
        if "player" in mob.name:
            continue
        body += f"execute positioned ~{mob.position_offset[0]} ~{mob.position_offset[1]} ~{mob.position_offset[2]} summon {mob.entity_type} run function ai:ai_modules/{module_name}/{mob.name}/setup\n"

    return body


def build_ai_module(
    module_name: str,
    mobs,
    dataset_path: str,
    position_offset=(0, 0, 0), # arena center within minecraft,
    inference_tree_depth=5
):

    dataset = {
        mob.name: PolicyDataset.load(dataset_path + f"/{mob.name}.npz") for mob in mobs
        if "player" not in mob.name
    }

    summon.compile(module_name=module_name, mobs=mobs)
    fetch.compile(module_name=module_name, mobs=mobs)
    step.compile(module_name=module_name, mobs=mobs)

    for mob in mobs:
        if "player" in mob.name:
            mob.fetch.compile(module_name=module_name)
            continue

        compile(mob, module_name=module_name)

        build_inference_module(
            module_name=module_name,
            mob_name=mob.name,
            dataset=dataset[mob.name],
            inference_tree_depth=inference_tree_depth
        )

