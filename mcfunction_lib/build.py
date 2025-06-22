from typing import List, Tuple, Dict, NamedTuple

from minecraft_ai.mc_types import MCObservable, MCAction
from mcfunction_lib.mcfunction import mcfunction, compile

@mcfunction
def step(
        module_name:str,
        mobs,
):
    body = ""
    for mob in mobs:
        if "player" in mob.name:
            continue
        body += f"$execute as $(uuid) if entity @s[tag={mob.name}] run return run function ai:ai_modules/{module_name}/{mob.name}/step\n"

    return body

@mcfunction
def fetch(
        module_name: str,
        mobs,
):
    body = ""
    for mob in mobs:
        if "player" in mob.name:
            body +=f"$execute as $(uuid) if entity @s[type=player] run return run function ai:ai_modules/{module_name}/{mob.name}/fetch\n"
        else:
            body += f"$execute as $(uuid) if entity @s[tag={mob.name}] run return run function ai:ai_modules/{module_name}/{mob.name}/fetch\n"

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
        body += f"execute summon {mob.entity_type} run function ai:ai_modules/{module_name}/{mob.name}/setup\n"


def build_ai_module(
    module_name: str,
    mobs,
    position_offset=(0, 0, 0) # arena center within minecraft
):

    summon.compile(module_name=module_name, mobs=mobs)
    fetch.compile(module_name=module_name, mobs=mobs)
    step.compile(module_name=module_name, mobs=mobs)

    for mob in mobs:
        compile(mob, module_name=module_name)

        # todo:
        # build inference tree here, with position offset