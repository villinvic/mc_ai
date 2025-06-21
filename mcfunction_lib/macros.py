from typing import List, Tuple, Dict, NamedTuple

from minecraft_ai.mc_types import MCObservable, MCAction
from mcfunction_lib.mc_function import mcfunction

class Entity(NamedTuple):
    name: str
    mob: str
    position_offset: Tuple[float, float, float] = (0., 0., 0.)
    nbt: str = "{}" # using string is more convenient
    attributes: dict = {} # such as movement speed, hitbox size, etc.

@mcfunction
def fetch(
        entity_name: str,
        observables: List[MCObservable]
):
    # fetch data
    fetch = """
execute unless entity @s[nbt={DeathTime:0s}] run tag @s add dying
execute if entity @s[tag=dying] run return fail
"""
    for obs in observables:
        fetch += obs.build_mc_function(entity_name)
    return fetch

@mcfunction
def summon(
        module_name: str,
        entities: List[Entity]
):
    summons = ""
    for entity in entities:

        summons += f"""
        data modify storage ai nbt set value {entity.nbt}
        execute positioned ~{entity.position_offset[0]} ~{entity.position_offset[1]} ~{entity.position_offset[2]} summon {entity.mob} run function ai:ai_modules/{module_name}/entities/{entity.name}/setup
"""

@mcfunction
def setup(
        module_name: str,
        entity: Entity,
        observables: List[MCObservable],
        tags: Tuple[str] = (),
):
    # function called upon summoning

    tags = "\n".join(["tag @s add {tag}" for tag in (entity.name, "ai") + tags])

    data_create = ['uuid:""']
    for observable in observables:
        data_create.append(f'{observable.name}:0')

    return f"""
# setup tags
{tags}

# register this entity as alive
function gu:generate
data modify storage ai alive append from storage gu:main out

# init ai storage for this entity
data modify storage ai {entity.name} set value {{{",".join(data_create)}}}
data modify storage ai {entity.name}.uuid set from storage gu:main out

# init scores
scoreboard players set @s toggled -1
scoreboard players set @s action 0

# merge entity attributes
data merge entity @s {entity.nbt}

# summon target (for moving)
 summon marker ~ ~ ~ {{Tags:[target,{entity.name}_target]}}
 
# in-game ai disabler
summon tadpole ~ ~ ~ {{Silent:1b,Invulnerable:1b,NoAI:1b,Tags:[ai_disabler,{entity.name}_disabler],active_effects:[{id:"minecraft:invisibility",amplifier:0,duration:-1,show_particles:0b}],attributes:[{id:"minecraft:armor",base:100},{id:"minecraft:scale",base:0}]}}
ride @n[type=bat, tag={entity.name}_disabler, distance=..1] mount @s

# init the observables for this entity
function ai:ai_modules/{module_name}/{entity.name}/fetch
"""

@mcfunction
def on_death(
        module_name:str,
        entity: Entity
):
    return f"""
kill @e[limit=1,type=marker, tag={entity.name}_target]
kill @e[limit=1,type=tadpole, tag={entity.name}_disabler]
"""

@mcfunction
def take_action(
        module_name:str,
        entity: Entity
):
    return f"""
$function ai:ai_modules/{module_name}/{entity.name}/action_$(id)
"""

def actions(
        module_name: str,
        entity: Entity,
        actions: List[MCAction]
):

    for index, mc_action in enumerate(actions):

        @mcfunction
        def action():
            return f"""
{mc_action.build_mc_function(index, entity_name=entity.name, **entity.attributes)}
"""
        action(mc_path=f"data/datapack_modules/{module_name}/{entity.name}", suffix=str(index))

        if action.toggle:
            @mcfunction
            def toggle():
                return f"""
            {mc_action.mc_toggle(index, entity_name=entity.name, **entity.attributes)}
            """

            toggle(mc_path=f"data/datapack_modules/{module_name}/{entity.name}", suffix=str(index))

@mcfunction
def act(
        module_name:str,
        entity: Entity,
):
    return f"""
# return if dying
execute if entity @s[tag=dying] run return run ai:ai_modules/{module_name}/{entity.name}/on_death

# infer action
function ai:ai_modules/{module_name}/{entity.name}/inference/node_0

# preprocess
execute store result score @s vx run data get entity @s Motion[0] 100000
execute store result score @s vy run data get entity @s Motion[1] 100000
execute store result score @s vz run data get entity @s Motion[2] 100000

execute store result score @s x run data get entity @s Pos[0] 100000
execute store result score @s y run data get entity @s Pos[1] 100000
execute store result score @s z run data get entity @s Pos[2] 100000

scoreboard players operation #n x = @s x
scoreboard players operation #n z = @s z

# the OnGround tag is not perfectly reliable
execute if entity @s[nbt={{OnGround:1b}}] if score @s vy matches -7841 run tag @s add on_ground

# take action
execute store result storage ai action.id int 1 run scoreboard players get @s action
function ai:ai_modules/{module_name}/{entity.name}/take_action with storage ai action 

# toggles
execute store result storage ai toggle.id int 1 run scoreboard players get @s toggled
function ai:ai_modules/{module_name}/{entity.name}/toggle with storage ai toggle 

# compute changes in velocity
scoreboard players operation #n x -= @s x
scoreboard players operation #n z -= @s z
scoreboard players operation @s vx += #n x
scoreboard players operation @s vz += #n z

# apply changes to entity
execute store result entity @s Motion[0] double 0.00001 run scoreboard players get @s vx
execute store result entity @s Motion[2] double 0.00001 run scoreboard players get @s vz
execute store result entity @s Motion[1] double 0.00001 run scoreboard players get @s vy

# post_process
data remove entity @s last_hurt_by_mob
tag @s remove on_ground
"""

@mcfunction
def act_global(
        module_name:str,
        entities: List[Entity],
):
    body = ""
    for entity in entities:
        if entity.name == "player":
            continue
        body += f"$execute as $(uuid) if entity @s[tag={entity.name}] function ai:ai_modules/{module_name}/{entity.name}/act\n"

    return body


@mcfunction
def fetch_global(
        module_name: str,
        entities: List[Entity],
):
    body = ""
    for entity in entities:
        body += f"$execute as $(uuid) if entity @s[tag={entity.name}] function ai:ai_modules/{module_name}/{entity.name}/fetch\n"

    return body

@mcfunction
def summon(
    module_name: str,
    entities: List[Entity],
):
    body = ""
    for entity in entities:
        if entity.name == "player":
            continue
        body += f"execute summon {entity.mob} run function ai:ai_modules/{module_name}/{entity.name}/setup\n"



def build_ai_module(
    module_name: str,
    entities: List[Entity],
    observables: Dict[str, List[MCObservable]]
):

    # global

    # per entity
