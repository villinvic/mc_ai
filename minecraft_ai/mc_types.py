from enum import Enum
from typing import Tuple, List, NamedTuple, Callable, Dict

import numpy as np

from mcfunction_lib.mc_function import mcfunction


class Action(Enum):
    # TODO:
    # actions from different entities will be different

    # Toggleable
    SPRINT = 0
    STRIFE_LEFT = 1
    STRIFE_RIGHT = 2
    WALK_BACK = 3

    JUMP = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6
    ROTATE_HARD_LEFT = 7
    ROTATE_HARD_RIGHT = 8
    ATTACK = 9
    #NOOP = 10

class Constants:
    # global
    GRAVITY_UP = 0.0828
    GRAVITY_DOWN = 0.075
    GRAVITY_GROUND = 0.0784

    # ground motion
    FRICTION = 0.45
    WALK_ACCELERATION = 0.0971
    SPRINT_ACCELERATION = WALK_ACCELERATION * 1.3
    JUMP_FORCE = 0.415 + GRAVITY_UP
    JUMP_BOOST = 0.2 # motion gained by jumping while sprinting

    # air motions
    FRICTION_AIR = 0.09 # for dpos out of hurt time, and motion in hurt time
    WALK_ACCELERATION_AIR = 0.0197
    SPRINT_ACCELERATION_AIR = WALK_ACCELERATION_AIR * 1.3

    # knockback motions
    # if on ground, knocked up plus horizontally, with dpos
    # multiplier if sprinting (but we can't apply it in game, we will just use base knockbacks)
    # if in air, no knock up
    # full knockback does not depend on distance, just the norm (dx,dz) vector.
    KNOCK_UP = 0.434
    KNOCK_BACK = 0.39
    DAMAGE_FRAME = 10

    # Lava / water physics ?
    FRICTION_WATER = 1


class MCObservable(NamedTuple):
    name: str
    raw: Callable
    py_getter: Callable
    mc_getter: str

    def build_mc_function(self, entity_name: str):
        return f"""
# setup observation {self.name}
{self.mc_getter}
data modify storage ai {entity_name}.{self.name} set from storage ai obs_tmp
"""

def get_x(_self, arena_dims, **kwargs):
    return _self.x / arena_dims[0] - 0.5
x_obs = MCObservable(
    name="x",
    raw=lambda player, offsets, **k: player.x - offsets[0],
    py_getter=get_x,
    mc_getter="data modify storage ai obs_tmp set from entity @s Pos[0]"
)

def get_y(_self, arena_dims, **kwargs):
    return (_self.y - 1) / (arena_dims[1] - 1)
y_obs = MCObservable(
    name="y",
    raw=lambda player, offsets, **k: player.y - offsets[1],
    py_getter=get_y,
    mc_getter="data modify storage ai obs_tmp set from entity @s Pos[1]"
)

def get_z(_self, arena_dims, **kwargs):
    return _self.z / arena_dims[2] - 0.5
z_obs = MCObservable(
    name="z",
    raw=lambda player, offsets, **k: player.z - offsets[2],
    py_getter=get_z,
    mc_getter="data modify storage ai obs_tmp set from entity @s Pos[2]"
)

def get_dx(_self, motion_scale, **kwargs):
    return _self.dx * motion_scale
get_dx_mc = """
execute store result score @s x run data get entity @s Pos[0] 100000
scoreboard players operation #tmp x = @s x
scoreboard players operation #tmp x -= @s prev_x
scoreboard players operation @s prev_x = @s x
execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp x
"""
dx_obs = MCObservable(
    name="dx",
    raw=lambda player, **k: player.dx,
    py_getter=get_dx,
    mc_getter=get_dx_mc
)

def get_dy(_self, motion_scale, **kwargs):
    return _self.dy * motion_scale
get_dy_mc = """
execute store result score @s x run data get entity @s Pos[1] 100000
scoreboard players operation #tmp y = @s y
scoreboard players operation #tmp y -= @s prev_y
scoreboard players operation @s prev_y = @s y
execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp y
"""
dy_obs = MCObservable(
    name="dy",
    raw=lambda player, **k: player.dy,
    py_getter=get_dy,
    mc_getter=get_dy_mc
)

def get_dz(_self, motion_scale, **kwargs):
    return _self.dz * motion_scale
get_dz_mc = """
execute store result score @s z run data get entity @s Pos[2] 100000
scoreboard players operation #tmp z = @s z
scoreboard players operation #tmp z -= @s prev_z
scoreboard players operation @s prev_z = @s z
execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp z
"""
dz_obs = MCObservable(
    name="dz",
    raw=lambda player, **k: player.dz,
    py_getter=get_dz,
    mc_getter=get_dz_mc
)

def get_health(_self, **kwargs):
    return _self.health / _self.max_health
get_health_mc = """
execute store result storage ai obs_tmp double 1 run scoreboard players get @s health
"""
health_obs = MCObservable(
    name="health",
    raw=lambda player, **k: round(player.health),
    py_getter=get_health,
    mc_getter=get_health_mc
)

def get_rot(_self, **kwargs):
    return (_self.rot - np.pi) / np.pi
get_rot_mc = """
data modify storage ai obs_tmp set from entity @s Rot[0]
"""
rot_obs = MCObservable(
    name="rot",
    raw=lambda player, **k: player.rot * 180 / np.pi,
    py_getter=get_rot,
    mc_getter=get_rot_mc
)

def get_hurt_time(_self, **kwargs):
    return _self.hurt_time * 0.1
get_hurt_time_mc = """
data modify storage ai obs_tmp set from entity @s HurtTime
"""
hurt_time_obs = MCObservable(
    name="hurt_time",
    raw=lambda player, **k: player.hurt_time,
    py_getter=get_hurt_time,
    mc_getter=get_hurt_time_mc
)

def get_on_ground(_self, **kwargs):
    return float(_self.on_ground)
get_on_ground_mc = """
execute store success storage ai obs_tmp int 1 if entity @s[tag=on_ground]
"""
on_ground_obs = MCObservable(
    name="on_ground",
    raw=lambda player, **k: int(player.on_ground),
    py_getter=get_on_ground,
    mc_getter=get_on_ground_mc
)

def get_toggle(_self, num_toggles, **kwargs):
    one_hot_toggle = [0] * num_toggles
    if _self.toggled is not None:
        one_hot_toggle[_self.toggled.value] = 1
    return one_hot_toggle
get_toggle_mc = """
execute store result storage ai obs_tmp int 1 run scoreboard players get @s toggled 
"""
toggle_obs = MCObservable(
    name="toggle",
    raw=lambda player, **k: -1 if player.toggled is None else player.toggled.value,
    py_getter=get_on_ground,
    mc_getter=get_on_ground_mc
)

default_observables = [
    x_obs, y_obs, z_obs,
    dx_obs, dy_obs, dz_obs,
    health_obs, rot_obs, hurt_time_obs, on_ground_obs,
    toggle_obs
]

class MCAction(NamedTuple):
    name: str
    toggle: bool
    py_func: Callable = lambda **k: None
    py_toggle: Callable = lambda **k: None
    mc_func: Callable = lambda **k: ""
    mc_toggle: Callable = lambda **k: ""

    def build_py_function(self, index: int):
        def func(_self, **kwargs):
            if self.toggle:
                _self.toggled = index
            self.py_func()
        return func

    def build_mc_function(self, index: int, **kwargs):

        body = ""
        if self.toggle:
            body += f"execute if score @s action matches {index} run scoreboard players set @s toggled {index}\n"

        body += self.mc_func(**kwargs) + "\n"

        return body


def py_sprint_toggle(_self, **kwargs):
    accel = Constants.SPRINT_ACCELERATION if _self.on_ground else Constants.SPRINT_ACCELERATION_AIR
    _self.apply_accel(accel * _self.movement_speed, _self.rot)

def mc_sprint_toggle(_self, entity_name, movement_speed=1, **kwargs):
    return f"""
# if on ground
execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={entity_name}_target] run return run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION * movement_speed :.5f}"}}
# else
execute as @e[limit=1, type=marker, tag={entity_name}_target] run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION_AIR * movement_speed :.5f}"}}
"""

sprint = MCAction(
    name="sprint",
    toggle=True,
    py_toggle=py_sprint_toggle,
    mc_toggle=mc_sprint_toggle,
)

def py_strife_left_toggle(_self, **kwargs):
    accel = Constants.WALK_ACCELERATION if _self.on_ground else Constants.WALK_ACCELERATION_AIR
    left = _self.rot + np.pi * 0.5
    _self.apply_accel(accel * _self.movement_speed, left)

def mc_strife_left_toggle(_self, entity_name, movement_speed=1, **kwargs):
    return f"""
# if on ground
execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={entity_name}_target] run return run function ai:move {{target:"^{Constants.WALK_ACCELERATION * movement_speed :.5f} ^ ^"}}
# else
execute as @e[limit=1, type=marker, tag={entity_name}_target] run function ai:move {{target:"^{Constants.WALK_ACCELERATION_AIR * movement_speed :.5f} ^ ^"}}
"""

strife_left = MCAction(
    name="strife_left",
    toggle=True,
    py_toggle=py_strife_left_toggle,
    mc_toggle=mc_strife_left_toggle,
)


def py_strife_right_toggle(_self, **kwargs):
    accel = Constants.WALK_ACCELERATION if _self.on_ground else Constants.WALK_ACCELERATION_AIR
    right = _self.rot - np.pi * 0.5
    _self.apply_accel(accel * _self.movement_speed, right)

def mc_strife_right_toggle(_self, entity_name, movement_speed=1, **kwargs):
    return f"""
# if on ground
execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={entity_name}_target] run return run function ai:move {{target:"^-{Constants.WALK_ACCELERATION * movement_speed :.5f} ^ ^"}}
# else
execute as @e[limit=1, type=marker, tag={entity_name}_target] run function ai:move {{target:"^-{Constants.WALK_ACCELERATION_AIR * movement_speed :.5f} ^ ^"}}
"""

strife_right = MCAction(
    name="strife_right",
    toggle=True,
    py_toggle=py_strife_right_toggle,
    mc_toggle=mc_strife_right_toggle,
)

def py_walk_back_toggle(_self, **kwargs):
    accel = Constants.WALK_ACCELERATION if _self.on_ground else Constants.WALK_ACCELERATION_AIR
    direction = _self.rot + np.pi
    _self.apply_accel(accel * _self.movement_speed, direction)

def mc_walk_back_toggle(_self, entity_name, movement_speed=1, **kwargs):
    return f"""
# if on ground
execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={entity_name}_target] run return run function ai:move {{target:"^ ^ ^-{Constants.WALK_ACCELERATION * movement_speed :.5f}"}}
# else
execute as @e[limit=1, type=marker, tag={entity_name}_target] run function ai:move {{target:"^ ^ ^-{Constants.WALK_ACCELERATION_AIR * movement_speed :.5f}"}}
"""

walk_back = MCAction(
    name="walk_back",
    toggle=True,
    py_toggle=py_walk_back_toggle,
    mc_toggle=mc_walk_back_toggle,
)

def py_jump_func(_self, **kwargs):
    if not _self.on_ground or _self.vy > 0:
        return
    _self.vy += _self.jump_height * Constants.JUMP_FORCE
    if _self.actions[_self.toggled].name == sprint.name:
        _self.apply_accel(Constants.JUMP_BOOST, _self.rot)

def mc_jump_func(_self, entity_name, jump_height=1, **kwargs):
    return f"""
execute if entity @s[tag=!on_ground] run return fail

# jump
scoreboard players add @s vy {round(50000 * jump_height)}

# velocity boost 
# 0 should be reserved for the sprint toggle
execute if score @s toggled matches 0 run function ai:jump_boost
"""

jump = MCAction(
    name="jump",
    toggle=False,
    py_func=py_jump_func,
    mc_func=mc_jump_func,
)


def py_rotate_left_func(_self, **kwargs):
    _self.new_rot = (_self.rot + _self.rot_speed) % (2 * np.pi)

def mc_rotate_left_func(_self, entity_name, rot_speed=360/16, **kwargs):
    return f"""
rotate @s ~{rot_speed:.5f} ~
"""

rotate_left = MCAction(
    name="rotate_left",
    toggle=False,
    py_func=py_rotate_left_func,
    mc_func=mc_rotate_left_func,
)


def py_rotate_right_func(_self, **kwargs):
    _self.new_rot = (_self.rot - _self.rot_speed) % (2 * np.pi)

def mc_rotate_right_func(_self, entity_name, rot_speed=360/16, **kwargs):
    return f"""
rotate @s ~{rot_speed:.5f} ~
"""

rotate_right = MCAction(
    name="rotate_right",
    toggle=False,
    py_func=py_rotate_right_func,
    mc_func=mc_rotate_right_func,
)


def py_rotate_hard_left_func(_self, **kwargs):
    _self.new_rot = (_self.rot + 4 * _self.rot_speed) % (2 * np.pi)

def mc_rotate_hard_left_func(_self, entity_name, rot_speed=360/16, **kwargs):
    return f"""
rotate @s ~{rot_speed * 4:5.f} ~
"""

rotate_hard_left = MCAction(
    name="rotate_hard_left",
    toggle=False,
    py_func=py_rotate_hard_left_func,
    mc_func=mc_rotate_hard_left_func,
)


def py_rotate_hard_right_func(_self, **kwargs):
    _self.new_rot = (_self.rot - 4 * _self.rot_speed) % (2 * np.pi)

def mc_rotate_hard_right_func(_self, entity_name, rot_speed=360/16, **kwargs):
    return f"""
rotate @s ~{rot_speed * 4:.5f} ~
"""

rotate_hard_right = MCAction(
    name="rotate_hard_right",
    toggle=False,
    py_func=py_rotate_hard_right_func,
    mc_func=mc_rotate_hard_right_func,
)


def py_attack_func(_self: "Player", players: List["Player"], arena: "Arena"= None, **kwargs):
        nearest = None
        nearest_hit = None
        d = np.inf
        for p in players:
            di, hit = _self.distance(p)
            if di < d:
                d = di
                nearest = p
                nearest_hit = hit

        if d > _self.base_attack_range:
            return
        blocking = arena.raycast(_self.eyes_pos, nearest_hit)

        if blocking is not None:
            return

        looking_at = _self.look_pos(_self.distance((nearest.x, nearest.y, nearest.z)))

        hit_distance = nearest.distance(looking_at)

        if hit_distance < (nearest.collision_box[0] * 0.6):
            nearest.damage(_self)

def mc_attack_func(_self, module_name, entity_name, eye_height=1.6, base_attack_range=3, base_attack_damage=2, **kwargs):
    # attacking is always w.r.t. human players here

    @mcfunction
    def hurt_player():
        return f"""
# check for blocks in the way
$execute store success storage ai raycast_success run function ai:ai_modules/{module_name}/{entity_name}/raycast {{target:$(target)}}
execute unless data storage ai raycast_success run return fail

# no block in the way, get the position we are looking at (at the same distance of the targeted player)
execute store result storage ai look.dist double 0.00001 run scoreboard players get #dist attack_dist
data modify storage ai whiff set value {{x:[0,0], y:[0,0]}}
execute as @e[limit=1, type=marker, tag={entity_name}_target] rotated ~ 0 run function ai:look_pos with storage ai look
data modify storage ai whiff.y[0] set from storage ai atk.y[0]
data modify storage ai whiff.y[1] set from storage ai atk.y[2]

execute store result score #dist attack_dist run function gm:distance with storage ai whiff

execute if score #dist attack_dist matches {round(0.6 * 0.6 * 100000)}.. run return fail

# we are close enough and pointing to the right direction
function ai:deal_damage {{damage:{round(base_attack_damage)}, crit_damage:{round(base_attack_damage * 1.51)}}}

# don't change pitch, only yaw
data modify storage ai rot set from entity @s Rotation[0]
$rotate @s facing entity @p $(target)
data modify entity @s Rotation[0] set from storage ai rot
return 1
"""

    base_attack_range = base_attack_range + 0.3

    @mcfunction
    def raycast():
        return f"""
scoreboard players operation #ray_dist count = #dist attack_dist
$execute at @s anchored eyes facing entity @p $(target) as @e[limit=1, type=marker, tag={entity_name}_target] run return run function ai:ai_modules/{module_name}/{entity_name}/raycast_loop {{increment:0.4}}
"""

    @mcfunction
    def raycast_loop():
        # increment until block found or exceeded distance
        return f"""
$execute unless block ^ ^ ^$(increment) #ai:air_like run return fail
$tp @s ^ ^ ^$(increment)
scoreboard players remove #ray_dist count 40000
execute if score #ray_dist count matches 40000.. run return run function ai:ai_modules/{module_name}/{entity_name}/raycast_loop
execute store result storage ai raytrace.increment double 0.00001 run scoreboard players get #ray_dist count
execute if score #ray_dist count matches 1..40000 run return run function ai:ai_modules/{module_name}/{entity_name}/raycast_loop with storage ai raytrace
return 1
"""

    hurt_player(
        mc_path=f"data/datapack_modules/{module_name}/{entity_name}"
    )
    raycast(
        mc_path=f"data/datapack_modules/{module_name}/{entity_name}"
    )
    raycast_loop(
        mc_path=f"data/datapack_modules/{module_name}/{entity_name}"
    )


    return f"""
# clearly out of range
execute unless entity @p[distance=..{base_attack_range + 0.2}] run return fail

# compute distance between attacker and nearest human player
data modify storage ai atk set value {{x:[0,0,0], y:[0,0,0]}}
execute store result storage ai atk.x[0] run scoreboard players get @s x
execute store result storage ai atk.x[2] run scoreboard players get @s z
scoreboard players operation #tmp y = @s y
# offset to eyes of entity
scoreboard players add #tmp y {round(eye_height * 100000)}
execute store result storage ai atk.x[1] double 1 run scoreboard players get #tmp y

# compute distance from head
execute as @p run function ai:player_head_to_input
function gm:distance with storage ai atk
execute store result score #dist attack_dist run data get storage gm:io out 1
scoreboard player set #threshold attack_dist {round(base_attack_range * 100000)}
execute if score #dist attack_dist matches ..{round(base_attack_range * 100000)} if function ai:ai_modules/{module_name}/{entity_name}/hurt_player {{target:"head"}} run return 1

# compute distance from feet
execute as @p run function ai:player_feet_to_input
function gm:distance with storage ai atk
execute store result score #dist attack_dist run data get storage gm:io out 1
execute if score #dist attack_dist matches ..{round(base_attack_range * 100000)} if function ai:ai_modules/{module_name}/{entity_name}/hurt_player {{target:"feet"}} run return 1

# neutral yaw
rotate @s ~ 0
return fail
"""

attack = MCAction(
    name="attack",
    toggle=False,
    py_func=py_attack_func,
    mc_func=mc_attack_func,
)

default_actions = [
    sprint, strife_left, strife_right, walk_back,
    jump, rotate_left, rotate_right, rotate_hard_left, rotate_hard_right,
    attack
]

class Player:
    def __init__(
            self,
            aid: str,
            movement_speed = 1.,
            jump_height = 1.,
            rot_speed = 360/16,
            base_attack_range = 3,
            base_attack_damage = 2,
            max_health = 20,
            collision_box = (0.6, 1.8, 0.6), # anchored at center feet
            eyes_height = 1.6,
            hitbox_width = 1,
            gravity = 1.,
            observables = default_observables,
            actions = default_actions

    ):
        self.aid = aid

        self.hurt_time = 0.

        self.x = 0
        self.y = 0 # y is height
        self.z = 0
        self.rot = - np.pi / 2 # in gradients
        self.new_rot = self.rot

        self.vx = 0
        self.vy = -Constants.GRAVITY_GROUND
        self.vz = 0

        # not used by the physics engine, but used for the ai input
        self.dx = 0
        self.dy = 0
        self.dz = 0

        self.was_on_ground = False
        self.on_ground = False
        self.toggled = None

        self.health_ratio = 1.
        self.movement_speed = movement_speed
        self.jump_height = jump_height
        self.rot_speed =  rot_speed * np.pi / 180
        self.base_attack_range = base_attack_range
        self.base_attack_damage = base_attack_damage
        self.health = max_health
        self.max_health = max_health
        self.collision_box = collision_box
        self.eyes_height = eyes_height
        self.hitbox_width = hitbox_width
        self.gravity = gravity

        self.observables = {
            obs.name: obs
            for obs in observables
        }

        self.actions: Dict[int, Callable] = {}
        self.toggles: Dict[int, Callable] = {}

        for index, action in enumerate(actions):
            self.actions[index] = action.build_py_function(index)
            if action.toggle:
                self.toggles[index] = action.py_toggle

        self.num_toggles = len(self.toggles)




    @property
    def eyes_pos(self):
        return (self.x, self.eyes_height + self.y, self.z)

    def move_to(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


    def apply_accel(self, accel, direction):
        self.vx += -accel * np.sin(direction)
        self.vz += -accel * np.cos(direction)

    def distance(self, other: "Player" | Tuple):
        if isinstance(other, Player):
            #distance from our eyes to the other player's hitbox.
            width_x = other.collision_box[0]
            width_z = other.collision_box[2]
            height = other.collision_box[1]

            # other hitbox
            min_x = other.x - width_x / 2
            max_x = other.x + width_x / 2
            min_y = other.y
            max_y = other.y + height
            min_z = other.z - width_z / 2
            max_z = other.z + width_z / 2

            # Eye position of self
            eye_x, eye_y, eye_z = self.eyes_pos

            # Clamp each coordinate to the hitbox range
            clamped_x = max(min_x, min(eye_x, max_x))
            clamped_y = max(min_y, min(eye_y, max_y))
            clamped_z = max(min_z, min(eye_z, max_z))

            # Compute Euclidean distance
            dx = eye_x - clamped_x
            dy = eye_y - clamped_y
            dz = eye_z - clamped_z

            return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5, (clamped_x, clamped_y, clamped_z)
        else:
            return np.sqrt(
                (self.x - other[0]) ** 2 + (self.z - other[2]) ** 2
            )

    def look_pos(self, distance: float):
        """
        Returns the position at `distance` units in the direction
        the player is currently facing (self.rot) on the XZ plane.
        """
        dx = -np.sin(self.rot) * distance
        dz = -np.cos(self.rot) * distance
        return self.x + dx, self.y, self.z + dz

    def damage(self, source: "Player"):
        if self.hurt_time > 0:
            return

        if self.on_ground and self.vy <= 0:
            self.vy += Constants.KNOCK_UP

        hit_angle = np.angle(complex(
            self.x - source.x,
            self.z - source.z
        ))

        self.vx = Constants.KNOCK_BACK * np.cos(hit_angle)
        self.vz = Constants.KNOCK_BACK * np.sin(hit_angle)

        self.hurt_time = 10
        dmg = source.base_attack_damage
        if source.vy < 0 and not source.on_ground:
            dmg *= 1.5 # crit
        self.health = np.maximum(0., self.health-dmg)

    def apply_friction(self):
        # friction is applied (computed) before any forces

        angle = np.angle(complex(self.vx, self.vz))
        total = np.sqrt(self.vx**2 + self.vz ** 2)

        coeff = Constants.FRICTION if self.was_on_ground else Constants.FRICTION_AIR

        friction = total * coeff

        self.vx -= np.cos(angle) * friction
        self.vz -= np.sin(angle) * friction

        self.was_on_ground = self.on_ground

    def apply_gravity(self):
        if self.vy == 0:
            self.vy = - Constants.GRAVITY_GROUND
        elif self.vy > 0:
            self.vy -= Constants.GRAVITY_UP
        else:
            self.vy -= Constants.GRAVITY_DOWN

    def act(self, action_index: int, players: List["Player"], arena: "Arena"):
        self.actions[action_index](self, players=players, arena=arena)

    def tick(self, arena: "Arena"):
        if self.hurt_time == 0:
            # toggles
            if self.toggled is not None:
                self.toggles[self.toggled](self)

        if self.hurt_time > 0:
            self.hurt_time -= 1

        x = self.x + self.vx
        y = np.maximum(self.y + self.vy, 0)
        z = self.z + self.vz

        nx, ny, nz = arena.handle_collision(self, x, y, z)

        self.rot = self.new_rot
        self.dx = nx - self.x
        self.dy = ny - self.y
        self.dz = nz - self.z
        self.x = nx
        self.y = ny
        self.z = nz

    def observe(
            self,
            arena_dims,
            motion_scale,
            num_toggles,
            exclude = (),
    ):
        observed = []
        for name, obs in self.observables.items():
            if name in exclude:
                continue

            x = obs.py_getter(self, arena_dims=arena_dims, motion_scale=motion_scale, num_toggles=num_toggles)
            if isinstance(x, list):
                observed.extend(x)
            else:
                observed.append(x)

        return observed
