import abc
from typing import Tuple, NamedTuple, Any

import numpy as np

from minecraft_ai.mc_types import Constants
import mcfunction_lib as mc


class Mob(abc.ABC, mc.Observable, mc.Controllable):

    def __init__(
            self,
            name: str,
            movement_speed = 1.,
            rot_speed = 360/16,
            max_health = 20,
            collision_box = (0.6, 1.8, 0.6), # anchored at center feet
            eyes_height = 1.6,
            gravity = 1.,
            position_offset = (0., 0., 0.),
            entity_type = "Zombie",
            **kwargs
    ):
        mc.Observable.__init__(self)
        mc.Controllable.__init__(self)

        self.name = name

        self.hurt_time = 0.

        self.position_offset = position_offset
        self.x, self.y, self.z = position_offset

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

        self.health_ratio = 1.
        self.movement_speed = movement_speed
        self.gravity = gravity
        self.rot_speed =  rot_speed * np.pi / 180
        self.health = max_health
        self.max_health = max_health
        self.collision_box = collision_box
        self.eyes_height = eyes_height
        self.entity_type = entity_type

    @property
    def eyes_pos(self):
        return self.x, self.eyes_height + self.y, self.z

    def init_at(self, x, y, z):
        self.x, self.y, self.z = self.position_offset
        self.x += x
        self.y += y
        self.z += z

    def apply_accel(self, accel, direction):
        self.vx += -accel * np.sin(direction)
        self.vz += -accel * np.cos(direction)

    def distance_to_hitbox(self, other: "Mob"):

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

    def horizontal_distance(self, other: "Mob" | Tuple):
        if isinstance(other, Mob):
            return (self.x - other.x) ** 2 + (self.z - other.z) ** 2
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

    def damage(self, source: "Mob", damage):
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
        self.health = np.maximum(0., self.health-damage)

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
            self.vy = - Constants.GRAVITY_GROUND * self.gravity
        elif self.vy > 0:
            self.vy -= Constants.GRAVITY_UP * self.gravity
        else:
            self.vy -= Constants.GRAVITY_DOWN * self.gravity

    def tick(self, arena: "Arena"):
        if self.toggled is not None:
            self.apply_toggle()

        if self.hurt_time > 0:
            self.hurt_time -= 1

        x = self.x + self.vx
        y = np.maximum(self.y + self.vy, 0)
        z = self.z + self.vz

        nx, ny, nz = arena.handle_collision(self, x, y, z)
        if self.gravity == 0:
            self.on_ground = False

        self.rot = self.new_rot
        self.dx = nx - self.x
        self.dy = ny - self.y
        self.dz = nz - self.z
        self.x = nx
        self.y = ny
        self.z = nz

    @mc.observation(
        raw=lambda self: self.x,
        mc_function=lambda **_: "data modify storage ai obs_tmp set from entity @s Pos[0]"
    )
    def pos_x(self, arena, **kwargs):
        return self.x / arena.dx - 0.5

    @mc.observation(
        raw=lambda self: self.y,
        mc_function=lambda **_: "data modify storage ai obs_tmp set from entity @s Pos[1]"
    )
    def pos_y(self, arena, **kwargs):
        return (self.y - 1) / (arena.dy[1] - 1)

    @mc.observation(
        raw=lambda self: self.z,
        mc_function=lambda **_: "data modify storage ai obs_tmp set from entity @s Pos[2]"
    )
    def pos_z(self, arena, **kwargs):
        return self.z / arena.dz - 0.5

    @mc.observation(
        raw=lambda self: self.dx,
        mc_function=lambda **_: """
        execute store result score @s x run data get entity @s Pos[0] 100000
        scoreboard players operation #tmp x = @s x
        scoreboard players operation #tmp x -= @s prev_x
        scoreboard players operation @s prev_x = @s x
        execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp x
        """
    )
    def vel_x(self, **kwargs):
        return self.dx * 2

    @mc.observation(
        raw=lambda self: self.dy,
        mc_function=lambda **_: """
        execute store result score @s x run data get entity @s Pos[1] 100000
        scoreboard players operation #tmp y = @s y
        scoreboard players operation #tmp y -= @s prev_y
        scoreboard players operation @s prev_y = @s y
        execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp y
        """
    )
    def vel_y(self, **kwargs):
        return self.dy * 2

    @mc.observation(
        raw=lambda self: self.dz,
        mc_function=lambda **_: """
        execute store result score @s z run data get entity @s Pos[2] 100000
        scoreboard players operation #tmp z = @s z
        scoreboard players operation #tmp z -= @s prev_z
        scoreboard players operation @s prev_z = @s z
        execute store result storage ai obs_tmp double 0.00001 run scoreboard players get #tmp z
        """
    )
    def vel_z(self, **kwargs):
        return self.dz * 2

    @mc.observation(
        raw=lambda self: self.health,
        mc_function=lambda **_: "execute store result storage ai obs_tmp double 1 run scoreboard players get @s health"
    )
    def current_health(self, **kwargs):
        return self.health / self.max_health

    @mc.observation(
        raw=lambda self: self.rot,
        mc_function=lambda **_: "data modify storage ai obs_tmp set from entity @s Rot[0]"
    )
    def rotation(self, **kwargs):
        return (self.rot - np.pi) / np.pi

    @mc.observation(
        raw=lambda self: self.hurt_time,
        mc_function=lambda **_: "data modify storage ai obs_tmp set from entity @s HurtTime"
    )
    def hurt_time_left(self, **kwargs):
        return self.hurt_time * 0.1

    @mc.observation(
        raw=lambda self: int(self.on_ground),
        mc_function=lambda **_: "execute store success storage ai obs_tmp int 1 if entity @s[tag=on_ground]"
    )
    def is_on_ground(self, **kwargs):
        return float(self.on_ground)

    @mc.observation(
        raw=lambda self: -1 if self.toggled is None else self.toggled,
        mc_function=lambda **_: "execute store result storage ai obs_tmp int 1 run scoreboard players get @s toggled",
        private=True,
    )
    def toggle_obs(self, **kwargs):
        one_hot_toggle = [0] * self.num_toggles
        if self.toggled is not None:
            one_hot_toggle[self.toggled] = 1
        return one_hot_toggle

    @mc.mcfunction
    def summon(self, module_name, **kwargs):
        return f"""
        execute positioned ~{self.position_offset[0]} ~{self.position_offset[1]} ~{self.position_offset[2]} summon {self.entity_type} run function ai:ai_modules/{module_name}/{self.name}/setup
        """

    @mc.mcfunction
    def setup(self, module_name, **kwargs):
        return f"""
        # setup tags
        tag @s add ai

        # register this entity as alive
        function gu:generate
        data modify storage ai alive append from storage gu:main out

        # init ai storage for this entity
        data modify storage ai {self.name}.uuid set from storage gu:main out

        # init scores
        scoreboard players set @s toggled -1
        scoreboard players set @s action 0

        # merge entity attributes
        data merge entity @s {{attributes:[{{id:"minecraft:attack_damage",base:0}},{{id:"minecraft:follow_range",base:0}},{{id:"minecraft:movement_speed",base:0}}]}}

        # summon target (for moving)
         summon marker ~ ~ ~ {{Tags:[target,{self.name}_target]}}

        # in-game ai disabler
        summon tadpole ~ ~ ~ {{Silent:1b,Invulnerable:1b,NoAI:1b,Tags:[ai_disabler,{self.name}_disabler],active_effects:[{{id:"minecraft:invisibility",amplifier:0,duration:-1,show_particles:0b}}],attributes:[{{id:"minecraft:armor",base:100}},{{id:"minecraft:scale",base:0}}]}}
        ride @n[type=bat, tag={self.name}_disabler, distance=..1] mount @s

        # init the observables for this entity
        function ai:ai_modules/{module_name}/{self.name}/fetch
        """

    @mc.mcfunction
    def on_death(self, **kwargs):
        return f"""
    kill @e[limit=1,type=marker, tag={self.name}_target]
    kill @e[limit=1,type=tadpole, tag={self.name}_disabler]
    """


class PlayerLike(Mob):

    def __init__(
            self,
            jump_height=1.,
            base_attack_range=3,
            base_attack_damage=2,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.jump_height = jump_height
        self.base_attack_range = base_attack_range
        self.base_attack_damage = base_attack_damage

    @mc.mcfunction
    def setup(self, module_name, **kwargs):
        setup = super().setup(module_name, **kwargs)
        setup += f"data modify storage ai {self.name}.attack_attributes set value {{damage:{self.base_attack_damage}, crit_damage:{round(self.base_attack_damage * 1.51)}}}"
        return setup

    @mc.toggle(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={mob.name}_target] run return run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION * mob.movement_speed :.5f}"}}
        execute as @e[limit=1, type=marker, tag={mob.name}_target] run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION_AIR * mob.movement_speed :.5f}"}}
        """
    )
    def sprint(self, **kwargs):
        accel = Constants.SPRINT_ACCELERATION if self.on_ground else Constants.SPRINT_ACCELERATION_AIR
        self.apply_accel(accel * self.movement_speed, self.rot)

    @mc.toggle(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={mob.name}_target] run return run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION * mob.movement_speed :.5f}"}}
        execute as @e[limit=1, type=marker, tag={mob.name}_target] run function ai:move {{target:"^ ^ ^{Constants.SPRINT_ACCELERATION_AIR * mob.movement_speed :.5f}"}}
        """
    )
    def sprint(self, **kwargs):
        accel = Constants.SPRINT_ACCELERATION if self.on_ground else Constants.SPRINT_ACCELERATION_AIR
        self.apply_accel(accel * self.movement_speed, self.rot)

    @mc.toggle(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={mob.name}_target] run return run function ai:move {{target:"^{Constants.WALK_ACCELERATION * mob.movement_speed :.5f} ^ ^"}}
        execute as @e[limit=1, type=marker, tag={mob.name}_target] run function ai:move {{target:"^{Constants.WALK_ACCELERATION_AIR * mob.movement_speed :.5f} ^ ^"}}
        """
    )
    def strife_left(self, **kwargs):
        accel = Constants.WALK_ACCELERATION if self.on_ground else Constants.WALK_ACCELERATION_AIR
        left = self.rot + np.pi * 0.5
        self.apply_accel(accel * self.movement_speed, left)

    @mc.toggle(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={mob.name}_target] run return run function ai:move {{target:"^-{Constants.WALK_ACCELERATION * mob.movement_speed :.5f} ^ ^"}}
        execute as @e[limit=1, type=marker, tag={mob.name}_target] run function ai:move {{target:"^-{Constants.WALK_ACCELERATION_AIR * mob.movement_speed :.5f} ^ ^"}}
        """
    )
    def strife_right(self, **kwargs):
        accel = Constants.WALK_ACCELERATION if self.on_ground else Constants.WALK_ACCELERATION_AIR
        right = self.rot - np.pi * 0.5
        self.apply_accel(accel * self.movement_speed, right)

    @mc.toggle(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=on_ground] as @e[limit=1, type=marker, tag={mob.name}_target] run return run function ai:move {{target:"^ ^ ^-{Constants.WALK_ACCELERATION * mob.movement_speed :.5f}"}}
        execute as @e[limit=1, type=marker, tag={mob.name}_target] run function ai:move {{target:"^ ^ ^-{Constants.WALK_ACCELERATION_AIR * mob.movement_speed :.5f}"}}
        """
    )
    def walk_back(self, **kwargs):
        accel = Constants.WALK_ACCELERATION if self.on_ground else Constants.WALK_ACCELERATION_AIR
        direction = self.rot + np.pi
        self.apply_accel(accel * self.movement_speed, direction)

    @mc.action(
        mc_function=lambda mob, **kwargs: f"""
        execute if entity @s[tag=!on_ground] run return fail
        # jump
        scoreboard players add @s vy {round(50000 * mob.jump_height)}
        
        # velocity boost 
        execute if score @s toggled matches {mob.jump.index} run function ai:jump_boost
        """
    )
    def jump(self, **kwargs):
        if not self.on_ground or self.vy > 0:
            return
        self.vy += self.jump_height * Constants.JUMP_FORCE
        if self.toggled is not None and self.actions[self.toggled].name == "sprint":
            self.apply_accel(Constants.JUMP_BOOST, self.rot)

    @mc.action(
        mc_function=lambda mob, **kwargs: f"rotate @s ~{mob.rot_speed * 180 / np.pi:.5f} ~",
    )
    def rotate_left(self, **kwargs):
        self.new_rot = (self.rot + self.rot_speed) % (2 * np.pi)

    @mc.action(
        mc_function=lambda mob, **kwargs: f"rotate @s ~-{mob.rot_speed * 180 / np.pi:.5f} ~",
    )
    def rotate_right(self, **kwargs):
        self.new_rot = (self.rot - self.rot_speed) % (2 * np.pi)

    @mc.action(
        mc_function=lambda mob, **kwargs: f"rotate @s ~{4 * mob.rot_speed * 180 / np.pi:.5f} ~",
        cost=0.05
    )
    def rotate_hard_left(self, **kwargs):
        self.new_rot = (self.rot + 4 * self.rot_speed) % (2 * np.pi)

    @mc.action(
        mc_function=lambda mob, **kwargs: f"rotate @s ~-{4 * mob.rot_speed * 180 / np.pi:.5f} ~",
        cost=0.05
    )
    def rotate_hard_right(self, **kwargs):
        self.new_rot = (self.rot - 4 * self.rot_speed) % (2 * np.pi)

    @mc.action(mc_function=lambda mob, **kwargs: f"""
    # this applies to only human players
    # clearly out of range
    execute unless entity @p[distance=..{mob.base_attack_range + 0.4}] run return fail
    
    # compute distance between attacker and nearest human player
    data modify storage ai atk set value {{x:[0,0,0], y:[0,0,0]}}
    execute store result storage ai atk.x[0] run scoreboard players get @s x
    execute store result storage ai atk.x[2] run scoreboard players get @s z
    scoreboard players operation #tmp y = @s y
    # offset to eyes of entity
    scoreboard players add #tmp y {round(mob.eye_height * 100000)}
    execute store result storage ai atk.x[1] double 1 run scoreboard players get #tmp y
    
    # compute distance from head
    execute as @p run function ai:player_head_to_input
    function gm:distance with storage ai atk
    execute store result score #dist attack_dist run data get storage gm:io out 1
    scoreboard player set #threshold attack_dist {round((mob.base_attack_range+0.3) * 100000)}
    data modify storage ai attack.attributes set from storage ai {mob.name}.attack_attributes
    data modify storage ai attack.target set string "head"
    execute if score #dist attack_dist matches ..{round((mob.base_attack_range+0.3) * 100000)} if function ai:hurt_player run return 1
    
    # compute distance from feet
    execute as @p run function ai:player_feet_to_input
    function gm:distance with storage ai atk
    execute store result score #dist attack_dist run data get storage gm:io out 1
    data modify storage ai attack.target set string "feet"
    execute if score #dist attack_dist matches ..{round((mob.base_attack_range+0.3) * 100000)} if function ai:hurt_player run return 1
    
    # neutral yaw
    rotate @s ~ 0
    return fail
    """
    )
    def attack(self, opponents, arena, **kwargs):
        nearest = None
        nearest_hit = None
        d = np.inf
        for p in opponents:
            di, hit = self.distance_to_hitbox(p)
            if di < d:
                d = di
                nearest = p
                nearest_hit = hit

        if d > self.base_attack_range:
            return
        blocking = arena.raycast(self.eyes_pos, nearest_hit)

        if blocking is not None:
            return

        looking_at = self.look_pos(self.horizontal_distance(nearest))
        hit_distance = nearest.distance(looking_at)

        if hit_distance < (nearest.collision_box[0] * 0.6):
            damage = self.base_attack_damage
            if self.vy < 0 and not self.on_ground:
                damage *= 1.5  # crit
            nearest.damage(self, self.base_attack_damage)