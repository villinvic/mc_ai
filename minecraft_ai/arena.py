import os
from enum import Enum
from typing import List, Dict, NamedTuple
import numpy as np
import nbtlib
from collections import defaultdict
import pyvista as pv

from minecraft_ai.mc_types import Player
from minecraft_ai.view import GameRenderer

class Arena:
    def __init__(self, layout: str):

        self.dx, self.dy, self.dz, self.blocks = load(layout)

        self.center = (self.dx /2, 1, self.dz / 2)

        self.renderer = GameRenderer(self.blocks)

    def render(
            self,
            players: Dict[str, Player],
    ):
        self.renderer.update_scene(players)
        self.renderer.render()

    def handle_collision(
            self,
            player: Player,
            x, y, z,
            step = 0.03
    ):
        # Ignore player hitbox collisions ? we probably can
        # returns "clipped" position

        # increment progressively along vector
        # if block met for some x/y/z set motion to 0
        # keep going until end of vector or if blocked along all axes

        # feet collision is 0,3125 blocks wide


        # TODO: handle stairs, slabs

        dx = x - player.x
        dy = y - player.y
        dz = z - player.z

        def is_blocking(px, py, pz, axis):
            """
            Checks if any block inside the player's bounding box at (px, py, pz)
            is solid. px/pz are the center of the box on the ground, py is at feet level.
            """
            width_x = player.collision_box[0]
            width_z = player.collision_box[2]
            height = player.collision_box[1]

            min_x = px - width_x / 2
            max_x = px + width_x / 2
            min_y = py
            max_y = py + height
            min_z = pz - width_z / 2
            max_z = pz + width_z / 2

            # Iterate over all blocks inside the AABB
            target = [px, py, pz][axis]
            origin = [player.x, player.y, player.z][axis]
            for ix in range(int(np.floor(min_x)), int(np.ceil(max_x))):
                for iy in range(int(np.floor(min_y)), int(np.ceil(max_y))):
                    for iz in range(int(np.floor(min_z)), int(np.ceil(max_z))):
                        if (0 <= ix < self.blocks.shape[0] and
                                0 <= iy < self.blocks.shape[1] and
                                0 <= iz < self.blocks.shape[2]):
                            block = self.blocks[ix, iy, iz]
                            if block.name != "air":
                                # position when against this block:
                                v = target - origin
                                if axis == 0:
                                    if v > 0:
                                        return ix - width_x/2
                                    else:
                                        return ix + 1 + width_x/2
                                elif axis == 1:
                                    if v > 0:
                                        return iy - height
                                    else:
                                        return iy + 1
                                else:
                                    if v > 0:
                                        return iz -width_z / 2
                                    else:
                                        return iz + 1 + width_z / 2
            return None

        pos = np.array([player.x, player.y, player.z], dtype=float)
        target = np.array([x, y, z], dtype=float)
        delta = target - pos
        dist = np.linalg.norm(delta)

        if dist < 1e-6:
            return tuple(pos)  # No movement

        direction = delta / dist
        steps = int(dist / step) + 1
        last_free = pos

        free_axes = {
            'x': 0,
            'y': 1,
            'z': 2
        }
        met_ground = False
        for i in range(steps):
            current = last_free.copy()
            to_pop = set()
            for axis_name, axis in free_axes.items():
                new = current.copy()

                #if abs(new[axis] + direction[axis] * step) < abs(target[axis]):
                if i == steps - 1:
                    new[axis] = target[axis]
                else:
                    new[axis] = new[axis] + direction[axis] * step
                # else:
                #     new[axis] = target[axis]

                collided = is_blocking(*new, axis)
                if collided is None:
                    last_free[axis] = new[axis]
                else:
                    # clip to block

                    # TODO: how does this reset ?
                    if axis_name == "y" and delta[axis] < 0: # falling:
                        met_ground = True
                    #last_free[axis] = collided
                    setattr(player, f"v{axis_name}", 0)
                    to_pop.add(axis_name)

            for axis in to_pop:
                free_axes.pop(axis)
            if len(free_axes) == 0:
                break


        player.on_ground = met_ground
        return tuple(last_free.tolist())

    def raycast(self, start, end, max_steps=5):
        if isinstance(start, Player):
            start = start.x, start.y, start.z
        if isinstance(end, Player):
            end = end.x, end.y, end.z
        x0, y0, z0 = start
        x1, y1, z1 = end

        # Direction of the ray
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        # Normalize direction
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0:
            return None  # start and end are the same

        dx /= length
        dy /= length
        dz /= length

        # Initial voxel
        ix, iy, iz = int(np.floor(x0)), int(np.floor(y0)), int(np.floor(z0))

        # Step direction
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        step_z = 1 if dz > 0 else -1

        # Distance to first voxel boundary
        tx = ((ix + (step_x > 0)) - x0) / dx if dx != 0 else float('inf')
        ty = ((iy + (step_y > 0)) - y0) / dy if dy != 0 else float('inf')
        tz = ((iz + (step_z > 0)) - z0) / dz if dz != 0 else float('inf')

        # Distance to next voxel boundary
        dtx = abs(1 / dx) if dx != 0 else float('inf')
        dty = abs(1 / dy) if dy != 0 else float('inf')
        dtz = abs(1 / dz) if dz != 0 else float('inf')

        for _ in range(max_steps):
            # Check if current voxel blocks
            if (0 <= ix < len(self.blocks) and
                    0 <= iy < len(self.blocks[0]) and
                    0 <= iz < len(self.blocks[0][0])):
                block = self.blocks[ix][iy][iz]
                if block.name != "air":
                    return (ix, iy, iz)  # hit a blocking block

            # Step to next voxel
            if tx < ty and tx < tz:
                tx += dtx
                ix += step_x
            elif ty < tz:
                ty += dty
                iy += step_y
            else:
                tz += dtz
                iz += step_z

            # Stop if we passed the end point
            if ((ix - x1) * dx > 0 or
                    (iy - y1) * dy > 0 or
                    (iz - z1) * dz > 0):
                break

        return None  # No block in the way



class SlabType(Enum):
    BOTTOM = 0
    TOP= 1


class Orientation(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Block(NamedTuple):
    name: str
    slab_type: None | SlabType  = None
    orientation: None | Orientation = None
    waterlogged: bool = False



def load(layout):
    nbt_struct = nbtlib.load(f"data/arenas/{layout}.nbt")
    nbt_blocks = nbt_struct["blocks"]
    nbt_palette = nbt_struct["palette"]

    # infer size of schematic
    dx, dy, dz = nbt_struct["size"]

    schematic = np.empty(
        (dx, dy, dz),
        dtype=object,
    )

    for block in nbt_blocks:
        info = nbt_palette[block['state']]

        input_dict = dict(
            name=info['Name'].replace('minecraft:', "")
        )

        if 'Properties' in info:
            for field, value in info['Properties'].items():
                # TODO: reversed stairs ?
                if "half" == field:
                    if 'type' in info['Properties'] and info['Properties']['type'] == 'double':
                        continue
                    if value == "bottom":
                        input_dict["slab_type"] = SlabType.BOTTOM
                    else:
                        input_dict["slab_type"] = SlabType.TOP
                elif "facing" == field:
                    input_dict["orientation"] = Orientation[value.upper()]
                elif "waterlogged" == field and value == "true":
                    input_dict["orientation"] = True
                #else:
                #    print(field, value)

        schematic[*block['pos']] = Block(**input_dict)

    return dx, dy, dz, schematic



if __name__ == '__main__':
    arena = Arena("arena_test")

    p1 = Player("0")
    p1.x = 11/2
    p1.y = 1
    p1.z = 11/2
    p1.health = 10
    p1.rot = 0

    p2 = Player("1")
    p2.x = 11/2 + 1
    p2.y = 1
    p2.z = 11/2 + 2
    p2.rot = 0
    p2.hurt_time = 5

    players = {
        0: p1,
        1: p2
    }

    arena.render(players, name="test.png")
