import os

import pyvista as pv
import numpy as np
from minecraft_ai.mc_types import Player
from typing import List, Dict

class GameRenderer:
    def __init__(self, blocks):
        self.blocks = blocks
        self.pl = pv.Plotter(lighting=None, off_screen=True)
        self.actor_map = {}
        self.player_labels = {}
        self.camera_position = [(12, 20, 12), (5.5, 2, 5.5), (0, 0, 0)]
        self.frames = []
        self._setup_static_scene()

    def _setup_static_scene(self):
        cube = pv.Cube()
        unique_blocks = np.unique(self.blocks)

        for block in unique_blocks:
            if block.name == "air":
                continue

            mask = np.vectorize(lambda x: block == x)(self.blocks)
            mesh = pv.PolyData(np.argwhere(mask))
            glyphs = mesh.glyph(geom=cube)

            texture = pv.read_texture(self._get_texture_path(block.name))
            actor = self.pl.add_mesh(glyphs, texture=texture)
            self.actor_map[f'block_{block.name}'] = actor

        self.pl.enable_3_lights()
        self.pl.background_color = 'w'
        self.pl.camera_position = self.camera_position

    def _get_texture_path(self, block_type):
        path = f"data/textures/block/{block_type}.png"

        if not os.path.exists(path):
            if "_wood" in block_type:
                block_type = block_type.replace("wood", "log")
            elif "slab" in block_type:
                if "oak" in block_type:
                    block_type = block_type.replace("_slab", "_planks")
                else:
                    block_type = block_type.replace("_slab", "")
            elif "stairs" in block_type:
                if "oak" in block_type:
                    block_type = block_type.replace("_stairs", "_planks")
                else:
                    block_type = block_type.replace("_stairs", "")
            else:
                block_type = block_type + "_top"
            path = f"data/textures/block/{block_type}.png"
        return path


    def update_labels(self, i:int, player: Player):
        font_size = 20

        y_offset = i * font_size * 1.15

        label = pv.Text(position=(10, y_offset), name=f"{player.aid}_label")
        label.prop.font_size = font_size
        if player.aid not in self.player_labels:
            self.pl.add_actor(label)
            self.player_labels[player.aid] = label

        label_text = f"HP({player.aid}): {player.health}/{player.max_health}"
        self.player_labels[player.aid].SetInput(label_text)



    def update_scene(self, players: Dict[str, Player]):

        for i, (aid, player) in enumerate(players.items()):
            # Either reuse or load the zombie mesh
            if aid not in self.actor_map:
                mesh = pv.read("data/entities/zombie.gltf")
                mesh = mesh.scale(1)
                texture = pv.read_texture("data/entities/zombie.png")
                actors = []
                for m in mesh.recursive_iterator():
                    actor = self.pl.add_mesh(m, texture=texture)
                    actors.append(actor)
                self.actor_map[aid] = actors
            else:
                actors = self.actor_map[aid]

            # Apply transformation
            position = (player.x - 0.5, player.y - 0.5, player.z - 0.5)
            rotation = player.rot * 180 / np.pi
            for actor in actors:
                actor.SetPosition(position)
                actor.SetOrientation(0, rotation, 0)

                if player.hurt_time > 0:
                    actor.GetProperty().SetColor(1, 0.2, 0.2)
                else:
                    actor.GetProperty().SetColor(1, 1, 1)
            self.update_labels(i, player)


    def render(self, down_scale: int = 2):
        self.pl.render()
        self.frames.append(self.pl.screenshot("/tmp/frame.png", return_img=True)[::down_scale, ::down_scale])