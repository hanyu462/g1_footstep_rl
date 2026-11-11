"""Footstep events (Isaac Lab)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedEnv
  from isaaclab.managers import EventTermCfg, SceneEntityCfg


class randomize_feet_friction(ManagerTermBase):
  """One friction coefficient per env, shared by every collision shape of the selected bodies (static = dynamic).

  Matches the mjlab foot_friction event (geom_friction, operation "abs", shared_random=True).  The ground material
  has friction 1 with combine mode "multiply", so the contact friction equals the foot value, as in mjlab where the
  foot geoms have the higher contact priority.  (Isaac Lab's randomize_rigid_body_material samples per shape.)
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.asset = env.scene[asset_cfg.name]
    view = self.asset.root_physx_view
    per_body = [self.asset._physics_sim_view.create_rigid_body_view(p).max_shapes for p in view.link_paths[0]]
    assert sum(per_body) == view.max_shapes, (sum(per_body), view.max_shapes)
    starts = [sum(per_body[:i]) for i in range(len(per_body))]
    self.shape_ids = torch.tensor([s for b in asset_cfg.body_ids for s in range(starts[b], starts[b] + per_body[b])])
    assert len(self.shape_ids) > 0

  def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
               friction_range: tuple[float, float]):
    env_ids = torch.arange(env.scene.num_envs) if env_ids is None else env_ids.cpu()
    mats = self.asset.root_physx_view.get_material_properties()                  # (N, shapes, 3) on cpu
    mu = sample_uniform(*friction_range, (len(env_ids), 1), "cpu")
    mats[env_ids[:, None], self.shape_ids[None, :], 0] = mu
    mats[env_ids[:, None], self.shape_ids[None, :], 1] = mu
    self.asset.root_physx_view.set_material_properties(mats, env_ids)
