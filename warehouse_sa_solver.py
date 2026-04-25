from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm

import rectirization_py


EPS = 1e-8
OBJECTIVE_CAP = 10.0


@dataclass
class CaseData:
    warehouse_polygon: list[tuple[float, float]]
    bay_types: list[dict[str, float]]
    obstacles: list[tuple[float, float, float, float]]
    ceiling_profile: list[tuple[float, float]]


@dataclass
class PreprocessedCase:
    warehouse_boxes: torch.Tensor
    obstacle_boxes: torch.Tensor
    warehouse_area: float
    warehouse_bbox: torch.Tensor
    ceiling_boxes: torch.Tensor
    ceiling_heights: torch.Tensor
    bay_widths: torch.Tensor
    bay_depths: torch.Tensor
    bay_gap_depths: torch.Tensor
    bay_heights: torch.Tensor
    bay_capacities: torch.Tensor
    bay_costs: torch.Tensor
    bay_polygons_local: torch.Tensor
    access_polygons_local: torch.Tensor
    access_bounds_local: torch.Tensor
    bay_areas: torch.Tensor
    access_areas: torch.Tensor
    rot: torch.Tensor
    inv_rot: torch.Tensor
    objective_scale: float
    empty_state_objective: float
    n_slots: int
    device: torch.device


@dataclass
class AnnealingState:
    active: torch.Tensor
    type_id: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    theta_idx: torch.Tensor


@dataclass
class Evaluation:
    score: torch.Tensor
    objective_raw: torch.Tensor
    objective_norm: torch.Tensor
    area_used: torch.Tensor
    access_area: torch.Tensor
    total_capacity: torch.Tensor
    total_cost: torch.Tensor
    pair_overlap: torch.Tensor
    obstacle_overlap: torch.Tensor
    exterior_overlap: torch.Tensor
    ceiling_overlap: torch.Tensor
    feasible: torch.Tensor


@dataclass
class SolverParams:
    n_chains: int = 1024
    n_steps: int = 200
    target_initial_acceptance: float = 0.8
    final_temperature_ratio: float = 0.02
    objective_weight: float = 1.0
    penalty_weight: float = 4.0
    w_pair: float = 1.0
    w_obstacle: float = 1.0
    w_exterior: float = 1.0
    w_ceiling: float = 1.0
    translate_step_fraction: float = 0.05
    toggle_probability: float = 0.05
    seed: int | None = None
    temperature_samples: int = 8
    show_progress: bool = False
    snapshot_every_steps: int | None = None


@dataclass
class SnapshotRecord:
    step_index: int
    temperature: float
    acceptance_rate: float
    best_state: AnnealingState
    best_evaluation: Evaluation


@dataclass
class SolveResult:
    best_state: AnnealingState
    best_evaluation: Evaluation
    initial_evaluation: Evaluation
    final_evaluation: Evaluation
    temperatures: torch.Tensor
    acceptance_rates: torch.Tensor
    snapshots: list[SnapshotRecord]
    device: torch.device


CSV_SCHEMAS = {
    "warehouse": ["x", "y"],
    "types_of_bays": [
        "type_id",
        "width",
        "depth",
        "height",
        "min_spacing",
        "capacity",
        "cost",
    ],
    "obstacles": ["x", "y", "width", "height"],
    "ceiling": ["x", "height"],
}


def _parse_value(value: str) -> float:
    value = value.strip()
    try:
        return float(int(value))
    except ValueError:
        return float(value)


def _read_csv_rows(csv_path: Path, columns: list[str]) -> list[dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing expected file: {csv_path}")

    with csv_path.open(newline="") as file:
        reader = csv.reader(file, skipinitialspace=True)
        return [
            {column: _parse_value(value) for column, value in zip(columns, row)}
            for row in reader
            if row
        ]


def load_case(case_dir: str | Path) -> CaseData:
    case_path = Path(case_dir)
    warehouse_rows = _read_csv_rows(case_path / "warehouse.csv", CSV_SCHEMAS["warehouse"])
    bay_rows = _read_csv_rows(case_path / "types_of_bays.csv", CSV_SCHEMAS["types_of_bays"])
    obstacle_rows = _read_csv_rows(case_path / "obstacles.csv", CSV_SCHEMAS["obstacles"])
    ceiling_rows = _read_csv_rows(case_path / "ceiling.csv", CSV_SCHEMAS["ceiling"])

    warehouse_polygon = [(row["x"], row["y"]) for row in warehouse_rows]
    bay_types = [
        {
            "type_id": row["type_id"],
            "width": row["width"],
            "depth": row["depth"],
            "height": row["height"],
            "gap_depth": row["min_spacing"],
            "capacity": row["capacity"],
            "cost": row["cost"],
        }
        for row in bay_rows
    ]
    obstacles = [
        (row["x"], row["y"], row["width"], row["height"])
        for row in obstacle_rows
    ]
    ceiling_profile = [(row["x"], row["height"]) for row in ceiling_rows]

    return CaseData(
        warehouse_polygon=warehouse_polygon,
        bay_types=bay_types,
        obstacles=obstacles,
        ceiling_profile=ceiling_profile,
    )


def _rect_to_corners(x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def _make_box_tensor(boxes: list[tuple[float, float, float, float]], device: torch.device) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 4, 2), dtype=torch.float32, device=device)
    corners = [_rect_to_corners(*box) for box in boxes]
    return torch.tensor(corners, dtype=torch.float32, device=device)


def _shoelace_area(points: torch.Tensor, counts: torch.Tensor | None = None) -> torch.Tensor:
    """Compute polygon area with the shoelace formula on a fixed-capacity vertex buffer."""
    if points.shape[-2] == 0:
        return torch.zeros(points.shape[:-2], dtype=points.dtype, device=points.device)

    if counts is None:
        shifted = torch.roll(points, shifts=-1, dims=-2)
        cross = points[..., 0] * shifted[..., 1] - points[..., 1] * shifted[..., 0]
        return 0.5 * torch.abs(cross.sum(dim=-1))

    max_vertices = points.shape[-2]
    vertex_indices = torch.arange(max_vertices, device=points.device)
    valid = vertex_indices.view(*((1,) * (points.ndim - 2)), -1) < counts.unsqueeze(-1)
    next_indices = torch.where(
        vertex_indices.view(*((1,) * (points.ndim - 2)), -1) + 1 < counts.unsqueeze(-1),
        vertex_indices.view(*((1,) * (points.ndim - 2)), -1) + 1,
        torch.zeros_like(valid, dtype=torch.long),
    )
    next_points = points.gather(-2, next_indices.unsqueeze(-1).expand(*next_indices.shape, 2))
    cross = points[..., 0] * next_points[..., 1] - points[..., 1] * next_points[..., 0]
    cross = torch.where(valid, cross, torch.zeros_like(cross))
    return 0.5 * torch.abs(cross.sum(dim=-1))


def preprocess_case(case_data: CaseData, device: str | torch.device) -> PreprocessedCase:
    device = torch.device(device)
    warehouse_boxes_raw = rectirization_py.rectirize(case_data.warehouse_polygon)
    warehouse_boxes = _make_box_tensor(
        [(box[0], box[1], box[2], box[3]) for box in warehouse_boxes_raw], device=device
    )
    warehouse_polygon = torch.tensor(case_data.warehouse_polygon, dtype=torch.float32, device=device)

    xs = warehouse_polygon[:, 0]
    ys = warehouse_polygon[:, 1]
    min_x, max_x = float(xs.min().item()), float(xs.max().item())
    min_y, max_y = float(ys.min().item()), float(ys.max().item())
    warehouse_area = float((torch.tensor([box[2] * box[3] for box in warehouse_boxes_raw])).sum().item())
    warehouse_bbox = torch.tensor([min_x, max_x, min_y, max_y], dtype=torch.float32, device=device)
    sorted_ceiling = sorted(case_data.ceiling_profile, key=lambda item: item[0])
    if not sorted_ceiling:
        raise ValueError("ceiling_profile must not be empty")
    ceiling_boxes_list: list[tuple[float, float, float, float]] = []
    ceiling_heights: list[float] = []
    for index, (x_start, height) in enumerate(sorted_ceiling):
        x_end = sorted_ceiling[index + 1][0] if index + 1 < len(sorted_ceiling) else max_x
        start = max(x_start, min_x)
        end = min(x_end, max_x)
        if end > start:
            ceiling_boxes_list.append((start, min_y, end - start, max_y - min_y))
            ceiling_heights.append(height)
    ceiling_boxes = _make_box_tensor(ceiling_boxes_list, device=device)
    ceiling_heights_tensor = torch.tensor(ceiling_heights, dtype=torch.float32, device=device)

    obstacles = _make_box_tensor(case_data.obstacles, device=device)

    bay_widths = torch.tensor([row["width"] for row in case_data.bay_types], dtype=torch.float32, device=device)
    bay_depths = torch.tensor([row["depth"] for row in case_data.bay_types], dtype=torch.float32, device=device)
    bay_gap_depths = torch.tensor([row["gap_depth"] for row in case_data.bay_types], dtype=torch.float32, device=device)
    bay_heights = torch.tensor([row["height"] for row in case_data.bay_types], dtype=torch.float32, device=device)
    bay_capacities = torch.tensor([row["capacity"] for row in case_data.bay_types], dtype=torch.float32, device=device)
    bay_costs = torch.tensor([row["cost"] for row in case_data.bay_types], dtype=torch.float32, device=device)

    bay_xmin = -0.5 * bay_widths
    bay_xmax = 0.5 * bay_widths
    bay_ymin = -0.5 * bay_depths
    bay_ymax = 0.5 * bay_depths
    access_xmin = bay_xmin
    access_xmax = bay_xmax
    access_ymin = bay_ymin
    access_ymax = 0.5 * bay_depths + bay_gap_depths

    bay_polygons_local = torch.stack(
        [
            torch.stack([bay_xmin, bay_ymin], dim=-1),
            torch.stack([bay_xmax, bay_ymin], dim=-1),
            torch.stack([bay_xmax, bay_ymax], dim=-1),
            torch.stack([bay_xmin, bay_ymax], dim=-1),
        ],
        dim=1,
    )
    access_polygons_local = torch.stack(
        [
            torch.stack([access_xmin, access_ymin], dim=-1),
            torch.stack([access_xmax, access_ymin], dim=-1),
            torch.stack([access_xmax, access_ymax], dim=-1),
            torch.stack([access_xmin, access_ymax], dim=-1),
        ],
        dim=1,
    )
    access_bounds_local = torch.stack([access_xmin, access_xmax, access_ymin, access_ymax], dim=-1)

    bay_areas = bay_widths * bay_depths
    access_areas = bay_widths * (bay_depths + bay_gap_depths)

    angles_deg = torch.arange(0, 360, 15, dtype=torch.float32, device=device)
    angles_rad = torch.deg2rad(angles_deg)
    cos = torch.cos(angles_rad)
    sin = torch.sin(angles_rad)
    rot = torch.stack(
        [
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ],
        dim=-2,
    )
    inv_rot = rot.transpose(-1, -2).contiguous()

    min_bay_area = float(torch.min(bay_areas).item())
    n_slots = min(128, math.ceil(1.25 * warehouse_area / max(min_bay_area, EPS)))
    ratio = bay_costs / torch.clamp(bay_capacities, min=EPS)
    objective_scale = max(float(torch.max(ratio.pow(2)).item()), EPS)
    empty_state_objective = objective_scale * OBJECTIVE_CAP

    preprocessed = PreprocessedCase(
        warehouse_boxes=warehouse_boxes,
        obstacle_boxes=obstacles,
        warehouse_area=warehouse_area,
        warehouse_bbox=warehouse_bbox,
        ceiling_boxes=ceiling_boxes,
        ceiling_heights=ceiling_heights_tensor,
        bay_widths=bay_widths,
        bay_depths=bay_depths,
        bay_gap_depths=bay_gap_depths,
        bay_heights=bay_heights,
        bay_capacities=bay_capacities,
        bay_costs=bay_costs,
        bay_polygons_local=bay_polygons_local,
        access_polygons_local=access_polygons_local,
        access_bounds_local=access_bounds_local,
        bay_areas=bay_areas,
        access_areas=access_areas,
        rot=rot,
        inv_rot=inv_rot,
        objective_scale=objective_scale,
        empty_state_objective=empty_state_objective,
        n_slots=n_slots,
        device=device,
    )
    return preprocessed


def _gather_type_tensor(preprocessed: PreprocessedCase, tensor: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
    flat = tensor.index_select(0, type_id.reshape(-1))
    return flat.reshape(*type_id.shape, *tensor.shape[1:])


def _polygon_world(local_polygons: torch.Tensor, rot: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    rotated = torch.matmul(local_polygons, rot.transpose(-1, -2))
    return rotated + centers.unsqueeze(-2)


def _transform_world_to_local(points: torch.Tensor, centers: torch.Tensor, inv_rot: torch.Tensor) -> torch.Tensor:
    shifted = points - centers.unsqueeze(-2)
    return torch.matmul(shifted, inv_rot.transpose(-1, -2))


def _clip_against_axis_boundary(
    points: torch.Tensor,
    counts: torch.Tensor,
    bound: torch.Tensor,
    axis: int,
    keep_greater: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_vertices = points.shape[-2]
    vertex_indices = torch.arange(max_vertices, device=points.device)
    valid = vertex_indices.view(1, -1) < counts.unsqueeze(-1)

    next_indices = torch.where(
        vertex_indices.view(1, -1) + 1 < counts.unsqueeze(-1),
        vertex_indices.view(1, -1) + 1,
        torch.zeros((points.shape[0], max_vertices), dtype=torch.long, device=points.device),
    )
    nxt = points.gather(1, next_indices.unsqueeze(-1).expand(-1, -1, 2))

    if keep_greater:
        current_inside = points[..., axis] >= bound.unsqueeze(-1) - EPS
        next_inside = nxt[..., axis] >= bound.unsqueeze(-1) - EPS
    else:
        current_inside = points[..., axis] <= bound.unsqueeze(-1) + EPS
        next_inside = nxt[..., axis] <= bound.unsqueeze(-1) + EPS

    denom_raw = nxt[..., axis] - points[..., axis]
    denom = torch.where(torch.abs(denom_raw) < EPS, torch.full_like(denom_raw, EPS), denom_raw)
    t = (bound.unsqueeze(-1) - points[..., axis]) / denom
    intersections = points + t.unsqueeze(-1) * (nxt - points)

    emit_intersection = valid & (current_inside ^ next_inside)
    emit_next = valid & next_inside
    emit_count = emit_intersection.to(torch.long) + emit_next.to(torch.long)
    base_pos = torch.cumsum(emit_count, dim=1) - emit_count

    out_points = torch.zeros_like(points)
    row_ids = torch.arange(points.shape[0], device=points.device).unsqueeze(-1).expand(-1, max_vertices)

    out_points[row_ids[emit_intersection], base_pos[emit_intersection]] = intersections[emit_intersection]
    next_pos = base_pos + emit_intersection.to(torch.long)
    out_points[row_ids[emit_next], next_pos[emit_next]] = nxt[emit_next]

    out_counts = emit_count.sum(dim=1)
    return out_points, out_counts


def _clip_quad_to_axis_rect(local_polygons: torch.Tensor, rect_bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_polygons = local_polygons.reshape(-1, local_polygons.shape[-2], 2)
    broadcast_bounds = torch.broadcast_to(rect_bounds, (*local_polygons.shape[:-2], 4))
    flat_bounds = broadcast_bounds.reshape(-1, 4)

    max_vertices = 8
    points = torch.zeros((flat_polygons.shape[0], max_vertices, 2), dtype=flat_polygons.dtype, device=flat_polygons.device)
    points[:, : flat_polygons.shape[1]] = flat_polygons
    counts = torch.full((flat_polygons.shape[0],), flat_polygons.shape[1], dtype=torch.long, device=flat_polygons.device)

    for axis, keep_greater, bound_index in ((0, True, 0), (0, False, 1), (1, True, 2), (1, False, 3)):
        points, counts = _clip_against_axis_boundary(points, counts, flat_bounds[:, bound_index], axis, keep_greater)

    out_shape = (*local_polygons.shape[:-2], max_vertices, 2)
    return points.reshape(out_shape), counts.reshape(local_polygons.shape[:-2])


def _intersection_area_quad_rect(local_polygons: torch.Tensor, rect_bounds: torch.Tensor) -> torch.Tensor:
    clipped_points, clipped_counts = _clip_quad_to_axis_rect(local_polygons, rect_bounds)
    return _shoelace_area(clipped_points, clipped_counts)


def _pairwise_local_polygons(world_polygons: torch.Tensor, centers: torch.Tensor, inv_rot: torch.Tensor) -> torch.Tensor:
    return _transform_world_to_local(world_polygons.unsqueeze(1), centers.unsqueeze(2), inv_rot.unsqueeze(2))


def _world_boxes_in_slot_local_frame(world_boxes: torch.Tensor, centers: torch.Tensor, inv_rot: torch.Tensor) -> torch.Tensor:
    return _transform_world_to_local(world_boxes.unsqueeze(0).unsqueeze(0), centers.unsqueeze(2), inv_rot.unsqueeze(2))


def _sum_intersection_with_boxes(
    world_boxes: torch.Tensor,
    centers: torch.Tensor,
    inv_rot: torch.Tensor,
    rect_bounds: torch.Tensor,
) -> torch.Tensor:
    if world_boxes.shape[0] == 0:
        return torch.zeros(rect_bounds.shape[:-1], dtype=rect_bounds.dtype, device=rect_bounds.device)

    boxes_local = _world_boxes_in_slot_local_frame(world_boxes, centers, inv_rot)
    intersections = _intersection_area_quad_rect(boxes_local, rect_bounds.unsqueeze(2))
    return intersections.sum(dim=-1)


def _normalize_overlap(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    safe_scale = torch.clamp(scale, min=EPS)
    return 1.0 - torch.exp(-value / safe_scale)


def _clone_state(state: AnnealingState) -> AnnealingState:
    return AnnealingState(
        active=state.active.clone(),
        type_id=state.type_id.clone(),
        x=state.x.clone(),
        y=state.y.clone(),
        theta_idx=state.theta_idx.clone(),
    )


def _state_where(mask: torch.Tensor, lhs: AnnealingState, rhs: AnnealingState) -> AnnealingState:
    mask_long = mask.unsqueeze(-1)
    return AnnealingState(
        active=torch.where(mask_long, lhs.active, rhs.active),
        type_id=torch.where(mask_long, lhs.type_id, rhs.type_id),
        x=torch.where(mask, lhs.x, rhs.x),
        y=torch.where(mask, lhs.y, rhs.y),
        theta_idx=torch.where(mask_long, lhs.theta_idx, rhs.theta_idx),
    )


def _eval_where(mask: torch.Tensor, lhs: Evaluation, rhs: Evaluation) -> Evaluation:
    return Evaluation(
        score=torch.where(mask, lhs.score, rhs.score),
        objective_raw=torch.where(mask, lhs.objective_raw, rhs.objective_raw),
        objective_norm=torch.where(mask, lhs.objective_norm, rhs.objective_norm),
        area_used=torch.where(mask, lhs.area_used, rhs.area_used),
        access_area=torch.where(mask, lhs.access_area, rhs.access_area),
        total_capacity=torch.where(mask, lhs.total_capacity, rhs.total_capacity),
        total_cost=torch.where(mask, lhs.total_cost, rhs.total_cost),
        pair_overlap=torch.where(mask, lhs.pair_overlap, rhs.pair_overlap),
        obstacle_overlap=torch.where(mask, lhs.obstacle_overlap, rhs.obstacle_overlap),
        exterior_overlap=torch.where(mask, lhs.exterior_overlap, rhs.exterior_overlap),
        ceiling_overlap=torch.where(mask, lhs.ceiling_overlap, rhs.ceiling_overlap),
        feasible=torch.where(mask, lhs.feasible, rhs.feasible),
    )


def evaluate_state_gpu(state: AnnealingState, preprocessed: PreprocessedCase, params: SolverParams | None = None) -> Evaluation:
    if params is None:
        params = SolverParams()

    device = preprocessed.device
    active = state.active.squeeze(-1).to(dtype=torch.float32)
    type_id = state.type_id.squeeze(-1)
    theta_idx = state.theta_idx.squeeze(-1)
    centers = torch.stack([state.x, state.y], dim=-1)

    bay_local = _gather_type_tensor(preprocessed, preprocessed.bay_polygons_local, type_id)
    access_bounds = _gather_type_tensor(preprocessed, preprocessed.access_bounds_local, type_id)
    bay_area_slot = _gather_type_tensor(preprocessed, preprocessed.bay_areas, type_id)
    access_area_slot = _gather_type_tensor(preprocessed, preprocessed.access_areas, type_id)
    bay_heights = _gather_type_tensor(preprocessed, preprocessed.bay_heights, type_id)
    bay_capacities = _gather_type_tensor(preprocessed, preprocessed.bay_capacities, type_id)
    bay_costs = _gather_type_tensor(preprocessed, preprocessed.bay_costs, type_id)

    rot = preprocessed.rot.index_select(0, theta_idx.reshape(-1)).reshape(*theta_idx.shape, 2, 2)
    inv_rot = preprocessed.inv_rot.index_select(0, theta_idx.reshape(-1)).reshape(*theta_idx.shape, 2, 2)

    bay_world = _polygon_world(bay_local, rot, centers)

    area_used = (bay_area_slot * active).sum(dim=1)
    access_area = (access_area_slot * active).sum(dim=1)
    total_capacity = (bay_capacities * active).sum(dim=1)
    total_cost = (bay_costs * active).sum(dim=1)

    contained = _sum_intersection_with_boxes(preprocessed.warehouse_boxes, centers, inv_rot, access_bounds)
    contained = torch.clamp(contained, min=0.0)
    contained = torch.minimum(contained, access_area_slot + 1e-4)
    exterior_overlap = (active * torch.clamp(access_area_slot - contained, min=0.0)).sum(dim=1)

    obstacle_hits = _sum_intersection_with_boxes(preprocessed.obstacle_boxes, centers, inv_rot, access_bounds)
    obstacle_overlap = (active * obstacle_hits).sum(dim=1)

    if preprocessed.ceiling_boxes.shape[0] > 0:
        ceiling_boxes_local = _world_boxes_in_slot_local_frame(preprocessed.ceiling_boxes, centers, inv_rot)
        ceiling_area = _intersection_area_quad_rect(ceiling_boxes_local, access_bounds.unsqueeze(2))
        low_ceiling_mask = (
            preprocessed.ceiling_heights.view(1, 1, -1) < bay_heights.unsqueeze(-1) - EPS
        ).to(torch.float32)
        ceiling_overlap = (active * (ceiling_area * low_ceiling_mask).sum(dim=-1)).sum(dim=1)
    else:
        ceiling_overlap = torch.zeros_like(exterior_overlap)

    pair_local = _pairwise_local_polygons(bay_world, centers, inv_rot)
    pair_area = _intersection_area_quad_rect(pair_local, access_bounds.unsqueeze(2))
    n_slots = type_id.shape[1]
    off_diagonal = (~torch.eye(n_slots, dtype=torch.bool, device=device)).unsqueeze(0).to(dtype=pair_area.dtype)
    pair_mask = active.unsqueeze(1) * active.unsqueeze(2) * off_diagonal
    pair_overlap = (pair_area * pair_mask).sum(dim=(1, 2))

    area_used_ratio = area_used / max(preprocessed.warehouse_area, EPS)
    capacity_safe = torch.clamp(total_capacity, min=EPS)
    objective_defined = total_capacity > EPS
    objective_raw = torch.where(
        objective_defined,
        (total_cost / capacity_safe) ** (2.0 - area_used_ratio),
        torch.full_like(total_capacity, preprocessed.empty_state_objective),
    )
    objective_norm = torch.clamp(objective_raw / max(preprocessed.objective_scale, EPS), max=OBJECTIVE_CAP)

    p_pair = _normalize_overlap(pair_overlap, access_area)
    p_obstacle = _normalize_overlap(obstacle_overlap, access_area)
    p_exterior = _normalize_overlap(exterior_overlap, access_area)
    p_ceiling = _normalize_overlap(ceiling_overlap, access_area)

    penalty_norm = (
        params.w_pair * p_pair
        + params.w_obstacle * p_obstacle
        + params.w_exterior * p_exterior
        + params.w_ceiling * p_ceiling
    )
    score = params.objective_weight * objective_norm + params.penalty_weight * penalty_norm
    feasible = (
        (pair_overlap <= 1e-4)
        & (obstacle_overlap <= 1e-4)
        & (exterior_overlap <= 1e-4)
        & (ceiling_overlap <= 1e-4)
        & objective_defined
    )

    return Evaluation(
        score=score,
        objective_raw=objective_raw,
        objective_norm=objective_norm,
        area_used=area_used,
        access_area=access_area,
        total_capacity=total_capacity,
        total_cost=total_cost,
        pair_overlap=pair_overlap,
        obstacle_overlap=obstacle_overlap,
        exterior_overlap=exterior_overlap,
        ceiling_overlap=ceiling_overlap,
        feasible=feasible,
    )


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _rand(shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def _randint(low: int, high: int, shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randint(low, high, shape, device=device, generator=generator)


def initialize_state(preprocessed: PreprocessedCase, params: SolverParams) -> AnnealingState:
    device = preprocessed.device
    generator = _make_generator(device, params.seed)
    n_chains = params.n_chains
    n_slots = preprocessed.n_slots
    bbox = preprocessed.warehouse_bbox
    x = bbox[0] + (bbox[1] - bbox[0]) * _rand((n_chains, n_slots), device, generator)
    y = bbox[2] + (bbox[3] - bbox[2]) * _rand((n_chains, n_slots), device, generator)
    type_id = _randint(0, preprocessed.bay_widths.shape[0], (n_chains, n_slots), device, generator)
    theta_idx = _randint(0, preprocessed.rot.shape[0], (n_chains, n_slots), device, generator)
    active = torch.ones((n_chains, n_slots), dtype=torch.bool, device=device)
    return AnnealingState(
        active=active.unsqueeze(-1),
        type_id=type_id.unsqueeze(-1),
        x=x,
        y=y,
        theta_idx=theta_idx.unsqueeze(-1),
    )


def propose_mutation(
    state: AnnealingState,
    preprocessed: PreprocessedCase,
    params: SolverParams,
    temperature: float,
    generator: torch.Generator | None = None,
) -> AnnealingState:
    device = preprocessed.device
    candidate = _clone_state(state)
    n_chains, n_slots, _ = state.active.shape
    slot_index = _randint(0, n_slots, (n_chains,), device, generator)
    move_kind = _randint(0, 4, (n_chains,), device, generator)

    bbox = preprocessed.warehouse_bbox
    x_span = float((bbox[1] - bbox[0]).item())
    y_span = float((bbox[3] - bbox[2]).item())
    base_step = params.translate_step_fraction * max(x_span, y_span)
    scaled_step = base_step * max(temperature, EPS)

    rows = torch.arange(n_chains, device=device)

    translate_mask = move_kind == 0
    if torch.any(translate_mask):
        dx = torch.randn((n_chains,), device=device, generator=generator) * scaled_step
        dy = torch.randn((n_chains,), device=device, generator=generator) * scaled_step
        candidate.x[rows[translate_mask], slot_index[translate_mask]] += dx[translate_mask]
        candidate.y[rows[translate_mask], slot_index[translate_mask]] += dy[translate_mask]

    rotate_mask = move_kind == 1
    if torch.any(rotate_mask):
        delta_options = torch.tensor([-2, -1, 1, 2], device=device, dtype=candidate.theta_idx.dtype)
        delta = delta_options[_randint(0, 4, (n_chains,), device, generator)]
        candidate.theta_idx[rows[rotate_mask], slot_index[rotate_mask], 0] = (
            candidate.theta_idx[rows[rotate_mask], slot_index[rotate_mask], 0] + delta[rotate_mask]
        ) % preprocessed.rot.shape[0]

    type_mask = move_kind == 2
    if torch.any(type_mask):
        new_types = _randint(0, preprocessed.bay_widths.shape[0], (n_chains,), device, generator)
        candidate.type_id[rows[type_mask], slot_index[type_mask], 0] = new_types[type_mask]

    toggle_mask = (move_kind == 3) & (_rand((n_chains,), device, generator) <= params.toggle_probability)
    if torch.any(toggle_mask):
        candidate.active[rows[toggle_mask], slot_index[toggle_mask], 0] = ~candidate.active[rows[toggle_mask], slot_index[toggle_mask], 0]

    return candidate


def _temperature_from_deltas(deltas: torch.Tensor, target_acceptance: float) -> float:
    positive = deltas[deltas > EPS]
    if positive.numel() == 0:
        return 1.0
    representative = torch.median(positive).item()
    acceptance = min(max(target_acceptance, 1e-3), 1 - 1e-3)
    return max(-representative / math.log(acceptance), EPS)


def _make_temperature_schedule(t0: float, n_steps: int, final_ratio: float, device: torch.device) -> torch.Tensor:
    if n_steps <= 1:
        return torch.tensor([t0], dtype=torch.float32, device=device)
    tf = max(t0 * final_ratio, EPS)
    exponents = torch.linspace(0.0, 1.0, n_steps, device=device)
    return t0 * torch.exp(exponents * math.log(tf / t0))


def _choose_overall_best(best_eval: Evaluation) -> int:
    feasible = best_eval.feasible
    if feasible.any():
        scores = torch.where(feasible, best_eval.score, torch.full_like(best_eval.score, float("inf")))
    else:
        scores = best_eval.score
    return int(torch.argmin(scores).item())


def _index_state(state: AnnealingState, index: int) -> AnnealingState:
    return AnnealingState(
        active=state.active[index : index + 1].clone(),
        type_id=state.type_id[index : index + 1].clone(),
        x=state.x[index : index + 1].clone(),
        y=state.y[index : index + 1].clone(),
        theta_idx=state.theta_idx[index : index + 1].clone(),
    )


def _index_eval(evaluation: Evaluation, index: int) -> Evaluation:
    return Evaluation(
        score=evaluation.score[index : index + 1].clone(),
        objective_raw=evaluation.objective_raw[index : index + 1].clone(),
        objective_norm=evaluation.objective_norm[index : index + 1].clone(),
        area_used=evaluation.area_used[index : index + 1].clone(),
        access_area=evaluation.access_area[index : index + 1].clone(),
        total_capacity=evaluation.total_capacity[index : index + 1].clone(),
        total_cost=evaluation.total_cost[index : index + 1].clone(),
        pair_overlap=evaluation.pair_overlap[index : index + 1].clone(),
        obstacle_overlap=evaluation.obstacle_overlap[index : index + 1].clone(),
        exterior_overlap=evaluation.exterior_overlap[index : index + 1].clone(),
        ceiling_overlap=evaluation.ceiling_overlap[index : index + 1].clone(),
        feasible=evaluation.feasible[index : index + 1].clone(),
    )


def run_simulated_annealing_gpu(
    preprocessed: PreprocessedCase,
    params: SolverParams | None = None,
    snapshot_callback: Callable[[SnapshotRecord], None] | None = None,
) -> SolveResult:
    params = params or SolverParams()
    device = preprocessed.device
    generator = _make_generator(device, params.seed)

    current_state = initialize_state(preprocessed, params)
    current_eval = evaluate_state_gpu(current_state, preprocessed, params)
    initial_best_index = _choose_overall_best(current_eval)
    initial_eval = _index_eval(current_eval, initial_best_index)

    delta_samples: list[torch.Tensor] = []
    probe_state = _clone_state(current_state)
    probe_eval = current_eval
    for _ in range(params.temperature_samples):
        candidate = propose_mutation(probe_state, preprocessed, params, temperature=1.0, generator=generator)
        candidate_eval = evaluate_state_gpu(candidate, preprocessed, params)
        delta_samples.append(candidate_eval.score - probe_eval.score)
        probe_state = candidate
        probe_eval = candidate_eval
    t0 = _temperature_from_deltas(torch.cat(delta_samples), params.target_initial_acceptance)
    temperatures = _make_temperature_schedule(t0, params.n_steps, params.final_temperature_ratio, device)

    best_state = _clone_state(current_state)
    best_eval = current_eval
    acceptance_rates = []
    snapshots: list[SnapshotRecord] = []
    progress_iterable = tqdm(
        enumerate(temperatures),
        total=int(temperatures.shape[0]),
        disable=not params.show_progress,
        desc="annealing",
    )

    for step_index, temperature in progress_iterable:
        candidate_state = propose_mutation(current_state, preprocessed, params, float(temperature.item()) / max(t0, EPS), generator)
        candidate_eval = evaluate_state_gpu(candidate_state, preprocessed, params)
        delta = candidate_eval.score - current_eval.score
        accept_probability = torch.exp(torch.clamp(-delta / temperature, max=80.0))
        accept_mask = (delta <= 0) | (_rand(delta.shape, device, generator) < accept_probability)
        current_state = _state_where(accept_mask.unsqueeze(-1), candidate_state, current_state)
        current_eval = _eval_where(accept_mask, candidate_eval, current_eval)
        acceptance_rate = accept_mask.to(torch.float32).mean()
        acceptance_rates.append(acceptance_rate)

        better_mask = ((~best_eval.feasible) & current_eval.feasible) | (
            (best_eval.feasible == current_eval.feasible) & (current_eval.score < best_eval.score)
        )
        best_state = _state_where(better_mask.unsqueeze(-1), current_state, best_state)
        best_eval = _eval_where(better_mask, current_eval, best_eval)

        best_index = _choose_overall_best(best_eval)
        best_score = float(best_eval.score[best_index].item())
        feasible_count = int(best_eval.feasible.sum().item())
        if params.show_progress:
            progress_iterable.set_postfix(
                best_score=f"{best_score:.4f}",
                feasible=feasible_count,
                accept=f"{float(acceptance_rate.item()):.3f}",
            )

        should_snapshot = (
            params.snapshot_every_steps is not None
            and params.snapshot_every_steps > 0
            and ((step_index + 1) % params.snapshot_every_steps == 0 or step_index == 0 or step_index + 1 == len(temperatures))
        )
        if should_snapshot:
            snapshot = SnapshotRecord(
                step_index=step_index,
                temperature=float(temperature.item()),
                acceptance_rate=float(acceptance_rate.item()),
                best_state=_index_state(best_state, best_index),
                best_evaluation=_index_eval(best_eval, best_index),
            )
            snapshots.append(snapshot)
            if snapshot_callback is not None:
                snapshot_callback(snapshot)

    overall_best_index = _choose_overall_best(best_eval)
    final_best_index = _choose_overall_best(current_eval)
    return SolveResult(
        best_state=_index_state(best_state, overall_best_index),
        best_evaluation=_index_eval(best_eval, overall_best_index),
        initial_evaluation=initial_eval,
        final_evaluation=_index_eval(current_eval, final_best_index),
        temperatures=temperatures.detach().clone(),
        acceptance_rates=torch.stack(acceptance_rates) if acceptance_rates else torch.empty(0, device=device),
        snapshots=snapshots,
        device=device,
    )


__all__ = [
    "AnnealingState",
    "CaseData",
    "Evaluation",
    "PreprocessedCase",
    "SolveResult",
    "SolverParams",
    "SnapshotRecord",
    "EPS",
    "OBJECTIVE_CAP",
    "evaluate_state_gpu",
    "initialize_state",
    "load_case",
    "preprocess_case",
    "propose_mutation",
    "run_simulated_annealing_gpu",
]
