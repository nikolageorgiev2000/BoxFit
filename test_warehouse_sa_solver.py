from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

import warehouse_sa_solver as solver
from warehouse_sa_solver import (
    AnnealingState,
    CaseData,
    SolverParams,
    _intersection_area_quad_rect,
    _polygon_world,
    _shoelace_area,
    _transform_world_to_local,
    evaluate_state_gpu,
    load_case,
    propose_mutation,
    preprocess_case,
    run_simulated_annealing_gpu,
)


def _rect_polygon(xmin: float, xmax: float, ymin: float, ymax: float) -> list[tuple[float, float]]:
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def _poly_area(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        total += point[0] * nxt[1] - point[1] * nxt[0]
    return abs(total) * 0.5


def _clip_polygon_axis_rect(poly: list[tuple[float, float]], bounds: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    def clip(points: list[tuple[float, float]], edge: str, bound: float) -> list[tuple[float, float]]:
        if not points:
            return []
        result: list[tuple[float, float]] = []
        for idx, current in enumerate(points):
            nxt = points[(idx + 1) % len(points)]
            if edge == "xmin":
                current_inside = current[0] >= bound
                next_inside = nxt[0] >= bound
                axis = 0
            elif edge == "xmax":
                current_inside = current[0] <= bound
                next_inside = nxt[0] <= bound
                axis = 0
            elif edge == "ymin":
                current_inside = current[1] >= bound
                next_inside = nxt[1] >= bound
                axis = 1
            else:
                current_inside = current[1] <= bound
                next_inside = nxt[1] <= bound
                axis = 1

            if current_inside != next_inside:
                denom = nxt[axis] - current[axis]
                if abs(denom) < 1e-8:
                    t = 0.0
                else:
                    t = (bound - current[axis]) / denom
                intersection = (
                    current[0] + t * (nxt[0] - current[0]),
                    current[1] + t * (nxt[1] - current[1]),
                )
                result.append(intersection)
            if next_inside:
                result.append(nxt)
        return result

    result = poly
    for edge, bound in (("xmin", bounds[0]), ("xmax", bounds[1]), ("ymin", bounds[2]), ("ymax", bounds[3])):
        result = clip(result, edge, bound)
    return result


def _reference_evaluate_state(state: AnnealingState, pre: solver.PreprocessedCase, params: SolverParams) -> solver.Evaluation:
    active = state.active.squeeze(-1).to(dtype=torch.float32)
    type_id = state.type_id.squeeze(-1)
    theta_idx = state.theta_idx.squeeze(-1)
    centers = torch.stack([state.x, state.y], dim=-1)

    bay_local = pre.bay_polygons_local.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape, 4, 2)
    access_bounds = pre.access_bounds_local.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape, 4)
    bay_area_slot = pre.bay_areas.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape)
    access_area_slot = pre.access_areas.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape)
    bay_heights = pre.bay_heights.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape)
    bay_capacities = pre.bay_capacities.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape)
    bay_costs = pre.bay_costs.index_select(0, type_id.reshape(-1)).reshape(*type_id.shape)

    rot = pre.rot.index_select(0, theta_idx.reshape(-1)).reshape(*theta_idx.shape, 2, 2)
    inv_rot = pre.inv_rot.index_select(0, theta_idx.reshape(-1)).reshape(*theta_idx.shape, 2, 2)
    bay_world = _polygon_world(bay_local, rot, centers)

    area_used = (bay_area_slot * active).sum(dim=1)
    access_area = (access_area_slot * active).sum(dim=1)
    total_capacity = (bay_capacities * active).sum(dim=1)
    total_cost = (bay_costs * active).sum(dim=1)

    n_chains, n_slots = type_id.shape
    obstacle_overlap = torch.zeros(n_chains, dtype=torch.float32, device=pre.device)
    exterior_overlap = torch.zeros_like(obstacle_overlap)
    ceiling_overlap = torch.zeros_like(obstacle_overlap)
    pair_overlap = torch.zeros_like(obstacle_overlap)

    for slot_index in range(n_slots):
        active_i = active[:, slot_index]
        centers_i = centers[:, slot_index, :]
        inv_rot_i = inv_rot[:, slot_index, :, :]
        access_bounds_i = access_bounds[:, slot_index, :]
        access_area_i = access_area_slot[:, slot_index]
        bay_height_i = bay_heights[:, slot_index]

        if pre.warehouse_boxes.shape[0] > 0:
            warehouse_local = _transform_world_to_local(
                pre.warehouse_boxes.unsqueeze(0).expand(n_chains, -1, -1, -1).reshape(-1, 4, 2),
                centers_i.unsqueeze(1).expand(-1, pre.warehouse_boxes.shape[0], -1).reshape(-1, 2),
                inv_rot_i.unsqueeze(1).expand(-1, pre.warehouse_boxes.shape[0], -1, -1).reshape(-1, 2, 2),
            )
            contained = _intersection_area_quad_rect(
                warehouse_local,
                access_bounds_i.unsqueeze(1).expand(-1, pre.warehouse_boxes.shape[0], -1).reshape(-1, 4),
            ).reshape(n_chains, pre.warehouse_boxes.shape[0]).sum(dim=1)
            contained = torch.clamp(contained, min=0.0)
            contained = torch.minimum(contained, access_area_i + 1e-4)
            exterior_overlap = exterior_overlap + active_i * torch.clamp(access_area_i - contained, min=0.0)

        if pre.obstacle_boxes.shape[0] > 0:
            obstacle_local = _transform_world_to_local(
                pre.obstacle_boxes.unsqueeze(0).expand(n_chains, -1, -1, -1).reshape(-1, 4, 2),
                centers_i.unsqueeze(1).expand(-1, pre.obstacle_boxes.shape[0], -1).reshape(-1, 2),
                inv_rot_i.unsqueeze(1).expand(-1, pre.obstacle_boxes.shape[0], -1, -1).reshape(-1, 2, 2),
            )
            obstacle_area = _intersection_area_quad_rect(
                obstacle_local,
                access_bounds_i.unsqueeze(1).expand(-1, pre.obstacle_boxes.shape[0], -1).reshape(-1, 4),
            ).reshape(n_chains, pre.obstacle_boxes.shape[0]).sum(dim=1)
            obstacle_overlap = obstacle_overlap + active_i * obstacle_area

        if pre.ceiling_boxes.shape[0] > 0:
            low_mask = pre.ceiling_heights.unsqueeze(0) < bay_height_i.unsqueeze(1) - solver.EPS
            ceiling_local = _transform_world_to_local(
                pre.ceiling_boxes.unsqueeze(0).expand(n_chains, -1, -1, -1).reshape(-1, 4, 2),
                centers_i.unsqueeze(1).expand(-1, pre.ceiling_boxes.shape[0], -1).reshape(-1, 2),
                inv_rot_i.unsqueeze(1).expand(-1, pre.ceiling_boxes.shape[0], -1, -1).reshape(-1, 2, 2),
            )
            ceiling_area = _intersection_area_quad_rect(
                ceiling_local,
                access_bounds_i.unsqueeze(1).expand(-1, pre.ceiling_boxes.shape[0], -1).reshape(-1, 4),
            ).reshape(n_chains, pre.ceiling_boxes.shape[0])
            ceiling_overlap = ceiling_overlap + active_i * (ceiling_area * low_mask.to(torch.float32)).sum(dim=1)

        bay_local_all = _transform_world_to_local(
            bay_world.reshape(-1, 4, 2),
            centers_i.unsqueeze(1).expand(-1, n_slots, -1).reshape(-1, 2),
            inv_rot_i.unsqueeze(1).expand(-1, n_slots, -1, -1).reshape(-1, 2, 2),
        )
        pair_area = _intersection_area_quad_rect(
            bay_local_all,
            access_bounds_i.unsqueeze(1).expand(-1, n_slots, -1).reshape(-1, 4),
        ).reshape(n_chains, n_slots)
        active_other = active.clone()
        active_other[:, slot_index] = 0.0
        pair_overlap = pair_overlap + active_i * (pair_area * active_other).sum(dim=1)

    area_used_ratio = area_used / max(pre.warehouse_area, solver.EPS)
    capacity_safe = torch.clamp(total_capacity, min=solver.EPS)
    objective_defined = total_capacity > solver.EPS
    objective_raw = torch.where(
        objective_defined,
        (total_cost / capacity_safe) ** (2.0 - area_used_ratio),
        torch.full_like(total_capacity, pre.empty_state_objective),
    )
    objective_norm = torch.clamp(objective_raw / max(pre.objective_scale, solver.EPS), max=solver.OBJECTIVE_CAP)
    penalty_norm = (
        params.w_pair * (1.0 - torch.exp(-pair_overlap / torch.clamp(access_area, min=solver.EPS)))
        + params.w_obstacle * (1.0 - torch.exp(-obstacle_overlap / torch.clamp(access_area, min=solver.EPS)))
        + params.w_exterior * (1.0 - torch.exp(-exterior_overlap / torch.clamp(access_area, min=solver.EPS)))
        + params.w_ceiling * (1.0 - torch.exp(-ceiling_overlap / torch.clamp(access_area, min=solver.EPS)))
    )
    score = params.objective_weight * objective_norm + params.penalty_weight * penalty_norm
    feasible = (
        (pair_overlap <= 1e-4)
        & (obstacle_overlap <= 1e-4)
        & (exterior_overlap <= 1e-4)
        & (ceiling_overlap <= 1e-4)
        & objective_defined
    )

    return solver.Evaluation(
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test suite")
class WarehouseSASolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def test_load_and_preprocess_public_case(self) -> None:
        case = load_case(Path("PublicTestCases/Case2"))
        pre = preprocess_case(case, self.device)

        self.assertEqual(pre.device.type, "cuda")
        self.assertGreater(pre.warehouse_boxes.shape[0], 0)
        self.assertEqual(pre.rot.shape, (24, 2, 2))
        self.assertTrue(pre.bay_widths.is_cuda)
        self.assertTrue(pre.ceiling_boxes.is_cuda)

    def test_rotation_matrices_are_orthonormal(self) -> None:
        case = load_case(Path("PublicTestCases/Case0"))
        pre = preprocess_case(case, self.device)
        identity = torch.eye(2, device=self.device).expand(24, -1, -1)
        gram = torch.matmul(pre.rot.transpose(-1, -2), pre.rot)
        self.assertTrue(torch.allclose(gram, identity, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(pre.inv_rot, pre.rot.transpose(-1, -2), atol=1e-6, rtol=1e-6))

    def test_shoelace_area_matches_reference_polygon_area(self) -> None:
        points = torch.tensor(
            [[[-2.0, -1.0], [1.0, -2.0], [3.0, 0.0], [2.0, 3.0], [-1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        counts = torch.tensor([5], dtype=torch.long, device=self.device)
        expected = _poly_area([(-2.0, -1.0), (1.0, -2.0), (3.0, 0.0), (2.0, 3.0), (-1.0, 2.0)])
        self.assertAlmostEqual(_shoelace_area(points, counts).item(), expected, places=5)

    def test_local_frame_overlap_matches_reference_clipper(self) -> None:
        bounds = torch.tensor([[-2.0, 2.0, -1.0, 3.0]], device=self.device)
        polygon = torch.tensor(
            [[[-3.0, -0.5], [0.0, -2.0], [3.0, 1.0], [0.0, 4.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        area = _intersection_area_quad_rect(polygon, bounds).item()
        clipped = _clip_polygon_axis_rect(
            [tuple(row) for row in polygon[0].cpu().tolist()],
            tuple(bounds[0].cpu().tolist()),
        )
        expected = _poly_area(clipped)
        self.assertAlmostEqual(area, expected, places=4)

    def test_intersection_area_handles_full_exclusion(self) -> None:
        bounds = torch.tensor([[-1.0, 1.0, -1.0, 1.0]], device=self.device)
        polygon = torch.tensor(
            [[[3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]]],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertAlmostEqual(_intersection_area_quad_rect(polygon, bounds).item(), 0.0, places=6)

    def test_gap_gap_overlap_is_not_penalized(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-10.0, 10.0, -10.0, 10.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 2.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                }
            ],
            obstacles=[],
            ceiling_profile=[(-10.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        state = AnnealingState(
            active=torch.tensor([[[True], [True]]], device=self.device),
            type_id=torch.tensor([[[0], [0]]], device=self.device),
            x=torch.tensor([[0.0, 0.0]], device=self.device),
            y=torch.tensor([[0.0, 4.0]], device=self.device),
            theta_idx=torch.tensor([[[0], [12]]], device=self.device),
        )
        eval_result = evaluate_state_gpu(state, pre, SolverParams())
        self.assertAlmostEqual(eval_result.pair_overlap.item(), 0.0, places=5)

    def test_directed_pair_overlap_counts_only_access_vs_bay(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-10.0, 10.0, -10.0, 10.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 2.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                }
            ],
            obstacles=[],
            ceiling_profile=[(-10.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        state = AnnealingState(
            active=torch.tensor([[[True], [True]]], device=self.device),
            type_id=torch.tensor([[[0], [0]]], device=self.device),
            x=torch.tensor([[0.0, 0.0]], device=self.device),
            y=torch.tensor([[0.0, 2.5]], device=self.device),
            theta_idx=torch.tensor([[[0], [0]]], device=self.device),
        )
        eval_result = evaluate_state_gpu(state, pre, SolverParams())
        self.assertAlmostEqual(eval_result.pair_overlap.item(), 3.0, places=4)

    def test_ceiling_penalty_uses_access_polygon(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-10.0, 10.0, -10.0, 10.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 3.0,
                    "gap_depth": 2.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                }
            ],
            obstacles=[],
            ceiling_profile=[(-10.0, 4.0), (1.0, 2.0)],
        )
        pre = preprocess_case(case, self.device)
        state = AnnealingState(
            active=torch.tensor([[[True]]], device=self.device),
            type_id=torch.tensor([[[0]]], device=self.device),
            x=torch.tensor([[0.0]], device=self.device),
            y=torch.tensor([[0.0]], device=self.device),
            theta_idx=torch.tensor([[[18]]], device=self.device),
        )
        eval_result = evaluate_state_gpu(state, pre, SolverParams())
        self.assertGreater(eval_result.ceiling_overlap.item(), 0.0)

    def test_local_transform_matches_world_overlap(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-5.0, 5.0, -5.0, 5.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 1.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                }
            ],
            obstacles=[],
            ceiling_profile=[(-5.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        centers = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, device=self.device)
        theta_idx = torch.tensor([[3]], dtype=torch.long, device=self.device)
        rot = pre.rot.index_select(0, theta_idx.reshape(-1)).reshape(1, 1, 2, 2)
        inv_rot = pre.inv_rot.index_select(0, theta_idx.reshape(-1)).reshape(1, 1, 2, 2)
        access_world = _polygon_world(pre.access_polygons_local[:1].unsqueeze(0), rot, centers)[0, 0]
        target_world = torch.tensor(
            [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )
        target_local = _transform_world_to_local(target_world.unsqueeze(0), centers[:, 0, :], inv_rot[:, 0, :, :])[0]
        area_local = _intersection_area_quad_rect(target_local.unsqueeze(0), pre.access_bounds_local[:1]).item()
        clipped = _clip_polygon_axis_rect(
            [tuple(row) for row in target_local.cpu().tolist()],
            tuple(pre.access_bounds_local[0].cpu().tolist()),
        )
        self.assertAlmostEqual(area_local, _poly_area(clipped), places=4)
        self.assertGreater(_poly_area([tuple(row) for row in access_world.cpu().tolist()]), 0.0)

    def test_vectorized_evaluation_matches_reference_implementation(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-8.0, 8.0, -6.0, 6.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 2.0,
                    "gap_depth": 1.0,
                    "capacity": 2.0,
                    "cost": 1.0,
                },
                {
                    "type_id": 1.0,
                    "width": 3.0,
                    "depth": 1.5,
                    "height": 4.0,
                    "gap_depth": 1.5,
                    "capacity": 3.0,
                    "cost": 2.0,
                },
            ],
            obstacles=[(-1.0, -1.0, 2.0, 2.0), (3.0, -2.0, 1.5, 3.0)],
            ceiling_profile=[(-8.0, 5.0), (0.0, 3.0)],
        )
        pre = preprocess_case(case, self.device)
        params = SolverParams(n_chains=3, seed=13)
        state = AnnealingState(
            active=torch.tensor(
                [
                    [[True], [True], [False]],
                    [[True], [True], [True]],
                    [[False], [True], [True]],
                ],
                device=self.device,
            ),
            type_id=torch.tensor(
                [
                    [[0], [1], [0]],
                    [[1], [0], [1]],
                    [[0], [1], [1]],
                ],
                device=self.device,
            ),
            x=torch.tensor(
                [
                    [-3.0, 2.0, 0.0],
                    [-4.0, 0.0, 4.5],
                    [1.0, -2.0, 3.0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            y=torch.tensor(
                [
                    [-2.5, 0.0, 1.0],
                    [2.0, -1.0, 1.5],
                    [0.0, 2.0, -2.0],
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            theta_idx=torch.tensor(
                [
                    [[0], [6], [12]],
                    [[3], [9], [15]],
                    [[18], [21], [4]],
                ],
                device=self.device,
            ),
        )

        expected = _reference_evaluate_state(state, pre, params)
        actual = evaluate_state_gpu(state, pre, params)

        self.assertTrue(torch.allclose(actual.score, expected.score, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(actual.pair_overlap, expected.pair_overlap, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(actual.obstacle_overlap, expected.obstacle_overlap, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(actual.exterior_overlap, expected.exterior_overlap, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(actual.ceiling_overlap, expected.ceiling_overlap, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.equal(actual.feasible, expected.feasible))

    def test_cleanup_removes_unused_helpers_and_fields(self) -> None:
        self.assertFalse(hasattr(solver, "_polygon_area_xy"))
        self.assertFalse(hasattr(solver, "_make_snapshot"))
        self.assertNotIn("warehouse_polygon", solver.PreprocessedCase.__annotations__)
        self.assertNotIn("y_span", solver.PreprocessedCase.__annotations__)
        self.assertNotIn("angles_deg", solver.PreprocessedCase.__annotations__)

    def test_mutation_probabilities_can_disable_type_switches(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-5.0, 5.0, -5.0, 5.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 1.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                },
                {
                    "type_id": 1.0,
                    "width": 3.0,
                    "depth": 1.0,
                    "height": 1.0,
                    "gap_depth": 1.0,
                    "capacity": 2.0,
                    "cost": 2.0,
                },
            ],
            obstacles=[],
            ceiling_profile=[(-5.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        params = SolverParams(
            n_chains=32,
            seed=123,
            mutation_probabilities=[1.0, 0.0, 0.0, 0.0],
            translate_step_fraction=0.1,
        )
        state = solver.initialize_state(pre, params)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(123)
        candidate = propose_mutation(state, pre, params, temperature=1.0, generator=generator)

        self.assertTrue(torch.equal(candidate.type_id, state.type_id))
        self.assertTrue(torch.equal(candidate.theta_idx, state.theta_idx))
        self.assertTrue(torch.equal(candidate.active, state.active))
        self.assertTrue(
            torch.any(torch.abs(candidate.x - state.x) > 1e-6).item()
            or torch.any(torch.abs(candidate.y - state.y) > 1e-6).item()
        )

    def test_slot_mutation_probability_can_mutate_multiple_slots_per_chain(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-5.0, 5.0, -5.0, 5.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 1.0,
                    "capacity": 1.0,
                    "cost": 1.0,
                },
            ],
            obstacles=[],
            ceiling_profile=[(-5.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        params = SolverParams(
            n_chains=2,
            seed=321,
            slot_mutation_probability=1.0,
            mutation_probabilities=[1.0, 0.0, 0.0, 0.0],
            translate_step_fraction=0.1,
        )
        state = solver.initialize_state(pre, params)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(321)
        candidate = propose_mutation(state, pre, params, temperature=1.0, generator=generator)

        moved_slots = (torch.abs(candidate.x - state.x) > 1e-6) | (torch.abs(candidate.y - state.y) > 1e-6)
        self.assertTrue(torch.all(moved_slots).item())

    def test_annealing_run_returns_gpu_result(self) -> None:
        case = CaseData(
            warehouse_polygon=_rect_polygon(-3.0, 3.0, -3.0, 3.0),
            bay_types=[
                {
                    "type_id": 0.0,
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 1.0,
                    "gap_depth": 1.0,
                    "capacity": 2.0,
                    "cost": 1.0,
                }
            ],
            obstacles=[],
            ceiling_profile=[(-3.0, 10.0)],
        )
        pre = preprocess_case(case, self.device)
        snapshots = []
        params = SolverParams(n_chains=16, n_steps=10, seed=7, temperature_samples=3, show_progress=False, snapshot_every_steps=3)
        result = run_simulated_annealing_gpu(pre, params, snapshot_callback=snapshots.append)
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.best_state.x.device.type, "cuda")
        self.assertEqual(result.acceptance_rates.shape[0], params.n_steps)
        self.assertTrue(torch.isfinite(result.best_evaluation.score).all())
        self.assertGreaterEqual(len(result.snapshots), 1)
        self.assertEqual(len(result.snapshots), len(snapshots))
        self.assertEqual(result.snapshots[-1].best_state.x.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
