"""The concept-connection network of tutorial 03: how the papers link media bias to neighbouring terms (tutorials/README.md).

`network_inputs` takes the subtype hierarchy (`queries.hierarchy`) and the requested terms; `connections_figure` keeps the union of all shortest undirected paths between them, restores every directed relation among the kept terms, and draws them at the fixed publication positions of `network_layout.json`, with arrows from subtype to supertype labelled by their distinct supporting papers.
The figure code checks its own geometry (no overlapping boxes, arrows pointing into their target) and fails rather than drawing a figure that breaks it.
"""

import heapq
import json
import math
import textwrap
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PlotPath

from mbdef.analysis.plot_style import (
    BLUE,
    BLUE_LIGHT,
    EMPHASIS_FONT_SCALE,
    FIGURE_WIDTH,
    FONT_FAMILY,
    FONT_SIZE,
    INK,
)
from mbdef.analysis.queries import find_concept, hierarchy

LAYOUT = Path(__file__).with_name("network_layout.json")
ANCHORS = (
    "media bias",
    "bias",
    "misinformation",
    "disinformation",
    "fake news",
    "propaganda",
)
# Labels drawn on one line, kept with the bottom edge they would have wrapped (the publication figure).
SINGLE_LINE = {"cognitive biases", "statement bias"}


def network_inputs(data, anchors=ANCHORS):
    """(evidence, requested concept ids, label of every concept id, self-relations) for `connections_figure`."""
    evidence, self_relations = hierarchy(data)
    requested = [find_concept(data, term) for term in anchors]
    labels = dict(data["concepts"].select("concept_id", "label").iter_rows())
    return evidence, requested, labels, self_relations


def compact_layout(terms, edges, labels, font_size=None):
    """The fixed centre of each term's box, from LAYOUT (keyed by label, ignoring case), and the layout's other settings."""
    specification = json.loads(LAYOUT.read_text())
    positions = {
        label.casefold(): point for label, point in specification["positions"].items()
    }
    missing = {
        labels.get(term, term)
        for term in terms
        if labels.get(term, term).casefold() not in positions
    }
    if missing:
        raise ValueError(
            f"Concepts need explicit positions in {LAYOUT.name}: {sorted(missing)}"
        )
    coords = {
        term: tuple(positions[labels.get(term, term).casefold()]) for term in terms
    }
    return coords, {
        key: value for key, value in specification.items() if key != "positions"
    }


def connections_figure(
    output_dir,
    evidence,
    requested,
    audit_path=None,
    font_size=FONT_SIZE,
    node_labels=None,
):
    """Draw the union of all shortest undirected paths between the REQUESTED terms, with every directed relation among the selected terms; write the figure (SVG, PNG) and its audit (JSON) to OUTPUT_DIR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as PlotPath

    matplotlib.rcParams.update({"svg.fonttype": "path", "font.family": FONT_FAMILY})

    anchors = set(requested)
    node_labels = node_labels or {}

    def display(term):
        return node_labels.get(term, term)

    adjacency = defaultdict(set)
    for a, b in evidence:
        adjacency[a].add(b)
        adjacency[b].add(a)
    missing = anchors - adjacency.keys()
    if missing:
        raise ValueError(
            f"Connection terms have no hierarchy relations: {sorted(missing)}"
        )
    distances = {}
    for anchor in sorted(anchors):
        distance = {anchor: 0}
        queue = deque([anchor])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        distances[anchor] = distance
    nodes = set(anchors)
    disconnected = []
    for a, b in combinations(sorted(anchors), 2):
        if b not in distances[a]:
            disconnected.append([a, b])
            continue
        nodes.update(
            n
            for n in distances[a]
            if n in distances[b]
            and distances[a][n] + distances[b][n] == distances[a][b]
        )
    edges = {edge: claims for edge, claims in evidence.items() if set(edge) <= nodes}
    terms = sorted(nodes)
    coords, layout_report = compact_layout(
        terms, edges, node_labels, font_size=font_size
    )
    canvas_width, canvas_height = layout_report["canvas_inches"]
    # Match typography at equal manuscript width despite the wider network canvas.
    font_size = font_size * canvas_width / FIGURE_WIDTH
    fig, ax = plt.subplots(figsize=(canvas_width, canvas_height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    labels = {}
    for term, (x, y) in coords.items():
        anchor = term in anchors
        labels[term] = ax.text(
            x,
            y,
            textwrap.fill(display(term), 13 if display(term) in SINGLE_LINE else 18),
            ha="center",
            va="center",
            fontsize=font_size * EMPHASIS_FONT_SCALE if anchor else font_size,
            color="white" if anchor else INK,
            fontweight="bold" if anchor else "normal",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": BLUE if anchor else BLUE_LIGHT,
                "edgecolor": BLUE if anchor else BLUE,
                "linewidth": 1,
            },
        )
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.invert_yaxis()
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # ty: ignore[unresolved-attribute]  # the Agg canvas has one

    def extent(term):
        """The window extent of TERM's box (every label is drawn with one)."""
        return labels[term].get_bbox_patch().get_window_extent(renderer)  # ty: ignore[unresolved-attribute]

    bottom_alignment = {}
    for term in terms:
        if display(term) not in SINGLE_LINE:
            continue
        before = extent(term).y0
        labels[term].set_text(display(term))
        fig.canvas.draw()
        after = extent(term).y0
        center = ax.transData.transform(labels[term].get_position())
        center[1] += before - after
        labels[term].set_position(tuple(ax.transData.inverted().transform(center)))
        coords[term] = labels[term].get_position()
        fig.canvas.draw()
        final = extent(term).y0
        if abs(final - before) > 1e-5:
            raise ValueError("Single-line box bottom moved")
        bottom_alignment[term] = {
            "before_pixels": before,
            "after_pixels": final,
            "single_line": True,
        }
    by_name = {display(term): term for term in terms}

    def box(name):
        return extent(by_name[name])

    def shift(name, dx=0, dy=0):
        term = by_name[name]
        center = ax.transData.transform(labels[term].get_position())
        labels[term].set_position(
            tuple(ax.transData.inverted().transform(center + np.array([dx, dy])))
        )
        coords[term] = labels[term].get_position()
        fig.canvas.draw()

    ideological_top = box("ideological stand bias").y1
    labels[by_name["ideological stand bias"]].set_text("ideological stand bias")
    fig.canvas.draw()
    shift(
        "ideological stand bias", dy=ideological_top - box("ideological stand bias").y1
    )
    shift("informational bias", dy=ideological_top - box("informational bias").y1)
    shift(
        "paltering",
        dx=box("disinformation").x1 - box("paltering").x1,
        dy=ideological_top - box("paltering").y1,
    )
    shift(
        "coverage bias",
        dx=box("fake news").x0 - box("coverage bias").x1,
        dy=-0.35 * fig.dpi,
    )
    edge_alignment = {
        "ideological_top_before": ideological_top,
        "ideological_top_after": box("ideological stand bias").y1,
        "informational_top": box("informational bias").y1,
        "paltering_top": box("paltering").y1,
        "paltering_right": box("paltering").x1,
        "disinformation_right": box("disinformation").x1,
        "coverage_right": box("coverage bias").x1,
        "fake_news_left": box("fake news").x0,
        "coverage_down_inches": 0.35,
    }
    if (
        max(
            abs(edge_alignment[key] - ideological_top)
            for key in ["ideological_top_after", "informational_top", "paltering_top"]
        )
        > 1e-5
    ):
        raise ValueError("Top-boundary alignment failed")
    if (
        abs(edge_alignment["paltering_right"] - edge_alignment["disinformation_right"])
        > 1e-5
    ):
        raise ValueError("Right-boundary alignment failed")
    if abs(edge_alignment["coverage_right"] - edge_alignment["fake_news_left"]) > 1e-5:
        raise ValueError("Coverage/fake-news boundary alignment failed")
    for a, b in combinations(terms, 2):
        box_a = extent(a)
        box_b = extent(b)
        if box_a.overlaps(box_b):
            raise ValueError(f"Reference labels overlap: {display(a)} / {display(b)}")
    # Route in display coordinates: curvature and box clearance must remain
    # correct even in a wide, shallow figure with unequal axis scales.
    display_boxes = {term: extent(term) for term, label in labels.items()}
    centers = {term: ax.transData.transform(coords[term]) for term in terms}

    def display_boundary(term, toward):
        center = centers[term]
        delta = toward - center
        box = display_boxes[term]
        scale = min(
            (box.width / 2 + 5) / abs(delta[0]) if delta[0] else float("inf"),
            (box.height / 2 + 5) / abs(delta[1]) if delta[1] else float("inf"),
        )
        return center + delta * scale

    occupied = [box.expanded(1.04, 1.08) for box in display_boxes.values()]
    outer_nodes = (
        nodes - anchors - {term for term in nodes if display(term) == "framing"}
    )
    route_hits = {}
    route_audit = {}
    routed_segments = []

    def route_score(route):
        base, vertices, codes, samples = route
        points = np.array(samples[::5] + [samples[-1]])
        u, v = points[:-1], points[1:]
        length = float(np.linalg.norm(v - u, axis=1).sum())
        if not routed_segments:
            return length / 100 + base * 0.1
        prior = np.array(routed_segments)
        a, b = u[:, None, :], v[:, None, :]
        c, d = prior[None, :, 0, :], prior[None, :, 1, :]

        def cross(x, y):
            return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

        proper = (cross(b - a, c - a) * cross(b - a, d - a) < -1e-5) & (
            cross(d - c, a - c) * cross(d - c, b - c) < -1e-5
        )
        ab = b - a
        cd = d - c
        norm = np.maximum(np.linalg.norm(ab, axis=2), 1e-6)
        norm_cd = np.maximum(np.linalg.norm(cd, axis=2), 1e-6)
        parallel = abs(cross(ab, cd)) / (norm * norm_cd) < 0.05
        close = abs(cross(ab, c - a)) / norm < 4
        axis = ab / np.maximum(norm[:, :, None], 1e-6)
        t0 = np.sum((c - a) * axis, axis=2)
        t1 = np.sum((d - a) * axis, axis=2)
        overlap = np.minimum(np.maximum(t0, t1), norm) - np.maximum(
            np.minimum(t0, t1), 0
        )
        shared = parallel & close & (overlap > 10)
        return length / 100 + base * 0.3 + 0.5 * proper.sum() + 4 * shared.sum()

    for (a, b), claims in sorted(
        edges.items(),
        key=lambda item: (
            item[0][0] not in outer_nodes,
            centers[item[0][0]][0] if item[0][0] in outer_nodes else 0,
            -len({c["paper_id"] for c in item[1]}),
            item[0],
        ),
    ):
        count = len({c["paper_id"] for c in claims})
        start, end = centers[a], centers[b]
        delta = end - start
        reciprocal = (b, a) in edges
        curvatures = (
            [0.1 * i for i in range(1, 19)]
            if reciprocal
            else [0] + [sign * 0.1 * i for i in range(1, 19) for sign in [1, -1]]
        )
        routes = []
        for rad in curvatures:
            control = (start + end) / 2 + rad * delta[::-1] * [1, -1]
            if display_boxes[a].padded(6).contains(*control) or display_boxes[b].padded(
                6
            ).contains(*control):
                continue
            p0, p2 = display_boundary(a, control), display_boundary(b, control)
            samples = [
                (1 - t) ** 2 * p0 + 2 * (1 - t) * t * control + t * t * p2
                for t in (i / 100 for i in range(101))
            ]
            hits = sum(
                any(box.expanded(1.03, 1.08).contains(*point) for point in samples)
                for term, box in display_boxes.items()
                if term not in (a, b)
            )
            if hits == 0 and all(ax.bbox.contains(*point) for point in samples):
                routes.append(
                    (
                        abs(rad),
                        [p0, control, p2],
                        [PlotPath.MOVETO, PlotPath.CURVE3, PlotPath.CURVE3],
                        samples,
                    )
                )

        def add_cubic_routes(
            strength_pairs, a=a, b=b, start=start, end=end, routes=routes
        ):
            normals = [
                np.array(v, dtype=float) for v in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            ]
            parameters = np.linspace(0, 1, 121)[:, None]
            for source_normal in normals:
                p0 = display_boundary(a, start + source_normal)
                for target_normal in normals:
                    p3 = display_boundary(b, end + target_normal)
                    distance = float(np.linalg.norm(p3 - p0))
                    for source_strength, target_strength in strength_pairs:
                        p1 = p0 + source_normal * max(15, distance * source_strength)
                        p2 = p3 + target_normal * max(15, distance * target_strength)
                        t = parameters
                        sample_array = (
                            (1 - t) ** 3 * p0
                            + 3 * (1 - t) ** 2 * t * p1
                            + 3 * (1 - t) * t * t * p2
                            + t**3 * p3
                        )
                        if (
                            sample_array[:, 0].min() < ax.bbox.x0
                            or sample_array[:, 0].max() > ax.bbox.x1
                            or sample_array[:, 1].min() < ax.bbox.y0
                            or sample_array[:, 1].max() > ax.bbox.y1
                        ):
                            continue
                        blocked = False
                        for box in display_boxes.values():
                            if np.any(
                                (sample_array[:, 0] >= box.x0 - 2)
                                & (sample_array[:, 0] <= box.x1 + 2)
                                & (sample_array[:, 1] >= box.y0 - 2)
                                & (sample_array[:, 1] <= box.y1 + 2)
                            ):
                                blocked = True
                                break
                        if not blocked:
                            samples = list(sample_array)
                            routes.append(
                                (
                                    2 + source_strength + target_strength,
                                    [p0, p1, p2, p3],
                                    [PlotPath.MOVETO] + [PlotPath.CURVE4] * 3,
                                    samples,
                                )
                            )

        add_cubic_routes([(s, s) for s in [0.15, 0.25, 0.4, 0.6, 0.85, 1.2]])
        if not routes:
            strengths = [0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.85, 1.2]
            add_cubic_routes(
                [(left, right) for left in strengths for right in strengths]
            )
        label = ax.text(
            0,
            0,
            f"({count})",
            ha="center",
            va="center",
            fontsize=font_size,
            color=INK,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        label.set_gid(f"paper-count:{a}->{b};papers={count}")
        chosen = None
        source_outer = a in outer_nodes
        windows = (
            [(0.12, 1 / 3, 0.24), (1 / 3, 0.5, 0.36), (0.5, 0.75, 0.55)]
            if source_outer
            else [(0.25, 0.75, 0.5)]
        )
        for attempt in range(3):
            if attempt == 1:
                strengths = [0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.85, 1.2]
                add_cubic_routes(
                    [(left, right) for left in strengths for right in strengths]
                )
            if attempt == 2:
                for vertices, codes, samples in smooth_detours(
                    display_boxes, centers, a, b, display_boundary
                ):
                    points = np.array(samples)
                    if (
                        points[:, 0].min() < ax.bbox.x0
                        or points[:, 0].max() > ax.bbox.x1
                        or points[:, 1].min() < ax.bbox.y0
                        or points[:, 1].max() > ax.bbox.y1
                    ):
                        continue
                    if any(
                        np.any(
                            (points[:, 0] >= box.x0 - 1)
                            & (points[:, 0] <= box.x1 + 1)
                            & (points[:, 1] >= box.y0 - 1)
                            & (points[:, 1] <= box.y1 + 1)
                        )
                        for box in display_boxes.values()
                    ):
                        continue
                    routes.append((20, vertices, codes, samples))
            for low, high, preferred in windows:
                for _, vertices, codes, samples in sorted(routes, key=route_score):
                    lengths = np.r_[
                        0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))
                    ]
                    fractions = lengths / max(lengths[-1], 1e-9)
                    candidates = sorted(
                        (
                            i
                            for i, fraction in enumerate(fractions)
                            if low <= fraction <= high
                        ),
                        key=lambda i: abs(fractions[i] - preferred),
                    )
                    for i in candidates:
                        point = samples[i]
                        label.set_position(
                            tuple(ax.transData.inverted().transform(point))
                        )
                        label_box = label.get_window_extent(renderer).expanded(
                            1.04, 1.08
                        )
                        if (
                            ax.bbox.contains(label_box.x0, label_box.y0)
                            and ax.bbox.contains(label_box.x1, label_box.y1)
                            and not any(label_box.overlaps(other) for other in occupied)
                        ):
                            chosen = (vertices, codes)
                            route_points = samples[::5] + [samples[-1]]
                            label_fraction = float(fractions[i])
                            occupied.append(label_box)
                            route_hits[f"{a}->{b}"] = 0
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                f"Cannot place paper count without overlap: {display(a)} -> {display(b)}"
            )
        routed_segments.extend(zip(route_points, route_points[1:], strict=False))
        vertices, codes = chosen
        single_curve = codes in (
            [PlotPath.MOVETO, PlotPath.CURVE3, PlotPath.CURVE3],
            [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4],
        )
        if not single_curve:
            if codes[0] != PlotPath.MOVETO or any(
                code != PlotPath.CURVE4 for code in codes[1:]
            ):
                raise ValueError("Arrow path contains a sharp line join")
            for i in range(3, len(vertices) - 1, 3):
                incoming = vertices[i] - vertices[i - 1]
                outgoing = vertices[i + 1] - vertices[i]
                cosine = float(
                    np.dot(incoming, outgoing)
                    / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
                )
                if cosine < 0.99999:
                    raise ValueError("Arrow path has a non-smooth tangent join")
        tangent = vertices[-1] - vertices[-2]
        toward_center = centers[b] - vertices[-1]
        alignment = float(
            np.dot(tangent, toward_center)
            / (np.linalg.norm(tangent) * np.linalg.norm(toward_center))
        )
        if alignment < 0.999:
            raise ValueError(
                f"Arrow does not point into target: {display(a)} -> {display(b)}"
            )
        route_audit[f"{a}->{b}"] = {
            "single_curve": single_curve,
            "smooth_joins": True,
            "target_alignment_cosine": alignment,
            "label_arc_fraction": label_fraction,
            "source_outer": source_outer,
            "label_in_first_third": label_fraction <= 1 / 3,
            "rounded_segments": int(codes.count(PlotPath.CURVE3) // 2),
            "single_cubic": codes == [PlotPath.MOVETO] + [PlotPath.CURVE4] * 3,
        }
        vertices = ax.transData.inverted().transform(vertices)
        path = PlotPath(vertices, codes)
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=26,
            linewidth=1.0 + 0.45 * math.log2(count),
            color=INK,
            alpha=0.85,
            zorder=1,
        )
        ax.add_patch(arrow)
        arrow.set_gid(f"relation:{a}->{b};papers={count}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "media-bias-concept-connection"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=150)
    plt.close(fig)
    payload = {
        "selection": "All shortest undirected paths between every pair of requested concepts; induced directed relations.",
        "typography": {
            "node_font_size": font_size,
            "arrow_label_font_size": font_size,
            "anchor_font_size": font_size * EMPHASIS_FONT_SCALE,
            "reference_canvas_width": FIGURE_WIDTH,
        },
        "layout": layout_report,
        "positions": {term: list(point) for term, point in coords.items()},
        "requested_concepts": sorted(anchors),
        "disconnected_pairs": disconnected,
        "box_edge_alignment": edge_alignment,
        "box_bottom_alignment": bottom_alignment,
        "arrow_geometry": route_audit,
        "routing_node_intersections": route_hits,
        "min_distinct_papers": 1,
        "nodes": terms,
        "node_labels": {t: display(t) for t in terms},
        "edges": [
            {
                "source": a,
                "target": b,
                "distinct_papers": len({c["paper_id"] for c in claims}),
                "evidence": claims,
            }
            for (a, b), claims in sorted(edges.items())
        ],
    }
    if audit_path:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
    (audit_path or base.with_suffix(".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Concept connections: {base.with_suffix('.svg')} ({len(nodes)} terms, {len(edges)} relations)"
    )


def smooth_detours(boxes, centers, source, target, boundary):
    points = []
    for box in boxes.values():
        outer = box.padded(36)
        points.extend(
            np.array(p)
            for p in [
                (outer.x0, outer.y0),
                (outer.x0, outer.y1),
                (outer.x1, outer.y0),
                (outer.x1, outer.y1),
            ]
        )

    def ports(term):
        box = boxes[term]
        x, y = centers[term]
        return [
            np.array(p)
            for p in [
                (box.x0 - 45, y),
                (box.x1 + 45, y),
                (x, box.y0 - 45),
                (x, box.y1 + 45),
            ]
        ]

    starts = list(range(len(points), len(points) + 4))
    points.extend(ports(source))
    ends = list(range(len(points), len(points) + 4))
    points.extend(ports(target))
    obstacles = [box.padded(8) for box in boxes.values()]

    def clear(p, q):
        for box in obstacles:
            low, high = 0.0, 1.0
            for axis, lower, upper in [(0, box.x0, box.x1), (1, box.y0, box.y1)]:
                delta = q[axis] - p[axis]
                if abs(delta) < 1e-9:
                    if not lower <= p[axis] <= upper:
                        low, high = 1.0, 0.0
                        break
                else:
                    u, v = sorted(
                        ((lower - p[axis]) / delta, (upper - p[axis]) / delta)
                    )
                    low, high = max(low, u), min(high, v)
            if low <= high:
                return False
        return True

    graph = {i: [] for i in range(len(points))}
    for i, j in combinations(range(len(points)), 2):
        if clear(points[i], points[j]):
            length = float(np.linalg.norm(points[i] - points[j])) + 25
            graph[i].append((j, length))
            graph[j].append((i, length))
    paths = []
    for start in starts:
        distances = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            for neighbor, length in graph[node]:
                candidate = distance + length
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        for end in ends:
            if end not in distances:
                continue
            path = [end]
            while path[-1] in previous:
                path.append(previous[path[-1]])
            polyline = [points[i] for i in reversed(path)]
            polyline.insert(0, boundary(source, polyline[0]))
            polyline.append(boundary(target, polyline[-1]))
            rounded = round_path(polyline)
            if rounded is not None:
                paths.append((distances[end], rounded))
    return [route for _, route in sorted(paths, key=lambda pair: pair[0])]


def round_path(points):
    vertices = [points[0]]
    codes = [PlotPath.MOVETO]
    samples = [points[0]]

    def cubic(c1, c2, end):
        start = vertices[-1]
        vertices.extend([c1, c2, end])
        codes.extend([PlotPath.CURVE4] * 3)
        samples.extend(
            [
                (1 - t) ** 3 * start
                + 3 * (1 - t) ** 2 * t * c1
                + 3 * (1 - t) * t * t * c2
                + t**3 * end
                for t in np.linspace(0, 1, 30)[1:]
            ]
        )

    def straight(end):
        start = vertices[-1]
        delta = end - start
        if np.linalg.norm(delta) > 1e-6:
            cubic(start + delta / 3, start + 2 * delta / 3, end)

    for before, corner, after in zip(points, points[1:], points[2:], strict=False):
        incoming, outgoing = corner - before, after - corner
        left, right = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(left, right) < 1e-6:
            continue
        if np.dot(incoming, outgoing) / (left * right) < -0.90:
            return None
        radius = min(70, left * 0.45, right * 0.45)
        entry = corner - incoming / left * radius
        exit = corner + outgoing / right * radius
        straight(entry)
        cubic(entry + 2 * (corner - entry) / 3, exit + 2 * (corner - exit) / 3, exit)
    straight(points[-1])
    return vertices, codes, samples
