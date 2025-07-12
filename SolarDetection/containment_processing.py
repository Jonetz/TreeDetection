
from SolarDetection.utilities import compute_rectangularity
from SolarDetection.config import Config

from rtree import index
import copy
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

def filter_containment(features, containment_threshold=0.5):
    """Recursively merge all contained polygons that fulfill the thresholds."""

    config = Config()

    updated_features = copy.deepcopy(features)
    polygons = [shape(f['geometry']) for f in updated_features]
    keep_flags = [True] * len(features)
    idx = index.Index((i, poly.bounds, None) for i, poly in enumerate(polygons))

    def should_merge(conf_a, conf_b, union_poly):
        if conf_a >= 0.99 and conf_b >= 0.99:
            return True
        if conf_a >= 0.90 and conf_b >= 0.90:
            return compute_rectangularity(union_poly) >= 0.7
        return False

    def containment_ratio(poly1, poly2):
        intersection = poly1.intersection(poly2)
        if intersection.is_empty:
            return 0.0
        return intersection.area / min(poly1.area, poly2.area)

    for i, (feat_a, poly_a) in enumerate(zip(updated_features, polygons)):
        if not keep_flags[i] or poly_a.is_empty:
            continue

        # Iterative merging
        has_merged = True
        while has_merged:
            has_merged = False
            for j in list(idx.intersection(poly_a.bounds)):
                if i == j or not keep_flags[j]:
                    continue

                poly_b = polygons[j]
                if poly_b.is_empty:
                    continue

                ratio = containment_ratio(poly_a, poly_b)
                if ratio < containment_threshold:
                    continue

                conf_a = float(feat_a['properties']['Confidence_score'])
                conf_b = float(updated_features[j]['properties']['Confidence_score'])
                candidate_union = poly_a.union(poly_b)

                if should_merge(conf_a, conf_b, candidate_union):
                    poly_a = candidate_union
                    polygons[i] = poly_a
                    feat_a['geometry'] = mapping(poly_a)
                    feat_a['properties']['merged'] = True

                    keep_flags[j] = False
                    polygons[j] = shape({'type': 'Polygon', 'coordinates': []})
                    idx.delete(j, poly_b.bounds)
                    has_merged = True
                    break  # restart merging

        # Containment metadata
        num_contained = 0
        contained_geoms = []

        for j, (feat_b, poly_b) in enumerate(zip(updated_features, polygons)):
            if i == j or not keep_flags[j] or poly_b.is_empty or poly_b.area >= poly_a.area:
                continue

            inter = poly_a.intersection(poly_b)
            if inter.is_empty:
                continue

            ratio = inter.area / poly_b.area
            if ratio >= containment_threshold:
                num_contained += 1
                contained_geoms.append(poly_b)
                prev_ratio = feat_b['properties'].get('containment_ratio', 0)
                feat_b['properties']['containment_ratio'] = round(max(prev_ratio, ratio), 3)

        if contained_geoms:
            union_geom = unary_union(contained_geoms)
            inter = union_geom.intersection(poly_a)
            containment_coverage = inter.area / poly_a.area if poly_a.area > 0 else 0.0
        else:
            containment_coverage = 0.0

        # Small-panel pruning
        if (
            num_contained >= 2
            and poly_a.area < config.large_panels_threshold
            and containment_coverage > 0.7
        ):
            keep_flags[i] = False
            continue

        feat_a['properties']['num_contained'] = num_contained
        feat_a['properties']['is_contained'] = False
        feat_a['properties']['containment_coverage'] = round(containment_coverage, 3)

    # Final containment cleanup: keep only most outlying
    final_keep_flags = [True] * len(updated_features)
    sorted_indices = sorted(
        range(len(updated_features)),
        key=lambda i: updated_features[i]['properties'].get('num_contained', 0),
        reverse=True
    )

    for i in sorted_indices:
        if not final_keep_flags[i]:
            continue
        poly_i = shape(updated_features[i]['geometry'])

        for j in sorted_indices:
            if i == j or not final_keep_flags[j]:
                continue
            poly_j = shape(updated_features[j]['geometry'])
            if poly_j.is_empty or poly_j.area >= poly_i.area:
                continue

            inter = poly_i.intersection(poly_j)
            if inter.is_empty:
                continue

            ratio = inter.area / poly_j.area
            if ratio > containment_threshold:
                final_keep_flags[j] = False

    return [f for i, f in enumerate(updated_features) if final_keep_flags[i]]