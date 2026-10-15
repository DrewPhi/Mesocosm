"""Versioned, lazy height-field voxel terrain. Never uses Python's salted hash."""
import hashlib
import math

VERSION = "heightfield-1"


def noise(seed, *parts):
    raw = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def height(seed, x, z):
    return math.floor(7 + 3 * math.sin((x + seed % 97) / 19) + 2 * math.cos(z / 23) + math.sin((x + z) / 7))


def cell(seed, x, z):
    y = height(seed, x, z)
    biome = "water" if y <= 3 else "forest" if noise(seed, x // 8, z // 8) > .43 else "grassland"
    return {"x": x, "y": y, "z": z, "biome": biome}


def voxel(state, x, y, z):
    key = f"{x},{y},{z}"
    if key in state["voxels"]:
        return state["voxels"][key]
    top = height(state["seed"], x, z)
    if y <= top:
        return "stone" if y < top - 2 else "soil"
    return "water" if y <= 3 else "air"


def surface(state, x, z):
    top = height(state["seed"], x, z)
    for key, material in state["voxels"].items():
        vx, vy, vz = map(int, key.split(","))
        if vx == math.floor(x) and vz == math.floor(z) and material not in {"air", "water"}:
            top = max(top, vy)
    while voxel(state, math.floor(x), top, math.floor(z)) == "air":
        top -= 1
    return top + 1


def position(state, x, z):
    return [float(x), float(surface(state, x, z)), float(z)]


def distance(a, b):
    return math.dist(a, b)


def path(state, start, x, z):
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, z)):
        raise ValueError("Destination must be finite coordinates")
    length = math.hypot(x - start[0], z - start[2])
    if length > 200:
        raise ValueError("Choose a next destination within 200 metres")
    points = [start]
    for i in range(1, max(1, math.ceil(length)) + 1):
        fraction = i / max(1, math.ceil(length))
        px, pz = start[0] + fraction * (x - start[0]), start[2] + fraction * (z - start[2])
        point = position(state, px, pz)
        if point[1] <= 4 or abs(point[1] - points[-1][1]) > 1.5:
            raise ValueError("Direct route is obstructed by water or a steep step; choose a waypoint")
        points.append(point)
    return points
