"""
Simple 3D point dataclass (used by visualization and flow field).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    timestamp: float = 0.0

    def subtract(self, other: Point3D) -> Point3D:
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z, self.timestamp)

    def cross(self, other: Point3D) -> Point3D:
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            self.timestamp,
        )

    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
