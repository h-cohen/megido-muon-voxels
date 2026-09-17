"""Detector constants supplied by the engineers, 2026-09-17.

Source: docs/parameters.data, docs/detector_info.txt (filed under "# Detector 3";
the Megiddo campaign used that unit).

These values are NOT fitted. megido.validate falsification-tests each of them
against data; a failure is escalated to the engineers, never silently replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

# Lookup tables: index is bar position (0-based), value is the DAQ channel.
# detector_info.txt: "channel 28 of ASIC_0 connected to bar number 1".
ASIC_CHANNELS: dict[int, tuple[int, ...]] = {
    0: (28, 30, 3, 31, 29, 0, 24, 26, 7, 27, 25, 4, 20, 22, 11, 23, 21, 8, 16, 18, 15, 19, 17),
    1: (3, 31, 2, 0, 28, 1, 7, 27, 6, 4, 24, 5, 11, 23, 10, 8, 20, 9, 15, 19, 14, 12, 16),
    2: (19, 15, 18, 16, 12, 17, 23, 11, 22, 20, 8, 21, 27, 7, 26, 24, 4, 25, 31, 3, 30, 28, 0),
    3: (12, 14, 19, 15, 13, 16, 8, 10, 23, 11, 9, 20, 4, 6, 27, 7, 5, 24, 0, 2, 31, 3, 1),
}

ASIC_TO_LAYER: tuple[int, ...] = (1, 3, 2, 0)
LAYER_Z_CM: tuple[float, ...] = (0.0, 6.2, 31.5, 37.7)
LAYER_COORD: tuple[str, ...] = ("x", "y", "x", "y")  # bottom-up


@dataclass(frozen=True)
class BarGeometry:
    """Triangular extruded scintillator bars, interleaved apex-up / apex-down.

    Bar `b` spans [b*pitch, b*pitch + base]; consecutive bars overlap by `pitch`.
    """

    n_bars: int = 23
    base_cm: float = 3.2
    height_cm: float = 1.7
    side_cm: float = 2.3345
    length_cm: float = 40.0

    @property
    def pitch_cm(self) -> float:
        """Centre-to-centre spacing. Interleaved triangles overlap by half a base."""
        return self.base_cm / 2.0

    @property
    def active_width_cm(self) -> float:
        return (self.n_bars - 1) * self.pitch_cm + self.base_cm

    def bar_center_cm(self, bar: int) -> float:
        if not 0 <= bar < self.n_bars:
            raise ValueError(f"bar {bar} out of range 0..{self.n_bars - 1}")
        return (bar + 1) * self.pitch_cm

    def check_consistency(self) -> None:
        """side == sqrt((base/2)^2 + height^2) for an isoceles triangle."""
        expected = hypot(self.base_cm / 2.0, self.height_cm)
        if abs(expected - self.side_cm) > 1e-3:
            raise ValueError(f"side_cm {self.side_cm} inconsistent with base/height ({expected:.4f})")


@dataclass(frozen=True)
class DetectorGeometry:
    asic_channels: dict[int, tuple[int, ...]]
    asic_to_layer: tuple[int, ...]
    layer_z_cm: tuple[float, ...]
    layer_coord: tuple[str, ...]
    bar: BarGeometry = field(default_factory=BarGeometry)

    @classmethod
    def megiddo(cls) -> "DetectorGeometry":
        bar = BarGeometry()
        bar.check_consistency()
        return cls(
            asic_channels=dict(ASIC_CHANNELS),
            asic_to_layer=ASIC_TO_LAYER,
            layer_z_cm=LAYER_Z_CM,
            layer_coord=LAYER_COORD,
            bar=bar,
        )

    def __post_init__(self) -> None:
        for asic, chans in self.asic_channels.items():
            if len(set(chans)) != self.bar.n_bars:
                raise ValueError(f"ASIC {asic}: expected {self.bar.n_bars} distinct channels")

    # --- channel <-> bar -------------------------------------------------

    def channel_to_bar(self, asic: int, channel: int) -> int | None:
        """Bar index for a DAQ channel, or None if the channel is unmapped."""
        try:
            return self.asic_channels[asic].index(channel)
        except ValueError:
            return None

    def bar_to_channel(self, asic: int, bar: int) -> int:
        return self.asic_channels[asic][bar]

    def unmapped_channels(self, asic: int) -> tuple[int, ...]:
        used = set(self.asic_channels[asic])
        return tuple(c for c in range(32) if c not in used)

    # --- layer geometry --------------------------------------------------

    def layer_of_asic(self, asic: int) -> int:
        return self.asic_to_layer[asic]

    def z_of_asic(self, asic: int) -> float:
        return self.layer_z_cm[self.layer_of_asic(asic)]

    def coord_of_asic(self, asic: int) -> str:
        return self.layer_coord[self.layer_of_asic(asic)]

    def asics_for_coord(self, coord: str) -> tuple[int, int]:
        """The two ASICs measuring `coord`, ordered by ascending z."""
        asics = [a for a in range(4) if self.coord_of_asic(a) == coord]
        if len(asics) != 2:
            raise ValueError(f"expected exactly 2 ASICs for coord {coord!r}, got {asics}")
        asics.sort(key=self.z_of_asic)
        return asics[0], asics[1]

    def dz_cm(self, coord: str) -> float:
        lo, hi = self.asics_for_coord(coord)
        return self.z_of_asic(hi) - self.z_of_asic(lo)

    def max_tan(self) -> float:
        """Geometric acceptance limit: a track must cross both layers of a coordinate."""
        return self.bar.active_width_cm / min(self.dz_cm("x"), self.dz_cm("y"))
