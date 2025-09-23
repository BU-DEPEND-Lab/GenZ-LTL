"""LTL level 5."""

from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask
from safety_gymnasium.world import World


class LtlSafetyLevel5(LtlBaseTask):
    """Green, yellow, blue, magenta, cyan, red, orange, purple, lime, and teal zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.35)
        print(f"LtlSafetyLevel5, |AP| = 10")

        self._add_geoms(Zones(color='green', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='cyan', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='red', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='orange', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='purple', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='lime', size=self.zone_size, num=2, keepout=0.4))
        self._add_geoms(Zones(color='teal', size=self.zone_size, num=2, keepout=0.4))
