"""LTL level 3."""

from safety_gymnasium.assets.geoms import Zones
from safety_gymnasium.tasks.ltl.ltl_base_task import LtlBaseTask
from safety_gymnasium.world import World


class LtlSafetyLevel3(LtlBaseTask):
    """Green, yellow, blue, magenta, cyan, and red zones."""

    def __init__(self, config) -> None:
        super().__init__(config=config, zone_size=0.4)
        print(f"LtlSafetyLevel3, |AP| = 6")

        self._add_geoms(Zones(color='green', size=self.zone_size, num=2))
        self._add_geoms(Zones(color='yellow', size=self.zone_size, num=2))
        self._add_geoms(Zones(color='blue', size=self.zone_size, num=2))
        self._add_geoms(Zones(color='magenta', size=self.zone_size, num=2))
        self._add_geoms(Zones(color='cyan', size=self.zone_size, num=2))
        self._add_geoms(Zones(color='red', size=self.zone_size, num=2))
