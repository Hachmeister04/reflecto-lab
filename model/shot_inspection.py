"""What there is to know about a shot before it is loaded: its bands and the linearization tables for them.

No Qt here: the band selection dialog only shows what this says.
"""
from dataclasses import dataclass, field
from typing import Optional

from constants import BANDS, SIDES

# States of a band on a side
USABLE = 'usable'          # it has a raw signal and its band has linearization table entries
NO_TABLE = 'no_table'      # it has a raw signal but its band has no linearization table entries: it cannot be processed
NO_SIGNAL = 'no_signal'    # the shot has no raw signal for it


@dataclass
class ShotInspection:
    shot: int
    file_path: str
    linearization_shot: int                 # the shot whose linearization tables are checked, and then used
    raw_bands: list                         # rpspy.BandSignal for each raw signal file found in the shot folder
    table_entries: dict = field(default_factory=dict)   # band name -> number of entries in the tables ({} if unreadable)
    table_error: Optional[str] = None       # why the tables could not be read, if so

    def entries(self, band):
        """Number of entries of `band` in the linearization tables."""
        return self.table_entries.get(band, 0)

    def has_signal(self, band, side):
        return any(b.name == band and b.side == side for b in self.raw_bands)

    def state(self, band, side):
        """USABLE, NO_TABLE or NO_SIGNAL."""
        if not self.has_signal(band, side):
            return NO_SIGNAL
        return USABLE if self.entries(band) > 0 else NO_TABLE

    def usable_bands(self):
        """The bands that can be processed: raw signal and linearization entries."""
        return [b for b in self.raw_bands if self.state(b.name, b.side) == USABLE]

    def default_selection(self):
        """All the bands that can be processed, as a set of (band name, side)."""
        return {(b.name, b.side) for b in self.usable_bands()}

    def rows(self):
        """One row per band of the diagnostic: (band name, {side: state}, number of entries)."""
        return [(band, {side: self.state(band, side) for side in SIDES}, self.entries(band)) for band in BANDS]

    def bands_without_table(self):
        """Names of the bands that have a raw signal but no linearization entries."""
        return [band for band in BANDS
                if self.entries(band) == 0 and any(self.has_signal(band, side) for side in SIDES)]

    def report(self):
        """Text for the user: the bands of the shot and the entries of each band in the linearization tables."""
        signals = ', '.join(f"{b.name}-{b.side}" for b in self.raw_bands) or 'none'
        lines = [f"Shot {self.shot}. Available bands: {signals}."]
        if self.table_error is not None:
            lines.append(f"Linearization tables of shot {self.linearization_shot}: {self.table_error}")
        else:
            entries = ', '.join(
                f"{band} band - {self.entries(band)} entries" if self.entries(band) else f"{band} band - no entries available"
                for band in BANDS)
            lines.append(f"Linearization tables of shot {self.linearization_shot}: {entries}.")
        if self.linearization_shot != self.shot:
            lines.append(f"The tables of shot {self.linearization_shot} are used instead of those of shot {self.shot}.")
        return '\n'.join(lines)
