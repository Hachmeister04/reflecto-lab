# ReflectoLab
![reflecto-lab](https://github.com/user-attachments/assets/06bcd75a-4bfc-4809-b314-115b375d4b1d)

ReflectoLab is a GUI to analyze plasma reflectometry signals. It is was designed to work with the ASDEX Upgrade database and with the future SPARC database.

## What you see is what Reconstruct Shot computes

The spectrograms, group delays and profiles shown are computed by the same `rpspy` functions, with the same settings,
that `Reconstruct Shot` runs. Do not reimplement a processing step in `model/shot_model.py`: call `rpspy`.

A reconstruction only computes the instants of its own time grid (Start Time, End Time, Time Step and burst size).
`Current sweep in reconstruction` says whether the sweep being shown is one of them. At any other sweep the preview is
what the reconstruction would compute if it had an instant there.

Tick `Snap to reconstruction grid` to make the sweep slider, the sweep number and the timestamp move only through those instants.

Loading a shot reads the linearization tables and the Q-HFS signal of the whole shot once, which can take a few seconds.

## Tests

```cmd
python3 -m unittest discover -s tests -v
```

compares the preview with a reconstruction on a synthetic shot. It needs the `rpspy` repository next to this one, or in `$RPSPY_PATH`.

## Contributors

Filipe Hachmeister Ferreira da Costa, IST-UL
Daniel Hachmeister Ferreira da Costa, MIT-PSFC
