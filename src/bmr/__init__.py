"""bmr: coupled Battery (PyBaMM DFN) - Motor (PMSM) - Rotor (BEMT) system model.

The package is organised by physical subsystem:

* ``bmr.battery``    - high-fidelity DFN cell model (electrochemistry + thermal + ageing)
* ``bmr.motor``      - PMSM dq-axis motor model
* ``bmr.rotor``      - blade-element momentum theory (BEMT) rotor aerodynamics
* ``bmr.aircraft``   - airframe trim (total thrust -> per-rotor thrust)
* ``bmr.mission``    - mission profile parsing / time discretisation
* ``bmr.coupling``   - builds the battery power/current demand profile and the
                       optional bus-voltage feedback iteration
* ``bmr.postprocess``- plotting and result export
"""

__version__ = "0.1.0"
