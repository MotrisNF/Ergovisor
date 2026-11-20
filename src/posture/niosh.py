"""The revised NIOSH lifting equation, as the INSST describes it in its
technical note NTP 477 (Waters et al., NIOSH 1994).

The INSST's guide on manual handling of loads (2024 edition) lists it
first among the methods to evaluate lifting and lowering, now that the
old "INSST method" has been withdrawn.

The recommended weight limit (LPR, in English RWL) is the weight that
nearly all healthy workers could lift in those conditions for up to 8
hours without a higher risk of low back pain:

    LPR = LC x HM x VM x DM x AM x FM x CM

and the lifting index (IL) is the real weight divided by it. Every number
below is copied from NTP 477 and checked against it.
"""
from dataclasses import dataclass
from typing import Literal

Duration = Literal["corta", "moderada", "larga"]
Grip = Literal["bueno", "regular", "malo"]

# NTP 477: load constant, kg.
LC = 23.0
# NTP 477: the limits of each distance (cm) and angle (degrees).
H_MIN, H_MAX = 25.0, 63.0
V_BEST, V_MAX = 75.0, 175.0
D_MIN, D_MAX = 25.0, 175.0
A_MAX = 135.0
# NTP 477: below one lift every 5 minutes, 0.2 lifts a minute is used.
F_MIN = 0.2

# NTP 477, frequency factor (FM): lifts a minute -> for each duration,
# (V < 75 cm, V >= 75 cm). Above 15 lifts a minute it is 0.
FREQUENCY: dict[float, dict[Duration, tuple[float, float]]] = {
    0.2: {"corta": (1.00, 1.00), "moderada": (0.95, 0.95),
          "larga": (0.85, 0.85)},
    0.5: {"corta": (0.97, 0.97), "moderada": (0.92, 0.92),
          "larga": (0.81, 0.81)},
    1: {"corta": (0.94, 0.94), "moderada": (0.88, 0.88),
        "larga": (0.75, 0.75)},
    2: {"corta": (0.91, 0.91), "moderada": (0.84, 0.84),
        "larga": (0.65, 0.65)},
    3: {"corta": (0.88, 0.88), "moderada": (0.79, 0.79),
        "larga": (0.55, 0.55)},
    4: {"corta": (0.84, 0.84), "moderada": (0.72, 0.72),
        "larga": (0.45, 0.45)},
    5: {"corta": (0.80, 0.80), "moderada": (0.60, 0.60),
        "larga": (0.35, 0.35)},
    6: {"corta": (0.75, 0.75), "moderada": (0.50, 0.50),
        "larga": (0.27, 0.27)},
    7: {"corta": (0.70, 0.70), "moderada": (0.42, 0.42),
        "larga": (0.22, 0.22)},
    8: {"corta": (0.60, 0.60), "moderada": (0.35, 0.35),
        "larga": (0.18, 0.18)},
    9: {"corta": (0.52, 0.52), "moderada": (0.30, 0.30),
        "larga": (0.00, 0.15)},
    10: {"corta": (0.45, 0.45), "moderada": (0.26, 0.26),
         "larga": (0.00, 0.13)},
    11: {"corta": (0.41, 0.41), "moderada": (0.00, 0.23),
         "larga": (0.00, 0.00)},
    12: {"corta": (0.37, 0.37), "moderada": (0.00, 0.21),
         "larga": (0.00, 0.00)},
    13: {"corta": (0.00, 0.34), "moderada": (0.00, 0.00),
         "larga": (0.00, 0.00)},
    14: {"corta": (0.00, 0.31), "moderada": (0.00, 0.00),
         "larga": (0.00, 0.00)},
    15: {"corta": (0.00, 0.28), "moderada": (0.00, 0.00),
         "larga": (0.00, 0.00)},
}
# NTP 477, coupling factor (CM): grip -> (V < 75 cm, V >= 75 cm).
COUPLING: dict[Grip, tuple[float, float]] = {
    "bueno": (1.00, 1.00), "regular": (0.95, 1.00), "malo": (0.90, 0.90)}
# NTP 477: how much rest makes a duration short or moderate.
DURATIONS = {
    "corta": "1 hora o menos, seguida de un descanso de 1,2 veces ese "
             "tiempo",
    "moderada": "de 1 a 2 horas, seguida de un descanso de 0,3 veces ese "
                "tiempo",
    "larga": "más de 2 horas (hasta 8)",
}


@dataclass(frozen=True)
class Niosh:
    """The recommended weight limit of one lift and its lifting index.

    Attributes:
        hm, vm, dm, am, fm, cm: The six factors, from 0 to 1.
        limit: The recommended weight limit (LPR), kg.
        index: The lifting index (IL): the load over the limit;
            ``None`` without a load, or with a limit of 0 kg (any load
            is too much: "el límite de peso recomendado será igual a
            cero", NTP 477).
        risk: What NTP 477 says of that index; with a limit of 0 kg and
            a load given, an "incremento acusado del riesgo". Without a
            load there is no risk to name: NTP 477 reads the risk from
            the index, and the video does not show whether the hands
            held anything (raising the empty arms is a risk of posture,
            scored by REBA, not of the load).
    """

    hm: float
    vm: float
    dm: float
    am: float
    fm: float
    cm: float
    limit: float
    index: float | None
    risk: str


def horizontal(h: float) -> float:
    """HM = 25 / H; 1 up to 25 cm, 0 beyond 63 cm."""
    if h <= H_MIN:
        return 1.0
    return 0.0 if h > H_MAX else H_MIN / h


def vertical(v: float) -> float:
    """VM = 1 - 0.003 |V - 75|; 0 above 175 cm."""
    if v > V_MAX:
        return 0.0
    return max(0.0, 1 - 0.003 * abs(max(v, 0.0) - V_BEST))


def displacement(d: float) -> float:
    """DM = 0.82 + 4.5 / D; 1 below 25 cm, 0 beyond 175 cm."""
    if d < D_MIN:
        return 1.0
    return 0.0 if d > D_MAX else 0.82 + 4.5 / d


def asymmetry(a: float) -> float:
    """AM = 1 - 0.0032 A; 0 beyond 135 degrees."""
    return 0.0 if a > A_MAX else 1 - 0.0032 * abs(a)


def frequency(lifts_per_minute: float, duration: Duration,
              v: float) -> float:
    """FM from the table: the row of the next frequency at or above the
    real one (the safer side), and the column of the height."""
    rate = max(lifts_per_minute, F_MIN)
    if rate > 15:
        return 0.0
    row = min(f for f in FREQUENCY if f >= rate)
    return FREQUENCY[row][duration][0 if v < V_BEST else 1]


def coupling(grip: Grip, v: float) -> float:
    """CM from the table, by the grip and the height."""
    return COUPLING[grip][0 if v < V_BEST else 1]


def risk_of(index: float) -> str:
    """NTP 477's reading of a lifting index."""
    if index <= 1:
        return "Riesgo limitado"
    if index <= 3:
        return "Incremento moderado del riesgo"
    return "Incremento acusado del riesgo"


def evaluate(h: float, v: float, d: float, a: float,
             lifts_per_minute: float, duration: Duration, grip: Grip,
             load_kg: float | None) -> Niosh:
    """The recommended weight limit and lifting index of one position.

    Args:
        h: Horizontal distance of the hands from the ankles, cm.
        v: Height of the hands above the floor, cm.
        d: Vertical travel of the lift, cm.
        a: Twist from the front, degrees.
        lifts_per_minute: How often the task lifts.
        duration: How long the task lasts (see ``DURATIONS``).
        grip: How good the grip on the load is.
        load_kg: The weight lifted, or ``None`` if not known.
    """
    factors = (horizontal(h), vertical(v), displacement(d), asymmetry(a),
               frequency(lifts_per_minute, duration, v), coupling(grip, v))
    limit = LC
    for factor in factors:
        limit *= factor
    index = None
    if load_kg is not None and load_kg > 0 and limit > 0:
        index = load_kg / limit
    hm, vm, dm, am, fm, cm = factors
    loaded = load_kg is not None and load_kg > 0
    risk = (risk_of(index) if index is not None
            else risk_of(float("inf")) if limit <= 0 and loaded else "")
    return Niosh(hm=hm, vm=vm, dm=dm, am=am, fm=fm, cm=cm,
                 limit=round(limit, 2),
                 index=None if index is None else round(index, 2),
                 risk=risk)


# Why a factor is 0 (NTP 477), so the limit is 0 kg.
ZERO_FACTORS = {"HM": f"la carga a más de {H_MAX:.0f} cm del cuerpo",
                "VM": f"las manos a más de {V_MAX:.0f} cm del suelo",
                "DM": f"un recorrido vertical de más de {D_MAX:.0f} cm",
                "AM": f"un giro de más de {A_MAX:.0f}°",
                "FM": "una frecuencia por encima de la tabla de la NTP 477"}
