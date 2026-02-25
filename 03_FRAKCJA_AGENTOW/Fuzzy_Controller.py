import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyCombatDecider:

    def __init__(
        self,
        cooldown_ticks=30,
        ammo_full_scale=16
    ):
        self.cooldown_ticks = cooldown_ticks
        self.ammo_full_scale = ammo_full_scale
        self._build_system()

        self.last_decision = "attack"
        self.last_decision_tick = -10_000

    def _build_system(self):
        hp = ctrl.Antecedent(np.linspace(0, 1, 101), "hp")
        ammo = ctrl.Antecedent(np.linspace(0, 1, 101), "ammo")
        decision = ctrl.Consequent(np.linspace(0, 1, 101), "decision")

        # HP Antecedents
        hp["low"] = fuzz.trapmf(hp.universe, [0.0, 0.0, 0.20, 0.40])
        hp["medium"] = fuzz.trimf(hp.universe, [0.25, 0.50, 0.75])
        hp["high"] = fuzz.trapmf(hp.universe, [0.60, 0.80, 1.0, 1.0])

        # AMMO Antecedents
        ammo["low"] = fuzz.trapmf(ammo.universe, [0.0, 0.0, 0.20, 0.45])
        ammo["medium"] = fuzz.trimf(ammo.universe, [0.25, 0.55, 0.80])
        ammo["high"] = fuzz.trapmf(ammo.universe, [0.65, 0.85, 1.0, 1.0])

        # DECISION Consequents
        decision["escape"] = fuzz.trapmf(decision.universe, [0.0, 0.0, 0.25, 0.40])
        decision["power_up"] = fuzz.trapmf(decision.universe, [0.30, 0.50, 0.50, 0.70])
        decision["attack"] = fuzz.trapmf(decision.universe, [0.60, 0.75, 1.0, 1.0])

        # RULES MATRIX
        rules = [
            # Critical Status -> Escape
            ctrl.Rule(hp["low"] & ammo["low"], decision["escape"]),
            ctrl.Rule(hp["low"] & ammo["medium"], decision["escape"]),

            # Deficit Status -> Power Up
            ctrl.Rule(hp["low"] & ammo["high"], decision["power_up"]),
            ctrl.Rule(hp["medium"] & ammo["low"], decision["power_up"]),
            ctrl.Rule(hp["medium"] & ammo["medium"], decision["power_up"]),
            ctrl.Rule(hp["high"] & ammo["low"], decision["power_up"]),

            # Optimal Status -> Attack
            ctrl.Rule(hp["medium"] & ammo["high"], decision["attack"]),
            ctrl.Rule(hp["high"] & ammo["medium"], decision["attack"]),
            ctrl.Rule(hp["high"] & ammo["high"], decision["attack"]),
        ]

        system = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(system)

    def decide(self, agent):
        now = int(getattr(agent, "current_tick", 0))

        # Respect cooldown to prevent command oscillation
        if (now - self.last_decision_tick) < int(self.cooldown_ticks):
            return self.last_decision

        self._sim.input["hp"] = self._hp_ratio(agent)
        self._sim.input["ammo"] = self._ammo_ratio(agent)
        self._sim.compute()
        
        out = float(self._sim.output["decision"])

        # Defuzzification
        if out <= 0.35:
            decision = "escape"
        elif out >= 0.65:
            decision = "attack"
        else:
            decision = "power_up"

        self.last_decision = decision
        self.last_decision_tick = now
        return decision

    def _hp_ratio(self, agent):
        dynamic = getattr(agent, "dynamic_info", {}) or {}
        static = getattr(agent, "static_info", {}) or {}

        hp = float(dynamic.get("hp", 0))
        max_hp = float(static.get("max_hp", 1))

        return max(0.0, min(1.0, hp / max_hp))

    def _ammo_ratio(self, agent):
        fn = getattr(agent, "_ammo_inventory", None)
        if not fn:
            return 0.0
            
        inv = fn()
        total = float(inv.get("HEAVY", 0)) + float(inv.get("LIGHT", 0)) + float(inv.get("LONG_DISTANCE", 0))

        return max(0.0, min(1.0, total / float(self.ammo_full_scale)))