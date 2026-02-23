import logging
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyCombatDecider:

    def __init__(
        self,
        cooldown_ticks=200,
        ammo_full_scale = 16,
        attack_threshold=0.55
    ):
        self.cooldown_ticks = cooldown_ticks
        self.ammo_full_scale = ammo_full_scale
        self.attack_threshold = attack_threshold

        self._build_system()

        self.last_decision = "attack"
        self.last_decision_tick = -10_000

    def _build_system(self):
        hp = ctrl.Antecedent(np.linspace(0, 1, 101), "hp")
        ammo = ctrl.Antecedent(np.linspace(0, 1, 101), "ammo")
        decision = ctrl.Consequent(np.linspace(0, 1, 101), "decision")

        # HP
        hp["low"] = fuzz.trapmf(hp.universe, [0.0, 0.0, 0.20, 0.40])
        hp["medium"] = fuzz.trimf(hp.universe, [0.25, 0.50, 0.75])
        hp["high"] = fuzz.trapmf(hp.universe, [0.60, 0.80, 1.0, 1.0])

        # AMMO
        ammo["low"] = fuzz.trapmf(ammo.universe, [0.0, 0.0, 0.20, 0.45])
        ammo["medium"] = fuzz.trimf(ammo.universe, [0.25, 0.55, 0.80])
        ammo["high"] = fuzz.trapmf(ammo.universe, [0.65, 0.85, 1.0, 1.0])

        decision["escape"] = fuzz.trapmf(decision.universe, [0.0, 0.0, 0.25, 0.45])
        decision["attack"] = fuzz.trapmf(decision.universe, [0.60, 0.75, 1.0, 1.0])

        rules = [
            ctrl.Rule(ammo["low"] & hp["low"], decision["escape"]),
            ctrl.Rule(ammo["low"] & hp["medium"], decision["escape"]),
            ctrl.Rule(ammo["medium"] & hp["low"], decision["escape"]),

            ctrl.Rule(ammo["high"] & hp["low"], decision["attack"]),
            ctrl.Rule(ammo["high"] & hp["medium"], decision["attack"]),
            ctrl.Rule(ammo["high"] & hp["high"], decision["attack"]),

            ctrl.Rule(ammo["medium"] & hp["high"], decision["attack"]),
            ctrl.Rule(ammo["medium"] & hp["medium"], decision["attack"]),

            ctrl.Rule(ammo["low"] & hp["high"], decision["attack"]),
        ]

        system = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(system)

    def decide(self, agent):
        now = int(getattr(agent, "current_tick", 0))

        #Abandon all logic and if the ammo is zero then put into the escape mode
        fn = getattr(agent, "_ammo_inventory", None)
        if fn:
            inv = fn()
            total_ammo = float(inv.get("HEAVY", 0) + inv.get("LIGHT", 0) + inv.get("LONG_DISTANCE", 0))
            if total_ammo <= 0.0:
                self.last_decision = "escape"
                self.last_decision_tick = now
                return "escape"

        # cooldown
        if (now - self.last_decision_tick) < int(self.cooldown_ticks):
            return self.last_decision

        hp_r = self._hp_ratio(agent)      
        ammo_r = self._ammo_ratio(agent)

        self._sim.input["hp"] = hp_r
        self._sim.input["ammo"] = ammo_r
        self._sim.compute()
        out = float(self._sim.output["decision"])

        decision = "attack" if out >= float(self.attack_threshold) else "escape"

        self.last_decision = decision
        self.last_decision_tick = now
        return decision

    def _hp_ratio(self, agent):
        dynamic = getattr(agent, "dynamic_info", {}) or {}
        static = getattr(agent, "static_info", {}) or {}

        hp = float(dynamic["hp"])
        max_hp = float(static["max_hp"])

        r = hp / max_hp
        if r < 0.0:
            r = 0.0
        if r > 1.0:
            r = 1.0
        return r

    def _ammo_ratio(self, agent):
        fn = getattr(agent, "_ammo_inventory", None)
       
        inv = fn()  
        total = float(inv["HEAVY"]) + float(inv["LIGHT"]) + float(inv["LONG_DISTANCE"])

        scale = float(self.ammo_full_scale)
        r = total / scale
        # clamp
        if r < 0.0:
            r = 0.0
        if r > 1.0:
            r = 1.0
        return r