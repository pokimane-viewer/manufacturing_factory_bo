#!/usr/bin/env python3
import json, pprint, importlib, sys, os

def cause_and_effect_graph():
    return {
        "ManufacturingProcessModules": {
            "RawMaterials": "Metals, composites, electronics",
            "IntermediateComponents": "Machined parts, subassemblies",
            "IntegrationStage": "Assembly lines, robotics"
        },
        "CADModularComponents": {
            "GeometryDefinition": "Dimensions, shape constraints",
            "SimulationAndAnalysis": "Finite element, stress tests",
            "OptimizationAndRefinement": "Iterative shape changes"
        },
        "UpgradeSolution": {
            "UnknownUpgradeRequirement": "New performance spec",
            "ProposedCADAdjustments": "Modified geometry, materials",
            "ManufacturingFeasibility": "Verifying tooling, cost"
        },
        "PerformanceVehicleModules": {
            "Propulsion": "Engines, thrusters, missile motors",
            "Avionics": "Guidance, sensor integration",
            "Aerodynamics": "Shape, control surfaces"
        },
        "TargetComposition": {
            "OperationalParameters": "Range, altitude, velocity",
            "EngagementContext": "CRT-based conflict scenario",
            "IntendedOutcome": "Successful interception/defense"
        }
    }

def logical_clauses():
    return [
        "∀u(Upgrade(u)→Supply(ManufacturingModules,BaselineSubassemblies(u)))",
        "CADSystem→(Define(GeometricConstraints)∧Simulate(Feasibility(Upgrade)))",
        "CRTAnalysis→Integrate(ConflictProbabilities,DesignPriorities(PerformanceModules))",
        "(AerodynamicsModules∧PropulsionModules∧AvionicsModules)→DependOn(TestedConfigurations(CADOutputs))",
        "(TargetComposition∧MissionScenario)→(FinalizeDesignConstraints(UnknownUpgrade)∧Meet(OperationalDemands))"
    ]

def _clauses_ascii():
    tbl=str.maketrans({"∀":"forall ","→":"->","∧":" and ","∨":" or ","¬":"not ","⊢":"|-"})
    return [c.translate(tbl) for c in logical_clauses()]

def _dbg():
    pprint.pprint({"graph":cause_and_effect_graph(),"clauses":_clauses_ascii()})

def _wrap(fn):
    def w(*a,**k):
        _dbg()
        return fn(*a,**k)
    return w

_mod=importlib.import_module("pl15_j20_sim_run")
_mod.main=_wrap(_mod.main)
_mod.run_taiwan_war_game_live=_wrap(_mod.run_taiwan_war_game_live)
_mod.run_taiwan_conflict_100v100=_wrap(_mod.run_taiwan_conflict_100v100)

if __name__=="__main__":
    if os.getenv("RUN_LIVE_VIS","0")=="1":
        _mod.run_taiwan_war_game_live()
    elif os.getenv("RUN_100v100","0")=="1":
        _mod.run_taiwan_conflict_100v100()
    else:
        _mod.main()
