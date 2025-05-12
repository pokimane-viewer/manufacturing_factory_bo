#!/usr/bin/env python3
import cadquery as cq

def create_manufacturing_friendly_part():
    result = (
        cq.Workplane("XY")
        .tag("base")
        .box(120, 80, 40)
        .faces(">Z").workplane()
        .rect(40, 20, forConstruction=True)
        .vertices()
        .hole(5)
        .workplaneFromTagged("base")
        .transformed(offset=(0,0,20))
        .circle(15).cutBlind(-15)
        .edges("|Z").fillet(2)
        .faces("<Z")
        .workplane()
        .rect(20, 20)
        .cutBlind(-10)
    )
    cq.exporters.export(result, "manufacturing_friendly_part.step")

if __name__ == "__main__":
    create_manufacturing_friendly_part()
