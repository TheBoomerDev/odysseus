"use client";

import * as React from "react";
import { Code2, FileText, Rocket, Wrench } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const FEATURES = [
  { icon: Code2, label: "Code Generation", desc: "Full-stack projects from descriptions" },
  { icon: FileText, label: "Auto-Documentation", desc: "PRD, specs, roadmap — automatically" },
  { icon: Wrench, label: "Smart Scaffolding", desc: "Next.js, Express, custom starters" },
  { icon: Rocket, label: "Git-Ready", desc: "Auto-init with conventional commits" },
];

export function FeatureChips() {
  return (
    <div className="mx-auto grid w-full max-w-4xl grid-cols-2 gap-3 md:grid-cols-4">
      {FEATURES.map((f) => (
        <Card key={f.label} className="border-border/60 bg-card/50">
          <CardContent className="flex flex-col items-center gap-2 p-4 text-center">
            <f.icon className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="text-xs font-semibold">{f.label}</p>
              <p className="text-[10px] text-muted-foreground">{f.desc}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
