"use client";

import * as React from "react";
import { Sparkles } from "lucide-react";
import { TopNav } from "@/components/layout/top-nav";
import { CommandPrompt } from "@/components/chat/command-prompt";
import { FeatureChips } from "@/components/home/feature-chips";
import {
  RecentTasksSection,
  type RecentProject,
} from "@/components/home/recent-tasks";

export default function HomePage() {
  const [loading, setLoading] = React.useState(false);
  const [projects, setProjects] = React.useState<RecentProject[]>([]);

  const handleSubmit = async (prompt: string) => {
    setLoading(true);
    try {
      const resp = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: prompt.slice(0, 40), prompt }),
      });
      if (resp.ok) {
        const data = await resp.json();
        // TODO: navigate to workspace
        console.log("Created project:", data.id);
      }
    } catch (err) {
      console.error("Failed to create project:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <TopNav />

      {/* Hero + Prompt */}
      <main className="mx-auto flex max-w-7xl flex-col items-center gap-12 px-6 py-20 md:py-28">
        {/* Hero */}
        <div className="flex flex-col items-center gap-4 text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-background/60 px-3 py-1 text-xs font-medium text-muted-foreground backdrop-blur">
            <Sparkles className="h-3 w-3" />
            Powered by multi-agent AI
          </span>
          <h1 className="max-w-2xl text-4xl font-semibold tracking-tight md:text-6xl">
            Describe it.
            <br />
            <span className="bg-gradient-to-r from-foreground to-foreground/50 bg-clip-text text-transparent">
              Watch it build.
            </span>
          </h1>
          <p className="max-w-lg text-base leading-relaxed text-muted-foreground md:text-lg">
            Generate full-stack projects from natural language. Code, docs, and
            git — all in one step.
          </p>
        </div>

        {/* Command Prompt */}
        <CommandPrompt onSubmit={handleSubmit} loading={loading} />

        {/* Feature Chips */}
        <FeatureChips />

        {/* Recent Projects */}
        <RecentTasksSection
          projects={projects}
          onOpen={(id) => console.log("Open:", id)}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 text-xs text-muted-foreground">
          <span>Odysseus — Open Source AI Builder</span>
          <span>MIT License</span>
        </div>
      </footer>
    </div>
  );
}
